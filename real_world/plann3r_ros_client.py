#!/usr/bin/env python3
"""ROS client that sends RealSense RGB frames to a Plann3r server and publishes cmd_vel.

Runs on the robot machine (ROS Noetic, no torch). Each cycle it JPEG-encodes the
latest RGB frame, sends it with the latest odometry to the server's /predict
endpoint, scales and clips the returned (v, w), publishes it as a Twist, holds it
for --execute-cmd-time seconds, then publishes a stop. It drops stale frames,
odometry and responses, and exits once the server reports the goal is within
--goal-distance-threshold metres. See docs/real-world.md.

Run inside the robot container (start with --dry-run):

    python3 real_world/plann3r_ros_client.py --server-url $PLANN3R_SERVER_URL \
        --require-odom --dry-run
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from threading import Lock

import cv2
import requests
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from PIL import ImageDraw
from sensor_msgs.msg import Image


class Plann3rRosClient:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.bridge = CvBridge()
        self.lock = Lock()
        self.latest_rgb = None
        self.latest_stamp = None
        self.latest_odom = None
        self.seq = 0
        self.last_ok_time = 0.0
        self.vis_dir = Path(args.vis_dir).expanduser().resolve() if args.vis_dir else None
        self.vis_counter = 0
        if self.vis_dir is not None:
            self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.cmd_pub = rospy.Publisher(args.cmd_topic, Twist, queue_size=1)
        rospy.Subscriber(args.rgb_topic, Image, self.rgb_callback, queue_size=1, buff_size=2**24)
        if args.odom_topic:
            rospy.Subscriber(args.odom_topic, Odometry, self.odom_callback, queue_size=10)

        self.log_file = None
        if args.log_jsonl:
            log_path = Path(args.log_jsonl).expanduser().resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = log_path.open("a", encoding="utf-8")

    def rgb_callback(self, msg: Image) -> None:
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        with self.lock:
            self.latest_rgb = rgb.copy()
            self.latest_stamp = msg.header.stamp.to_sec()

    def odom_callback(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        with self.lock:
            self.latest_odom = {
                "stamp": msg.header.stamp.to_sec(),
                "position": [p.x, p.y, p.z],
                "orientation_xyzw": [q.x, q.y, q.z, q.w],
            }

    def encode_latest(self):
        with self.lock:
            if self.latest_rgb is None:
                return None, None, None
            rgb = self.latest_rgb.copy()
            stamp = self.latest_stamp

        if self.args.width > 0 and self.args.height > 0:
            rgb = cv2.resize(rgb, (self.args.width, self.args.height), interpolation=cv2.INTER_AREA)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality)])
        if not ok:
            raise RuntimeError("Failed to JPEG-encode RGB frame")
        return base64.b64encode(encoded.tobytes()).decode("ascii"), stamp, rgb

    def latest_odom_copy(self):
        with self.lock:
            odom = self.latest_odom
        return dict(odom) if odom is not None else None

    def latest_stamp_copy(self):
        with self.lock:
            return self.latest_stamp

    @staticmethod
    def age_seconds(stamp):
        if stamp is None:
            return None
        stamp = float(stamp)
        if stamp <= 0.0:
            return None
        now = rospy.Time.now().to_sec()
        if now <= 0.0:
            now = time.time()
        age = now - stamp
        if age < -1.0:
            return None
        return max(0.0, age)

    def publish_stop(self) -> None:
        if self.args.dry_run:
            rospy.loginfo("dry-run stop cmd")
            return
        self.cmd_pub.publish(Twist())

    def publish_cmd(self, v: float, w: float) -> dict:
        twist = Twist()
        scaled_v = float(v) * float(self.args.linear_scale)
        scaled_w = float(w) * float(self.args.angular_scale) * float(self.args.angular_sign)
        twist.linear.x = max(min(scaled_v, self.args.max_v), -self.args.max_v)
        twist.angular.z = max(min(scaled_w, self.args.max_w), -self.args.max_w)
        cmd = {
            "server_v": float(v),
            "server_w": float(w),
            "published_v": float(twist.linear.x),
            "published_w": float(twist.angular.z),
            "angular_sign": float(self.args.angular_sign),
            "dry_run": bool(self.args.dry_run),
            "published": not bool(self.args.dry_run),
        }
        if self.args.dry_run:
            rospy.loginfo("dry-run cmd: v=%.3f w=%.3f", twist.linear.x, twist.angular.z)
            return cmd
        self.cmd_pub.publish(twist)
        return cmd

    def stop_record(self, reason: str) -> dict:
        if self.args.stop_on_error:
            self.publish_stop()
        return {
            "server_v": 0.0,
            "server_w": 0.0,
            "published_v": 0.0,
            "published_w": 0.0,
            "angular_sign": float(self.args.angular_sign),
            "dry_run": bool(self.args.dry_run),
            "published": bool(self.args.stop_on_error) and not bool(self.args.dry_run),
            "stop_reason": reason,
        }

    def request_prediction(self, rgb_b64: str, stamp: float, odom) -> dict:
        payload = {
            "seq": self.seq,
            "stamp": stamp,
            "rgb_jpeg_b64": rgb_b64,
        }
        if odom is not None:
            payload["odom"] = odom
        response = requests.post(
            self.args.server_url.rstrip("/") + "/predict",
            json=payload,
            timeout=float(self.args.timeout),
        )
        response.raise_for_status()
        return response.json()

    def reset_server(self) -> None:
        if not self.args.reset_server_on_start:
            return
        with self.lock:
            odom = self.latest_odom
        payload = {}
        if odom is not None:
            payload["odom"] = odom
        if self.args.reset_map_frame >= 0:
            payload["map_frame"] = int(self.args.reset_map_frame)
        try:
            response = requests.post(
                self.args.server_url.rstrip("/") + "/reset",
                json=payload,
                timeout=float(self.args.timeout),
            )
            response.raise_for_status()
            rospy.loginfo("Reset Plann3r server: %s", response.json())
        except Exception as exc:  # pylint: disable=broad-except
            rospy.logwarn("Could not reset Plann3r server: %s", exc)

    def log_result(self, record: dict) -> None:
        if self.log_file is None:
            return
        self.log_file.write(json.dumps(record) + "\n")
        self.log_file.flush()

    def maybe_save_visualization(
        self,
        rgb,
        result,
        cmd,
        frame_age_s,
        odom_age_s,
        request_latency_s,
        command_age_s,
        reason="",
    ) -> None:
        if self.vis_dir is None or int(self.args.vis_every) <= 0:
            return

        self.vis_counter += 1
        if (self.vis_counter - 1) % int(self.args.vis_every) != 0:
            return

        panel_size = (320, 240)
        query_panel = PILImage.fromarray(rgb).resize(panel_size)
        draw = ImageDraw.Draw(query_panel)
        draw.rectangle((0, 0, panel_size[0], 18), fill=(0, 0, 0))
        draw.text((4, 3), f"sent query seq={self.seq}", fill=(255, 255, 255))

        debug_panel = PILImage.new("RGB", panel_size, (250, 250, 250))
        draw = ImageDraw.Draw(debug_panel)
        debug = result.get("debug", {}) if isinstance(result, dict) else {}
        lines = [
            f"seq: {self.seq}",
            f"reason: {reason or 'ok'}",
            f"frame_age_s: {frame_age_s if frame_age_s is not None else 'na'}",
            f"odom_age_s: {odom_age_s if odom_age_s is not None else 'na'}",
            f"request_latency_s: {request_latency_s if request_latency_s is not None else 'na'}",
            f"command_age_s: {command_age_s if command_age_s is not None else 'na'}",
            f"execute_cmd_time_s: {self.args.execute_cmd_time}",
            f"server_query_stamp: {debug.get('query_stamp', 'na')}",
            f"server v,w: {cmd.get('server_v'):.3f}, {cmd.get('server_w'):.3f}",
            f"pub v,w: {cmd.get('published_v'):.3f}, {cmd.get('published_w'):.3f}",
            f"published: {cmd.get('published')} dry_run: {cmd.get('dry_run')}",
            f"nearest: {debug.get('nearest_frame', 'na')} anchor: {debug.get('anchor_frame', 'na')}",
            f"server_vis: {debug.get('vis_path', 'na')}",
        ]
        draw.rectangle((0, 0, panel_size[0], 18), fill=(0, 0, 0))
        draw.text((4, 3), "client command debug", fill=(255, 255, 255))
        for i, line in enumerate(lines):
            draw.text((8, 28 + i * 18), str(line)[:82], fill=(0, 0, 0))

        canvas = PILImage.new("RGB", (panel_size[0] * 2, panel_size[1]), (255, 255, 255))
        canvas.paste(query_panel, (0, 0))
        canvas.paste(debug_panel, (panel_size[0], 0))

        out_path = self.vis_dir / f"client_vis_{self.vis_counter:06d}_seq_{self.seq:06d}.jpg"
        canvas.save(out_path, quality=90)
        canvas.save(self.vis_dir / "latest.jpg", quality=90)

    def run(self) -> None:
        rate = rospy.Rate(float(self.args.hz))
        rospy.loginfo("Plann3r ROS client sending %s -> %s -> %s", self.args.rgb_topic, self.args.server_url, self.args.cmd_topic)
        self.reset_server()

        while not rospy.is_shutdown():
            try:
                rgb_b64, stamp, rgb = self.encode_latest()
                if rgb_b64 is None:
                    rate.sleep()
                    continue

                frame_age_s = self.age_seconds(stamp)
                if (
                    frame_age_s is not None
                    and self.args.max_frame_age > 0.0
                    and frame_age_s > self.args.max_frame_age
                ):
                    cmd = self.stop_record("stale_rgb_frame")
                    rospy.logwarn(
                        "Skipping stale RGB frame: age=%.3fs max=%.3fs",
                        frame_age_s,
                        self.args.max_frame_age,
                    )
                    self.maybe_save_visualization(rgb, None, cmd, frame_age_s, None, None, frame_age_s, "stale_rgb_frame")
                    rate.sleep()
                    continue

                odom = self.latest_odom_copy()
                odom_age_s = self.age_seconds(odom.get("stamp")) if odom is not None else None
                if odom is None and self.args.require_odom:
                    cmd = self.stop_record("missing_odom")
                    rospy.logwarn("Skipping command because --require-odom is set and no odom has arrived")
                    self.maybe_save_visualization(rgb, None, cmd, frame_age_s, None, None, frame_age_s, "missing_odom")
                    rate.sleep()
                    continue
                if (
                    odom_age_s is not None
                    and self.args.max_odom_age > 0.0
                    and odom_age_s > self.args.max_odom_age
                ):
                    cmd = self.stop_record("stale_odom")
                    rospy.logwarn(
                        "Skipping stale odom: age=%.3fs max=%.3fs",
                        odom_age_s,
                        self.args.max_odom_age,
                    )
                    self.maybe_save_visualization(rgb, None, cmd, frame_age_s, odom_age_s, None, frame_age_s, "stale_odom")
                    rate.sleep()
                    continue

                request_start = time.time()
                result = self.request_prediction(rgb_b64, stamp, odom)
                request_latency_s = time.time() - request_start
                command_age_s = self.age_seconds(stamp)
                latest_stamp_after_response = self.latest_stamp_copy()

                if result.get("seq") is not None and int(result.get("seq")) != int(self.seq):
                    cmd = self.stop_record("seq_mismatch")
                    rospy.logwarn("Dropping response with seq=%s for client seq=%d", result.get("seq"), self.seq)
                    self.maybe_save_visualization(
                        rgb,
                        result,
                        cmd,
                        frame_age_s,
                        odom_age_s,
                        request_latency_s,
                        command_age_s,
                        "seq_mismatch",
                    )
                    self.seq += 1
                    rate.sleep()
                    continue

                result_debug = result.get("debug", {})
                result_query_stamp = result_debug.get("query_stamp")
                if result_query_stamp is not None and stamp is not None and abs(float(result_query_stamp) - float(stamp)) > 1e-3:
                    cmd = self.stop_record("query_stamp_mismatch")
                    rospy.logwarn(
                        "Dropping response with query_stamp=%s for sent stamp=%.6f",
                        result_query_stamp,
                        float(stamp),
                    )
                    self.maybe_save_visualization(
                        rgb,
                        result,
                        cmd,
                        frame_age_s,
                        odom_age_s,
                        request_latency_s,
                        command_age_s,
                        "query_stamp_mismatch",
                    )
                    self.seq += 1
                    rate.sleep()
                    continue

                if (
                    command_age_s is not None
                    and self.args.max_response_age > 0.0
                    and command_age_s > self.args.max_response_age
                ):
                    cmd = self.stop_record("stale_response")
                    rospy.logwarn(
                        "Dropping stale Plann3r response: image-to-command age=%.3fs max=%.3fs",
                        command_age_s,
                        self.args.max_response_age,
                    )
                    self.maybe_save_visualization(
                        rgb,
                        result,
                        cmd,
                        frame_age_s,
                        odom_age_s,
                        request_latency_s,
                        command_age_s,
                        "stale_response",
                    )
                    self.log_result(
                        {
                            "seq": self.seq,
                            "stamp": stamp,
                            "result": result,
                            "cmd": cmd,
                            "frame_age_s": frame_age_s,
                            "odom_age_s": odom_age_s,
                            "request_latency_s": request_latency_s,
                            "command_age_s": command_age_s,
                        }
                    )
                    self.seq += 1
                    rate.sleep()
                    continue

                debug = result.get("debug", {})
                goal_distance_m = debug.get("retrieval", {}).get("goal_distance_m")
                if (
                    goal_distance_m is not None
                    and self.args.goal_distance_threshold > 0.0
                    and float(goal_distance_m) <= float(self.args.goal_distance_threshold)
                ):
                    cmd = self.stop_record("goal_reached")
                    rospy.loginfo(
                        "Reached goal: odom distance %.3fm <= %.3fm",
                        float(goal_distance_m),
                        float(self.args.goal_distance_threshold),
                    )
                    self.maybe_save_visualization(
                        rgb,
                        result,
                        cmd,
                        frame_age_s,
                        odom_age_s,
                        request_latency_s,
                        command_age_s,
                        "goal_reached",
                    )
                    self.log_result(
                        {
                            "seq": self.seq,
                            "stamp": stamp,
                            "latest_stamp_after_response": latest_stamp_after_response,
                            "result": result,
                            "cmd": cmd,
                            "frame_age_s": frame_age_s,
                            "odom_age_s": odom_age_s,
                            "request_latency_s": request_latency_s,
                            "command_age_s": command_age_s,
                        }
                    )
                    break

                v = float(result.get("v", 0.0))
                w = float(result.get("w", 0.0))
                cmd = self.publish_cmd(v, w)
                self.last_ok_time = time.time()
                self.maybe_save_visualization(
                    rgb,
                    result,
                    cmd,
                    frame_age_s,
                    odom_age_s,
                    request_latency_s,
                    command_age_s,
                )
                self.log_result(
                    {
                        "seq": self.seq,
                        "stamp": stamp,
                        "latest_stamp_after_response": latest_stamp_after_response,
                        "result": result,
                        "cmd": cmd,
                        "frame_age_s": frame_age_s,
                        "odom_age_s": odom_age_s,
                        "request_latency_s": request_latency_s,
                        "command_age_s": command_age_s,
                    }
                )

                rospy.loginfo(
                    "seq=%d server_v=%.3f server_w=%.3f pub_v=%.3f pub_w=%.3f age=%.3fs latency=%.3fs nearest=%s anchor=%s goal_dist=%s",
                    self.seq,
                    v,
                    w,
                    cmd["published_v"],
                    cmd["published_w"],
                    command_age_s if command_age_s is not None else -1.0,
                    request_latency_s,
                    debug.get("nearest_frame", "na"),
                    debug.get("anchor_frame", "na"),
                    goal_distance_m if goal_distance_m is not None else "na",
                )
                if not self.args.dry_run and self.args.execute_cmd_time > 0.0:
                    rospy.sleep(float(self.args.execute_cmd_time))
                    self.publish_stop()
                self.seq += 1
            except Exception as exc:  # pylint: disable=broad-except
                rospy.logwarn("Plann3r request failed: %s", exc)
                if self.args.stop_on_error:
                    self.publish_stop()
            rate.sleep()

        self.publish_stop()
        if self.log_file is not None:
            self.log_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-url",
        default=os.environ.get("PLANN3R_SERVER_URL", ""),
        help="Plann3r server URL, for example http://<gpu-host>:8088. Defaults to $PLANN3R_SERVER_URL.",
    )
    parser.add_argument("--rgb-topic", default="/camera/color/image_raw")
    parser.add_argument("--odom-topic", default="/RosAria/pose")
    parser.add_argument("--cmd-topic", default="/RosAria/cmd_vel")
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-v", type=float, default=0.20)
    parser.add_argument("--max-w", type=float, default=0.60)
    parser.add_argument("--linear-scale", type=float, default=3.0)
    parser.add_argument("--angular-scale", type=float, default=3.0)
    parser.add_argument("--angular-sign", type=float, default=-1.0, help="Multiply server angular output before publishing. ROS uses -1.0 for the current controller convention.")
    parser.add_argument("--max-frame-age", type=float, default=0.75, help="Drop RGB frames older than this many seconds. <=0 disables.")
    parser.add_argument("--max-response-age", type=float, default=1.50, help="Drop commands if image-to-command age exceeds this. <=0 disables.")
    parser.add_argument("--max-odom-age", type=float, default=0.75, help="Drop stale odometry when odom is present. <=0 disables.")
    parser.add_argument("--execute-cmd-time", type=float, default=0.35, help="After each prediction, execute that command for this many seconds, stop, then send the next image. <=0 disables the stop/sleep cycle.")
    parser.add_argument("--goal-distance-threshold", type=float, default=1.0, help="Stop and exit once server-reported odometry distance to the goal is within this many meters. <=0 disables.")
    parser.add_argument("--require-odom", action="store_true", help="Do not publish nav commands until fresh odometry is available.")
    parser.add_argument("--vis-dir", default="", help="Optional robot-side client visualization directory.")
    parser.add_argument("--vis-every", type=int, default=0, help="Save one client visualization every N requests. 0 disables.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-server-on-start", dest="reset_server_on_start", action="store_true", default=True)
    parser.add_argument("--no-reset-server-on-start", dest="reset_server_on_start", action="store_false")
    parser.add_argument("--reset-map-frame", type=int, default=-1, help="Map frame corresponding to robot pose at startup.")
    parser.add_argument("--stop-on-error", dest="stop_on_error", action="store_true", default=True)
    parser.add_argument("--no-stop-on-error", dest="stop_on_error", action="store_false")
    parser.add_argument("--log-jsonl", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.server_url:
        raise SystemExit("Set --server-url or the PLANN3R_SERVER_URL environment variable.")
    rospy.init_node("plann3r_ros_client", anonymous=False)
    Plann3rRosClient(args).run()


if __name__ == "__main__":
    main()
