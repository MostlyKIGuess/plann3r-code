#!/usr/bin/env python3
"""Record a RealSense traversal into the image layout used by Plann3r mapping.

Runs on the robot machine while the robot is driven by teleop. It saves resized RGB
frames to <out-dir>/images/NNNNNN.jpg at --hz, one JSON line per frame with the
P3DX odometry and last cmd_vel to frames.jsonl, and map_meta.json on exit
(plus goal.json when --goal-frame is given). See docs/real-world.md.

Run inside the robot container:

    python3 real_world/record_realsense_map.py --out-dir /data/plann3r_real/my_map --hz 1.5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image


class RealSenseMapRecorder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.bridge = CvBridge()
        self.lock = Lock()
        self.latest_rgb = None
        self.latest_rgb_stamp = None
        self.latest_depth = None
        self.latest_depth_stamp = None
        self.latest_odom = None
        self.latest_cmd = None
        self.frame_idx = 0
        self.last_save_time = 0.0

        self.out_dir = Path(args.out_dir).expanduser().resolve()
        self.image_dir = self.out_dir / "images"
        self.depth_dir = self.out_dir / "depth"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        if args.save_depth:
            self.depth_dir.mkdir(parents=True, exist_ok=True)

        self.frames_path = self.out_dir / "frames.jsonl"
        mode = "a" if args.append else "w"
        self.frames_file = self.frames_path.open(mode, encoding="utf-8")

        rospy.Subscriber(args.rgb_topic, Image, self.rgb_callback, queue_size=1, buff_size=2**24)
        if args.save_depth:
            rospy.Subscriber(args.depth_topic, Image, self.depth_callback, queue_size=1, buff_size=2**24)
        if args.odom_topic:
            rospy.Subscriber(args.odom_topic, Odometry, self.odom_callback, queue_size=10)
        if args.cmd_topic:
            rospy.Subscriber(args.cmd_topic, Twist, self.cmd_callback, queue_size=10)

    def rgb_callback(self, msg: Image) -> None:
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        with self.lock:
            self.latest_rgb = rgb.copy()
            self.latest_rgb_stamp = msg.header.stamp.to_sec()

    def depth_callback(self, msg: Image) -> None:
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with self.lock:
            self.latest_depth = depth.copy()
            self.latest_depth_stamp = msg.header.stamp.to_sec()

    def odom_callback(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        with self.lock:
            self.latest_odom = {
                "stamp": msg.header.stamp.to_sec(),
                "position": [p.x, p.y, p.z],
                "orientation_xyzw": [q.x, q.y, q.z, q.w],
            }

    def cmd_callback(self, msg: Twist) -> None:
        with self.lock:
            self.latest_cmd = {
                "linear": [msg.linear.x, msg.linear.y, msg.linear.z],
                "angular": [msg.angular.x, msg.angular.y, msg.angular.z],
            }

    def should_save(self, now: float) -> bool:
        if self.args.max_frames > 0 and self.frame_idx >= self.args.max_frames:
            return False
        return now - self.last_save_time >= 1.0 / max(float(self.args.hz), 1e-6)

    def save_latest(self) -> bool:
        with self.lock:
            if self.latest_rgb is None:
                return False
            rgb = self.latest_rgb.copy()
            rgb_stamp = self.latest_rgb_stamp
            depth = None if self.latest_depth is None else self.latest_depth.copy()
            depth_stamp = self.latest_depth_stamp
            odom = self.latest_odom
            cmd = self.latest_cmd

        if self.args.width > 0 and self.args.height > 0:
            rgb = cv2.resize(rgb, (self.args.width, self.args.height), interpolation=cv2.INTER_AREA)

        image_name = f"{self.frame_idx:06d}.jpg"
        image_path = self.image_dir / image_name
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality)])

        depth_name = None
        if self.args.save_depth and depth is not None:
            if self.args.width > 0 and self.args.height > 0:
                depth = cv2.resize(depth, (self.args.width, self.args.height), interpolation=cv2.INTER_NEAREST)
            depth_name = f"{self.frame_idx:06d}.png"
            cv2.imwrite(str(self.depth_dir / depth_name), depth)

        record = {
            "frame_idx": self.frame_idx,
            "stamp": rgb_stamp,
            "image": f"images/{image_name}",
            "depth": f"depth/{depth_name}" if depth_name else None,
            "depth_stamp": depth_stamp,
            "odom": odom,
            "cmd": cmd,
        }
        self.frames_file.write(json.dumps(record) + "\n")
        self.frames_file.flush()

        self.frame_idx += 1
        self.last_save_time = time.time()
        return True

    def write_meta(self) -> None:
        meta = {
            "created_unix": time.time(),
            "rgb_topic": self.args.rgb_topic,
            "depth_topic": self.args.depth_topic if self.args.save_depth else None,
            "odom_topic": self.args.odom_topic,
            "cmd_topic": self.args.cmd_topic,
            "frame_count": self.frame_idx,
            "image_dir": "images",
            "depth_dir": "depth" if self.args.save_depth else None,
            "width": self.args.width,
            "height": self.args.height,
            "hz": self.args.hz,
            "goal": {
                "frame": self.args.goal_frame,
                "pixel": [self.args.goal_pixel_x, self.args.goal_pixel_y],
            },
        }
        (self.out_dir / "map_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if self.args.goal_frame >= 0:
            goal = {
                "image_idx": int(self.args.goal_frame),
                "pixel_x": int(self.args.goal_pixel_x),
                "pixel_y": int(self.args.goal_pixel_y),
            }
            (self.out_dir / "goal.json").write_text(json.dumps(goal, indent=2), encoding="utf-8")

    def run(self) -> None:
        rate = rospy.Rate(max(float(self.args.poll_hz), 1.0))
        rospy.loginfo("Recording Plann3r map to %s", self.out_dir)
        while not rospy.is_shutdown():
            now = time.time()
            if self.should_save(now):
                saved = self.save_latest()
                if saved and self.frame_idx % 10 == 0:
                    rospy.loginfo("Recorded %d frames", self.frame_idx)
                if self.args.max_frames > 0 and self.frame_idx >= self.args.max_frames:
                    rospy.loginfo("Reached max_frames=%d", self.args.max_frames)
                    break
            rate.sleep()
        self.write_meta()
        self.frames_file.close()
        rospy.loginfo("Saved map metadata to %s", self.out_dir / "map_meta.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Output map directory. Must contain/create images/.")
    parser.add_argument("--rgb-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--odom-topic", default="/RosAria/pose")
    parser.add_argument("--cmd-topic", default="/RosAria/cmd_vel")
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--hz", type=float, default=1.5, help="Frame recording rate.")
    parser.add_argument("--poll-hz", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--append", action="store_true", help="Append to frames.jsonl instead of overwriting it.")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--goal-frame", type=int, default=-1, help="Optional goal image index for propagation mapping.")
    parser.add_argument("--goal-pixel-x", type=int, default=160)
    parser.add_argument("--goal-pixel-y", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rospy.init_node("plann3r_map_recorder", anonymous=False)
    RealSenseMapRecorder(args).run()


if __name__ == "__main__":
    main()
