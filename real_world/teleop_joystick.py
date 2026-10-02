#!/usr/bin/env python3
"""Direct joystick teleop for P3DX/RosAria.

Reads Linux joystick events from /dev/input/js0 and publishes geometry_msgs/Twist.
Default mapping is Xbox 360 style:
  left stick Y  axis 1 -> linear.x
  right stick X axis 3 -> angular.z
Used to drive the robot while recording a map, and as a manual override during
live navigation (--idle-no-publish). See docs/real-world.md.

Run inside the robot container:

    real_world/teleop both --device /dev/input/js0 --cmd-topic /RosAria/cmd_vel
"""

from __future__ import annotations

import argparse
import contextlib
import os
import select
import struct
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from typing import Dict

import rospy
from geometry_msgs.msg import Twist

JS_EVENT_FORMAT = "IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80


@dataclass
class JoystickState:
    axes: Dict[int, float] = field(default_factory=dict)
    buttons: Dict[int, int] = field(default_factory=dict)
    center: Dict[int, float] = field(default_factory=dict)


def apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0
    scaled = (abs(value) - deadzone) / max(1.0 - deadzone, 1e-6)
    return scaled if value > 0 else -scaled


def remove_center(value: float, center: float) -> float:
    value = max(-1.0, min(1.0, value))
    center = max(-0.95, min(0.95, center))
    if value >= center:
        return max(0.0, min(1.0, (value - center) / max(1.0 - center, 1e-6)))
    return min(0.0, max(-1.0, (value - center) / max(center + 1.0, 1e-6)))


@contextlib.contextmanager
def raw_terminal(enabled: bool):
    if not enabled:
        yield
        return
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class JoystickTeleop:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.state = JoystickState()
        self.last_event_time = time.time()
        self.keyboard_linear = 0.0
        self.keyboard_angular = 0.0
        self.last_keyboard_time = 0.0
        self.was_publishing = False
        self.pub = rospy.Publisher(args.cmd_topic, Twist, queue_size=1)

    def read_event(self, dev) -> bool:
        data = dev.read(JS_EVENT_SIZE)
        if len(data) != JS_EVENT_SIZE:
            return False

        _, value, event_type, number = struct.unpack(JS_EVENT_FORMAT, data)
        event_type = event_type & ~JS_EVENT_INIT
        if event_type == JS_EVENT_AXIS:
            self.state.axes[int(number)] = max(-1.0, min(1.0, float(value) / 32767.0))
            self.last_event_time = time.time()
        elif event_type == JS_EVENT_BUTTON:
            self.state.buttons[int(number)] = int(value)
            self.last_event_time = time.time()
        return True

    def drain_events(self, dev) -> bool:
        saw_event = False
        while True:
            ready, _, _ = select.select([dev], [], [], 0.0)
            if not ready:
                return saw_event
            saw_event = self.read_event(dev) or saw_event

    def calibrate(self, dev) -> None:
        if self.args.calibrate_seconds <= 0:
            return

        rospy.loginfo("Calibrating joystick center for %.2fs; leave sticks untouched", self.args.calibrate_seconds)
        end_time = time.time() + float(self.args.calibrate_seconds)
        while time.time() < end_time and not rospy.is_shutdown():
            self.drain_events(dev)
            time.sleep(0.01)

        for axis in {self.args.linear_axis, self.args.angular_axis}:
            self.state.center[axis] = self.state.axes.get(axis, 0.0)
        rospy.loginfo("Joystick center offsets: %s", self.state.center)

    def read_keyboard(self) -> None:
        if not self.args.keyboard:
            return

        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                return
            key = sys.stdin.read(1).lower()
            if key in {"w", "i"}:
                self.keyboard_linear = 1.0
                self.keyboard_angular = 0.0
            elif key in {"s", "k"}:
                self.keyboard_linear = -1.0
                self.keyboard_angular = 0.0
            elif key in {"a", "j"}:
                self.keyboard_linear = 0.0
                self.keyboard_angular = 1.0
            elif key in {"d", "l"}:
                self.keyboard_linear = 0.0
                self.keyboard_angular = -1.0
            elif key in {" ", "x"}:
                self.keyboard_linear = 0.0
                self.keyboard_angular = 0.0
            elif key in {"q", "\x03"}:
                rospy.signal_shutdown("keyboard requested shutdown")
            else:
                continue
            self.last_keyboard_time = time.time()

    def build_twist(self) -> Twist:
        linear_raw = remove_center(
            self.state.axes.get(self.args.linear_axis, 0.0),
            self.state.center.get(self.args.linear_axis, 0.0),
        )
        angular_raw = remove_center(
            self.state.axes.get(self.args.angular_axis, 0.0),
            self.state.center.get(self.args.angular_axis, 0.0),
        )

        linear = apply_deadzone(linear_raw, self.args.deadzone)
        angular = apply_deadzone(angular_raw, self.args.deadzone)

        if self.args.invert_linear:
            linear *= -1.0
        if self.args.invert_angular:
            angular *= -1.0

        if self.args.stale_timeout > 0 and time.time() - self.last_event_time > self.args.stale_timeout:
            linear = 0.0
            angular = 0.0

        if self.args.keyboard and time.time() - self.last_keyboard_time <= self.args.keyboard_timeout:
            linear = self.keyboard_linear
            angular = self.keyboard_angular

        twist = Twist()
        twist.linear.x = max(-self.args.max_v, min(self.args.max_v, linear * self.args.max_v))
        twist.angular.z = max(-self.args.max_w, min(self.args.max_w, angular * self.args.max_w))

        if self.args.enable_button >= 0 and self.state.buttons.get(self.args.enable_button, 0) == 0:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        return twist

    def should_publish(self, twist: Twist) -> bool:
        if not self.args.idle_no_publish:
            return True
        is_active = abs(twist.linear.x) > 1e-6 or abs(twist.angular.z) > 1e-6
        return is_active or self.was_publishing

    def run(self) -> None:
        if not os.path.exists(self.args.device):
            raise FileNotFoundError(f"Joystick device not found: {self.args.device}")

        rospy.loginfo(
            "Joystick teleop %s -> %s linear_axis=%d angular_axis=%d",
            self.args.device,
            self.args.cmd_topic,
            self.args.linear_axis,
            self.args.angular_axis,
        )
        if self.args.enable_button >= 0:
            rospy.loginfo("Hold button %d to enable motion", self.args.enable_button)
        if self.args.keyboard:
            rospy.loginfo("Keyboard enabled: w/s forward/back, a/d turn, space stop, q quit")

        with open(self.args.device, "rb", buffering=0) as dev:
            self.drain_events(dev)
            self.calibrate(dev)
            rate = rospy.Rate(float(self.args.hz))
            last_log = 0.0
            with raw_terminal(self.args.keyboard):
                try:
                    while not rospy.is_shutdown():
                        self.drain_events(dev)
                        self.read_keyboard()

                        twist = self.build_twist()
                        if self.should_publish(twist):
                            self.pub.publish(twist)
                            self.was_publishing = abs(twist.linear.x) > 1e-6 or abs(twist.angular.z) > 1e-6
                        else:
                            self.was_publishing = False

                        now = time.time()
                        if now - last_log > self.args.log_period:
                            rospy.loginfo(
                                "teleop cmd: v=%.3f w=%.3f axes=%s center=%s",
                                twist.linear.x,
                                twist.angular.z,
                                self.state.axes,
                                self.state.center,
                            )
                            last_log = now
                        rate.sleep()
                finally:
                    self.pub.publish(Twist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/input/js0")
    parser.add_argument("--cmd-topic", default="/RosAria/cmd_vel")
    parser.add_argument("--linear-axis", type=int, default=1, help="Default: left stick vertical on Xbox 360.")
    parser.add_argument("--angular-axis", type=int, default=3, help="Default: right stick horizontal on Xbox 360.")
    parser.add_argument("--invert-linear", action="store_true", default=True)
    parser.add_argument("--no-invert-linear", dest="invert_linear", action="store_false")
    parser.add_argument("--invert-angular", action="store_true", default=True)
    parser.add_argument("--no-invert-angular", dest="invert_angular", action="store_false")
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--max-v", type=float, default=0.20)
    parser.add_argument("--max-w", type=float, default=0.75)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--stale-timeout", type=float, default=0.0)
    parser.add_argument("--enable-button", type=int, default=-1, help="Optional deadman button index. -1 disables.")
    parser.add_argument("--calibrate-seconds", type=float, default=0.75)
    parser.add_argument("--keyboard", action="store_true", help="Also accept keyboard teleop from this terminal.")
    parser.add_argument("--keyboard-timeout", type=float, default=0.35)
    parser.add_argument(
        "--idle-no-publish",
        action="store_true",
        help="Do not publish neutral commands while idle; publish one stop after active teleop.",
    )
    parser.add_argument("--log-period", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rospy.init_node("plann3r_joystick_teleop", anonymous=False)
    JoystickTeleop(args).run()


if __name__ == "__main__":
    main()
