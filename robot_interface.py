from __future__ import annotations

"""PC/Raspberry Pi side interface for the Arduino Nano rescue robot.

This module is intentionally outside the SNN/network code. The SNN should see
only normalized observations and integer actions; this file owns serial I/O,
raw sensor parsing, motor commands, Bluetooth/serial I/O, and calibration.

Expected Arduino JSON lines
---------------------------
Sensor line example:
    {"ok":true,"front_mm":312,"left_mm":180,"right_mm":640,
     "r":120,"g":32,"b":25,"c":440,"color":"blue","sound_raw":612}
Action ack example:
    {"ok":true,"action":"forward","elapsed_ms":520,"collision":false}
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import json
import time

try:
    import serial  # type: ignore
except Exception:  # pragma: no cover
    serial = None  # type: ignore


ACTION_ID_TO_NAME = {
    0: "forward",
    1: "turn_left",
    2: "turn_right",
    3: "stay",
}


@dataclass
class RobotCalibration:
    """Values that must be measured on the real robot/map."""

    port: str = "/dev/ttyUSB0"
    # HC-06 default baudrate is usually 9600. Keep this matched with Arduino Serial.begin(...).
    baudrate: int = 9600
    timeout_s: float = 1.0

    # ToF distance normalization range.
    clear_min_mm: float = 40.0
    clear_max_mm: float = 450.0

    # Safety stop threshold. This does not choose a replacement action.
    obstacle_stop_mm: float = 90.0

    # Color/victim detection.
    victim_color_names: Tuple[str, ...] = ("red", "blue", "green")
    min_clear_channel: int = 40

    # DFR0034 analog sound sensor normalization.
    enable_sound_sensor: bool = True
    sound_raw_min: int = 350
    sound_raw_max: int = 800

    # Allow serial line settling after open/reset.
    arduino_boot_wait_s: float = 2.0


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def normalize_distance_mm(distance_mm: float, cal: RobotCalibration) -> float:
    """Convert raw distance to the encoder's 0..1 clearance convention."""
    if distance_mm is None or distance_mm <= 0:
        return 0.0
    span = max(1e-9, float(cal.clear_max_mm - cal.clear_min_mm))
    return _clip01((float(distance_mm) - cal.clear_min_mm) / span)


def normalize_sound_raw(sound_raw: float, cal: RobotCalibration) -> float:
    """Convert DFR0034 analog peak/average value to 0..1 sound_signal."""
    if not cal.enable_sound_sensor:
        return 0.0
    span = max(1e-9, float(cal.sound_raw_max - cal.sound_raw_min))
    return _clip01((float(sound_raw) - float(cal.sound_raw_min)) / span)


@dataclass
class RobotRawReading:
    front_mm: float
    left_mm: float
    right_mm: float
    r: int = 0
    g: int = 0
    b: int = 0
    c: int = 0
    color: str = "unknown"
    sound_raw: int = 0
    raw: Optional[Dict[str, Any]] = None

    def to_observation(self, cal: RobotCalibration) -> Dict[str, float]:
        color_name = str(self.color).lower().strip()
        color_confident = int(self.c) >= int(cal.min_clear_channel)
        victim_colors = {str(name).lower().strip() for name in cal.victim_color_names}
        victim_signal = 1.0 if color_confident and color_name in victim_colors else 0.0
        return {
            "front_clearance": normalize_distance_mm(self.front_mm, cal),
            "left_clearance": normalize_distance_mm(self.left_mm, cal),
            "right_clearance": normalize_distance_mm(self.right_mm, cal),
            "victim_signal": victim_signal,
            "sound_signal": normalize_sound_raw(self.sound_raw, cal),
        }


class ArduinoRobotInterface:
    """JSON-line serial interface used by real_robot_env.py."""

    def __init__(self, cal: Optional[RobotCalibration] = None) -> None:
        self.cal = RobotCalibration() if cal is None else cal
        if serial is None:
            raise RuntimeError("pyserial is not installed. Install with: pip install pyserial")
        self.ser = serial.Serial(
            self.cal.port,
            self.cal.baudrate,
            timeout=self.cal.timeout_s,
        )
        time.sleep(self.cal.arduino_boot_wait_s)
        self.flush()

    def flush(self) -> None:
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass

    def _write_json(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, separators=(",", ":")) + "\n"
        self.ser.write(line.encode("utf-8"))
        self.ser.flush()

    def _read_json_line(self) -> Dict[str, Any]:
        deadline = time.time() + self.cal.timeout_s
        last = b""
        while time.time() < deadline:
            line = self.ser.readline()
            if not line:
                continue
            last = line
            try:
                return json.loads(line.decode("utf-8", errors="replace").strip())
            except json.JSONDecodeError:
                continue
        raise TimeoutError(f"No valid JSON line from Arduino. Last line={last!r}")

    def request(self, cmd: str, **kwargs: Any) -> Dict[str, Any]:
        self._write_json({"cmd": cmd, **kwargs})
        resp = self._read_json_line()
        if not bool(resp.get("ok", False)):
            raise RuntimeError(f"Arduino command failed: cmd={cmd}, response={resp}")
        return resp

    def read_sensors(self) -> RobotRawReading:
        resp = self.request("read")
        return RobotRawReading(
            front_mm=float(resp.get("front_mm", 0.0)),
            left_mm=float(resp.get("left_mm", 0.0)),
            right_mm=float(resp.get("right_mm", 0.0)),
            r=int(resp.get("r", 0)),
            g=int(resp.get("g", 0)),
            b=int(resp.get("b", 0)),
            c=int(resp.get("c", 0)),
            color=str(resp.get("color", "unknown")),
            sound_raw=int(resp.get("sound_raw", 0)),
            raw=resp,
        )

    def get_observation(self) -> Dict[str, float]:
        return self.read_sensors().to_observation(self.cal)

    def execute_action(self, action: int) -> Dict[str, Any]:
        action_name = ACTION_ID_TO_NAME[int(action)]
        return self.request("action", action=action_name)

    def beep(self, duration_ms: int = 80) -> None:
        self.request("beep", duration_ms=int(duration_ms))
