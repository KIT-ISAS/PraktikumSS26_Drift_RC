#!/usr/bin/env python3
"""
Manual test script for the ESP32 driftcar controller.

Each test describes what it does, the expected Serial output on the ESP32,
and the expected LED behaviour. Watch both the Serial monitor and the LED
while running.

NOTE: The car only moves if the BOOT button has been pressed to ARM it.
      Most tests are safe to run unarmed; arm only when physically safe.

Usage:
    python test_esp32.py <ESP32_IP> [test ...]

    python test_esp32.py 192.168.1.42            # interactive menu
    python test_esp32.py 192.168.1.42 all        # run every test in order
    python test_esp32.py 192.168.1.42 timeout reconnect
"""

import argparse
import json
import socket
import struct
import sys
import time

PORT = 12345
COMMAND_TIMEOUT_S = 0.5   # must match firmware COMMAND_TIMEOUT_MS
SEND_RATE_HZ = 20


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

class Session:
    """One TCP connection to the ESP32."""

    def __init__(self, ip: str, port: int = PORT):
        self.ip = ip
        self.port = port
        self._sock: socket.socket | None = None
        self._seq = 0

    def connect(self, timeout: float = 3.0) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((self.ip, self.port))
        s.settimeout(None)
        self._sock = s
        print(f"  [connected to {self.ip}:{self.port}]")

    def send(
        self,
        steering: float = 0.0,
        throttle: float = 0.0,
        gyro_mode: str = "soft",
        gyro_gain: float = 0.0,
    ) -> None:
        self._seq += 1
        msg = {
            "steering": steering,
            "throttle": throttle,
            "gyro_mode": gyro_mode,
            "gyro_gain": gyro_gain,
            "seq": self._seq,
        }
        self._send_raw((json.dumps(msg) + "\n").encode())

    def send_raw(self, data: bytes) -> None:
        """Send arbitrary bytes — used to provoke parser errors."""
        self._send_raw(data)

    def _send_raw(self, data: bytes) -> None:
        if self._sock is None:
            raise RuntimeError("Not connected.")
        self._sock.sendall(data)

    def close(self) -> None:
        """Graceful close (FIN)."""
        if self._sock:
            self._sock.close()
            self._sock = None
            print("  [disconnected gracefully]")

    def rst(self) -> None:
        """Abrupt close (RST — no FIN, no TIME_WAIT)."""
        if self._sock:
            # SO_LINGER with l_onoff=1, l_linger=0 → RST on close
            self._sock.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_LINGER,
                struct.pack("ii", 1, 0),
            )
            self._sock.close()
            self._sock = None
            print("  [disconnected abruptly (RST)]")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def stream(session: Session, duration_s: float, **kwargs) -> None:
    """Send commands at SEND_RATE_HZ for duration_s seconds."""
    interval = 1.0 / SEND_RATE_HZ
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        t = time.monotonic()
        session.send(**kwargs)
        elapsed = time.monotonic() - t
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)


def header(title: str, what: str, serial: str, led: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  TEST: {title}")
    print("=" * width)
    print(f"  What : {what}")
    print(f"  Serial: {serial}")
    print(f"  LED  : {led}")
    print("-" * width)
    input("  Press Enter to run...")


def pause(msg: str = "  Press Enter to continue...") -> None:
    input(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_normal(ip: str) -> None:
    header(
        "normal — steady command stream",
        "Stream steering sweep + throttle=0 at 20 Hz for 3 s.",
        "No new output (state already Receiving after first message).",
        "Solid ON if armed, RapidBlink if not armed.",
    )
    s = Session(ip)
    s.connect()
    print("  Streaming 3 s …")
    for i in range(SEND_RATE_HZ * 3):
        steering = (i % (SEND_RATE_HZ * 2)) / (SEND_RATE_HZ * 2) * 2 - 1
        s.send(steering=steering, throttle=0.0, gyro_mode="soft", gyro_gain=0.3)
        time.sleep(1 / SEND_RATE_HZ)
    s.close()


def test_timeout(ip: str) -> None:
    header(
        "timeout — stop sending mid-stream",
        f"Stream 1 s, pause {COMMAND_TIMEOUT_S * 1000:.0f}+ ms, resume 2 s.",
        '"State: PeerConnected" after pause, "State: Receiving" on resume.',
        "DoublePulse during pause, then solid/rapid again.",
    )
    s = Session(ip)
    s.connect()
    print("  Phase 1: streaming 1 s …")
    stream(s, 1.0)
    print(f"  Phase 2: silence {COMMAND_TIMEOUT_S + 0.3:.1f} s …")
    time.sleep(COMMAND_TIMEOUT_S + 0.3)
    print("  Phase 3: streaming 2 s …")
    stream(s, 2.0)
    s.close()


def test_graceful_disconnect(ip: str) -> None:
    header(
        "graceful disconnect — clean FIN",
        "Stream 1 s then close the socket normally.",
        '"Peer disconnected." → "State: WifiConnected".',
        "Drops from solid/rapid → FastBlink.",
    )
    s = Session(ip)
    s.connect()
    stream(s, 1.0)
    s.close()


def test_abrupt_disconnect(ip: str) -> None:
    header(
        "abrupt disconnect — RST (no FIN)",
        "Stream 1 s then force a TCP reset.",
        "ESP32 detects broken connection on next read/write attempt. "
        '"Peer disconnected." may be delayed until it tries to read.',
        "Same as graceful but may take one loop iteration longer.",
    )
    s = Session(ip)
    s.connect()
    stream(s, 1.0)
    s.rst()


def test_reconnect(ip: str) -> None:
    header(
        "reconnect — disconnect then reconnect",
        "Stream, disconnect, wait 1 s, reconnect, stream again.",
        '"Peer disconnected." then "Peer connected." + "State: Receiving".',
        "FastBlink for ~1 s, then back to solid/rapid.",
    )
    s = Session(ip)
    s.connect()
    stream(s, 1.0)
    s.close()
    print("  Waiting 1 s …")
    time.sleep(1.0)
    s.connect()
    stream(s, 2.0)
    s.close()


def test_invalid_json(ip: str) -> None:
    header(
        "invalid JSON",
        'Send "{bad json}\\n" — not valid JSON.',
        '"JSON parse error: {bad json}" — no state change, command ignored.',
        "No LED change (state stays wherever it was).",
    )
    s = Session(ip)
    s.connect()
    stream(s, 0.5)   # get into Receiving first
    print("  Sending invalid JSON …")
    s.send_raw(b"{bad json}\n")
    time.sleep(0.2)
    stream(s, 1.0)   # confirm stream still works after the bad line
    s.close()


def test_invalid_gyro_mode(ip: str) -> None:
    header(
        "invalid gyro_mode",
        'Send gyro_mode="turbo" — not "soft" or "hard".',
        '"Invalid gyro_mode." — command rejected, no state change.',
        "No LED change.",
    )
    s = Session(ip)
    s.connect()
    stream(s, 0.5)
    print("  Sending bad gyro_mode …")
    bad = json.dumps({"steering": 0, "throttle": 0, "gyro_mode": "turbo",
                      "gyro_gain": 0, "seq": 999}) + "\n"
    s.send_raw(bad.encode())
    time.sleep(0.2)
    stream(s, 1.0)
    s.close()


def test_out_of_range(ip: str) -> None:
    header(
        "out-of-range values",
        "Send steering=5.0, throttle=-3.0, gyro_gain=2.0.",
        "No error — firmware clamps to ±1 / [0,1] silently.",
        "No LED change, outputs clamped to limits.",
    )
    s = Session(ip)
    s.connect()
    stream(s, 0.5)
    print("  Sending out-of-range values …")
    for _ in range(10):
        s.send(steering=5.0, throttle=-3.0, gyro_mode="soft", gyro_gain=2.0)
        time.sleep(0.05)
    stream(s, 1.0)
    s.close()


def test_overflow(ip: str) -> None:
    header(
        "RX buffer overflow (>512 bytes without newline)",
        "Send 600 bytes without \\n, then flush with \\n.",
        '"RX overflow, clearing buffer." — the oversized line is discarded.',
        "No LED change. Stream remains functional afterwards.",
    )
    s = Session(ip)
    s.connect()
    stream(s, 0.5)
    print("  Sending 600-byte line without newline …")
    s.send_raw(b"x" * 600)
    time.sleep(0.1)
    print("  Sending newline to flush …")
    s.send_raw(b"\n")
    time.sleep(0.1)
    stream(s, 1.0)
    s.close()


def test_no_newline(ip: str) -> None:
    header(
        "partial message — valid JSON, no trailing newline",
        "Send complete JSON without \\n, wait 0.8 s, then send \\n.",
        "No command processed until \\n arrives. State stays PeerConnected "
        "during the wait, then transitions to Receiving.",
        "DoublePulse while waiting, then solid/rapid after \\n.",
    )
    s = Session(ip)
    s.connect()
    # First get into Receiving, then let it time out so we start fresh
    stream(s, 0.5)
    time.sleep(COMMAND_TIMEOUT_S + 0.1)   # drop back to PeerConnected
    msg = json.dumps({"steering": 0, "throttle": 0,
                      "gyro_mode": "soft", "gyro_gain": 0, "seq": 1})
    print("  Sending JSON without newline …")
    s.send_raw(msg.encode())
    print(f"  Waiting {COMMAND_TIMEOUT_S + 0.3:.1f} s (state should stay PeerConnected) …")
    time.sleep(COMMAND_TIMEOUT_S + 0.3)
    print("  Sending newline — command should now be processed …")
    s.send_raw(b"\n")
    time.sleep(0.1)
    stream(s, 1.0)
    s.close()


def test_rapid_cycles(ip: str) -> None:
    header(
        "rapid connect/disconnect cycles",
        "10 × connect → 2 commands → RST, 100 ms apart.",
        "Rapid 'Peer connected.' / 'Peer disconnected.' churn. No crash.",
        "Flickers between FastBlink and DoublePulse/solid.",
    )
    for i in range(10):
        print(f"  Cycle {i + 1}/10 …")
        s = Session(ip)
        try:
            s.connect(timeout=2.0)
            s.send()
            s.send()
            s.rst()
        except OSError as e:
            print(f"  [connect failed: {e}]")
        time.sleep(0.1)
    print("  Done — give the ESP32 a moment to settle …")
    time.sleep(1.0)


def test_double_connect(ip: str) -> None:
    header(
        "second client while first is still connected",
        "Connect client A, stream, then connect client B simultaneously.",
        "ESP32 queues client B (TCP backlog). Client B gets no data read "
        "until client A disconnects. Then 'Peer connected.' for B.",
        "LED stays solid/rapid while A is streaming. Switches to A-based "
        "state only. After A disconnects, B takes over.",
    )
    a = Session(ip)
    a.connect()
    stream(a, 1.0)
    print("  Connecting second client while first is active …")
    b = Session(ip)
    try:
        b.connect(timeout=2.0)
        print("  Client B connected (queued in TCP backlog).")
        print("  Streaming from A for 1 s (B ignored by ESP32) …")
        stream(a, 1.0)
        print("  Disconnecting A …")
        a.close()
        time.sleep(0.2)
        print("  Streaming from B (ESP32 should now accept it) …")
        stream(b, 1.5)
        b.close()
    except OSError as e:
        print(f"  Client B failed to connect: {e}")
        a.close()


# ---------------------------------------------------------------------------
# Registry + runner
# ---------------------------------------------------------------------------

TESTS: dict[str, tuple[str, object]] = {
    "normal":             ("Steady 20 Hz command stream",            test_normal),
    "timeout":            ("Stop sending → failsafe after 500 ms",   test_timeout),
    "graceful_disconnect":("Clean TCP FIN disconnect",                test_graceful_disconnect),
    "abrupt_disconnect":  ("TCP RST disconnect",                      test_abrupt_disconnect),
    "reconnect":          ("Disconnect then reconnect",               test_reconnect),
    "invalid_json":       ("Malformed JSON line",                     test_invalid_json),
    "invalid_gyro_mode":  ("Unknown gyro_mode string",                test_invalid_gyro_mode),
    "out_of_range":       ("Values outside [-1, 1]",                  test_out_of_range),
    "overflow":           ("RX buffer overflow (>512 b, no newline)", test_overflow),
    "no_newline":         ("Valid JSON without trailing newline",      test_no_newline),
    "rapid_cycles":       ("10 rapid connect/disconnect cycles",      test_rapid_cycles),
    "double_connect":     ("Two clients simultaneously",              test_double_connect),
}


def print_menu() -> None:
    print("\nAvailable tests:")
    for i, (name, (desc, _)) in enumerate(TESTS.items(), 1):
        print(f"  {i:2d}. {name:<25} {desc}")
    print()


def run_test(name: str, ip: str) -> None:
    fn = TESTS[name][1]
    try:
        fn(ip)
        print(f"\n  ✓ {name} done.")
    except KeyboardInterrupt:
        print(f"\n  [interrupted]")
    except OSError as e:
        print(f"\n  [connection error: {e}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ip", help="ESP32 IP address")
    parser.add_argument("tests", nargs="*",
                        help="Test name(s) or 'all'. Omit for interactive menu.")
    args = parser.parse_args()

    if args.tests and args.tests[0] == "all":
        for name in TESTS:
            run_test(name, args.ip)
        return

    if args.tests:
        for name in args.tests:
            if name not in TESTS:
                print(f"Unknown test '{name}'. Available: {', '.join(TESTS)}")
                sys.exit(1)
            run_test(name, args.ip)
        return

    # Interactive menu
    print_menu()
    while True:
        choice = input("Enter test name (or number, 'all', 'q'): ").strip()
        if choice in ("q", "quit", "exit"):
            break
        if choice == "all":
            for name in TESTS:
                run_test(name, args.ip)
            continue
        if choice.isdigit():
            idx = int(choice) - 1
            names = list(TESTS.keys())
            if 0 <= idx < len(names):
                choice = names[idx]
            else:
                print("  Out of range.")
                continue
        if choice in TESTS:
            run_test(choice, args.ip)
        else:
            print(f"  Unknown: '{choice}'")


if __name__ == "__main__":
    main()
