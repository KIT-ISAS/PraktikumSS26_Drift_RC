#!/usr/bin/env python3
"""
Manual test script for the ESP32 driftcar controller (UDP protocol).

Each test describes what it does, the expected Serial output on the ESP32,
and the expected LED behaviour. Watch both the Serial monitor and the LED
while running.

NOTE: The car only moves if the BOOT button has been pressed to ARM it.
      Most tests are safe to run unarmed; arm only when physically safe.

Usage:
    python test_esp32.py <ESP32_IP> [test ...]

    python test_esp32.py 10.42.0.99            # interactive menu
    python test_esp32.py 10.42.0.99 all        # run every test in order
    python test_esp32.py 10.42.0.99 timeout status
"""

import argparse
import json
import math
import socket
import sys
import time

PORT = 12345
COMMAND_TIMEOUT_S = 0.20  # must match firmware COMMAND_TIMEOUT_MS
SEND_RATE_HZ = 20


# ---------------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------------

class Session:
    """UDP session for sending commands and receiving status from the ESP32."""

    def __init__(self, ip: str, port: int = PORT):
        self.ip = ip
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.1)
        self._seq = 0
        print(f"  [UDP socket open, will send to {ip}:{port}]")

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
        self._sock.sendto(json.dumps(msg).encode(), (self.ip, self.port))

    def send_raw(self, data: bytes) -> None:
        """Send arbitrary bytes — used to provoke parser errors."""
        self._sock.sendto(data, (self.ip, self.port))

    def recv_status(self) -> dict | None:
        """Return parsed status dict if a reply arrived, else None."""
        try:
            data, _ = self._sock.recvfrom(512)
            return json.loads(data.decode())
        except (socket.timeout, json.JSONDecodeError):
            return None

    def wait_for_status(self, timeout_s: float = 2.0) -> dict | None:
        """Block until a status arrives or timeout, then return it."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            s = self.recv_status()
            if s is not None:
                return s
        return None

    def close(self) -> None:
        self._sock.close()
        print("  [UDP socket closed]")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def stream(
    session: Session,
    duration_s: float,
    print_status: bool = False,
    **kwargs,
) -> None:
    """Send commands at SEND_RATE_HZ for duration_s seconds."""
    interval = 1.0 / SEND_RATE_HZ
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        t = time.monotonic()
        session.send(**kwargs)
        status = session.recv_status()
        if print_status and status:
            print(f"    ESP32: {status}")
        remaining = interval - (time.monotonic() - t)
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


def probe_bad_packet(ip: str, label: str, probe_fn) -> None:
    """Pre-stream 0.5 s, call probe_fn(session), post-stream 1 s."""
    s = Session(ip)
    stream(s, 0.5)
    print(f"  {label} …")
    probe_fn(s)
    stream(s, 1.0)
    s.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_normal(ip: str) -> None:
    header(
        "normal — full range sweep",
        "Runs four phases at 20 Hz (~32 s total):\n"
        "  1) Steering sweep -1→+1→-1 at throttle=0\n"
        "  2) Forward ramp  throttle 0→1→0, steering centered\n"
        "  3) Brake/reverse ramp  throttle 0→-1→0, steering centered\n"
        "  4) Combined: steering sweep while throttle oscillates",
        "No Serial output after initial 'State: Receiving'.",
        "Solid ON if armed, RapidBlink if not armed.",
    )
    s = Session(ip)
    interval = 1.0 / SEND_RATE_HZ

    def lerp(a: float, b: float, x: float) -> float:
        return a + (b - a) * max(0.0, min(1.0, x))

    def run_phase(label: str, duration_s: float, steering_fn, throttle_fn) -> None:
        print(f"  {label} ({duration_s:.0f} s) …")
        next_send = time.monotonic()
        t_start = time.monotonic()
        while True:
            now = time.monotonic()
            elapsed = now - t_start
            if elapsed >= duration_s:
                break
            s.send(
                steering=steering_fn(elapsed),
                throttle=throttle_fn(elapsed),
                gyro_mode="soft",
                gyro_gain=0.0,
            )
            next_send += interval
            remaining = next_send - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

    run_phase(
        "Phase 1: steering sweep", 8.0,
        lambda t: lerp(-1.0, 1.0, t / 4.0) if t < 4.0 else lerp(1.0, -1.0, (t - 4.0) / 4.0),
        lambda _: 0.0,
    )
    run_phase(
        "Phase 2: forward throttle ramp", 8.0,
        lambda _: 0.0,
        lambda t: lerp(0.0, 1.0, t / 4.0) if t < 4.0 else lerp(1.0, 0.0, (t - 4.0) / 4.0),
    )
    # ESC requires brake→neutral→brake again to arm reverse.
    # Phase 3a: brake to full, 3b: back to neutral (blip), 3c: ramp into reverse and back.
    run_phase(
        "Phase 3a: brake to arm", 2.0,
        lambda _: 0.0,
        lambda t: lerp(0.0, -1.0, t / 2.0),
    )
    run_phase(
        "Phase 3b: neutral blip", 0.3,
        lambda _: 0.0,
        lambda _: 0.0,
    )
    run_phase(
        "Phase 3c: reverse ramp", 5.7,
        lambda _: 0.0,
        lambda t: lerp(0.0, -1.0, t / 2.85) if t < 2.85 else lerp(-1.0, 0.0, (t - 2.85) / 2.85),
    )
    run_phase(
        "Phase 4: combined steering + throttle", 8.0,
        lambda t: math.sin(2 * math.pi * t / 4.0),
        lambda t: math.sin(2 * math.pi * t / 8.0) * 0.5,
    )

    s.close()


def test_status(ip: str) -> None:
    header(
        "status — verify bidirectional UDP",
        "Stream commands for 3 s and print every status reply from the ESP32.",
        "'State: Receiving' on first command. Status at ~10 Hz.",
        "Solid ON if armed, RapidBlink if not armed.",
    )
    s = Session(ip)
    stream(s, 3.0, print_status=True)
    s.close()


def test_timeout(ip: str) -> None:
    header(
        "timeout — stop sending mid-stream",
        f"Stream 1 s, pause {COMMAND_TIMEOUT_S * 1000:.0f}+ ms, resume 2 s.",
        '"State: WifiConnected" after pause, "State: Receiving" on resume.',
        "FastBlink during pause, then solid/rapid again.",
    )
    s = Session(ip)
    print("  Phase 1: streaming 1 s …")
    stream(s, 1.0)
    print(f"  Phase 2: silence {COMMAND_TIMEOUT_S + 0.3:.1f} s …")
    time.sleep(COMMAND_TIMEOUT_S + 0.3)
    print("  Phase 3: streaming 2 s …")
    stream(s, 2.0)
    s.close()


def test_reconnect(ip: str) -> None:
    header(
        "reconnect — new sender socket after silence",
        "Stream 1 s from socket A, close it, wait 1 s, open socket B, stream 2 s.",
        '"State: WifiConnected" during silence, "State: Receiving" on resume.\n'
        "ESP32 switches status target to socket B automatically.",
        "FastBlink for ~1 s, then back to solid/rapid.",
    )
    a = Session(ip)
    print("  Socket A: streaming 1 s …")
    stream(a, 1.0)
    a.close()
    print("  Waiting 1 s …")
    time.sleep(1.0)
    b = Session(ip)
    print("  Socket B: streaming 2 s …")
    stream(b, 2.0, print_status=True)
    b.close()


def test_invalid_json(ip: str) -> None:
    header(
        "invalid JSON",
        'Send one datagram with "{bad json}" — not valid JSON.',
        '"JSON parse error." — no state change, command ignored.',
        "No LED change (state stays wherever it was).",
    )
    def probe(s):
        s.send_raw(b"{bad json}")
        time.sleep(0.2)
    probe_bad_packet(ip, "Sending invalid JSON datagram", probe)


def test_invalid_gyro_mode(ip: str) -> None:
    header(
        "invalid gyro_mode",
        'Send gyro_mode="turbo" — not "soft" or "hard".',
        '"Invalid gyro_mode." — command rejected, no state change.',
        "No LED change.",
    )
    def probe(s):
        s.send_raw(json.dumps({"steering": 0, "throttle": 0, "gyro_mode": "turbo",
                               "gyro_gain": 0, "seq": 999}).encode())
        time.sleep(0.2)
    probe_bad_packet(ip, "Sending bad gyro_mode", probe)


def test_out_of_range(ip: str) -> None:
    header(
        "out-of-range values",
        "Send steering=5.0, throttle=-3.0, gyro_gain=2.0.",
        "No error — firmware clamps to ±1 / [0,1] silently.",
        "No LED change, outputs clamped to limits.",
    )
    def probe(s):
        for _ in range(10):
            s.send(steering=5.0, throttle=-3.0, gyro_mode="soft", gyro_gain=2.0)
            time.sleep(0.05)
    probe_bad_packet(ip, "Sending out-of-range values", probe)


def test_overflow(ip: str) -> None:
    header(
        "RX overflow — datagram >511 bytes",
        "Send a 600-byte datagram. Firmware discards it (too large for rxBuf).",
        '"Packet too large, discarded." — no state change.',
        "No LED change. Stream functional afterwards.",
    )
    def probe(s):
        s.send_raw(b"x" * 600)
        time.sleep(0.1)
    probe_bad_packet(ip, "Sending 600-byte datagram", probe)


def test_multiple_senders(ip: str) -> None:
    header(
        "multiple senders — two sockets compete",
        "Socket A and B alternate sends. ESP32 always replies to whichever\n"
        "sent the most recent datagram (no connection state).",
        "No state change. Status target alternates between A and B.",
        "Solid ON if armed, RapidBlink if not armed.",
    )
    a = Session(ip)
    b = Session(ip)
    interval = 1.0 / SEND_RATE_HZ
    end = time.monotonic() + 4.0
    toggle = True
    while time.monotonic() < end:
        t = time.monotonic()
        if toggle:
            a.send()
        else:
            b.send()
        toggle = not toggle
        # Both sockets try to recv; only the current target will get replies
        sa = a.recv_status()
        sb = b.recv_status()
        if sa:
            print(f"    A got: {sa}")
        if sb:
            print(f"    B got: {sb}")
        remaining = interval - (time.monotonic() - t)
        if remaining > 0:
            time.sleep(remaining)
    a.close()
    b.close()


def test_wifi_loss(ip: str) -> None:
    header(
        "WiFi loss simulation",
        "Stream 2 s, then power-cycle the WiFi AP or move out of range.\n"
        "Resume sending when AP is back.",
        '"State: NoWifi" during outage, "State: Receiving" on reconnect.',
        "SlowBlink during outage, solid/rapid after reconnect.",
    )
    s = Session(ip)
    print("  Streaming 2 s …")
    stream(s, 2.0, print_status=True)
    pause("  Now disable the WiFi AP. Press Enter when done …")
    print("  Waiting 4 s (ESP32 should detect loss and show SlowBlink) …")
    time.sleep(4.0)
    pause("  Re-enable the AP. Press Enter when ready to resume streaming …")
    print("  Streaming 5 s (ESP32 should reconnect and re-enter Receiving) …")
    stream(s, 5.0, print_status=True)
    s.close()


# ---------------------------------------------------------------------------
# Registry + runner
# ---------------------------------------------------------------------------

TESTS: dict[str, tuple[str, object]] = {
    "normal":           ("Steady 20 Hz command stream, full sweep",     test_normal),
    "status":           ("Print bidirectional status replies at 10 Hz", test_status),
    "timeout":          ("Stop sending → failsafe → resume",            test_timeout),
    "reconnect":        ("New sender socket after silence",             test_reconnect),
    "invalid_json":     ("Malformed JSON datagram",                     test_invalid_json),
    "invalid_gyro_mode":("Unknown gyro_mode string",                    test_invalid_gyro_mode),
    "out_of_range":     ("Values outside [-1, 1]",                      test_out_of_range),
    "overflow":         ("Datagram >511 bytes discarded",               test_overflow),
    "multiple_senders": ("Two UDP sockets alternating",                 test_multiple_senders),
    "wifi_loss":        ("Manually disable/enable AP mid-stream",       test_wifi_loss),
}


def print_menu() -> None:
    print("\nAvailable tests:")
    for i, (name, (desc, _)) in enumerate(TESTS.items(), 1):
        print(f"  {i:2d}. {name:<22} {desc}")
    print()


def run_test(name: str, ip: str) -> None:
    fn = TESTS[name][1]
    try:
        fn(ip)
        print(f"\n  ✓ {name} done.")
    except KeyboardInterrupt:
        print("\n  [interrupted]")
    except OSError as e:
        print(f"\n  [socket error: {e}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
