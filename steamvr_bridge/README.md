# SteamVR bridge (host-side)

Streams Vive tracker poses from the **host's** SteamVR runtime into the dev
container, where `steamvr_pkg`'s `steamvr_node` turns them into the standard
tracking-source outputs (TF `world -> tracker_link` + pseudo-IMU). Like
`esp32_controller/`, this directory is host-adjacent code — it is not part of
the colcon workspace and never runs inside the container (OpenVR needs local
IPC to `vrserver`, which the container does not have).

```
HOST                                     CONTAINER (host network)
SteamVR ──openvr──> steamvr_bridge.py ──UDP :15250──> steamvr_node
```

## Prerequisites (host)

- SteamVR installed and running, with the lighthouses and the tracker (via its
  USB dongle) set up. Headset-less operation requires the usual tweaks
  (`requireHmd=false`, null driver enabled) — use the existing working setup.
- Python 3 with the `openvr` package (`pip install openvr`).
- Note: SteamVR and libsurvive both claim the tracker dongle — stop the
  holodeck/libsurvive stack before starting SteamVR (and vice versa).

## Run

```bash
python3 steamvr_bridge.py                 # defaults: 127.0.0.1:15250 @ 100 Hz
python3 steamvr_bridge.py --rate 50 --universe raw
```

- `--universe standing` (default) uses the floor-calibrated room setup; its
  origin can shift when SteamVR room setup is redone — recalibrate
  `world -> steamvr_world` in `steamvr_pkg` afterwards (see that package's
  README). `--universe raw` is the uncalibrated fallback.
- The bridge retries until SteamVR is up and reconnects if it goes away, so
  start order does not matter.

To sanity-check the datagrams without ROS:

```bash
python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(('127.0.0.1', 15250))
while True: print(s.recvfrom(2048)[0].decode())
"
```

## Wire format

Versioned JSON datagrams, one per device sample, carrying the **raw OpenVR**
matrix and tracking-space velocities. The schema is owned by
[`drift_ws/src/steamvr_pkg/steamvr_pkg/lib/protocol.py`](../drift_ws/src/steamvr_pkg/steamvr_pkg/lib/protocol.py)
(stdlib-only), which this script imports from the repo checkout — both ends of
the link always share one definition. All coordinate conversion happens
container-side in `steamvr_pkg` (unit-tested); keep it that way.
