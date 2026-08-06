---
name: verify
description: Verify ROS-side changes end-to-end without car hardware by running nodes against a fake ESP32 on loopback UDP.
---

# Verifying ROS changes without hardware

The procedure — build/source, the fake ESP32 contract, how to drive it, what to check
and the failure-mode probe — lives in `documentation/Testing.md`. Read that first; it
is the single source of truth and is maintained for humans and agents alike.

## Agent-specific gotcha

- `pkill -f <pattern>` also kills the Bash tool's own wrapper shell, because that
  shell's command line contains the pattern → exit 144. Use
  `kill $(pgrep -f "[p]attern")` instead.
