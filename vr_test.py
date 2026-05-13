import triad_openvr as triad_openvr

vr = triad_openvr.triad_openvr()
vr.print_discovered_objects()
while True:
    # Read pose from VR tracker
    pose = vr.devices["tracker_1"].get_pose_euler()
    x, y, yaw = pose[0], pose[1], pose[3]
    print(x, y)

