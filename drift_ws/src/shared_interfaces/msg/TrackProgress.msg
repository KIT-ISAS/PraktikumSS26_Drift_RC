std_msgs/Header header
string track_name
int32 lap                       # completed laps since start
float32 lap_progress            # current-lap fraction, 0..1
float32 distance                # arc length s along centerline (m)
float32 total_length            # full track length (m)
geometry_msgs/Pose projected_pose   # closest centerline point (+ track heading)
float32 lateral_error           # signed car->centerline distance (m), + = car left of path
