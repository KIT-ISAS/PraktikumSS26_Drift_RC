import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimulationNode(Node):
    def __init__(self):
        super().__init__('simulation_node')
        self.publisher_ = self.create_publisher(String, 'simulation_status', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.get_logger().info("Simulation node started!")

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello ROS 2 Jazzy! Simulation is running: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    simulation_node = SimulationNode()
    try:
        rclpy.spin(simulation_node)
    except KeyboardInterrupt:
        pass
    finally:
        simulation_node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
