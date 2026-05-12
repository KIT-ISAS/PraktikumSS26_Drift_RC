import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ExampleNode(Node):
    def __init__(self):
        super().__init__('example_node')
        self.publisher_ = self.create_publisher(String, 'example_status', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.get_logger().info('Example node started!')

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello ROS 2 Jazzy! Example is running: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    example_node = ExampleNode()
    try:
        rclpy.spin(example_node)
    except KeyboardInterrupt:
        pass
    finally:
        example_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
