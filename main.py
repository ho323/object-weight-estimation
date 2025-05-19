import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys
import select
import termios
import tty
from process_image import process_image, load_depth_model, load_segmentation_model
from ultralytics import YOLO
import cfg


class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')

        self.bridge = CvBridge()
        self.latest_image = None  # 최신 이미지 저장

        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.listener_callback,
            10
        )
        print('Node initialized')

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image  # 최신 이미지를 저장
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {str(e)}")

def get_key(timeout=0.1):
    """터미널에서 키 입력을 비동기로 읽는다."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
            return key
        else:
            return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()

    print("Loading models...")
    de_model = load_depth_model(cfg.DEVICE)
    od_model = YOLO(cfg.OD_MODEL_PATH).to(cfg.DEVICE)
    sg_model = load_segmentation_model(cfg.DEVICE)
    print("Model loading complete.")

    print("Press 's' to save image, 'q' to quit.")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            key = get_key(timeout=0.01)
            if key == 's':
                if node.latest_image is not None:
                    temp_path = 'temp.png'
                    cv2.imwrite(temp_path, node.latest_image)
                    process_image(
                        temp_path, output_path='output.png',
                        de_model=de_model, od_model=od_model, sg_model=sg_model,
                        save_image=True
                    )
                    node.get_logger().info("Saved current image and processed output.png")
                else:
                    node.get_logger().warn("No image received yet.")
            elif key == 'q':
                node.get_logger().info("Quitting program.")
                break

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
