import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import calib
import aruco
from franka_msgs.msg import FrankaState
import rospy
from pytransform3d import transformations as pt
import vis


SHOW_IMAGES = False
SHOW_TCP = True
np.set_printoptions(precision=3)


def save(img, image_path):
    cv.imwrite(image_path, img)


def display(img, window_name="default", destroyable=True):
    imS = cv.resize(img, (1280, 720))
    cv.imshow(window_name, img)
    if destroyable is True:
        cv.waitKey(0)
        cv.destroyWindow(window_name)


def init_rs():
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    pipeline.start(config)
    return pipeline


class StateViewer:
    def __init__(self, rate=10, print_data=False):
        rospy.init_node('listener', anonymous=True)
        self.rate = rospy.Rate(rate)
        self.print_data = print_data
        self.joints = []
        self.tcp = []

        self.sub = rospy.Subscriber(
            "/franka_state_controller/franka_states", FrankaState, self.joint_callback)

    def joint_callback(self, franka_state):
        # print("Franka State: ", franka_state)
        self.joints = franka_state.q
        self.tcp = franka_state.O_T_EE
        self.tcp = np.reshape(self.tcp, (4, 4)).transpose()
        if SHOW_TCP:
            print("TCP: ")
            print(self.tcp)

    def get_tcp(self):
        return self.tcp


if __name__ == '__main__':
    K, dist_coeffs, tcp_to_cam = calib.load_calib()
    viewer = StateViewer()

    pipeline = init_rs()

    count = 0
    while True:
        frames = pipeline.wait_for_frames()
        color_image = np.asanyarray(frames.get_color_frame().get_data())
        color_image = cv.cvtColor(color_image, cv.COLOR_RGB2BGR)
        gray_image = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
        display(color_image, "live stream", False)

        T = aruco.detect_marker(gray_image, K, dist_coeffs, SHOW_IMAGES)
        if T is None:
            continue
        base_to_ee = viewer.get_tcp()
        base_to_aruco = base_to_ee @ tcp_to_cam @ pt.invert_transform(T)

        print("Base to EE: ")
        print(base_to_ee)
        print("TCP to Cam: ")
        print(tcp_to_cam)
        print("TCP to ArUco: ")
        print(T)

        print("Base to ArUco: ")
        print(base_to_aruco)
        # if base_to_aruco is not None:
        #     vis.view_poses(1, "patterns", [base_to_aruco])

        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            image_file = "image" + str(count) + ".png"
            print("Write image: " + image_file)
            save(color_image, "data\\capture\\" + image_file)
            count += 1

    # When everything done, release the capture
    cv.destroyAllWindows()
