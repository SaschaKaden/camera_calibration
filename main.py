import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import transformations as pt
from pytransform3d import rotations as rotations

import calib
import vis
import util
import aruco

CALIB_INTRINSIC = False
CALIB_HAND_EYE = True
UNDISTORT = True
SHOW_IMAGES = False

start_img = 0
end_img = 32


def undistort(image, K, coeffs):
    h, w = image.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, coeffs, (w, h), 1, (w, h))
    return cv2.undistort(image, new_K, coeffs)


if __name__ == '__main__':
    np.set_printoptions(precision=3)

    if CALIB_INTRINSIC:
        calib_images = []
        for i in range(0, 23):
            calib_images.append(cv2.imread(
                "data/intrinsic/{}.png".format(i), cv2.IMREAD_GRAYSCALE))

        # calibrate the camera
        obj_pts, img_pts = calib.detect_chessboard(calib_images, 10, 7, 0.022, SHOW_IMAGES)
        K, dist_coeffs = calib.calibrate_intrinsic(obj_pts, img_pts, calib_images[0])
        calib.save_calib(K, dist_coeffs)

    if CALIB_HAND_EYE:
        K, dist_coeffs, tcp_to_cam = calib.load_calib()
        hand_eye_images = []
        pattern_to_cam_Ts = []
        tcp_to_base_Ts = []
        base_to_tcp_Ts = []

        for i in range(start_img, end_img + 1):
            hand_eye_img = cv2.imread("data/eye-to-hand/{}.png".format(i), cv2.IMREAD_GRAYSCALE)
            if UNDISTORT:
                hand_eye_img = undistort(hand_eye_img, K, dist_coeffs)
            hand_eye_images.append(hand_eye_img)
            T = aruco.detect_marker(hand_eye_img, K, dist_coeffs, SHOW_IMAGES)
            if T is None:
                continue

            pattern_to_cam_Ts.append(T)

            tcp_to_base, base_to_tcp = util.load_transforms_file("data/eye-to-hand/{}.xml".format(i))
            tcp_to_base_Ts.append(tcp_to_base)
            base_to_tcp_Ts.append(base_to_tcp)

        cam_to_pattern_Ts = []
        for T in pattern_to_cam_Ts:
            cam_to_pattern_Ts.append(pt.invert_transform(T))

        tcp_to_cam = calib.calib_hand_eye(tcp_to_base_Ts, pattern_to_cam_Ts)
        calib.save_calib(K, dist_coeffs, tcp_to_cam)

        # vis.view_poses("grasps", base_to_tcp_Ts)

        base_to_pattern = pt.transform_from(np.eye(3), [-0.4, 0, 0])
        patterns = []
        poses = []
        tcps = []
        for i in range(len(base_to_tcp_Ts)):
            tcps.append(base_to_tcp_Ts[i] @ tcp_to_cam)
            patterns.append(base_to_tcp_Ts[i] @
                            tcp_to_cam @ cam_to_pattern_Ts[i])
            poses.append(base_to_pattern @ pattern_to_cam_Ts[i])
        vis.view_poses(end_img, "patterns", patterns, "p")
        vis.view_poses(5, "cam", tcps, "b",
                       patterns, "pattern", poses, "p")

        # error = 0
        # last_board_pose = base_to_tcp_Ts[0] @ tcp_to_cam @ cam_to_pattern_Ts[0]
        # for i in range(1, len(base_to_tcp_Ts)):
        #     board_pose = base_to_tcp_Ts[i] @ tcp_to_cam @ cam_to_pattern_Ts[i]
        #     error += np.linalg.norm(last_board_pose[0:3,
        #                             3] - board_pose[0:3, 3])
        #     last_board_pose = board_pose
        # print(error)
        print(tcp_to_cam)
        plt.show()
