import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import transformations as pt
from pytransform3d import rotations as rotations

import calib
import vis
import util

CALIB_INTRINSIC = False
CALIB_HAND_EYE = True
UNDISTORT = False
SHOW_IMAGES = False

start_img = 12
end_img = 37


def undistort(image, K, coeffs):
    h, w = image.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, coeffs, (w, h), 1, (w, h))
    return cv2.undistort(image, new_K, coeffs)


if __name__ == '__main__':
    if CALIB_INTRINSIC:
        calib_images = []
        for i in range(0, 23):
            calib_images.append(cv2.imread(
                "data/intrinsic/{}.png".format(i), cv2.IMREAD_GRAYSCALE))

        # calibrate the camera
        obj_pts, img_pts = calib.detect_chessboard(
            calib_images, 10, 7, 0.022, SHOW_IMAGES)
        K, dist_coeffs = calib.calibrate_intrinsic(
            obj_pts, img_pts, calib_images[0])
        calib.save_calib(K, dist_coeffs)
        undistort_images = []

        # for calib_img in calib_images:
        #     undistort_images.append(undistort(calib_img, K, dist_coeffs))
        #     cv2.imshow("undistort", calib_img)
        #     cv2.waitKey(0)
        #     cv2.destroyWindow("undistort")

    if CALIB_HAND_EYE:
        K, dist_coeffs = calib.load_calib()
        hand_eye_images = []
        pattern_to_cam_Ts = []
        for i in range(start_img, end_img + 1):
            hand_eye_img = cv2.imread(
                "data/eye-to-hand/{}.png".format(i), cv2.IMREAD_GRAYSCALE)
            if UNDISTORT:
                hand_eye_img = undistort(hand_eye_img, K, dist_coeffs)
            hand_eye_images.append(hand_eye_img)
            obj_pts, img_pts = calib.detect_chessboard(
                [hand_eye_img], 8, 5, 0.0298, False)

            T = calib.calibrate_extrinsic(
                obj_pts[0], img_pts[0], K, dist_coeffs, hand_eye_img, SHOW_IMAGES)
            T[2, 3] = -T[2, 3]
            pattern_to_cam_Ts.append(T)

        tcp_to_base_Ts, base_to_tcp_Ts = util.load_transforms("data/eye-to-hand/", start_img, end_img)

        cam_to_pattern_Ts = []
        for T in pattern_to_cam_Ts:
            cam_to_pattern_Ts.append(T)

        tcp_to_cam = calib.calib_hand_eye(tcp_to_base_Ts, pattern_to_cam_Ts)

        # tcp_to_cam[0, 3] = -0.05
        # tcp_to_cam[1, 3] = -0.05
        # tcp_to_cam[2, 3] = -0.05

        # vis.view_poses("grasps", base_to_tcp_Ts)

        rot = rotations.matrix_from_euler([np.pi, 0, 0], 0, 1, 2, True)
        base_to_pattern = pt.transform_from(rot, [0.8, 0, 0])
        for i in range(len(pattern_to_cam_Ts)):
            # pattern_to_cam_Ts[i][0, 3] = -pattern_to_cam_Ts[i][0, 3]
            # pattern_to_cam_Ts[i][1, 3] = -pattern_to_cam_Ts[i][1, 3]
            # pattern_to_cam_Ts[i][2, 3] = -pattern_to_cam_Ts[i][2, 3]
            pattern_to_cam_Ts[i] = base_to_pattern @ pattern_to_cam_Ts[i]
        vis.view_poses("calib", base_to_tcp_Ts, "grasp", pattern_to_cam_Ts, "pattern", [base_to_pattern], "base_to_pattern")

        poses = []
        for i in range(len(base_to_tcp_Ts)):
            poses.append(base_to_tcp_Ts[i] @ tcp_to_cam @ cam_to_pattern_Ts[i])
        vis.view_poses("poses", poses)

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
