import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import transformations as pt

import calib
import vis
import util

CALIB_INTRINSIC = True
CALIB_HAND_EYE = True
SHOW_IMAGES = True

start_img = 0
end_img = 25


def undistort(image, K, coeffs):
    h, w = image.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, coeffs, (w, h), 1, (w, h))
    return cv2.undistort(image, new_K, coeffs)


if __name__ == '__main__':
    if CALIB_INTRINSIC:
        calib_images = []
        for i in range(0, 6):
            # calib_images.append(cv2.imread("data/intrinsic/calib_img{}.png".format(i), cv2.IMREAD_GRAYSCALE))
            calib_images.append(cv2.imread("data/eye-to-hand/{}.png".format(i), cv2.IMREAD_GRAYSCALE))

        # calibrate the camera
        # obj_pts, img_pts = calib.detect_chessboard(calib_images, 10, 7, 0.022, SHOW_IMAGES)
        obj_pts, img_pts = calib.detect_chessboard(calib_images, 8, 5, 0.0298, SHOW_IMAGES)
        K, dist_coeffs = calib.calibrate_intrinsic(obj_pts, img_pts, calib_images[0])
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
        cam_to_pattern_Ts = []
        for i in range(start_img, end_img + 1):
            hand_eye_img = cv2.imread("data/eye-to-hand/{}.png".format(i), cv2.IMREAD_GRAYSCALE)
            # hand_eye_img = undistort(hand_eye_img, K, dist_coeffs)
            hand_eye_images.append(hand_eye_img)
            obj_pts, img_pts = calib.detect_chessboard([hand_eye_img], 8, 5, 0.0298, False)
            transform = calib.calibrate_extrinsic(obj_pts[0], img_pts[0], K, dist_coeffs, hand_eye_img, SHOW_IMAGES)
            cam_to_pattern_Ts.append(transform)

        tcp_to_base_Ts = []
        for i in range(start_img, end_img + 1):
            base_to_ee, ee_to_base = util.load_transforms("data/eye-to-hand/{}.xml".format(i))
            tcp_to_base_Ts.append(ee_to_base)

        # vis.view_points(cam_to_pattern_Ts, "cam_to_pattern_Ts")
        # vis.view_points(base_to_tcp_Ts, "base_to_tcp_Ts")
        board_Ts = []
        for i in range(cam_to_pattern_Ts.__len__()):
            board_Ts.append(tcp_to_base_Ts[i] @ cam_to_pattern_Ts[i])
        # vis.view_tcp_boards(tcp_to_base_Ts, board_Ts, "boards")
        # plt.show()

        pattern_to_cam_Ts = []
        print("pattern_to_cam: ")
        for T in cam_to_pattern_Ts:
            pattern_to_cam_Ts.append(pt.invert_transform(T))

        # R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(cam_Rs, cam_ts, tcp_to_base_Rs, tcp_to_base_ts, method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)
        # R = R_gripper2cam
        # t = t_gripper2cam

        base_to_tcp_Ts = []
        for T in tcp_to_base_Ts:
            base_to_tcp_Ts.append(pt.invert_transform(T))

        tcp_to_cam = calib.calib_hand_eye(tcp_to_base_Ts, pattern_to_cam_Ts,  method=cv2.CALIB_HAND_EYE_PARK, base_to_tcp_Ts=base_to_tcp_Ts, cam_to_pattern_Ts=cam_to_pattern_Ts)
        tcp_to_cam = calib.calib_hand_eye(tcp_to_base_Ts, pattern_to_cam_Ts,  method=cv2.CALIB_HAND_EYE_HORAUD, base_to_tcp_Ts=base_to_tcp_Ts, cam_to_pattern_Ts=cam_to_pattern_Ts)
        tcp_to_cam = calib.calib_hand_eye(tcp_to_base_Ts, pattern_to_cam_Ts,  method=cv2.CALIB_HAND_EYE_ANDREFF, base_to_tcp_Ts=base_to_tcp_Ts, cam_to_pattern_Ts=cam_to_pattern_Ts)
        tcp_to_cam = calib.calib_hand_eye(tcp_to_base_Ts, pattern_to_cam_Ts,  method=cv2.CALIB_HAND_EYE_DANIILIDIS, base_to_tcp_Ts=base_to_tcp_Ts, cam_to_pattern_Ts=cam_to_pattern_Ts)
        tcp_to_cam = calib.calib_hand_eye(tcp_to_base_Ts, pattern_to_cam_Ts,  method=cv2.CALIB_HAND_EYE_TSAI, base_to_tcp_Ts=base_to_tcp_Ts, cam_to_pattern_Ts=cam_to_pattern_Ts)

        new_board_Ts = []
        for i in range(board_Ts.__len__()):
            new_board_Ts.append(base_to_tcp_Ts[i] @ tcp_to_cam @ cam_to_pattern_Ts[i])

        # vis.view_boards(base_to_tcp_Ts, board_Ts, new_board_Ts, "new boards")
        vis.view_poses(base_to_tcp_Ts, tcp_to_cam, cam_to_pattern_Ts, "board poses")
        # vis.view_points(new_board_Ts, "new boards")
        plt.show()

