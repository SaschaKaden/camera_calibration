import cv2
import matplotlib.pyplot as plt
from pytransform3d import transformations as pt

import calib
import vis
import util

CALIB_INTRINSIC = True
CALIB_HAND_EYE = True
UNDISTORT = True
SHOW_IMAGES = False

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
        pattern_to_cam_Ts = []
        for i in range(start_img, end_img + 1):
            hand_eye_img = cv2.imread("data/eye-to-hand/{}.png".format(i), cv2.IMREAD_GRAYSCALE)
            if UNDISTORT:
                hand_eye_img = undistort(hand_eye_img, K, dist_coeffs)
            hand_eye_images.append(hand_eye_img)
            obj_pts, img_pts = calib.detect_chessboard([hand_eye_img], 8, 5, 0.0298, False)
            pattern_to_cam_Ts.append(calib.calibrate_extrinsic(obj_pts[0], img_pts[0], K, dist_coeffs, hand_eye_img, SHOW_IMAGES))

        tcp_to_base_Ts = []
        base_to_tcp_Ts = []
        for i in range(start_img, end_img + 1):
            base_to_tcp, tcp_to_base = util.load_transforms("data/eye-to-hand/{}.xml".format(i))
            tcp_to_base_Ts.append(tcp_to_base)
            base_to_tcp_Ts.append(base_to_tcp)

        cam_to_pattern_Ts = []
        for T in pattern_to_cam_Ts:
            cam_to_pattern_Ts.append(pt.invert_transform(T))

        tcp_to_cam = calib.calib_hand_eye(tcp_to_base_Ts, pattern_to_cam_Ts, base_to_tcp_Ts, cam_to_pattern_Ts)

        vis.view_poses(base_to_tcp_Ts, tcp_to_cam, cam_to_pattern_Ts, "board poses")
        plt.show()
