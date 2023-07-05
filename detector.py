import cv2
import cv2.aruco as aruco
import numpy as np
from pytransform3d import transformations as pt


ARUCO_DICT = {
    aruco.DICT_4X4_50: "DICT_4X4_50",
    aruco.DICT_4X4_100: "DICT_4X4_100",
    aruco.DICT_4X4_250: "DICT_4X4_250",
    aruco.DICT_4X4_1000: "DICT_4X4_1000",
    aruco.DICT_5X5_50: "DICT_5X5_50",
    aruco.DICT_5X5_100: "DICT_5X5_100",
    aruco.DICT_5X5_250: "DICT_5X5_250",
    aruco.DICT_5X5_1000: "DICT_5X5_1000",
    aruco.DICT_6X6_50: "DICT_6X6_50",
    aruco.DICT_6X6_100: "DICT_6X6_100",
    aruco.DICT_6X6_250: "DICT_6X6_250",
    aruco.DICT_6X6_1000: "DICT_6X6_1000",
    aruco.DICT_7X7_50: "DICT_7X7_50",
    aruco.DICT_7X7_100: "DICT_7X7_100",
    aruco.DICT_7X7_250: "DICT_7X7_250",
    aruco.DICT_7X7_1000: "DICT_7X7_1000",
    aruco.DICT_ARUCO_ORIGINAL: "DICT_ARUCO_ORIGINAL",
    aruco.DICT_APRILTAG_16h5: "DICT_APRILTAG_16h5",
    aruco.DICT_APRILTAG_25h9: "DICT_APRILTAG_25h9",
    aruco.DICT_APRILTAG_36h10: "DICT_APRILTAG_36h10",
    aruco.DICT_APRILTAG_36h11: "DICT_APRILTAG_36h11"
}


def detect_chessboard(images, num_x, num_y, square_size, show_images=True):
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    obj_p = np.zeros((num_y * num_x, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)
    obj_p *= square_size  # add the square size of the chessboard

    for chess_img in images:
        ret, corners = cv2.findChessboardCorners(
            chess_img, (num_x, num_y), None)
        if ret is True:
            obj_points.append(obj_p)
            corners = cv2.cornerSubPix(
                chess_img, corners, (8, 8), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_points.append(corners)
            if show_images:
                color_image = cv2.cvtColor(chess_img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(
                    color_image, (num_x, num_y), corners, ret)
                cv2.imshow('img', color_image)
                cv2.waitKey(0)

    return obj_points, img_points


def detect_aruco(img, K, dist_coeffs, show_image=False, dictionary=aruco.DICT_5X5_250):

    aruco_dict = aruco.Dictionary_get(dictionary)
    aruco_params = aruco.DetectorParameters_create()
    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    aruco_params.cornerRefinementMinAccuracy = 0.05
    aruco_params.cornerRefinementWinSize = 5
    # detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    # Detect ArUco markers in the image
    corners, ids, rejected = aruco.detectMarkers(
        img, aruco_dict, parameters=aruco_params)
    # corners, ids, rejected = detector.detectMarkers(img)

    if ids is not None:

        rvec, tvec, obj_pts = aruco.estimatePoseSingleMarkers(
            corners, 0.154, K, dist_coeffs)
        rot_mat, jacobian = cv2.Rodrigues(rvec)

        # Draw detected markers on the image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        aruco.drawDetectedMarkers(img, corners, ids)
        cv2.drawFrameAxes(img, K, dist_coeffs, rvec, tvec, 130.0)

        if show_image:
            cv2.imshow('Detected Markers', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return pt.transform_from(rot_mat, tvec[0])
    else:
        # print('No ArUco markers detected.')
        return None


def detect_charuco_board(img, K, dist_coeffs, x, y, show_image=False, dictionary=aruco.DICT_5X5_250):

    aruco_dict = aruco.Dictionary_get(dictionary)
    board = aruco.CharucoBoard_create(x, y, 1, 0.5, aruco_dict)
    params = aruco.DetectorParameters_create()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementMinAccuracy = 0.05

    corners, ids, rejected = aruco.detectMarkers(img, aruco_dict, parameters=params)
    retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, img, board)

    if charuco_corners is not None and charuco_ids is not None:
        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, K, dist_coeffs)
        rot_mat, jacobian = cv2.Rodrigues(rvec)

        # Draw detected markers on the image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        aruco.drawDetectedMarkers(img, corners, ids)
        aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
        cv2.drawFrameAxes(img, K, dist_coeffs, rvec, tvec, 130.0)

        if show_image:
            cv2.imshow('Detected Markers', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return pt.transform_from(rot_mat, tvec[0])
    else:
        print('No markers detected.')
        return None