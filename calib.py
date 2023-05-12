import cv2
import numpy as np
from pytransform3d import transformations as pt

import util


def detect_chessboard(images, num_x, num_y, square_size, show_images=True):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_p = np.zeros((num_y * num_x, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)
    obj_p *= square_size  # add the square size of the chessboard

    for chess_img in images:
        # Find the chess board corners and append them to the point lists
        ret, corners = cv2.findChessboardCorners(
            chess_img, (num_x, num_y), None)
        # If found, add object points, image points (after refining them)
        if ret is True:
            obj_points.append(obj_p)
            corners = cv2.cornerSubPix(
                chess_img, corners, (8, 8), (-1, -1), criteria)
            img_points.append(corners)
            if show_images:  # Draw and display the corners
                color_image = cv2.cvtColor(chess_img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(
                    color_image, (num_x, num_y), corners, ret)
                cv2.imshow('img', color_image)
                # cv2.imwrite("calib_img.png", color_image)
                cv2.waitKey(0)

    # return object and image points
    return obj_points, img_points


def calibrate_intrinsic(obj_points, img_points, img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start the calibration
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img.shape[::-1], None, None)

    # calculate the mean re-projection error and print it
    mean_error = 0
    for i in range(len(obj_points)):
        img_points_projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], K, dist_coeffs)
        error = cv2.norm(
            img_points[i], img_points_projected, cv2.NORM_L2) / len(img_points_projected)
        mean_error += error
    print("total error: {}".format(mean_error / len(obj_points)))
    # ... end

    # return camera matrix K and the distortion parameters
    return K, dist_coeffs


def calibrate_extrinsic(obj_points, img_points, K, dist_coeffs, image, show_images=True):
    retval, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    cv2.solvePnPRefineLM(obj_points, img_points, K, dist_coeffs, rvec, tvec)
    rot_mat, jacobian = cv2.Rodrigues(rvec)

    if show_images:
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawFrameAxes(color_image, K, dist_coeffs, rot_mat, tvec, 0.1)
        cv2.imshow('img', color_image)
        # cv2.imwrite("calib_img.png", color_image)
        cv2.waitKey(0)

    return pt.transform_from(rot_mat, tvec.transpose())


def calib_hand_eye(tcp_to_base_Ts, pattern_to_cam_Ts, method, base_to_tcp_Ts=None, cam_to_pattern_Ts=None):
    tcp_to_base_Rs = []
    tcp_to_base_ts = []
    pattern_to_cam_Rs = []
    pattern_to_cam_ts = []

    for T in tcp_to_base_Ts:
        tcp_to_base_Rs.append(T[0:3, 0:3])
        tcp_to_base_ts.append(T[0:3, 3])

    for T in pattern_to_cam_Ts:
        pattern_to_cam_Rs.append(T[0:3, 0:3])
        pattern_to_cam_ts.append(T[0:3, 3])
    rot, t = cv2.calibrateHandEye(
        tcp_to_base_Rs, tcp_to_base_ts, pattern_to_cam_Rs, pattern_to_cam_ts, method=method)
    tcp_to_cam = pt.transform_from(rot, t.transpose())
    print(np.linalg.det(rot))
    if np.linalg.det(rot) < 0:
        return tcp_to_cam
    tcp_to_cam = pt.invert_transform(tcp_to_cam)

    error = 0
    if base_to_tcp_Ts is not None and cam_to_pattern_Ts is not None:
        last_board_pose = base_to_tcp_Ts[0] @ tcp_to_cam @ cam_to_pattern_Ts[0]
        for i in range(1, base_to_tcp_Ts.__len__()):
            board_pose = base_to_tcp_Ts[i] @ tcp_to_cam @ cam_to_pattern_Ts[i]
            error += np.linalg.norm(last_board_pose[1:3,
                                    3] - board_pose[1:3, 3])
            last_board_pose = board_pose
        error /= base_to_tcp_Ts.__len__()
    else:
        print("error: base_to_tcp_Ts or cam_to_pattern_Ts is None")

    method_str = ""
    if method is cv2.CALIB_HAND_EYE_PARK:
        method_str = "PARK"
    elif method is cv2.CALIB_HAND_EYE_TSAI:
        method_str = "TSAI"
    elif method is cv2.CALIB_HAND_EYE_HORAUD:
        method_str = "HORAUD"
    elif method is cv2.CALIB_HAND_EYE_ANDREFF:
        method_str = "ANDREFF"
    elif method is cv2.CALIB_HAND_EYE_DANIILIDIS:
        method_str = "DANIILIDIS"
    print("method: ", method_str)
    print("tcp_to_cam")
    print(tcp_to_cam)
    print("Euclidean error between boards: ", error)
    return tcp_to_cam


def save_calib(K, coeffs):
    cv_file = cv2.FileStorage("data/calib.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", K)
    cv_file.write("dist_coeffs", coeffs)
    cv_file.release()
    print("K: ", K)
    print("distortion coefficients: ", coeffs)


import cv2
import numpy as np
from pytransform3d import transformations as pt

import util


def detect_chessboard(images, num_x, num_y, square_size, show_images=True):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_p = np.zeros((num_y * num_x, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)
    obj_p *= square_size  # add the square size of the chessboard

    for chess_img in images:
        # Find the chess board corners and append them to the point lists
        ret, corners = cv2.findChessboardCorners(chess_img, (num_x, num_y), None)
        # If found, add object points, image points (after refining them)
        if ret is True:
            obj_points.append(obj_p)
            corners = cv2.cornerSubPix(chess_img, corners, (8, 8), (-1, -1), criteria)
            img_points.append(corners)
            if show_images:  # Draw and display the corners
                color_image = cv2.cvtColor(chess_img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(color_image, (num_x, num_y), corners, ret)
                cv2.imshow('img', color_image)
                # cv2.imwrite("calib_img.png", color_image)
                cv2.waitKey(0)

    # return object and image points
    return obj_points, img_points


def calibrate_intrinsic(obj_points, img_points, img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start the calibration
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img.shape[::-1], None, None)

    # calculate the mean re-projection error and print it
    mean_error = 0
    for i in range(len(obj_points)):
        img_points_projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist_coeffs)
        error = cv2.norm(img_points[i], img_points_projected, cv2.NORM_L2) / len(img_points_projected)
        mean_error += error
    print("total error: {}".format(mean_error / len(obj_points)))
    # ... end

    # return camera matrix K and the distortion parameters
    return K, dist_coeffs


def calibrate_extrinsic(obj_points, img_points, K, dist_coeffs, image, show_images=True):
    retval, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    cv2.solvePnPRefineLM(obj_points, img_points, K, dist_coeffs, rvec, tvec)
    rot_mat, jacobian = cv2.Rodrigues(rvec)

    if show_images:
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawFrameAxes(color_image, K, dist_coeffs, rot_mat, tvec, 0.1)
        cv2.imshow('img', color_image)
        # cv2.imwrite("calib_img.png", color_image)
        cv2.waitKey(0)

    return pt.transform_from(rot_mat, tvec.transpose())


def calib_hand_eye_method(tcp_to_base_Ts, pattern_to_cam_Ts, method, base_to_tcp_Ts, cam_to_pattern_Ts):
    method_str = ""
    if method is cv2.CALIB_HAND_EYE_PARK:
        method_str = "PARK"
    elif method is cv2.CALIB_HAND_EYE_TSAI:
        method_str = "TSAI"
    elif method is cv2.CALIB_HAND_EYE_HORAUD:
        method_str = "HORAUD"
    elif method is cv2.CALIB_HAND_EYE_ANDREFF:
        method_str = "ANDREFF"
    elif method is cv2.CALIB_HAND_EYE_DANIILIDIS:
        method_str = "DANIILIDIS"

    tcp_to_base_Rs = []
    tcp_to_base_ts = []
    pattern_to_cam_Rs = []
    pattern_to_cam_ts = []

    for T in tcp_to_base_Ts:
        tcp_to_base_Rs.append(T[0:3, 0:3])
        tcp_to_base_ts.append(T[0:3, 3])
    for T in pattern_to_cam_Ts:
        pattern_to_cam_Rs.append(T[0:3, 0:3])
        pattern_to_cam_ts.append(T[0:3, 3])

    rot, t = cv2.calibrateHandEye(tcp_to_base_Rs, tcp_to_base_ts, pattern_to_cam_Rs, pattern_to_cam_ts, method=method)
    cam_to_tcp = pt.transform_from(rot, t.transpose())
    if np.linalg.det(rot) < 0.5:
        return {
            "method": method_str,
            "error": 999999999,
            "tcp_to_cam": cam_to_tcp
        }
    tcp_to_cam = pt.invert_transform(cam_to_tcp)

    error = 0
    if base_to_tcp_Ts is not None and cam_to_pattern_Ts is not None:
        last_board_pose = base_to_tcp_Ts[0] @ tcp_to_cam @ cam_to_pattern_Ts[0]
        for i in range(1, base_to_tcp_Ts.__len__()):
            board_pose = base_to_tcp_Ts[i] @ tcp_to_cam @ cam_to_pattern_Ts[i]
            error += np.linalg.norm(last_board_pose[1:3, 3] - board_pose[1:3, 3])
            last_board_pose = board_pose
        error /= base_to_tcp_Ts.__len__()
    else:
        print("error: base_to_tcp_Ts or cam_to_pattern_Ts is None")

    print("method: ", method_str)
    print("tcp_to_cam")
    print(tcp_to_cam)
    print("Euclidean error between boards: ", error)

    return {
        "method": method_str,
        "error": error,
        "tcp_to_cam": tcp_to_cam
    }


def calib_hand_eye(tcp_to_base_Ts, pattern_to_cam_Ts, base_to_tcp_Ts, cam_to_pattern_Ts):
    results = []
    for method in [cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_HORAUD,
                   cv2.CALIB_HAND_EYE_ANDREFF, cv2.CALIB_HAND_EYE_DANIILIDIS]:
        results.append(
            calib_hand_eye_method(tcp_to_base_Ts, pattern_to_cam_Ts, method, base_to_tcp_Ts, cam_to_pattern_Ts))

    best_error = results[0]["error"]
    best_index = 0
    for i in range(results.__len__()):
        if results[i]["error"] < best_error:
            best_error = results[i]["error"]
            best_index = i

    print("best method: " + results[best_index]["method"])
    return results[best_index]["tcp_to_cam"]


def save_calib(K, coeffs):
    cv_file = cv2.FileStorage("data/calib.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", K)
    cv_file.write("dist_coeffs", coeffs)
    cv_file.release()
    print("K: ", K)
    print("distortion coefficients: ", coeffs)


def load_calib():
    cv_file = cv2.FileStorage("data/calib.xml", cv2.FILE_STORAGE_READ)
    K = cv_file.getNode("K").mat()
    dist_coeffs = cv_file.getNode("dist_coeffs").mat()
    cv_file.release()
    return K, dist_coeffs


def load_calib():
    cv_file = cv2.FileStorage("data/calib.xml", cv2.FILE_STORAGE_READ)
    K = cv_file.getNode("K").mat()
    dist_coeffs = cv_file.getNode("dist_coeffs").mat()
    cv_file.release()
    return K, dist_coeffs
