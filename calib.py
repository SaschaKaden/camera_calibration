import cv2
import numpy as np
from pytransform3d import transformations as pt

import util


def calibrate_intrinsic(obj_points, img_points, img):
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img.shape[::-1], None, None)

    # calculate the mean re-projection error
    mean_error = 0
    for i in range(len(obj_points)):
        img_points_projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], K, dist_coeffs)
        error = cv2.norm(
            img_points[i], img_points_projected, cv2.NORM_L2) / len(img_points_projected)
        mean_error += error
    print("total error: {}".format(mean_error / len(obj_points)))

    return K, dist_coeffs


def calibrate_extrinsic(obj_points, img_points, K, dist_coeffs, image, show_images=True, write_images=False):
    retval, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    cv2.solvePnPRefineLM(obj_points, img_points, K, dist_coeffs, rvec, tvec)
    rot_mat, jacobian = cv2.Rodrigues(rvec)

    if show_images:
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawFrameAxes(color_image, K, dist_coeffs, rot_mat, tvec, 0.1)
        cv2.imshow('img', color_image)
        if write_images:
            cv2.imwrite("calib_img.png", color_image)
        cv2.waitKey(0)

    img_points_projected, jacobian = cv2.projectPoints(
        obj_points, rvec, tvec, K, dist_coeffs)
    error = cv2.norm(img_points, img_points_projected,
                     cv2.NORM_L2) / len(img_points_projected)
    # print("total error: {}".format(error))

    return pt.transform_from(rot_mat, tvec.transpose())


def calib_hand_eye_method(gripper_to_base_Ts, pattern_to_cam_Ts, method):
    len = gripper_to_base_Ts.__len__()
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

    gripper_to_base_Rs = []
    gripper_to_base_ts = []
    pattern_to_cam_Rs = []
    pattern_to_cam_ts = []
    base_to_tcp = []
    cam_to_pattern = []

    for T in gripper_to_base_Ts:
        gripper_to_base_Rs.append(T[0:3, 0:3])
        gripper_to_base_ts.append(T[0:3, 3])
        base_to_tcp.append(pt.invert_transform(T))
    for T in pattern_to_cam_Ts:
        pattern_to_cam_Rs.append(T[0:3, 0:3])
        pattern_to_cam_ts.append(T[0:3, 3])
        cam_to_pattern.append(pt.invert_transform(T))

    rot, t = cv2.calibrateHandEye(
        gripper_to_base_Rs, gripper_to_base_ts, pattern_to_cam_Rs, pattern_to_cam_ts, method=method)
    cam_to_tcp = pt.transform_from(rot, t.transpose())
    if np.linalg.det(rot) < 0.5 or np.isnan(rot).any():
        return {
            "method": method_str,
            "error": 999999999,
            "tcp_to_cam": cam_to_tcp
        }

    tcp_to_cam = pt.invert_transform(cam_to_tcp)

    error = 0
    last_board = base_to_tcp[0] @ tcp_to_cam @ cam_to_pattern[0]
    for i in range(1, len):
        board = base_to_tcp[i] @ tcp_to_cam @ cam_to_pattern[i]
        error += np.linalg.norm(last_board[0:3, 3] - board[0:3, 3])
        last_board = board
    error /= len

    print("method: ", method_str)
    print("tcp_to_cam: ")
    print(tcp_to_cam)
    print("Euclidean error between boards: ", error)

    return {
        "method": method_str,
        "error": error,
        "tcp_to_cam": tcp_to_cam
    }


def calib_hand_eye(tcp_to_base_Ts, pattern_to_cam_Ts):
    results = []
    for method in [cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_HORAUD,
                   cv2.CALIB_HAND_EYE_ANDREFF, cv2.CALIB_HAND_EYE_DANIILIDIS]:
        results.append(
            calib_hand_eye_method(tcp_to_base_Ts, pattern_to_cam_Ts, method))

    best_error = results[0]["error"]
    best_index = 0
    for i in range(results.__len__()):
        if results[i]["error"] < best_error:
            best_error = results[i]["error"]
            best_index = i

    print("The best method is: " + results[best_index]["method"])
    return results[best_index]["tcp_to_cam"]


def save_calib(K, coeffs, tcp_to_cam=None):
    cv_file = cv2.FileStorage("data/calib.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", K)
    cv_file.write("dist_coeffs", coeffs)
    if tcp_to_cam is not None:
        cv_file.write("tcp_to_cam", tcp_to_cam)
    cv_file.release()
    print("K: ", K)
    print("distortion coefficients: ", coeffs)


def load_calib():
    cv_file = cv2.FileStorage("data/calib.xml", cv2.FILE_STORAGE_READ)
    K = cv_file.getNode("K").mat()
    dist_coeffs = cv_file.getNode("dist_coeffs").mat()
    tcp_to_cam = cv_file.getNode("tcp_to_cam").mat()
    cv_file.release()
    if tcp_to_cam is None:
        tcp_to_cam = np.eye(4)
    return K, dist_coeffs, tcp_to_cam
