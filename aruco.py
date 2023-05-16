import cv2
import cv2.aruco as aruco
import numpy as np
from pytransform3d import transformations as pt


def detect_marker(img, K, dist_coeffs, show_image=False, dictionary=aruco.DICT_5X5_100):

    aruco_dict = aruco.Dictionary_get(dictionary)
    aruco_params = aruco.DetectorParameters_create()
    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    aruco_params.cornerRefinementMinAccuracy = 0.01
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
        cv2.drawFrameAxes(img, K, dist_coeffs, rvec, tvec, 190.0)

        if show_image:
            cv2.imshow('Detected Markers', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return pt.transform_from(rot_mat, tvec[0])
    else:
        print('No ArUco markers detected.')
        return None
