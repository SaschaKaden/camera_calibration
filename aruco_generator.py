import numpy as np
import cv2

ARUCO_DICT = {
    cv2.aruco.DICT_4X4_50: "DICT_4X4_50",
    cv2.aruco.DICT_4X4_100: "DICT_4X4_100",
    cv2.aruco.DICT_4X4_250: "DICT_4X4_250",
    cv2.aruco.DICT_4X4_1000: "DICT_4X4_1000",
    cv2.aruco.DICT_5X5_50: "DICT_5X5_50",
    cv2.aruco.DICT_5X5_100: "DICT_5X5_100",
    cv2.aruco.DICT_5X5_250: "DICT_5X5_250",
    cv2.aruco.DICT_5X5_1000: "DICT_5X5_1000",
    cv2.aruco.DICT_6X6_50: "DICT_6X6_50",
    cv2.aruco.DICT_6X6_100: "DICT_6X6_100",
    cv2.aruco.DICT_6X6_250: "DICT_6X6_250",
    cv2.aruco.DICT_6X6_1000: "DICT_6X6_1000",
    cv2.aruco.DICT_7X7_50: "DICT_7X7_50",
    cv2.aruco.DICT_7X7_100: "DICT_7X7_100",
    cv2.aruco.DICT_7X7_250: "DICT_7X7_250",
    cv2.aruco.DICT_7X7_1000: "DICT_7X7_1000",
    cv2.aruco.DICT_ARUCO_ORIGINAL: "DICT_ARUCO_ORIGINAL",
    cv2.aruco.DICT_APRILTAG_16h5: "DICT_APRILTAG_16h5",
    cv2.aruco.DICT_APRILTAG_25h9: "DICT_APRILTAG_25h9",
    cv2.aruco.DICT_APRILTAG_36h10: "DICT_APRILTAG_36h10",
    cv2.aruco.DICT_APRILTAG_36h11: "DICT_APRILTAG_36h11"
}

id = 24
type = cv2.aruco.DICT_5X5_100
output = "data/tags/"
size = 1000

if __name__ == '__main__':
    dictionary = cv2.aruco.getPredefinedDictionary(type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    print("Generating ArUCo tag of type '{}' with ID '{}'".format(type, id))
    marker_img = cv2.aruco.generateImageMarker(dictionary, id, size)

    # Save the tag generated
    tag_name = output + ARUCO_DICT[type] + "_ID_" + str(id) + ".png"
    cv2.imwrite(tag_name, marker_img)
    cv2.imshow("ArUCo Tag", marker_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
