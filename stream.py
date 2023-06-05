import cv2 as cv
import numpy as np
import pyrealsense2 as rs


def save(img, image_path):
    cv.imwrite(image_path, img)


def display(img, window_name="default", destroyable=True):
    cv.imshow(window_name, img)
    if destroyable is True:
        cv.waitKey(0)
        cv.destroyWindow(window_name)


if __name__ == '__main__':
    # show live streams from camera
    cap = cv.VideoCapture(0)  # Define general video driver

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    count = 0
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame

        image = frame  # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display(image, "live stream", False)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            image_file = "image" + str(count) + ".png"
            print("Write image: " + image_file)
            save(image, "data\\capture\\" + image_file)
            count += 1

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
