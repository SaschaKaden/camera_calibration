import cv2
import numpy as np


def save(img, image_path):
    cv2.imwrite(image_path, img)


def display(img, window_name="default", destroyable=True):
    cv2.imshow(window_name, img)
    if destroyable is True:
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)


if __name__ == '__main__':
    cap = cv2.VideoCapture(1)  # Define general video driver
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    count = 0
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame

        image = frame  # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display(image, "live stream", False)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            image_file = "image" + str(count) + ".png"
            print("Write image: " + image_file)
            save(image, "data\\capture\\" + image_file)
            count += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
