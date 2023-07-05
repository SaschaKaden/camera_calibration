import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import calib
import detector
from pytransform3d import transformations as pt


SHOW_IMAGES = False
SAVE_PC = False


def save(img, image_path):
    cv.imwrite(image_path, img)


def display(img, window_name="default", destroyable=True):
    img = cv.resize(img, (1280, 720))
    cv.imshow(window_name, img)
    if destroyable is True:
        cv.waitKey(0)
        cv.destroyWindow(window_name)


def init_rs():
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(config)
    return pipeline


if __name__ == '__main__':
    K, dist_coeffs, tcp_to_cam = calib.load_calib()
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2)
    decimation_filter = rs.decimation_filter()
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()

    pipeline = init_rs()

    count = 0
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_frame = decimate.process(depth_frame)
        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv.cvtColor(color_image, cv.COLOR_RGB2BGR)
        gray_image = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
        display(color_image, "live stream", False)

        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        points.export_to_ply("data/1.ply", color_frame);

        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            image_file = "image" + str(count) + ".png"
            print("Write image: " + image_file)
            save(color_image, "data\\capture\\" + image_file)
            count += 1

    # When everything done, release the capture
    cv.destroyAllWindows()
