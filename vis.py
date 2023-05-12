import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transform_manager import TransformManager
from pytransform3d import transformations as pt


def set_label(ax, title):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    # ax.set_xlim3d(-1, 1)
    # ax.set_ylim3d(-1, 1)
    # ax.set_zlim3d(-0.1, 1)
    ax.set_proj_type('persp')  # FOV = 90 deg


def view_points(transforms, title):
    fig = plt.figure(figsize=(10, 5))
    ax = make_3d_axis(1, 121, unit="m")

    for transform in transforms:
        pt.plot_transform(ax, A2B=transform)

    ax.view_init(elev=30, azim=20)
    ax.set_title(title)
    fig.show()


def view_tcp_boards(base_to_grasp, boards, title):
    tm = TransformManager()
    count = 0
    for transform in base_to_grasp:
        tm.add_transform("base", "grasp" + str(count), transform)
        count += 1

    count = 0
    for transform in boards:
        tm.add_transform("base", "board" + str(count), transform)
        count += 1

    fig = plt.figure(figsize=(10, 5))
    ax = make_3d_axis(1, 121)
    ax = tm.plot_frames_in("base", ax=ax, alpha=0.4, s=0.2)
    ax.view_init(elev=30, azim=20)

    set_label(ax, title)
    fig.show()


def view_boards(base_to_grasp, boards, new_boards, title):
    tm = TransformManager()
    count = 0
    for transform in base_to_grasp:
        tm.add_transform("base", "grasp" + str(count), transform)
        count += 1

    count = 0
    for transform in boards:
        tm.add_transform("base", "board" + str(count), transform)
        count += 1

    for transform in new_boards:
        tm.add_transform("base", "new_board" + str(count), transform)
        count += 1

    fig = plt.figure(figsize=(10, 5))
    ax = make_3d_axis(1, 121)
    ax = tm.plot_frames_in("base", ax=ax, alpha=0.6, s=0.2)
    ax.view_init(elev=30, azim=20)

    set_label(ax, title)
    fig.show()


def view_poses(base_to_tcp_Ts, tcp_to_cam, cam_to_pattern_Ts, title):
    fig = plt.figure(figsize=(10, 5))
    ax = make_3d_axis(1, 121, unit="m")

    count = 0
    for i in range(len(base_to_tcp_Ts)):
        pt.plot_transform(ax, A2B=base_to_tcp_Ts[i], name="t" + str(count))
        pt.plot_transform(
            ax, A2B=base_to_tcp_Ts[i] @ tcp_to_cam @ cam_to_pattern_Ts[i], name="b" + str(count))
        count += 1

    ax.view_init(elev=30, azim=20)
    ax.set_title(title)
    fig.show()
