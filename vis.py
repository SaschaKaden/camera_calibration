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


def view_poses(end_index, title, poses_1, suffix_1="a", poses_2=None, suffix_2="b", poses_3=None, suffix_3="c"):
    fig = plt.figure(figsize=(10, 5))
    ax = make_3d_axis(1, 121, unit="m")

    for i in range(end_index):
        pt.plot_transform(ax, A2B=poses_1[i], name=suffix_1 + str(i))

    if poses_2 is not None:
        for i in range(end_index):
            pt.plot_transform(ax, A2B=poses_2[i], name=suffix_2 + str(i))

    if poses_3 is not None:
        for i in range(end_index):
            pt.plot_transform(ax, A2B=poses_3[i], name=suffix_3 + str(i))

    ax.view_init(elev=30, azim=20)
    ax.set_title(title)
    fig.show()
