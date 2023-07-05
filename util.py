import numpy as np
import cv2
from pytransform3d import transformations as pt


def load_transforms_file(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    base_to_ee = cv_file.getNode("base_to_ee").mat()
    ee_to_base = cv_file.getNode("ee_to_base").mat()
    base_to_ee[:3, :3] = np.linalg.qr(base_to_ee[:3, :3], mode='complete')[0]
    ee_to_base[:3, :3] = np.linalg.qr(base_to_ee[:3, :3], mode='complete')[0]

    # print(base_to_ee @ ee_to_base)
    return base_to_ee, ee_to_base


def load_transforms(folder, start_index, end_index):
    tcp_to_base_Ts = []
    base_to_tcp_Ts = []

    for i in range(start_index, end_index + 1):
        base_to_tcp, tcp_to_base = load_transforms_file(folder + "{}.xml".format(i))
        tcp_to_base_Ts.append(tcp_to_base)
        base_to_tcp_Ts.append(base_to_tcp)

    return tcp_to_base_Ts, base_to_tcp_Ts
