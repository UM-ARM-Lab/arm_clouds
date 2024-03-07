import pytest

import torch
import numpy as np

from arm_clouds import PointCloudList


def test_point_cloud_list_initialization_np_array():
    """Test initialization of point cloud list with numpy data"""
    xyzs = np.random.rand(10, 3, 5)
    pcl = PointCloudList(xyzs)

    assert len(pcl) == 10
    assert not pcl.is_torch()
    assert not pcl.has_rgb()


def test_point_cloud_list_initialization_torch_tensor():
    """Test initialization of point cloud list with torch data"""
    xyzs = torch.rand(10, 3, 5)
    pcl = PointCloudList(xyzs)

    assert len(pcl) == 10
    assert pcl.is_torch()
    assert not pcl.has_rgb()


def test_point_cloud_list_append_create_cloud():
    """Test appending a new cloud to the list"""
    pcl = PointCloudList()
    xyz = np.random.rand(3, 5)
    rgb = np.random.rand(3, 5)
    pcl.append_create_cloud(xyz, rgb)

    assert len(pcl) == 1
    assert not pcl.is_torch()
    assert pcl.has_rgb()
