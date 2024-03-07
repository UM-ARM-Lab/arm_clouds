import pytest

import numpy as np
import torch

from arm_clouds import PointCloud

PC_SHAPE = (3, 20)


def test_cloud_initialization_np():
    """Test initialization of point cloud with numpy data"""
    xyz = np.random.rand(*PC_SHAPE)
    rgb = np.random.rand(*PC_SHAPE)

    pc = PointCloud(xyz, rgb)

    assert pc.xyz.shape == PC_SHAPE
    assert pc.rgb.shape == PC_SHAPE
    assert not pc.is_torch
    assert pc.has_rgb
    assert pc.is_rgb_normalized


def test_cloud_initialization_torch():
    """Test initialization of point cloud with torch data"""
    xyz = torch.rand(*PC_SHAPE)
    rgb = torch.rand(*PC_SHAPE)

    pc = PointCloud(xyz, rgb)

    assert pc.xyz.shape == PC_SHAPE
    assert pc.rgb.shape == PC_SHAPE
    assert pc.is_torch
    assert pc.has_rgb
    assert pc.is_rgb_normalized


def test_cloud_initialization_np_no_rgb():
    """Test initialization of point cloud with numpy data and no RGB"""
    xyz = np.random.rand(*PC_SHAPE)

    pc = PointCloud(xyz)

    assert pc.xyz.shape == PC_SHAPE
    assert pc.rgb is None
    assert not pc.is_torch
    assert not pc.has_rgb


def test_cloud_initialization_torch_no_rgb():
    """Test initialization of point cloud with torch data and no RGB"""
    xyz = torch.rand(*PC_SHAPE)

    pc = PointCloud(xyz)

    assert pc.xyz.shape == PC_SHAPE
    assert pc.rgb is None
    assert pc.is_torch
    assert not pc.has_rgb


def test_cloud_downsample():
    # Creates 5 duplicates points to ensure that downsampling returns a single point.
    xyz = torch.zeros(3, 5)
    xyz[0, :] = 1
    xyz[1, :] = 2
    xyz[2, :] = 3
    rgb = (torch.rand(3, 5) * 255).to(torch.uint8)

    pc = PointCloud(xyz, rgb)
    pc.downsample()

    assert pc.xyz.shape == (3, 1)
    assert pc.rgb.shape == (3, 1)
    assert not pc.is_rgb_normalized


if __name__ == "__main__":
    test_cloud_downsample()
