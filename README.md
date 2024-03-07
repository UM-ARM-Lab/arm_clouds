# `arm_clouds`

A simple module for convenient usage of both NumPy and
PyTorch point clouds. Functionality includes:
- Visualization in Plotly, Open3D, and Rerun with one function call.
- Tracks and enforces consistency between XYZ and (optional) RGB PyTorch data type and device.
- Methods for (un)normalization of RGB values.

# Installation

To install `arm_clouds`, clone the repository and install with `pip install -e .`.

# Usage

This repo includes classes for both single point clouds (`PointCloud`) and point cloud lists (`PointCloudList`).

## `PointCloud` Usage

An example `PointCloud`:
```python
# Your XYZ and RGB data
xyz = torch.rand(3, 100)
rgb = torch.rand(3, 100)  # optional

pc = PointCloud(xyz, rgb)

# One-call-visualization. Can also visualize in rerun and open3d.
pc.visualize_plotly()
```

## `PointCloudList` Usage

You can initialize a `PointCloudList` from a NumPy array or PyTorch tensor of all point clouds:
```python
xyz = np.random.rand(100, 3, 25)
pcl = PointCloudList(xyz)
pcl.visualize_plotly()
```

Or if you're dealing with point clouds with different numbers of points:
```python
pcl = PointCloudList()

# If you haven't already created a `PointCloud` from the raw data:
for xyz, rgb in zip(xyzs, rgbs):
    pcl.append_create_cloud(xyz, rgb)

# If you already have your data in the `PointCloud` format, you can do:
for pc in point_cloud_python_list:
    pcl.append(pc)
# But if you already have multiple `PointCloud` objects in a raw Python list,
# you can directly initialize the `PointCloudList`:
pcl = PointCloudList(point_cloud_python_list)
```

For further usage examples, look at the [test functions](./tests/).