import os

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

def show_points(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def get_colors(n):
    colors = [plt.cm.Spectral(each)[:3] for each in np.linspace(0, 1, n)]
    colors_int = (np.asarray(colors)*255).astype(np.int32)
    colors_float = np.asarray(colors)
    return colors_float

def show_convechull(points, labels, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    objects = [pcd]
    for i in range(colors.shape[0]-1):
        obj_pcd = o3d.geometry.PointCloud()
        obj_points = points[labels==i]
        obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
        obj = obj_pcd.get_axis_aligned_bounding_box()
        obj.color = colors[i]
        objects.append(obj)
    o3d.visualization.draw_geometries(objects)