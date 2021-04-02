#!/usr/bin/python3
# -*- coding:utf8 -*-

import numpy as np
import open3d as o3d


def draw_geometries_with_back_face(geometries):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    render_option = visualizer.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.show_coordinate_frame = True
    # render_option.point_size = 20
    render_option.point_show_normal = True
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    # Initialize a HalfEdgeTriangleMesh from TriangleMesh
    # mesh = o3d.io.read_triangle_mesh('/home/wlz/catkin_ws/src/opnet/logs/iter12259-epoch40/val/2n8kARJN3HM_room0__0___13_f0target-mesh.ply')
    mesh = o3d.io.read_point_cloud('/home/wlz/vox_ws/src/opnet/src/torch_dense/logs/test/ur6pFq6Qu1A_room0__0___5_f0input-mesh.xyzrgb')
    # mesh = mesh.crop([-1, -1, -1], [1, 0.6, 1])
    # inf = 1000#显示完全
    # upper = 120#上界   # level1: 50, level2: 30, level3: 20
    # left_lower_corner = np.array([[-1], [-1], [-1]])#截取立方体的左下角
    # right_upper_corner = np.array([[inf], [inf], [upper]])#截取立方体的右上角
    # boundingBox = o3d.geometry.AxisAlignedBoundingBox(left_lower_corner, right_upper_corner)#截取立方体
    # mesh = mesh.crop(boundingBox)
    # het_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)
    # draw_geometries_with_back_face([het_mesh])
    draw_geometries_with_back_face([mesh])

