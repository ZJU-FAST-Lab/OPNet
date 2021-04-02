#!/usr/bin/python3
# -*- coding:utf8 -*-

import numpy as np
import open3d as o3d
import glob
# import threading


def draw_geometries_with_back_face(geometries, id, visualizer_set):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(id, 600, 600, 600*i+20, 100, True)
    render_option = visualizer.get_render_option()
    render_option.mesh_show_back_face = True
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    visualizer_set.append(visualizer)


def draw_geometries_with_back_face_multithreading(visualizer_set, i):
    visualizer_set[i].run()
    visualizer_set[i].destroy_window()


if __name__ == "__main__":
    # Initialize a HalfEdgeTriangleMesh from TriangleMesh
    dir_path = '/home/pn/forward_opnet/torch_dense/logs/test'
    building_room = '1pXnuDYAj8r_room12__0__'
    pcd_names = glob.glob(dir_path + "/*.ply")
    building_room_names = [s for s in pcd_names if building_room in s]
    # input_name = [s for s in building_room_names if 'input' in s]
    # pred_name = [s for s in building_room_names if 'pred' in s]
    # target_name = [s for s in building_room_names if 'target' in s]
    # plot_pcd_names = [input_name, pred_name, target_name]
    plot_pcd_names = [s for s in building_room_names if ('input' in s or 'pred' in s or 'target' in s and 'mesh' in s)]
    print(plot_pcd_names)

    visualizer_set = []
    # thread_set = []
    for i in range(len(plot_pcd_names)):
        mesh = o3d.io.read_point_cloud(plot_pcd_names[i])
        # mesh = mesh.crop([-1, -1, -1], [1, 0.6, 1])
        inf = 1000 # 显示完全
        upper = 120 #上界   # level1: 50, level2: 30, level3: 20
        left_lower_corner = np.array([[-1], [-1], [-1]])# 截取立方体的左下角
        right_upper_corner = np.array([[inf], [inf], [upper]]) #截取立方体的右上角
        boundingBox = o3d.geometry.AxisAlignedBoundingBox(left_lower_corner, right_upper_corner)#截取立方体
        mesh = mesh.crop(boundingBox)
        # het_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)
        # draw_geometries_with_back_face([het_mesh])
        draw_geometries_with_back_face([mesh], plot_pcd_names[i], visualizer_set)
    for i in range(len(visualizer_set)):
        visualizer_set[i].run()
    visualizer_set[0].destroy_window()

    #     thread = threading.Thread(target=draw_geometries_with_back_face_multithreading(visualizer_set, i))
    #     thread_set.append(thread)
    # thread_set[0].start()
    # thread_set[1].start()
    # thread_set[2].start()
    # thread_set[2].join()
    # thread_set[1].join()
    # thread_set[0].join()

