# OPNet & krrt-planner with map-prediction

Video at:
https://www.youtube.com/watch?v=Qb3ni_j0Dic

Preprint:
Learning-based 3D Occupancy Prediction for Autonomous Navigation in Occluded Environments

https://arxiv.org/abs/2011.03981


#

c++ realization of the paper: Kinodynamic RRT*: Asymptotically Optimal Motion Planning for Robots with Linear Dynamics

Building:
The depth_sensor_simulator package in uav_simulator is alternative to build with GPU or CPU to render the depth sensor measurement. By default, it is set to build with GPU in CMakeLists:

Dependencies:

1. ROS (I am using Ubuntu 16.04 and ROS Kinetic, other versions maybe also usable)

2. CUDA (I am using 10.2)

3. for branch "master" (inference using NVIDIA TensorRT):
    * TensorRT 7.0.0+cuda10.2 (with TensorRT ONNX  libraries)
     
4. for branch "torch" (inference using a Python node with Pytorch ):
    * python > 2.7
    * numpy, Ipython, tensorboardX
    * Pytorch (I am using 1.3.0, later versions are also usable)


# set(ENABLE_CUDA false)
set(ENABLE_CUDA true)

Remember to change the 'arch' and 'code' flags according to your graphics card devices. 
for branch "torch":
* Remember to change the model path in net_node.py -- init_param function

Run Simulation:
1. roslaunch state_machine rviz.launch  (to open rviz for visualization)
2. roslaunch state_machine bench_with_pred.launch    (generate environment, start simulator)
3. roslaunch state_machine bench_with_pred.launch  or  bench_aggres.launch    or    bench_safe.launch    (test)

