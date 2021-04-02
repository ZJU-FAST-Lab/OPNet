#ifndef _VISUALIZE_RVIZ_H_
#define _VISUALIZE_RVIZ_H_

#include "utils.h"

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

class VisualRviz{
public:
    VisualRviz();
    VisualRviz(ros::NodeHandle nh);
    
    ros::NodeHandle nh_;
    ros::Publisher tree_node_pos_point_pub_;
    ros::Publisher tree_node_vel_vec_pub_;
    ros::Publisher rand_sample_pos_point_pub_;
    ros::Publisher rand_sample_vel_vec_pub_;
    ros::Publisher rand_sample_acc_vec_pub_;
    ros::Publisher tree_trunks_pub_;
    ros::Publisher tree_traj_pos_point_pub_;
    ros::Publisher tree_traj_vel_vec_pub_;
    ros::Publisher tree_traj_acc_vec_pub_;
    ros::Publisher a_traj_pos_point_pub_;
    ros::Publisher a_traj_vel_vec_pub_;
    ros::Publisher a_traj_acc_vec_pub_;
    ros::Publisher first_traj_pos_point_pub_;
    ros::Publisher first_traj_vel_vec_pub_;
    ros::Publisher first_traj_acc_vec_pub_;
    ros::Publisher best_traj_pos_point_pub_;
    ros::Publisher best_traj_vel_vec_pub_;
    ros::Publisher best_traj_acc_vec_pub_;
    ros::Publisher bypass_traj_pos_point_pub_;
    ros::Publisher bypass_traj_vel_vec_pub_;
    ros::Publisher bypass_traj_acc_vec_pub_;
    ros::Publisher start_and_goal_pub_;
    ros::Publisher skeleton_pub_;
    ros::Publisher grad_pub_;
    ros::Publisher topo_pub_;
    ros::Publisher surface_pub_;
    ros::Publisher orphans_pos_pub_;
    ros::Publisher orphans_vel_vec_pub_;
    
    void visualizeAllTreeNode(RRTNode* root, ros::Time local_time);
    void visualizeSampledState(const State& node, ros::Time local_time);
    void visualizeTreeTraj(const std::vector<State>& x, const std::vector<Control>& u, ros::Time local_time);
    void visualizeATraj(const std::vector<State>& x, const std::vector<Control>& u, ros::Time local_time);
    void visualizeFirstTraj(const std::vector<State>& x, const std::vector<Control>& u, ros::Time local_time);
    void visualizeBestTraj(const std::vector<State>& x, const std::vector<Control>& u, ros::Time local_time);
    void visualizeBypassTraj(const std::vector<State>& x, const std::vector<Control>& u, ros::Time local_time);
    void visualizeStartAndGoal(State start, State goal, ros::Time local_time);
    void visualizeSkeleton(const std::vector<Eigen::Vector3d>& skeleton, const std::vector<Eigen::Vector3d>& grads, const vector< double >& dists, ros::Time local_time);
    void visualizeTrajCovering(const std::vector<Eigen::Vector3d>& covering_grids, double grid_len, ros::Time local_time);
    void visualizeTopo(const std::vector<Eigen::Vector3d>& p_head, const std::vector<Eigen::Vector3d>& tracks, ros::Time local_time);
    void visualizeOrphans(const std::vector<State>& ophs, ros::Time local_time);
    
    void visualizeTopoPaths(const vector<vector<Eigen::Vector3d>>& paths, int id, Eigen::Vector4d color, ros::Time local_time);
};


#endif