#ifndef _FSM_H_
#define _FSM_H_

#include "occ_grid/occ_map.h"
#include "krrtstar/krrtplanner.h"
#include "krrtstar/topo_prm.h"
#include "poly_opt/traj_optimizer.h"
#include "vis_utils/planning_visualization.h"
#include "quadrotor_msgs/PolynomialTrajectory.h"
#include "quadrotor_msgs/PositionCommand.h"

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Empty.h>

namespace fast_planner
{
class FSM
{
public:
  FSM();
  ~FSM();
  void init(ros::NodeHandle& nh);
    
private:
  bool searchForTraj(Vector3d start_pos, Vector3d start_vel, Vector3d start_acc,  
                     Vector3d end_pos, Vector3d end_vel, Vector3d end_acc, 
                     double search_time, bool bidirection);
  void sendTrajToServer(const RRTNodeVector& path_nodes);
  void sendEStopToServer();
  bool reachGoal(double radius);
  bool checkForReplan();
  Eigen::VectorXd getReplanStateFromPath(double t, const RRTNodeVector& path_nodes);
  /*
   * replan in t second from current state
   */
  bool replanOnce(double t);
  bool getBypass(RRTNodeVector& bypass_node_e2s);
  bool optimize(RRTNodeVector& path_node_g2s, const Eigen::Vector3d &init_acc);
  Eigen::Vector3d getRPY(const Eigen::Quaterniond& quat);
  Eigen::Vector3d getAccFromRPY(const Eigen::Vector3d& rpy);

  // map, environment, planner 
  OccMap::Ptr env_ptr_;
  TopologyPRM::Ptr topo_prm_;
  KRRTPlanner::KRRTPlannerPtr krrt_planner_ptr_;
  TrajOptimizer::Ptr optimizer_ptr_;
  PlanningVisualization::Ptr vis_;
  
  // ros 
  ros::NodeHandle nh_;
  ros::Subscriber goal_sub_, final_goal_sub_;
  ros::Subscriber qrcode_pose_sub_;
  ros::Subscriber ref_traj_sub_;
  ros::Subscriber track_err_trig_sub_;
  ros::Publisher traj_pub_;
  ros::Timer execution_timer_;
  ros::Timer receding_horizon_timer_;
  void qrcodeCallback(const geometry_msgs::PointStamped::ConstPtr& msg);
  void goalCallback(const quadrotor_msgs::PositionCommand::ConstPtr& goal_msg);
  void executionCallback(const ros::TimerEvent& event);
  void refTrajCallback(const quadrotor_msgs::PolynomialTrajectory& traj);
  void trackErrCallback(const std_msgs::Empty& msg);
  void rcdHrzCallback(const ros::TimerEvent& event);
  void finalGoalCallback(const geometry_msgs::PointStamped::ConstPtr& msg);
  bool computeInterGoalPos(Eigen::Vector3d& inte_pos, const Eigen::Vector3d& curr_pos, const Eigen::Vector3d& goal, double range);

  // execution states 
  enum MACHINE_STATE{
    INIT, 
    WAIT_GOAL, 
    GENERATE_TRAJ, 
    FOLLOW_TRAJ,
    REPLAN_TRAJ, 
    EMERGENCY_TRAJ,
    REFINE_REMAINING_TRAJ
  };
  MACHINE_STATE machine_state_;
  void changeState(MACHINE_STATE new_state);
  void printState();
  
  // params 
  int sense_grid_size_;
  double resolution_;
  bool track_err_replan_, allow_track_err_replan_, close_goal_traj_;
  bool new_goal_, started_, use_optimization_, replan_, bidirection_, get_final_goal_, conservative_;
  Eigen::Vector3d last_goal_pos_;
  double replan_time_;
  Eigen::Vector3d start_pos_, start_vel_, start_acc_, end_pos_, end_vel_, end_acc_;
  Eigen::Vector3d final_goal_pos_;
  ros::Time cuur_traj_start_time_;
  RRTNodeVector path_node_g2s_, emergency_path_node_g2s_, bypass_node_e2s_;
  RRTNodePtrVector traj_tree_;
  int traj_tree_node_num_;
  Eigen::Vector3d emergency_stop_pos_, pos_about_to_collide_;
  double remain_safe_time_, e_stop_time_margin_;

  // configuration for reference trajectory
  RRTNodeVector ref_traj_g2s_;
  double t_in_ref_traj_;
  int idx_;
  int n_segment_;
  Eigen::VectorXd time_;
  Eigen::MatrixXd coef_[3];
  vector<int> order_;
  ros::Time final_time_;
  ros::Time start_time_;
  bool receive_ref_traj_;

  // random goal
  uniform_real_distribution<double> angle_rand_, radius_rand_, height_rand_;
  mt19937_64 gen_;
};
    
    
}


#endif //_FSM_H_
