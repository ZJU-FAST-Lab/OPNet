#ifndef _WAYPOINT_OPT_H_
#define _WAYPOINT_OPT_H_

#include "occ_grid/occ_map.h"
#include "krrtstar/raycast.h"
#include <ros/ros.h>
#include "Eigen/Core"

using Eigen::Vector3d;
using Eigen::MatrixXd;
using Eigen::VectorXi;

namespace fast_planner
{
class wpOptimizer {
public:
  wpOptimizer(ros::NodeHandle& nh);
  ~wpOptimizer();
  void setWp(const vector<Vector3d>& wp) {wp_ = wp; wp_num_ = wp.size();};
  void setGrad(const vector<Vector3d>& grad) {grad_ = grad;};
  void setEnv(const OccMap::Ptr& env) {env_ptr_ = env;};
  void optimize();
  void getWp(vector<Vector3d>& wp) {wp = wp_;};
  
  typedef shared_ptr<wpOptimizer> Ptr;
  
private:
  OccMap::Ptr env_ptr_;
  
  vector<Vector3d> wp_;
  VectorXi wp_status_;
  vector<Vector3d> grad_;
  int wp_num_;
  double safety_radius_;
  double clear_radius_;
  double resolution_;
  
  bool calGrad(MatrixXd& grad, VectorXi& wp_status);
  
};
}

#endif