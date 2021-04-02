#ifndef _PLANNING_VISUALIZATION_H_
#define _PLANNING_VISUALIZATION_H_

#include <Eigen/Eigen>
#include <iostream>
#include <algorithm>
#include <vector>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

using std::vector;
namespace fast_planner
{
class PlanningVisualization
{
private:
  enum FAST_ID
  {
    GOAL = 1,
    PATH = 200,
    BSPLINE = 300,
    BSPLINE_CTRL_PT = 400,
    POLY_TRAJ = 500
  };

  enum TOPO_ID
  {
    GRAPH_NODE = 1,
    GRAPH_EDGE = 100,
    RAW_PATH = 200,
    FILTERED_PATH = 300,
    SELECT_PATH = 400
  };

  /* data */
  ros::NodeHandle node;
  ros::Publisher traj_pub_;     // 0
  ros::Publisher topo_pub_;     // 1
  ros::Publisher predict_pub_;  // 2
  
  ros::Publisher ref_traj_pos_point_pub_;
  ros::Publisher ref_traj_vel_vec_pub_;
  ros::Publisher ref_traj_acc_vec_pub_;
  ros::Publisher opti_traj_pos_point_pub_;
  ros::Publisher opti_traj_vel_vec_pub_;
  ros::Publisher opti_traj_acc_vec_pub_;
  ros::Publisher bspl_opti_traj_pos_point_pub_;
  ros::Publisher bspl_opti_traj_vel_vec_pub_;
  ros::Publisher bspl_opti_traj_acc_vec_pub_;
  
  ros::Publisher voxel_pub_;
  ros::Publisher cover_pub_;
  
  vector<ros::Publisher> pubs_;

  int last_graph_num_;
  int last_path_num_;
  int last_guide_num_;
  int last_bspline_num_;

public:
  PlanningVisualization(/* args */)
  {
  }
  ~PlanningVisualization()
  {
  }

  PlanningVisualization(ros::NodeHandle& nh);

  void displaySphereList(const vector<Eigen::Vector3d>& list, double resolution, Eigen::Vector4d color, int id,
                         int pub_id = 0);
  void displayCubeList(const vector<Eigen::Vector3d>& list, double resolution, Eigen::Vector4d color, int id,
                       int pub_id = 0);
  void displayLineList(const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2, double line_width,
                       Eigen::Vector4d color, int id, int pub_id = 0);

  void drawPath(const vector<Eigen::Vector3d>& path, double resolution, Eigen::Vector4d color, int id = 0);


  void drawGoal(Eigen::Vector3d goal, double resolution, Eigen::Vector4d color, int id = 0);

  void drawSelectTopoPaths(vector<vector<vector<Eigen::Vector3d>>>& paths, double line_width);

  void drawFilteredTopoPaths(vector<vector<vector<Eigen::Vector3d>>>& guides, double line_width);

  void visualizeOptiTraj(const std::vector<Eigen::Matrix<double,6,1>>& x, const std::vector<Eigen::Matrix<double,3,1>>& u, ros::Time local_time);
  void visualizeRefTraj(const std::vector<Eigen::Matrix<double,6,1>>& x, const std::vector<Eigen::Matrix<double,3,1>>& u, ros::Time local_time);
  void visualizeBsplOptiTraj(const std::vector<Eigen::Matrix<double,6,1>>& x, const std::vector<Eigen::Matrix<double,3,1>>& u, ros::Time local_time);
  void visualizeCollide(std::vector<Eigen::Vector3d> positions, ros::Time local_time);
  void visualizeTrajCovering(const vector< Eigen::Vector3d >& covering_grids, double grid_len, ros::Time local_time);

  
  Eigen::Vector4d getColor(double h, double alpha = 1.0);

  typedef std::shared_ptr<PlanningVisualization> Ptr;
};
}  // namespace fast_planner
#endif