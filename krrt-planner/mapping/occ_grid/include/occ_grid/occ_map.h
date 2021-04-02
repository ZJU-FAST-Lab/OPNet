#ifndef _OCC_MAP_H
#define _OCC_MAP_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Eigen>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <ocp_msgs/OdomPcl.h>
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <ocp_msgs/PredictPCL.h>
#include "occ_grid/raycast.h"
#include <std_msgs/Int8.h>

// #include <tf2_ros/transform_listener.h>

#include <queue>

#define logit(x) (log((x) / (1 - (x))))
#define INVALID_IDX -1

using namespace std;

namespace fast_planner
{
class OccMap
{
public:
  OccMap() {}
  ~OccMap() {}
  void init(ros::NodeHandle& nh);

  bool odomValid() { return have_odom_; }
  bool mapValid() { return (global_map_valid_||local_map_valid_); }
  Eigen::Vector3d get_curr_posi() { return curr_posi_; }
	Eigen::Vector3d get_curr_twist() {return curr_twist_; }
  Eigen::Vector3d get_curr_acc() {return curr_acc_; }
  Eigen::Quaterniond get_curr_quaternion() {return curr_q_; }
  double getResolution() { return resolution_; }
  Eigen::Vector3d getOrigin() { return origin_; }
  void resetBuffer(Eigen::Vector3d min, Eigen::Vector3d max);
  void setOccupancy(Eigen::Vector3d pos);
  int getVoxelState(Eigen::Vector3d pos);
  int getVoxelState(Eigen::Vector3i id);
  int getOriginalVoxelState(Eigen::Vector3d pos);
	ros::Time getLocalTime() { return latest_odom_time_; };

  void posToIndex(Eigen::Vector3d pos, Eigen::Vector3i& id);
  void indexToPos(Eigen::Vector3i id, Eigen::Vector3d& pos);
  
  typedef shared_ptr<OccMap> Ptr;
  
private:
  RayCaster raycaster_;
  std::vector<double> occupancy_buffer_;  // 0 is free, 1 is occupied
  std::vector<double> pred_occupancy_buffer_;  // 0 is free, 1 is occupied

  std::vector<int> known_buffer_;  // 0 is unknown, 1 is known
  std::vector<float> local_occupancy_buffer_;
  std::vector<Eigen::Vector3d> local_idx_buffer_;

  int known_threshold_;
  void predLocalOcc(const Eigen::Vector3d cur_position);
  void publishLocalOccCallback(const ros::TimerEvent& e);

  // map property
  Eigen::Vector3d min_range_, max_range_;  // map range in pos
  Eigen::Vector3i grid_size_, local_grid_size_;              // map size in index
  Eigen::Vector3d local_range_min_, local_range_max_;

  bool isInMap(Eigen::Vector3d pos);

  Eigen::Vector3d origin_, map_size_;
  double resolution_, resolution_inv_;
  Eigen::Matrix4d T_ic0_, T_ic1_, T_ic2_, T_ic3_;

  bool have_odom_;
	Eigen::Vector3d curr_posi_, curr_twist_, curr_acc_, raycast_posi_;
	Eigen::Quaterniond curr_q_;

  // ros
  ros::NodeHandle node_;
  ros::Subscriber indep_odom_sub_;
	ros::Subscriber global_cloud_sub_;
	ros::Subscriber cloud_odom_sub_;
  ros::Timer global_occ_vis_timer_, local_occ_vis_timer_, collision_check_timer_;
	
  // for vis
	ros::Time latest_odom_time_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr curr_view_cloud_ptr_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr history_view_cloud_ptr_, history_pred_cloud_ptr_;
	ros::Publisher curr_view_cloud_pub_, hist_view_cloud_pub_, hist_pred_cloud_pub_; 
	ros::Publisher pose_vis_pub_, twist_vis_pub_, acc_vis_pub_;
  ros::Publisher local_occ_pub_, added_occ_pub_;
  ros::ServiceClient pred_client_;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry> SyncPolicyImageOdom;
	typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImageOdom>> SynchronizerImageOdom;
  SynchronizerImageOdom sync_image_odom_;
  shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub_;
  shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
	void depthOdomCallback(const sensor_msgs::ImageConstPtr& disp_msg, 
                         const nav_msgs::OdometryConstPtr& odom, 
                         const Eigen::Matrix4d& T_ic, 
                         Eigen::Matrix4d& last_T_wc, 
                         cv::Mat& last_depth_image, 
                         const string& camera_name);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> SyncPolicyPclOdom;
	typedef shared_ptr<message_filters::Synchronizer<SyncPolicyPclOdom>> SynchronizerPclOdom;
  SynchronizerPclOdom sync_pcl_odom_;
  shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> pcl_sub_;
  // void pclOdomCallback(const sensor_msgs::PointCloud2ConstPtr& pointcloud, 
  //                       const nav_msgs::OdometryConstPtr& odom, 
  //                       const Eigen::Matrix4d& T_ic
  //                     );

  // shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> pcl_sub_;
  void pclOdomCallback(const ocp_msgs::OdomPclConstPtr& odom_pcl);

  cv::Mat depth_image_;
  cv::Mat last_depth0_image_;
  Eigen::Matrix4d last_T_wc0_;
  void projectDepthImage(const Eigen::Matrix3d& K, 
                         const Eigen::Matrix4d& T_wc, const cv::Mat& depth_image, 
                         Eigen::Matrix4d& last_T_wc, cv::Mat& last_depth_image, ros::Time r_s);
  void raycastProcess(const Eigen::Vector3d& t_wc);
  void raycastPclProcess(const Eigen::Vector3d& t_wc, const sensor_msgs::PointCloud2 pointcloud_msg); //ConstPtr&
  int setCacheOccupancy(Eigen::Vector3d pos, int occ);

  void indepOdomCallback(const nav_msgs::OdometryConstPtr& msg);
  void globalOccVisCallback(const ros::TimerEvent& e);
  void localOccVisCallback(const ros::TimerEvent& e);
	void globalCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

  bool has_global_cloud_, has_first_depth_, use_global_map_;
	bool global_map_valid_, local_map_valid_;

  // map fusion 
  double fx_, fy_, cx_, cy_;
  int rows_, cols_;
  vector<Eigen::Vector3d> proj_points_;
  int proj_points_cnt_;
  vector<int> cache_hit_, cache_all_;
  vector<int> cache_traverse_, cache_rayend_;
  int raycast_num_;
  queue<Eigen::Vector3i> cache_voxel_;
	int img_col_, img_row_;
  Eigen::Matrix3d K_depth_;
  Eigen::Vector3d sensor_range_;
  bool show_raw_depth_, show_filter_proj_depth_;
  bool fully_initialized_;

  /* projection filtering */
  double depth_filter_maxdist_, depth_filter_mindist_, depth_filter_tolerance_;
  int depth_filter_margin_;
  bool use_shift_filter_;
  double depth_scale_;
  int skip_pixel_;

  /* raycasting */
  double p_hit_, p_miss_, p_min_, p_max_, p_occ_;
  double prob_hit_log_, prob_miss_log_, clamp_min_log_, clamp_max_log_;
	double min_occupancy_log_;
  double min_ray_length_, max_ray_length_;
  
  /* local pointcloud */
  int local_dimx_, local_dimy_, local_dimz_;

	/* origin pcl show */
	void pubPointCloudFromDepth(const std_msgs::Header& header, 
                              const cv::Mat& depth_img, 
                              const Eigen::Matrix3d& intrinsic_K, 
                              const string& camera_name);
	ros::Publisher origin_pcl_pub_;
  ros::Publisher projected_pc_pub_;

  /* network param*/
  bool use_pred_;
  std::string model_path_;
  double pred_occ_thresord_;
  Eigen::Vector3d init_odom_;


  /* collision check*/
  bool unknown_as_free_, use_pred_for_collision_, raycast_fisttime_, vis_unknown_;
  int collisions_;
	ros::Publisher collision_pub_;
  bool collisionCheck();
  void publishCollisionNum(const ros::TimerEvent& e);
};
}  // namespace fast_planner

#endif
