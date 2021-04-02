#include "krrtstar/krrtplanner.h"
#include "krrtstar/raycast.h"
#include <float.h>
#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/console.h>

using Eigen::Vector3d;
using Eigen::Vector2d;

#ifdef TIMING
double check_path_time = 0.0;
int check_path_nums = 0;
double check_cost_time = 0.0;
int check_cost_nums = 0;
double calculate_state_control_time = 0.0;
int calculate_state_control_nums = 0;
double kd_insert_time = 0.0;
int kd_insert_nums = 0;
double kd_query_time = 0.0;
int kd_query_nums = 0;
double linear_search_parent_time = 0.0;
int linear_search_parent_nums = 0;
double linear_rewire_time = 0.0;
int linear_rewire_nums = 0;
double kd_search_parent_time = 0.0;
int kd_search_parent_nums = 0;
double kd_rewire_time = 0.0;
int kd_rewire_nums = 0;
double state_time = 0.0;
double new_time = 0.0;
double cal_forward_bound_time = 0.0;
int cal_forward_bound_nums = 0;
double cal_backward_bound_time = 0.0;
int cal_backward_bound_nums = 0;
double connect_time = 0.0;
int connect_nums = 0;
double solve_poly_time = 0.0;
int solve_poly_nums = 0;
#endif

namespace fast_planner
{
KRRTPlanner::KRRTPlanner(ros::NodeHandle& nh): vis(nh) 
{
}

KRRTPlanner::~KRRTPlanner()
{
  for (int i = 0; i < tree_node_nums_; i++) {
    delete start_tree_[i];
  }
  for (auto &ptr : path_on_first_search_) {
    delete ptr;
  }
  delete close_goal_node_;
}

void KRRTPlanner::init(ros::NodeHandle& nh)
{
  nh.param("sample/v_mag_sample", v_mag_sample_, 0.0);
  nh.param("sample/px_min", px_min_, -1.0);
  nh.param("sample/px_max", px_max_, -1.0);
  nh.param("sample/py_min", py_min_, -1.0);
  nh.param("sample/py_max", py_max_, -1.0);
  nh.param("sample/pz_min", pz_min_, -1.0);
  nh.param("sample/pz_max", pz_max_, -1.0);
  nh.param("sample/vx_min", vx_min_, -1.0);
  nh.param("sample/vx_max", vx_max_, -1.0);
  nh.param("sample/vy_min", vy_min_, -1.0);
  nh.param("sample/vy_max", vy_max_, -1.0);
  nh.param("sample/vz_min", vz_min_, -1.0);
  nh.param("sample/vz_max", vz_max_, -1.0);
  nh.param("sample/ax_min", ax_min_, -1.0);
  nh.param("sample/ax_max", ax_max_, -1.0);
  nh.param("sample/ay_min", ay_min_, -1.0);
  nh.param("sample/ay_max", ay_max_, -1.0);
  nh.param("sample/az_min", az_min_, -1.0);
  nh.param("sample/az_max", az_max_, -1.0);
  nh.param("sample/rou", rou_, -1.0);
  nh.param("sample/c0", c_[0], 0.0);
  nh.param("sample/c1", c_[1], 0.0);
  nh.param("sample/c2", c_[2], 0.0);
  nh.param("sample/c3", c_[3], 0.0);
  nh.param("sample/c4", c_[4], 0.0);
  nh.param("sample/c5", c_[5], 0.0);
  nh.param("sample/tree_node_nums", tree_node_nums_, 0);
  nh.param("sample/orphan_nums", orphan_nums_, 0);
  nh.param("sample/search_time", search_time_, 0.0);
  nh.param("sample/replan_hor_safe_radius", replan_hor_safe_radius_, 0.3);
  nh.param("sample/replan_ver_safe_radius", replan_ver_safe_radius_, 0.2);
  nh.param("sample/copter_diag_len", copter_diag_len_, 0.5);
  nh.param("sample/clear_radius", clear_radius_, 1.0);
  nh.param("sample/radius_cost_between_two_states", radius_cost_between_two_states_, 0.0);
  nh.param("sample/allow_orphan", allow_orphan_, false);
  nh.param("sample/allow_close_goal", allow_close_goal_, false);
  nh.param("sample/stop_after_first_traj_found", stop_after_first_traj_found_, false);
  nh.param("sample/resolution", resolution_, 0.0);
  nh.param("sample/dim_three", dim_three_, true);
  
  hor_safe_radius_ = replan_hor_safe_radius_ + resolution_;
  ver_safe_radius_ = replan_ver_safe_radius_ + resolution_;
  
  ROS_INFO("hor_safe_radius: %lf", replan_hor_safe_radius_);
  ROS_INFO("ver_safe_radius: %lf", replan_ver_safe_radius_);

  random_device rd;
  gen_ = mt19937_64(rd());
  px_rand_ = uniform_real_distribution<double>(px_min_, px_max_);
  py_rand_ = uniform_real_distribution<double>(py_min_, py_max_);
  pz_rand_ = uniform_real_distribution<double>(pz_min_, pz_max_);
  vx_rand_ = uniform_real_distribution<double>(vx_min_, vx_max_);
  vy_rand_ = uniform_real_distribution<double>(vy_min_, vy_max_);
  vz_rand_ = uniform_real_distribution<double>(vz_min_, vz_max_);
  
  pos_mean_rand_ = uniform_real_distribution<double>(0.0, 1.0);
  pos_hor_rand_ = normal_distribution<double>(0.0, 0.5);
  pos_ver_rand_ = normal_distribution<double>(0.0, 0.5);
  vel_mag_rand_ = normal_distribution<double>(v_mag_sample_, 0.5);
  vel_hor_dir_rand_ = normal_distribution<double>(0.0, 0.2);
  
  traj_duration_ = 0.0;
  valid_start_tree_node_nums_ = 0;
  valid_goal_tree_node_nums_ = 0;
  
  //pre allocate memory
  start_tree_.resize(tree_node_nums_);
  goal_tree_.resize(tree_node_nums_);
  for (int i = 0; i < tree_node_nums_; i++) 
  {
    start_tree_[i] = new RRTNode;
    goal_tree_[i] = new RRTNode;
  }
  close_goal_node_ = new RRTNode;
}

void KRRTPlanner::setEnv(const OccMap::Ptr& env)
{
  occ_map_ = env;
}

void KRRTPlanner::setTopoFinder(const TopologyPRM::Ptr& topo_prm)
{
  topo_prm_ = topo_prm;
}

inline void KRRTPlanner::findSamplingSpace(const State& x_init,
                                  const State& x_goal,
                                  const vector<pair<State, State>>& segs, 
                                  vector<pair<Vector3d, Vector3d>>& all_corners)
{
  pair<Vector3d, Vector3d> corner;
  for (const auto& s : segs) 
  {
    Vector2d middle_v = (s.first.head(2) + s.second.head(2)) / 2;
    Vector2d tail_diff_head = s.second.head(2) - s.first.head(2);
    double l = tail_diff_head.norm();
    Vector2d unit_dir = tail_diff_head / l;
    Vector3d corner_v = (s.first.head(3) + s.second.head(3)) / 2;
    Vector3d corner_theta = corner_v;
    // ROS_INFO("corner_m: %lf, %lf, %lf", corner_v[0], corner_v[1], corner_v[2]);
        
    int nums = 10;
    double delta_theta = M_PI / (double)nums;
    Vector2d delta_vector;
    
    bool out_of_bound = false;
    double len_delta_theta = 0.0;
    delta_vector << 0.0, 0.0;
    rotateClockwise2d(M_PI/2.0, unit_dir);
    int j = 1;
    double out_of_bound_v = false;
    int voxel_state = -1;
    bool inside_occ = true;
    Vector3d first_free_corner_theta;
    while (1) 
    {
      delta_vector = unit_dir * resolution_ * j;
      ++j;
      corner_theta.head(2) = middle_v + delta_vector;
      voxel_state = occ_map_->getVoxelState(corner_theta);
      if (voxel_state == -1) 
      {
        out_of_bound_v = true;
        break;
      }
      else if (voxel_state == 0)
      {
        if (inside_occ)
        {
          first_free_corner_theta = corner_theta;
          inside_occ = false;
        }
        else 
        {
          double l = (corner_theta - first_free_corner_theta).norm();
          if (l >= hor_safe_radius_ * 2.0)
            break;
        }
      }
      else 
      {
        if (!inside_occ)
        {
          inside_occ = true;
        }
      }
    }
    if (delta_vector.norm() >= len_delta_theta) 
    {
      if (out_of_bound_v) 
      {
        out_of_bound = true;
      }
      len_delta_theta = delta_vector.norm();
      corner_v.head(2) = corner_theta.head(2);
    }
    if (out_of_bound) 
    {
      // ROS_INFO("corner1 out of bound, 0.0, 0.0, -1.0");
      corner.first << 0.0,0.0,-1.0;
    }
    else 
    {
      // ROS_INFO("corner1: %lf, %lf, %lf", corner_v[0], corner_v[1], corner_v[2]);
      corner.first = corner_v;   
    }   
    
    {
      corner_v = (s.first.head(3) + s.second.head(3) ) / 2;
      out_of_bound = false;
      delta_vector << 0.0, 0.0;
      len_delta_theta = 0.0;
      rotateClockwise2d(M_PI, unit_dir);
      int j = 1;
      double out_of_bound_v = false;
      double voxel_state = -1;
      bool inside_occ = true;
      Vector3d first_free_corner_theta;
      while (1) 
      {
        delta_vector = unit_dir * resolution_ * j;
        ++j;
        corner_theta.head(2) = middle_v + delta_vector;
        voxel_state = occ_map_->getVoxelState(corner_theta);
        if (voxel_state == -1) 
        {
          out_of_bound_v = true;
          break;
        }
        else if (voxel_state == 0)
        {
          if (inside_occ)
          {
            first_free_corner_theta = corner_theta;
            inside_occ = false;
          }
          else 
          {
            double l = (corner_theta - first_free_corner_theta).norm();
            if (l >= hor_safe_radius_ * 2.0)
              break;
          }
        }
        else 
        {
          if (!inside_occ)
          {
            inside_occ = true;
          }
        }
      }
      if (delta_vector.norm() > len_delta_theta) 
      {
        if (out_of_bound_v) 
        {
          out_of_bound = true;
        }
        len_delta_theta = delta_vector.norm();
          corner_v.head(2) = corner_theta.head(2);
      }
      if (out_of_bound) 
      {
        // ROS_INFO("corner2 out of bound, 0.0, 0.0, -1.0");
        corner.second << 0.0,0.0,-1.0;
      }
      else 
      {
        // ROS_INFO("corner2: %lf, %lf, %lf", corner_v[0], corner_v[1], corner_v[2]);
        corner.second = corner_v;
      }
    }
    all_corners.push_back(corner);
  }
}

void KRRTPlanner::setupRandomSampling(const Vector3d& x0, const Vector3d& x1, 
                                      const vector<vector<Vector3d>>& paths,
                                      vector<Vector3d>& unit_tracks,
                                      vector<Vector3d>& p_head,
                                      vector<Vector3d>& tracks, 
                                      vector<Vector3d>& rotated_unit_tracks)
{
  size_t path_num = paths.size();
  Vector3d track, unit_track, rotated_unit_track;

  if (path_num == 0)
  {
    p_head.push_back(x0);
    track = x1 - x0;
    tracks.push_back(track);
    unit_track = track.normalized();
    unit_tracks.push_back(unit_track);
    rotated_unit_track = rotate90Clockwise3d(unit_track);
    rotated_unit_tracks.push_back(rotated_unit_track);
    return;
  }

  for (const auto& path : paths)
  {
    int wp_num = path.size();
    for (int i = 0; i < wp_num - 1; ++i)
    {
      p_head.push_back(path[i]);
      track = path[i + 1] - path[i];
      tracks.push_back(track);
      unit_track = track.normalized();
      unit_tracks.push_back(unit_track);
      rotated_unit_track = rotate90Clockwise3d(unit_track);
      rotated_unit_tracks.push_back(rotated_unit_track);
    }
  }
}

void KRRTPlanner::setupRandomSampling(const State& x_init, const State& x_goal, 
                            const vector<pair<Vector3d, Vector3d>>& all_corners,
                            vector<Vector3d>& unit_tracks,
                            vector<Vector3d>& p_head,
                            vector<Vector3d>& tracks, 
                            vector<Vector3d>& rotated_unit_tracks)
{
  // all_corners.size() is the same as segs.size()
  Vector3d x0 = x_init.head(3), x1 = x_goal.head(3);
  Vector3d track, unit_track, rotated_unit_track;
  
  //segs is empty
  if (all_corners.size() == 0)
  {
    p_head.push_back(x0);
    track = x1 - x0;
    tracks.push_back(track);
    unit_track = track.normalized();
    unit_tracks.push_back(unit_track);
    rotated_unit_track = rotate90Clockwise3d(unit_track);
    rotated_unit_tracks.push_back(rotated_unit_track);
    vis.visualizeTopo(p_head, tracks, occ_map_->getLocalTime());
    return;
  }
    
  if (all_corners.front().first[2] != -1.0) 
  {
    p_head.push_back(x0);
    track = all_corners.front().first - x0;
    tracks.push_back(track);
    unit_track = track.normalized();
    unit_tracks.push_back(unit_track);
    rotated_unit_track = rotate90Clockwise3d(unit_track);
    rotated_unit_tracks.push_back(rotated_unit_track);
  }
  if (all_corners.front().second[2] != -1.0) 
  {
    p_head.push_back(x0);
    track = all_corners.front().second - x0;
    tracks.push_back(track);
    unit_track = track.normalized();
    unit_tracks.push_back(unit_track);
    rotated_unit_track = rotate90Clockwise3d(unit_track);
    rotated_unit_tracks.push_back(rotated_unit_track);
  }
  for (int i=0; i<(int)all_corners.size()-1; ++i) 
  {
    if (all_corners[i+1].first[2] != -1.0 && all_corners[i].first[2] != -1.0) 
    {
      p_head.push_back(all_corners[i].first);
      track = all_corners[i+1].first - all_corners[i].first;
      tracks.push_back(track);
      unit_track = track.normalized();
      unit_tracks.push_back(unit_track);
      rotated_unit_track = rotate90Clockwise3d(unit_track);
      rotated_unit_tracks.push_back(rotated_unit_track);
    }
    /******** X style bridges  ************/
//     if (all_corners[i+1].first[2] != -1.0 && all_corners[i].second[2] != -1.0) {
//         p_head.push_back(all_corners[i].second);
//         track = all_corners[i+1].first - all_corners[i].second;
//         tracks.push_back(track);
//         unit_track = track.normalized();
//         unit_tracks.push_back(unit_track);
//         rotated_unit_track = rotate90Clockwise3d(unit_track);
//         rotated_unit_tracks.push_back(rotated_unit_track);
//     }
//     if (all_corners[i+1].second[2] != -1.0 && all_corners[i].first[2] != -1.0) {
//         p_head.push_back(all_corners[i].first);
//         track = all_corners[i+1].second - all_corners[i].first;
//         tracks.push_back(track);
//         unit_track = track.normalized();
//         unit_tracks.push_back(unit_track);
//         rotated_unit_track = rotate90Clockwise3d(unit_track);
//         rotated_unit_tracks.push_back(rotated_unit_track);
//     }
    /****      ****/
    if (all_corners[i+1].second[2] != -1.0 && all_corners[i].second[2] != -1.0) 
    {
      p_head.push_back(all_corners[i].second);
      track = all_corners[i+1].second - all_corners[i].second;
      tracks.push_back(track);
      unit_track = track.normalized();
      unit_tracks.push_back(unit_track);
      rotated_unit_track = rotate90Clockwise3d(unit_track);
      rotated_unit_tracks.push_back(rotated_unit_track);
    }
  }
  if (all_corners.back().first[2] != -1.0) 
  {
    p_head.push_back(all_corners.back().first);
    track = x1 - all_corners.back().first;
    tracks.push_back(track);
    unit_track = track.normalized();
    unit_tracks.push_back(unit_track);
    rotated_unit_track = rotate90Clockwise3d(unit_track);
    rotated_unit_tracks.push_back(rotated_unit_track);
  }
  if (all_corners.back().second[2] != -1.0) 
  {
    p_head.push_back(all_corners.back().second);
    track = x1 - all_corners.back().second;
    tracks.push_back(track);
    unit_track = track.normalized();
    unit_tracks.push_back(unit_track);
    rotated_unit_track = rotate90Clockwise3d(unit_track);
    rotated_unit_tracks.push_back(rotated_unit_track);
  }
  
  //segs is not empty but all_corners are out of bound
  if (tracks.size() == 0)
  {
    p_head.push_back(x0);
    track = x1 - x0;
    tracks.push_back(track);
    unit_track = track.normalized();
    unit_tracks.push_back(unit_track);
    rotated_unit_track = rotate90Clockwise3d(unit_track);
    rotated_unit_tracks.push_back(rotated_unit_track);
  }
  vis.visualizeTopo(p_head, tracks, occ_map_->getLocalTime());
//     for (int i=0; i<p_head.size(); ++i) {
//         ROS_INFO("p_head:              %lf, %lf, %lf", p_head[i][0],p_head[i][1], p_head[i][2]);
//         ROS_INFO("tracks:              %lf, %lf, %lf", tracks[i][0],tracks[i][1], tracks[i][2]);
//         ROS_INFO("unit_tracks:         %lf, %lf, %lf", unit_tracks[i][0],unit_tracks[i][1], unit_tracks[i][2]);
//         ROS_INFO("rotated_unit_tracks: %lf, %lf, %lf", rotated_unit_tracks[i][0],rotated_unit_tracks[i][1], rotated_unit_tracks[i][2]);
//     }
}

inline bool KRRTPlanner::samplingOnce(int i, State& rand_state,
                                      const vector<Vector3d>& unit_tracks,
                                      const vector<Vector3d>& p_head,
                                      const vector<Vector3d>& tracks, 
                                      const vector<Vector3d>& rotated_unit_tracks)
{
  double pos_mean = pos_mean_rand_(gen_);
  double pos_hor = pos_hor_rand_(gen_);
  double pos_ver(0.0);
  if (dim_three_)
    pos_ver = pos_ver_rand_(gen_);
  double vel_mag = vel_mag_rand_(gen_);
  if (vel_mag > v_mag_sample_) 
    vel_mag = v_mag_sample_ - (vel_mag-v_mag_sample_);
  if (vel_mag <= 0)
    return false;
  double vel_hor_dir = vel_hor_dir_rand_(gen_);
  Vector3d p_m = p_head[i] + tracks[i]*pos_mean;
  Vector3d p = p_m + rotated_unit_tracks[i]*pos_hor;
  rand_state[0] = p[0];
  rand_state[1] = p[1];
  rand_state[2] = p_m[2] + pos_ver;
  Vector3d pos;
  pos = rand_state.head(3);
  
  Vector3d v_m = unit_tracks[i]*vel_mag;
  rotateClockwise3d(vel_hor_dir, v_m);
  rand_state[3] = v_m[0];
  rand_state[4] = v_m[1];
  rand_state[5] = v_m[2];
  
  vector<Vector3d> line_grids;
  Vector3d acc;
  getCheckPos(pos, v_m, acc, line_grids, hor_safe_radius_, ver_safe_radius_);
  for (const auto& grid : line_grids)
  {
    if (occ_map_->getVoxelState(grid) != 0) {
      return false;
    }
  }

  return true;
}


//result() is called every time before plan();
void KRRTPlanner::reset()
{
  for (auto &ptr : path_on_first_search_) {
    delete ptr;
  }
  path_on_first_search_.clear();
  
  for (int i=0; i<tree_node_nums_; i++) {
    start_tree_[i]->parent = nullptr;
    start_tree_[i]->children.clear();
    goal_tree_[i]->parent = nullptr;
    goal_tree_[i]->children.clear();
  }
  
  if (allow_orphan_) {
    orphans_.clear();
  }
  
  path_.clear();
  patched_path_.clear();
  traj_duration_ = 0.0;
  valid_start_tree_node_nums_ = 0;
  valid_goal_tree_node_nums_ = 0;
}

int KRRTPlanner::plan(Vector3d start_pos, Vector3d start_vel, Vector3d start_acc, 
                      Vector3d end_pos, Vector3d end_vel, Vector3d end_acc, 
                      double search_time, bool bidirection)
{
  reset();
  if (occ_map_->getVoxelState(start_pos) != 0) 
  {
    ROS_ERROR("[KRRT]: Start pos collide or out of bound");
    return 3;
  }
  if (occ_map_->getVoxelState(end_pos) != 0) 
  // if (!validatePosSurround(end_pos)) 
  {
    Vector3d shift[20] = {Vector3d(-1.0,0.0,0.0), Vector3d(0.0,1.0,0.0), Vector3d(0.0,-1.0,0.0), 
                         Vector3d(0.0,0.0,1.0), Vector3d(0.0,0.0,-1.0), Vector3d(1.0,0.0,0.0), 
                         Vector3d(-2.0,0.0,0.0), Vector3d(0.0,2.0,0.0), Vector3d(0.0,-2.0,0.0), 
                         Vector3d(0.0,0.0,2.0), Vector3d(0.0,0.0,-2.0), Vector3d(2.0,0.0,0.0), 

                         Vector3d(1.0,1.0,1.0), Vector3d(1.0,1.0,-1.0), Vector3d(1.0,-1.0,1.0), 
                         Vector3d(1.0,-1.0,-1.0), Vector3d(-1.0,1.0,-1.0), Vector3d(-1.1,1.0,1.0), 
                         Vector3d(-1.0,-1.0,1.0), Vector3d(-1.0,-1.0,-1.0)};
    ROS_WARN("[KRRT]: End pos collide or out of bound, search for other safe end");
    int i = 0;
    for (; i < 20; ++i)
    {
      end_pos += shift[i] * replan_hor_safe_radius_;
      if (validatePosSurround(end_pos))
        break;
    }
    if (i == 20)
    {
      ROS_ERROR("found no valid end pos, plan fail");
      return 3;
    }
  }
  
  if (search_time > 0.0)
    search_time_ = search_time;
  
  if((start_pos-end_pos).norm()<1e-3 && (start_vel-end_vel).norm()<1e-4)
  {
    ROS_ERROR("[KRRT]: start state & end state too close");
    return 4;
  }
  
  /* construct start and goal nodes */
  start_node_ = start_tree_[1]; //init ptr
  start_node_->parent = nullptr;
  start_node_->x.head(3) = start_pos;
  start_node_->x.tail(3) = start_vel;
  start_node_->cost_from_start = 0.0;
  start_node_->tau_from_parent = 0.0;
  start_node_->tau_from_start = 0.0;
  start_node_->n_order = 0;
  goal_node_ = start_tree_[0]; //init ptr
  goal_node_->parent = nullptr;
  goal_node_->x.head(3) = end_pos;
  if (end_vel.norm() >= vx_max_)
  {
    end_vel.normalize();
    end_vel = end_vel * vx_max_;
  }
  goal_node_->x.tail(3) = end_vel;
  goal_node_->cost_from_start = DBL_MAX;
  goal_node_->tau_from_parent = DBL_MAX;
  goal_node_->tau_from_start = DBL_MAX;
  goal_node_->n_order = 0;
  for (int i=0; i<6; ++i) {
    start_node_->x_coeff[i] = 0;
    start_node_->y_coeff[i] = 0;
    start_node_->z_coeff[i] = 0;
    goal_node_->x_coeff[i] = 0;
    goal_node_->y_coeff[i] = 0;
    goal_node_->z_coeff[i] = 0;
  }
  close_goal_node_->cost_from_start = DBL_MAX;
  close_goal_node_->tau_from_parent = DBL_MAX;
  close_goal_node_->tau_from_start = DBL_MAX;
  close_goal_node_->n_order = 0;

  end_tree_goal_node_ = goal_tree_[1]; //init ptr
  end_tree_goal_node_->parent = nullptr;
  end_tree_goal_node_->x.head(3) = start_pos;
  end_tree_goal_node_->x.tail(3) = start_vel;
  end_tree_goal_node_->cost_from_start = DBL_MAX;
  end_tree_goal_node_->tau_from_parent = DBL_MAX;
  end_tree_goal_node_->tau_from_start = DBL_MAX;
  end_tree_goal_node_->n_order = 0;
  end_tree_start_node_ = goal_tree_[0]; //init ptr
  end_tree_start_node_->parent = nullptr;
  end_tree_start_node_->x.head(3) = end_pos;
  end_tree_start_node_->x.tail(3) = end_vel;
  end_tree_start_node_->cost_from_start = 0.0;
  end_tree_start_node_->tau_from_parent = 0.0;
  end_tree_start_node_->tau_from_start = 0.0;
  end_tree_start_node_->n_order = 0;
  for (int i=0; i<6; ++i) {
    end_tree_goal_node_->x_coeff[i] = 0;
    end_tree_goal_node_->y_coeff[i] = 0;
    end_tree_goal_node_->z_coeff[i] = 0;
    end_tree_start_node_->x_coeff[i] = 0;
    end_tree_start_node_->y_coeff[i] = 0;
    end_tree_start_node_->z_coeff[i] = 0;
  }
  vis.visualizeStartAndGoal(start_node_->x, goal_node_->x, occ_map_->getLocalTime());
  valid_start_tree_node_nums_ = 2; //start and goal already in start_tree_
  valid_goal_tree_node_nums_ = 2;
  int plan_result(0);
  if (!bidirection)
    plan_result = rrtStar(start_node_->x, goal_node_->x, start_acc, end_acc, tree_node_nums_, radius_cost_between_two_states_, true, EPSILON);
  else
    plan_result = rrtStarConnect(start_node_->x, goal_node_->x, start_acc, end_acc, tree_node_nums_, radius_cost_between_two_states_, true);

  return plan_result;
}

int KRRTPlanner::rrtStar(const State& x_init, const State& x_final, 
                          const Vector3d& u_init, const Vector3d& u_final, int n, 
                          double radius, const bool rewire, const float epsilon)
{ 
  ros::Time rrt_start_time = ros::Time::now();
  ros::Time first_goal_found_time, final_goal_found_time;
  
  /* local variables */
  int valid_sample_nums = 0; //random samples in obs free area
  int valid_orphan_nums = 0;
  list<int> orphan_idx_list;
  vector<State> vis_x;
  vector<Control> vis_u;
  bool first_time_find_goal = true;
  bool close_goal_found = false;
  double close_dist = DBL_MAX;
  bool goal_found = false;
  /* init sampling space */
  double best_cost, best_tau;
  vector<pair<State, State>> segs;
  double x_coeff[6], y_coeff[6], z_coeff[6];
  VectorXd x_u_init = VectorXd::Zero(9);
  x_u_init.head(6) = x_init;
  x_u_init.tail(3) = u_init;
  VectorXd x_u_final = VectorXd::Zero(9);
  x_u_final.head(6) = x_final;
  x_u_final.tail(3) = u_final;
	// ROS_INFO_STREAM("kino rrt* start");
  bool direc_connect = computeBestCost(x_u_init, x_u_final, segs, best_cost, best_tau, &vis_x, &vis_u, 
                         x_coeff, y_coeff, z_coeff);
  vis.visualizeBestTraj(vis_x, vis_u, occ_map_->getLocalTime());
  // ROS_INFO("[KRRT]: Best cost: %lf, best tau: %lf", best_cost, best_tau);
  //best traj is collision free
  if (direc_connect && segs.size() == 0) 
  {
    // ROS_WARN("Best traj collision free, one shot connected");
    goal_node_->cost_from_start = best_cost;
    goal_node_->parent = start_node_;
    goal_node_->tau_from_parent = best_tau;
    goal_node_->tau_from_start = best_tau;
    goal_node_->n_order = 5;
    for (int i=0; i<6; ++i) 
    {
      goal_node_->x_coeff[i] = x_coeff[i];
      goal_node_->y_coeff[i] = y_coeff[i];
      goal_node_->z_coeff[i] = z_coeff[i];
    }
    goal_found = true;
    fillPath(goal_node_, path_);
    patched_path_ = path_;
 
    /* for traj vis */
    vis.visualizeATraj(vis_x, vis_u, occ_map_->getLocalTime());
    double final_traj_len(0.0), final_traj_duration(0.0), final_traj_ctrl_cost(0.0);
    int final_traj_seg_nums(0);
    getTrajAttributes(path_, final_traj_duration, final_traj_ctrl_cost, final_traj_len, final_traj_seg_nums);
    double final_traj_use_time = ros::Time::now().toSec()-rrt_start_time.toSec();
    // ROS_INFO_STREAM("[KRRT]: [front-end best one shot]: " << endl 
    //      << "    -   seg nums: " << final_traj_seg_nums << endl 
    //      << "    -   use time: " << final_traj_use_time << endl 
    //      << "    -   ctrl cost: " << final_traj_ctrl_cost << endl
    //      << "    -   traj duration: " << final_traj_duration << endl 
    //      << "    -   path length: " << final_traj_len);
    
    return 1;
  }
  else 
  {
    // ROS_ERROR("Best traj one shot violates OR no solution");
  }
  
  /* setup sampling */
  // naive topo finding
  vector<pair<Vector3d, Vector3d>> all_corners;
  findSamplingSpace(x_init, x_final, segs, all_corners);
  vector<Vector3d> unit_tracks, p_head, tracks, rotated_unit_tracks;
  setupRandomSampling(x_init, x_final, all_corners, 
                      unit_tracks, p_head, tracks, rotated_unit_tracks);
  
  // use visibility prm to find topo
  // vector<Eigen::Vector3d> start_pts, end_pts;
  // list<GraphNode::Ptr> graph;
  // vector<vector<Eigen::Vector3d>> raw_paths, filtered_paths, select_paths;
  // topo_prm_->findTopoPaths(x_init.head(3), x_final.head(3), start_pts, end_pts, graph, raw_paths, filtered_paths, select_paths);
  // vis.visualizeTopoPaths(select_paths, 3, Eigen::Vector4d(0, 0, 1, 1), occ_map_->getLocalTime());
  // vector<Vector3d> unit_tracks, p_head, tracks, rotated_unit_tracks;
  // setupRandomSampling(x_init.head(3), x_final.head(3), select_paths, unit_tracks, p_head, tracks, rotated_unit_tracks);

  /* setup sampling */

  /* kd tree init */
  kdtree *kd_tree = kd_create(X_DIM);
  //Add start and goal nodes to kd tree
  double init_state[X_DIM];
  double final_state[X_DIM];
  for (int i = 0; i < X_DIM; ++i) 
  {
    init_state[i] = x_init[i];
    final_state[i] = x_final[i];
  }
  kd_insert(kd_tree, init_state, start_node_);
  kd_insert(kd_tree, final_state, goal_node_);
  
  /* main loop */
  int idx = 0;
  for (idx = 0; (ros::Time::now() - rrt_start_time).toSec() < search_time_ &&
                                        valid_start_tree_node_nums_ < n ; ++idx) 
  {
    /* biased random sampling */
    State x_rand;
    bool good_sample = samplingOnce(idx%tracks.size(), x_rand, 
                        unit_tracks, p_head, tracks, rotated_unit_tracks);
    if (!good_sample) 
    {
      continue;
    }
    
    /* just random sampling */
//     x_rand[0] = px_rand_(gen_);
//     x_rand[1] = py_rand_(gen_);
//     x_rand[2] = pz_rand_(gen_);
//     x_rand[3] = vx_rand_(gen_);
//     x_rand[4] = vy_rand_(gen_);
//     x_rand[5] = vz_rand_(gen_);
//     bool check_goal = false;
//     if (idx == 0) {// || (float)rand()/(float)RAND_MAX < epsilon) {
//       //try to connect start_start to goal_state at first
//       x_rand = x_final;
//       check_goal = true;
//     }
//     //check x_rand pos valid
//     Vector3d pos;
//     pos = x_rand.head(3);
//     if (occ_map_->getVoxelState(pos) != 0
//       || x_rand[2] > max(x_init[2], x_final[2]) || x_rand[2] < min(x_init[2], x_final[2])
//       || x_rand[0] > max(x_init[0], x_final[0]) || x_rand[0] < min(x_init[0], x_final[0])
//       || x_rand[1] > max(x_init[1], x_final[1]) || x_rand[1] < min(x_init[1], x_final[1])
// //       || (x_rand[3]*(x_final[0]-x_init[0]) + x_rand[4]*(x_final[1]-x_init[1]) + x_rand[5]*(x_final[2]-x_init[2])) / (sqrt(x_rand[3]*x_rand[3]+x_rand[4]*x_rand[4]+x_rand[5]*x_rand[5]) * sqrt((x_final[2]-x_init[2])*(x_final[2]-x_init[2])+(x_final[1]-x_init[1])*(x_final[1]-x_init[1])+(x_final[0]-x_init[0])*(x_final[0]-x_init[0]))) < 0.9
//       ) {
//       continue;
//     }
    /* end just random sampling */
    
    
    vis.visualizeSampledState(x_rand, occ_map_->getLocalTime());
    ++valid_sample_nums;
    
    /* kd_tree bounds search for parent */
    /* TODO diff range for diff dimention*/
    BOUNDS backward_reachable_bounds;
    calc_backward_reachable_bounds(x_rand, radius, backward_reachable_bounds);
    double center[X_DIM];
    for (int i = 0; i < X_DIM; ++i) 
      {center[i] = x_rand[i];}
    double range = 0;
    for (size_t i=0; i<backward_reachable_bounds.size(); ++i) 
    {
      double r = (backward_reachable_bounds[i].second - backward_reachable_bounds[i].first)/2;
      range = max(range, r);
    }
    struct kdres *presults;
    presults = kd_nearest_range(kd_tree, center, range);
    
    /* choose parent from kd tree range query result*/
    double min_dist = DBL_MAX;
    double tau_from_s = DBL_MAX;
    RRTNode* x_near = nullptr; //parent
    double cost, tau, actual_deltaT;
    double x_coeff[4], y_coeff[4], z_coeff[4];
    int bound_node_cnt = 0;
    while(!kd_res_end(presults)) 
    {
      bound_node_cnt++;
      /* get the data and position of the current result item */
      double pos[X_DIM];
      RRTNode* curr_node = (RRTNode*)kd_res_item(presults, pos);
      if (curr_node == goal_node_) 
      {
        // goal node can not be parent of any other node
        kd_res_next( presults );
        continue;
      }
      // if (PVHeu(curr_node->x, x_final) - 1 <= PVHeu(x_rand, x_final))
      // {
      //   // parents can not be closer to goal.
      //   kd_res_next( presults );
      //   continue;
      // }
      double cost1, tau1, x_coeff1[4], y_coeff1[4], z_coeff1[4];
      bool connected = HtfConnect(curr_node->x, x_rand, radius, cost1, tau1, x_coeff1, y_coeff1, z_coeff1);
      if (connected && min_dist > (curr_node->cost_from_start + cost1)) 
      {
        tau = tau1;
        for (int i=0; i<4; ++i)
        {
          x_coeff[i] = x_coeff1[i];
          y_coeff[i] = y_coeff1[i];
          z_coeff[i] = z_coeff1[i];
        }
        min_dist = curr_node->cost_from_start + cost1;
        tau_from_s = curr_node->tau_from_start + tau1;
        x_near = curr_node;
        // ROS_WARN("parent found");
      }
      kd_res_next(presults); //go to next in kd tree range query result
    }
    kd_res_free(presults); //reset kd tree range query
    if (x_near == nullptr) 
    {
      //no valid parent found, sample next
      if (!allow_orphan_)
        continue;
      else if (valid_orphan_nums < orphan_nums_)
      {
        orphan_idx_list.push_back(valid_orphan_nums);
        valid_orphan_nums++;
        orphans_.push_back(x_rand);
        // vector<State> vis_orphan;
        // for (auto it = orphan_idx_list.cbegin(); it != orphan_idx_list.cend(); ++it) 
        // {
        //   vis_orphan.push_back(orphans_[*it]);
        // }
        // vis.visualizeOrphans(vis_orphan, occ_map_->getLocalTime());
        // TODO stock orphans in kd-tree?
      }
    } 
    else 
    {
      //sample rejection
      double cost_star, tau_star;
      computeCostAndTime(x_rand, goal_node_->x, cost_star, tau_star);
      if (min_dist + cost_star >= goal_node_->cost_from_start) {
        continue;
      }
      
      /* parent found within radius, then: 
       * 1.add a node to rrt and kd_tree; 
       * 2.rewire. */

      /* 1.1 add the randomly sampled node to rrt_tree */
      RRTNode* sampled_node = start_tree_[valid_start_tree_node_nums_++]; 
      sampled_node->x = x_rand;
      sampled_node->parent = x_near;
      sampled_node->cost_from_start = min_dist;
      sampled_node->tau_from_start = tau_from_s;
      sampled_node->tau_from_parent = tau;
      sampled_node->n_order = 3;
      for (int i=0; i<4; ++i) 
      {
        sampled_node->x_coeff[i] = x_coeff[i];
        sampled_node->y_coeff[i] = y_coeff[i];
        sampled_node->z_coeff[i] = z_coeff[i];
      }
      x_near->children.push_back(sampled_node);
          
      /* 1.2 add the randomly sampled node to kd_tree */
      double sample_state[X_DIM];
      for (int i = 0; i < X_DIM; ++i) sample_state[i] = x_rand[i];
      kd_insert(kd_tree, sample_state, sampled_node);
      
      /* try to connect to goal after a valid tree node found */ 
      double cost1, tau1, x_coeff1[4], y_coeff1[4], z_coeff1[4];
      bool connected_to_goal = HtfConnect(x_rand, goal_node_->x, radius, cost1, tau1, x_coeff1, y_coeff1, z_coeff1);
      double dist_to_goal = dist(x_rand, goal_node_->x);
      if (connected_to_goal && min_dist + cost1 < goal_node_->cost_from_start) 
      {
        if (goal_node_->parent) 
        {
          goal_node_->parent->children.remove(goal_node_);
        }
        goal_node_->parent = sampled_node;
        goal_node_->cost_from_start = min_dist + cost1;
        goal_node_->tau_from_parent = tau1;
        goal_node_->tau_from_start = tau_from_s + tau1;
        goal_node_->n_order = 3;
        for (int i=0; i<4; ++i) 
        {
          goal_node_->x_coeff[i] = x_coeff1[i];
          goal_node_->y_coeff[i] = y_coeff1[i];
          goal_node_->z_coeff[i] = z_coeff1[i];
        }
        sampled_node->children.push_back(goal_node_);
        goal_found = true;
        // ROS_INFO_STREAM("[KRRT]: goal found");
        
        if (first_time_find_goal) 
        {
          first_goal_found_time = ros::Time::now();
          first_time_find_goal = false;
          RRTNodePtr node = goal_node_;
          while (node) 
          {
            //new a RRTNodePtr instead of use ptr in start_tree_ because as sampling goes on and rewire happens, 
            //parent, children and other attributes of start_tree_[i] may change.
            RRTNodePtr n_p = new RRTNode;
            n_p->children = node->children;
            n_p->cost_from_start = node->cost_from_start;
            n_p->parent = node->parent;
            n_p->x = node->x;
            n_p->tau_from_parent = node->tau_from_parent;
            n_p->tau_from_start = node->tau_from_start;
            n_p->n_order = node->n_order;
            for (int i=0; i<4; ++i) 
            {
              n_p->x_coeff[i] = node->x_coeff[i];
              n_p->y_coeff[i] = node->y_coeff[i];
              n_p->z_coeff[i] = node->z_coeff[i];
            }
            // path_vector[0] is goal node, path_vector[n-1] is start node
            path_on_first_search_.push_back(n_p); 
            node = node->parent;
          }
        }
        if (stop_after_first_traj_found_)
          break; //stop searching after first time find the goal?
      }
      //else if (isClose(x_rand, goal_node_->x) && min_dist < close_goal_node_->cost_from_start)
      else if(allow_close_goal_ && !goal_found && dist_to_goal < close_dist && tau_from_s >= 2)
      {
        close_dist = dist_to_goal;
        // ROS_INFO_STREAM("[KRRT]: close goal found, Euc dist to goal: " << dist_to_goal);
        close_goal_found = true;
        close_goal_node_->cost_from_start = sampled_node->cost_from_start;
        close_goal_node_->tau_from_parent = sampled_node->tau_from_parent;
        close_goal_node_->tau_from_start = sampled_node->tau_from_start;
        close_goal_node_->n_order = sampled_node->n_order;
        close_goal_node_->parent = sampled_node->parent;
        for (int i=0; i<6; ++i) 
        {
          close_goal_node_->x_coeff[i] = sampled_node->x_coeff[i];
          close_goal_node_->y_coeff[i] = sampled_node->y_coeff[i];
          close_goal_node_->z_coeff[i] = sampled_node->z_coeff[i];
        }
      }
      else if (valid_start_tree_node_nums_ == n && goal_node_->parent == nullptr) 
      {
        //TODO add what if not connected to goal after n samples
        // ROS_INFO_STREAM("[KRRT]: NOT CONNECTED TO GOAL after " << n << " nodes added to rrt-tree");
      }/* end of try to connect to goal */
      
      
      /* 2.rewire */
      if (rewire) 
      {
        //kd_tree bounds search
        BOUNDS forward_reachable_bounds;
        calc_forward_reachable_bounds(x_rand, radius, forward_reachable_bounds);
        double center[X_DIM];
        for (int i = 0; i < X_DIM; ++i) center[i] = x_rand[i];
        double range = 0;
        for (size_t i=0; i<forward_reachable_bounds.size(); ++i) 
        {
          double r = (forward_reachable_bounds[i].second - forward_reachable_bounds[i].first)/2;
          range = max(range, r);
        }
        struct kdres *presults;
        presults = kd_nearest_range(kd_tree, center, range);
        while(!kd_res_end(presults)) 
        {
          /* get the data and position of the current result item */
          double pos[X_DIM];
          RRTNode* curr_node = (RRTNode*)kd_res_item(presults, pos);
          if (curr_node == goal_node_ || curr_node == start_node_) 
          {
            // already tried to connect to goal from random sampled node
            kd_res_next( presults );
            continue;
          }
          // if (PVHeu(curr_node->x, x_final) + 1 >= PVHeu(x_rand, x_final))
          // {
          //   // children can not be further to goal.
          //   kd_res_next( presults );
          //   continue;
          // }
          double cost1, tau1, x_coeff1[4], y_coeff1[4], z_coeff1[4];
          bool connected = HtfConnect(x_rand, curr_node->x, radius, cost1, tau1, x_coeff1, y_coeff1, z_coeff1);
          if (connected && sampled_node->cost_from_start + cost1 < curr_node->cost_from_start) 
          {
            // If we can get to a node via the sampled_node faster than via it's existing parent then change the parent
            curr_node->parent->children.remove(curr_node);  //DON'T FORGET THIS, remove it form its parent's children list
            curr_node->parent = sampled_node;
            curr_node->cost_from_start = sampled_node->cost_from_start + cost1;
            curr_node->tau_from_parent = tau1;
            curr_node->tau_from_start = sampled_node->tau_from_start + tau1;
            curr_node->n_order = 3;
            for (int i=0; i<4; ++i) 
            {
              curr_node->x_coeff[i] = x_coeff1[i];
              curr_node->y_coeff[i] = y_coeff1[i];
              curr_node->z_coeff[i] = z_coeff1[i];
            }
            sampled_node->children.push_back(curr_node);
          }
          /* go to the next entry */
          kd_res_next(presults);
        }
        kd_res_free(presults);
      }/* end of rewire */
      
      /* check orphans */
      if (allow_orphan_)
      {
        vector<int> adopted_orphan_idx;
        for (auto it = orphan_idx_list.cbegin(); it != orphan_idx_list.cend(); ++it) 
        {
          // if (PVHeu(x_rand, x_final) <= PVHeu(orphans_[*it], x_final) + 1)
          // {
          //   // children can not be further to goal.
          //   continue;
          // }
          double cost1, tau1, x_coeff1[4], y_coeff1[4], z_coeff1[4];
          bool connected_to_orphan = HtfConnect(x_rand, orphans_[*it], radius, cost1, tau1, x_coeff1, y_coeff1, z_coeff1);
//           cout << "idx: " << *it << ", orphan state: " << orphans_.at(*it).transpose() << ", cost: " << cost1 << ", tau: " << tau1 << endl;
          if (connected_to_orphan) 
          {
            /* 1. add the orphan node to rrt_tree */
            RRTNode* orphan_node = start_tree_[valid_start_tree_node_nums_++]; 
            orphan_node->x = orphans_[*it];
            orphan_node->parent = sampled_node;
            orphan_node->cost_from_start = sampled_node->cost_from_start + cost1;
            orphan_node->tau_from_start = sampled_node->tau_from_start + tau1;
            orphan_node->tau_from_parent = tau1;
            orphan_node->n_order = 3;
            for (int i=0; i<4; ++i) 
            {
              orphan_node->x_coeff[i] = x_coeff1[i];
              orphan_node->y_coeff[i] = y_coeff1[i];
              orphan_node->z_coeff[i] = z_coeff1[i];
            }
            sampled_node->children.push_back(orphan_node);
                
            /* 2. add the orphan node to kd_tree */
            double orphan_state[X_DIM];
            for (int j = 0; j < X_DIM; ++j) orphan_state[j] = orphans_[*it](j,0);
            kd_insert(kd_tree, orphan_state, orphan_node);
            
            adopted_orphan_idx.push_back(*it);
//             ROS_WARN("orphan!");
//             break;
          }
        }
        /* 3. remove orphan list */
        for (int i : adopted_orphan_idx)
        {
          orphan_idx_list.remove(i);
        }
      }
      
      // vis.visualizeAllTreeNode(start_node_, occ_map_->getLocalTime()); //every time a new node is found, visualize it
      // vis_x.clear();
      // vis_u.clear();
      // findAllStatesAndControlInAllTrunks(start_node_, &vis_x, &vis_u);
      // vis.visualizeATraj(vis_x, vis_u, occ_map_->getLocalTime());
      
    }/* end of find parent */
    
  }/* end of sample once */
  
  vis.visualizeAllTreeNode(start_node_, occ_map_->getLocalTime());
  if (goal_found) 
  {
    final_goal_found_time = ros::Time::now();
    fillPath(goal_node_, path_);  
    ros::Time t_p = ros::Time::now();
    patching(path_, patched_path_, u_init);
    // ROS_ERROR("patch use time: %lf", (ros::Time::now()-t_p).toSec());
    /* for traj vis */
    vis_x.clear();
    vis_u.clear();
    getVisStateAndControl(path_on_first_search_, &vis_x, &vis_u);
    vis.visualizeFirstTraj(vis_x, vis_u, occ_map_->getLocalTime());
    vis_x.clear();
    vis_u.clear();
    getVisStateAndControl(path_, &vis_x, &vis_u);
    vis.visualizeATraj(vis_x, vis_u, occ_map_->getLocalTime());
    
    // vis_x.clear();
    // vis_u.clear();
    // findAllStatesAndControlInAllTrunks(start_node_, &vis_x, &vis_u);
    // vis.visualizeTreeTraj(vis_x, vis_u, occ_map_->getLocalTime());
    
    // ros::Time t_c = ros::Time::now();
    // vector<Vector3d> cover_grid;
    // getVisTrajCovering(path_, cover_grid);
    // cout << "[===========]: " << (ros::Time::now()-t_c).toSec() << endl;
    // vis.visualizeTrajCovering(cover_grid, resolution_, occ_map_->getLocalTime());
    
    double first_traj_len(0.0), first_traj_duration(0.0), first_traj_ctrl_cost(0.0);
    double final_traj_len(0.0), final_traj_duration(0.0), final_traj_ctrl_cost(0.0);
    int first_traj_seg_nums(0), final_traj_seg_nums(0);
    getTrajAttributes(path_on_first_search_, first_traj_duration, first_traj_ctrl_cost, first_traj_len, first_traj_seg_nums);
    getTrajAttributes(path_, final_traj_duration, final_traj_ctrl_cost, final_traj_len, final_traj_seg_nums);
    double first_traj_use_time = first_goal_found_time.toSec()-rrt_start_time.toSec();
    double final_traj_use_time = final_goal_found_time.toSec()-rrt_start_time.toSec();
    // ROS_INFO_STREAM("[KRRT]: total sample times: " << idx);
    // ROS_INFO_STREAM("[KRRT]: valid sample times: " << valid_sample_nums);
    // ROS_INFO_STREAM("[KRRT]: valid tree node nums: " << valid_start_tree_node_nums_);
    // ROS_INFO_STREAM("[KRRT]: [front-end first path]: " << endl 
    //      << "    -   seg nums: " << first_traj_seg_nums << endl 
    //      << "    -   time: " << first_traj_use_time << endl 
    //      << "    -   ctrl cost: " << first_traj_ctrl_cost << endl
    //      << "    -   traj duration: " << first_traj_duration << endl 
    //      << "    -   path length: " << first_traj_len);
        
    // ROS_INFO_STREAM("[KRRT]: [front-end final path]: " << endl 
    //      << "    -   seg nums: " << final_traj_seg_nums << endl 
    //      << "    -   time: " << final_traj_use_time << endl 
    //      << "    -   ctrl cost: " << final_traj_ctrl_cost << endl
    //      << "    -   traj duration: " << final_traj_duration << endl 
    //      << "    -   path length: " << final_traj_len);
    return 1;
  } 
  else if (close_goal_found)
  {
    ROS_ERROR("Not connectting to goal, but to close_goal");
    final_goal_found_time = ros::Time::now();
    fillPath(close_goal_node_, path_);  
    ros::Time t_p = ros::Time::now();
    patching(path_, patched_path_, u_init);
    // ROS_ERROR("patch use time: %lf", (ros::Time::now()-t_p).toSec());
    /* for traj vis */
    vis_x.clear();
    vis_u.clear();
    getVisStateAndControl(path_, &vis_x, &vis_u);
    vis.visualizeATraj(vis_x, vis_u, occ_map_->getLocalTime());
    
    // vis_x.clear();
    // vis_u.clear();
    // findAllStatesAndControlInAllTrunks(start_node_, &vis_x, &vis_u);
    // vis.visualizeTreeTraj(vis_x, vis_u, occ_map_->getLocalTime());
    
    // ros::Time t_c = ros::Time::now();
    // vector<Vector3d> cover_grid;
    // getVisTrajCovering(path_, cover_grid);
    // cout << "[===========]: " << (ros::Time::now()-t_c).toSec() << endl;
    // vis.visualizeTrajCovering(cover_grid, resolution_, occ_map_->getLocalTime());
    
    double final_traj_len(0.0), final_traj_duration(0.0), final_traj_ctrl_cost(0.0);
    int final_traj_seg_nums(0);
    getTrajAttributes(path_, final_traj_duration, final_traj_ctrl_cost, final_traj_len, final_traj_seg_nums);
    double final_traj_use_time = final_goal_found_time.toSec()-rrt_start_time.toSec();
    // ROS_INFO_STREAM("[KRRT]: total sample times: " << idx);
    // ROS_INFO_STREAM("[KRRT]: valid sample times: " << valid_sample_nums);
    // ROS_INFO_STREAM("[KRRT]: valid tree node nums: " << valid_start_tree_node_nums_);
    // ROS_INFO_STREAM("[KRRT]: [front-end final path]: " << endl 
    //      << "    -   seg nums: " << final_traj_seg_nums << endl 
    //      << "    -   time: " << final_traj_use_time << endl 
    //      << "    -   ctrl cost: " << final_traj_ctrl_cost << endl
    //      << "    -   traj duration: " << final_traj_duration << endl 
    //      << "    -   path length: " << final_traj_len);
    return 2;
  }
  else if (valid_start_tree_node_nums_ == n)
  {
    // ROS_INFO_STREAM("[KRRT]: NOT CONNECTED TO GOAL after " << n << " nodes added to rrt-tree");
    // ROS_INFO_STREAM("[KRRT]: total sample times: " << idx);
    // ROS_INFO_STREAM("[KRRT]: valid sample times: " << valid_sample_nums);
    // ROS_INFO_STREAM("[KRRT]: valid tree node nums: " << valid_start_tree_node_nums_);
    return 0;
  }
  else if ((ros::Time::now() - rrt_start_time).toSec() >= search_time_)
  {
    // ROS_INFO_STREAM("[KRRT]: NOT CONNECTED TO GOAL after " << (ros::Time::now() - rrt_start_time).toSec() << " seconds");
    // ROS_INFO_STREAM("[KRRT]: total sample times: " << idx);
    // ROS_INFO_STREAM("[KRRT]: valid sample times: " << valid_sample_nums);
    // ROS_INFO_STREAM("[KRRT]: valid tree node nums: " << valid_start_tree_node_nums_);
    return 0;
  }
}

int KRRTPlanner::rrtStarConnect(const State& x_init, const State& x_final, 
                          const Vector3d& u_init, const Vector3d& u_final, int n, 
                          double radius, const bool rewire)
{ 
  ros::Time rrt_start_time = ros::Time::now();
  ros::Time first_goal_found_time, final_goal_found_time;
  
  /* local variables */
  int valid_sample_nums = 0; //random samples in obs free area
  int valid_orphan_nums = 0;
  list<int> orphan_idx_list;
  vector<State> vis_x;
  vector<Control> vis_u;
  bool first_time_find_goal = true;
  bool goal_found = false;
  /* init sampling space */
  double best_cost, best_tau;
  vector<pair<State, State>> segs;
  double x_coeff[6], y_coeff[6], z_coeff[6];
  VectorXd x_u_init = VectorXd::Zero(9);
  x_u_init.head(6) = x_init;
  x_u_init.tail(3) = u_init;
  VectorXd x_u_final = VectorXd::Zero(9);
  x_u_final.head(6) = x_final;
  x_u_final.tail(3) = u_final;
  bool direc_connect = computeBestCost(x_u_init, x_u_final, segs, best_cost, best_tau, &vis_x, &vis_u, 
                         x_coeff, y_coeff, z_coeff);
  vis.visualizeBestTraj(vis_x, vis_u, occ_map_->getLocalTime());
  // ROS_INFO("[KRRT-connect]: Best cost: %lf, best tau: %lf", best_cost, best_tau);
  //best traj is collision free
  if (direc_connect && segs.size() == 0) 
  {
    ROS_WARN("Best traj collision free, one shot connected");
    goal_node_->cost_from_start = best_cost;
    goal_node_->parent = start_node_;
    goal_node_->tau_from_parent = best_tau;
    goal_node_->tau_from_start = best_tau;
    goal_node_->n_order = 5;
    for (int i=0; i<6; ++i) 
    {
      goal_node_->x_coeff[i] = x_coeff[i];
      goal_node_->y_coeff[i] = y_coeff[i];
      goal_node_->z_coeff[i] = z_coeff[i];
    }
    goal_found = true;
    fillPath(goal_node_, path_);
    patched_path_ = path_;
 
    /* for traj vis */
    vis.visualizeATraj(vis_x, vis_u, occ_map_->getLocalTime());
    double final_traj_len(0.0), final_traj_duration(0.0), final_traj_ctrl_cost(0.0);
    int final_traj_seg_nums(0);
    getTrajAttributes(path_, final_traj_duration, final_traj_ctrl_cost, final_traj_len, final_traj_seg_nums);
    double final_traj_use_time = ros::Time::now().toSec()-rrt_start_time.toSec();
    // ROS_INFO_STREAM("[KRRT-connect]: [front-end best one shot]: " << endl 
    //      << "    -   seg nums: " << final_traj_seg_nums << endl 
    //      << "    -   use time: " << final_traj_use_time << endl 
    //      << "    -   ctrl cost: " << final_traj_ctrl_cost << endl
    //      << "    -   traj duration: " << final_traj_duration << endl 
    //      << "    -   path length: " << final_traj_len);
    
    return 1;
  }
  else 
  {
    // ROS_ERROR("Best traj one shot violates OR no solution");
  }
  
  vector<pair<Vector3d, Vector3d>> all_corners;
  findSamplingSpace(x_init, x_final, segs, all_corners);
  vector<Vector3d> unit_tracks, p_head, tracks, rotated_unit_tracks;
  setupRandomSampling(x_init, x_final, all_corners, unit_tracks, p_head, 
                      tracks, rotated_unit_tracks);
  /* kd tree init */
  kdtree *start_kd_tree = kd_create(X_DIM);
  kdtree *goal_kd_tree = kd_create(X_DIM);
  //Add start and goal nodes to kd tree
  double init_state[X_DIM];
  double final_state[X_DIM];
  for (int i = 0; i < X_DIM; ++i) 
  {
    init_state[i] = x_init[i];
    final_state[i] = x_final[i];
  }
  kd_insert(start_kd_tree, init_state, start_node_);
  kd_insert(start_kd_tree, final_state, goal_node_);
  kd_insert(goal_kd_tree, init_state, end_tree_goal_node_);
  kd_insert(goal_kd_tree, final_state, end_tree_start_node_);
  
  /* main loop */
  int idx = 0;
  for (idx = 0; (ros::Time::now() - rrt_start_time).toSec() < search_time_ && valid_start_tree_node_nums_ < n && valid_goal_tree_node_nums_ < n ; ++idx) 
  {
    /* biased random sampling */
    State x_rand;
    bool good_sample = samplingOnce(idx%tracks.size(), x_rand, unit_tracks, 
                                    p_head, tracks, rotated_unit_tracks);
    if (!good_sample) 
    {
      continue;
    }
    
    /* just random sampling */
//     x_rand[0] = px_rand_(gen_);
//     x_rand[1] = py_rand_(gen_);
//     x_rand[2] = pz_rand_(gen_);
//     x_rand[3] = vx_rand_(gen_);
//     x_rand[4] = vy_rand_(gen_);
//     x_rand[5] = vz_rand_(gen_);
//     bool check_goal = false;
//     if (idx == 0) {// || (float)rand()/(float)RAND_MAX < epsilon) {
//       //try to connect start_start to goal_state at first
//       x_rand = x_final;
//       check_goal = true;
//     }
//     //check x_rand pos valid
//     Vector3d pos;
//     pos = x_rand.head(3);
//     if (occ_map_->getVoxelState(pos) != 0
//       || x_rand[2] > max(x_init[2], x_final[2]) || x_rand[2] < min(x_init[2], x_final[2])
//       || x_rand[0] > max(x_init[0], x_final[0]) || x_rand[0] < min(x_init[0], x_final[0])
//       || x_rand[1] > max(x_init[1], x_final[1]) || x_rand[1] < min(x_init[1], x_final[1])
// //       || (x_rand[3]*(x_final[0]-x_init[0]) + x_rand[4]*(x_final[1]-x_init[1]) + x_rand[5]*(x_final[2]-x_init[2])) / (sqrt(x_rand[3]*x_rand[3]+x_rand[4]*x_rand[4]+x_rand[5]*x_rand[5]) * sqrt((x_final[2]-x_init[2])*(x_final[2]-x_init[2])+(x_final[1]-x_init[1])*(x_final[1]-x_init[1])+(x_final[0]-x_init[0])*(x_final[0]-x_init[0]))) < 0.9
//       ) {
//       continue;
//     }
    /* end just random sampling */
    
    
    vis.visualizeSampledState(x_rand, occ_map_->getLocalTime());
    ++valid_sample_nums;
    RRTNode* x_near_in_start_tree = nullptr; //parent
    RRTNode* x_near_in_goal_tree = nullptr; 
    double min_cost_from_start_in_start_tree = DBL_MAX;
    double tau_from_start_in_start_tree = DBL_MAX;
    double min_cost_from_start_in_goal_tree = DBL_MAX;
    double tau_from_start_in_goal_tree = DBL_MAX;
    RRTNode* sampled_node_for_start_tree;
    RRTNode* sampled_node_for_goal_tree;
    double cost_start_tree_parent, tau_start_tree_parent;
    double cost_goal_tree_parent, tau_goal_tree_parent;

    //try finding parent in start tree
    {
      /* kd_tree bounds search for parent */
      /* TODO diff range for diff dimention*/
      BOUNDS backward_reachable_bounds;
      calc_backward_reachable_bounds(x_rand, radius, backward_reachable_bounds);
      double center[X_DIM];
      for (int i = 0; i < X_DIM; ++i) 
        {center[i] = x_rand[i];}
      double range = 0;
      for (size_t i=0; i<backward_reachable_bounds.size(); ++i) 
      {
        double r = (backward_reachable_bounds[i].second - backward_reachable_bounds[i].first)/2;
        range = max(range, r);
      }
      struct kdres *presults;
      presults = kd_nearest_range(start_kd_tree, center, range);
      
      /* choose parent from kd tree range query result*/
      double cost, tau, actual_deltaT;
      double x_coeff[4], y_coeff[4], z_coeff[4];
      while(!kd_res_end(presults)) 
      {
        /* get the data and position of the current result item */
        double pos[X_DIM];
        RRTNode* curr_node = (RRTNode*)kd_res_item(presults, pos);
        if (curr_node == goal_node_) 
        {
          // goal node can not be parent of any other node
          kd_res_next( presults );
          continue;
        }
        double cost1, tau1, x_coeff1[4], y_coeff1[4], z_coeff1[4];
        bool connected = HtfConnect(curr_node->x, x_rand, radius, cost1, tau1, x_coeff1, y_coeff1, z_coeff1);
        if (connected && min_cost_from_start_in_start_tree > (curr_node->cost_from_start + cost1)) 
        {
          tau = tau1;
          for (int i=0; i<4; ++i)
          {
            x_coeff[i] = x_coeff1[i];
            y_coeff[i] = y_coeff1[i];
            z_coeff[i] = z_coeff1[i];
          }
          min_cost_from_start_in_start_tree = curr_node->cost_from_start + cost1;
          tau_from_start_in_start_tree = curr_node->tau_from_start + tau1;
          x_near_in_start_tree = curr_node;
          cost_start_tree_parent = cost1; 
          tau_start_tree_parent = tau1;
        }
        kd_res_next(presults); //go to next in kd tree range query result
      }
      kd_res_free(presults); //reset kd tree range query

      if (x_near_in_start_tree != nullptr)
      {
        //sample rejection
        double cost_star, tau_star;
        computeCostAndTime(x_rand, goal_node_->x, cost_star, tau_star);
        if (min_cost_from_start_in_start_tree + cost_star >= goal_node_->cost_from_start) {
          continue;
        }
        /* parent found within radius in start tree, then: 
        * 1.add a node to start_tree_ and start_kd_tree; 
        * 2.rewire. 
        * 3.connect to goal_tree
        */
        /* 1.1 add the randomly sampled node to rrt_tree */
        sampled_node_for_start_tree = start_tree_[valid_start_tree_node_nums_++]; 
        sampled_node_for_start_tree->x = x_rand;
        sampled_node_for_start_tree->parent = x_near_in_start_tree;
        sampled_node_for_start_tree->cost_from_start = min_cost_from_start_in_start_tree;
        sampled_node_for_start_tree->tau_from_start = tau_from_start_in_start_tree;
        sampled_node_for_start_tree->tau_from_parent = tau;
        sampled_node_for_start_tree->n_order = 3;
        for (int i=0; i<4; ++i) 
        {
          sampled_node_for_start_tree->x_coeff[i] = x_coeff[i];
          sampled_node_for_start_tree->y_coeff[i] = y_coeff[i];
          sampled_node_for_start_tree->z_coeff[i] = z_coeff[i];
        }
        x_near_in_start_tree->children.push_back(sampled_node_for_start_tree);
            
        /* 1.2 add the randomly sampled node to start_kd_tree */
        double sample_state[X_DIM];
        for (int i = 0; i < X_DIM; ++i) sample_state[i] = x_rand[i];
        kd_insert(start_kd_tree, sample_state, sampled_node_for_start_tree);
        
        /* 2.rewire */
        if (rewire) 
        {
          //kd_tree bounds search
          BOUNDS forward_reachable_bounds;
          calc_forward_reachable_bounds(x_rand, radius, forward_reachable_bounds);
          double center[X_DIM];
          for (int i = 0; i < X_DIM; ++i) center[i] = x_rand[i];
          double range = 0;
          for (size_t i=0; i<forward_reachable_bounds.size(); ++i) 
          {
            double r = (forward_reachable_bounds[i].second - forward_reachable_bounds[i].first)/2;
            range = max(range, r);
          }
          struct kdres *presults;
          presults = kd_nearest_range(start_kd_tree, center, range);
          while(!kd_res_end(presults)) 
          {
            /* get the data and position of the current result item */
            double pos[X_DIM];
            RRTNode* curr_ndoe = (RRTNode*)kd_res_item(presults, pos);
            if (curr_ndoe == start_node_ || curr_ndoe == goal_node_) 
            {
              kd_res_next( presults );
              continue;
            }
            double cost1, tau1, x_coeff1[4], y_coeff1[4], z_coeff1[4];
            bool connected = HtfConnect(x_rand, curr_ndoe->x, radius, cost1, tau1, x_coeff1, y_coeff1, z_coeff1);
            if (connected && sampled_node_for_start_tree->cost_from_start + cost1 < curr_ndoe->cost_from_start) 
            {
              // If we can get to a node via the sampled_node_for_start_tree faster than via it's existing parent then change the parent
              curr_ndoe->parent->children.remove(curr_ndoe);  //DON'T FORGET THIS, remove it form its parent's children list
              curr_ndoe->parent = sampled_node_for_start_tree;
              curr_ndoe->cost_from_start = sampled_node_for_start_tree->cost_from_start + cost1;
              curr_ndoe->tau_from_parent = tau1;
              curr_ndoe->tau_from_start = sampled_node_for_start_tree->tau_from_start + tau1;
              curr_ndoe->n_order = 3;
              for (int i=0; i<4; ++i) 
              {
                curr_ndoe->x_coeff[i] = x_coeff1[i];
                curr_ndoe->y_coeff[i] = y_coeff1[i];
                curr_ndoe->z_coeff[i] = z_coeff1[i];
              }
              sampled_node_for_start_tree->children.push_back(curr_ndoe);
            }
            /* go to the next entry */
            kd_res_next(presults);
          }
          kd_res_free(presults);
        }/* end of rewire */
      }
    }
    
    //try finding parent in goal tree
    {
      BOUNDS forward_reachable_bounds;
      calc_forward_reachable_bounds(x_rand, radius, forward_reachable_bounds);
      double center[X_DIM];
      for (int i = 0; i < X_DIM; ++i) 
        {center[i] = x_rand[i];}
      double range = 0;
      for (size_t i=0; i<forward_reachable_bounds.size(); ++i) 
      {
        double r = (forward_reachable_bounds[i].second - forward_reachable_bounds[i].first)/2;
        range = max(range, r);
      }
      struct kdres *presults;
      presults = kd_nearest_range(goal_kd_tree, center, range);
      
      /* choose parent from kd tree range query result*/
      double cost, tau, actual_deltaT;
      double x_coeff[4], y_coeff[4], z_coeff[4];
      while(!kd_res_end(presults)) 
      {
        /* get the data and position of the current result item */
        double pos[X_DIM];
        RRTNode* curr_node = (RRTNode*)kd_res_item(presults, pos);
        if (curr_node == end_tree_goal_node_) 
        {
          // goal node can not be parent of any other node
          kd_res_next( presults );
          continue;
        }
        double cost1, tau1, x_coeff1[4], y_coeff1[4], z_coeff1[4];
        bool connected = HtfConnect(x_rand, curr_node->x, radius, cost1, tau1, x_coeff1, y_coeff1, z_coeff1);
        if (connected && min_cost_from_start_in_goal_tree > (curr_node->cost_from_start + cost1)) 
        {
          tau = tau1;
          for (int i=0; i<4; ++i)
          {
            x_coeff[i] = x_coeff1[i];
            y_coeff[i] = y_coeff1[i];
            z_coeff[i] = z_coeff1[i];
          }
          min_cost_from_start_in_goal_tree = curr_node->cost_from_start + cost1;
          tau_from_start_in_goal_tree = curr_node->tau_from_start + tau1;
          x_near_in_goal_tree = curr_node;
          cost_goal_tree_parent = cost1; 
          tau_goal_tree_parent = tau1;
        }
        kd_res_next(presults); //go to next in kd tree range query result
      }
      kd_res_free(presults); //reset kd tree range query

      if (x_near_in_goal_tree != nullptr)
      {
        //sample rejection
        double cost_star, tau_star;
        computeCostAndTime(start_node_->x, x_rand, cost_star, tau_star);
        if (min_cost_from_start_in_goal_tree + cost_star >= goal_node_->cost_from_start) {
          continue;
        }
        /* parent found within radius in goal tree, then: 
        * 1.add a node to goal_tree_ and goal_kd_tree; 
        * 2.rewire. 
        */
        /* 1.1 add the randomly sampled node to goal_tree_ */
        sampled_node_for_goal_tree = goal_tree_[valid_goal_tree_node_nums_++]; 
        sampled_node_for_goal_tree->x = x_rand;
        sampled_node_for_goal_tree->parent = x_near_in_goal_tree;
        sampled_node_for_goal_tree->cost_from_start = min_cost_from_start_in_goal_tree;
        sampled_node_for_goal_tree->tau_from_start = tau_from_start_in_goal_tree;
        sampled_node_for_goal_tree->tau_from_parent = tau;
        sampled_node_for_goal_tree->n_order = 3;
        for (int i=0; i<4; ++i) 
        {
          sampled_node_for_goal_tree->x_coeff[i] = x_coeff[i];
          sampled_node_for_goal_tree->y_coeff[i] = y_coeff[i];
          sampled_node_for_goal_tree->z_coeff[i] = z_coeff[i];
        }
        x_near_in_goal_tree->children.push_back(sampled_node_for_goal_tree);
        /* 1.2 add the randomly sampled node to goal_kd_tree */
        double sample_state[X_DIM];
        for (int i = 0; i < X_DIM; ++i) sample_state[i] = x_rand[i];
        kd_insert(goal_kd_tree, sample_state, sampled_node_for_goal_tree);
        /* 2.rewire */
        if (rewire) 
        {
          //kd_tree bounds search
          BOUNDS backward_reachable_bounds;
          calc_backward_reachable_bounds(x_rand, radius, backward_reachable_bounds);
          double center[X_DIM];
          for (int i = 0; i < X_DIM; ++i) center[i] = x_rand[i];
          double range = 0;
          for (size_t i=0; i<backward_reachable_bounds.size(); ++i) 
          {
            double r = (backward_reachable_bounds[i].second - backward_reachable_bounds[i].first)/2;
            range = max(range, r);
          }
          struct kdres *presults;
          presults = kd_nearest_range(goal_kd_tree, center, range);
          while(!kd_res_end(presults)) 
          {
            /* get the data and position of the current result item */
            double pos[X_DIM];
            RRTNode* curr_ndoe = (RRTNode*)kd_res_item(presults, pos);
            if (curr_ndoe == end_tree_start_node_ || curr_ndoe == end_tree_goal_node_) 
            {
              kd_res_next( presults );
              continue;
            }
            double cost1, tau1, x_coeff1[4], y_coeff1[4], z_coeff1[4];
            bool connected = HtfConnect(curr_ndoe->x, x_rand, radius, cost1, tau1, x_coeff1, y_coeff1, z_coeff1);
            if (connected && sampled_node_for_goal_tree->cost_from_start + cost1 < curr_ndoe->cost_from_start) 
            {
              curr_ndoe->parent->children.remove(curr_ndoe);  //DON'T FORGET THIS, remove it form its parent's children list
              curr_ndoe->parent = sampled_node_for_goal_tree;
              curr_ndoe->cost_from_start = sampled_node_for_goal_tree->cost_from_start + cost1;
              curr_ndoe->tau_from_parent = tau1;
              curr_ndoe->tau_from_start = sampled_node_for_goal_tree->tau_from_start + tau1;
              curr_ndoe->n_order = 3;
              for (int i=0; i<4; ++i) 
              {
                curr_ndoe->x_coeff[i] = x_coeff1[i];
                curr_ndoe->y_coeff[i] = y_coeff1[i];
                curr_ndoe->z_coeff[i] = z_coeff1[i];
              }
              sampled_node_for_goal_tree->children.push_back(curr_ndoe);
            }
            /* go to the next entry */
            kd_res_next(presults);
          }
          kd_res_free(presults);
        }/* end of rewire */
      }
    }

    if (x_near_in_start_tree != nullptr && x_near_in_goal_tree != nullptr && 
        goal_node_->cost_from_start > min_cost_from_start_in_start_tree + min_cost_from_start_in_goal_tree) 
    {
      /* connect two trees if both parents are found */   
      // add nodes to start_tree_ and start_kd_tree; 
      //reverse list
      goal_found = true;
      RRTNodePtr n_p = sampled_node_for_goal_tree;
      RRTNodePtr parent_p = sampled_node_for_start_tree;
      while (n_p->parent != nullptr)
      {
        RRTNodePtr new_start_tree_node;
        if (n_p->parent->parent == nullptr)
          new_start_tree_node = goal_node_;
        else
          new_start_tree_node = start_tree_[valid_start_tree_node_nums_++]; 
        new_start_tree_node->x = n_p->parent->x;
        new_start_tree_node->parent = parent_p;
        new_start_tree_node->cost_from_start = parent_p->cost_from_start + (n_p->cost_from_start - n_p->parent->cost_from_start);
        new_start_tree_node->tau_from_start = parent_p->tau_from_start + n_p->tau_from_parent;
        new_start_tree_node->tau_from_parent = n_p->tau_from_parent;
        new_start_tree_node->n_order = 3;
        //do mirror by t = tau/2, that is, substitute t for (tau - t)
        // mirror_coeff(new_start_tree_node->x_coeff, n_p->x_coeff, n_p->tau_from_parent);
        // mirror_coeff(new_start_tree_node->y_coeff, n_p->y_coeff, n_p->tau_from_parent);
        // mirror_coeff(new_start_tree_node->z_coeff, n_p->z_coeff, n_p->tau_from_parent);
        for (int i=0; i<4; ++i) 
        {
          new_start_tree_node->x_coeff[i] = n_p->x_coeff[i];
          new_start_tree_node->y_coeff[i] = n_p->y_coeff[i];
          new_start_tree_node->z_coeff[i] = n_p->z_coeff[i];
        }

        parent_p->children.push_back(new_start_tree_node);
        /* add the node to goal_kd_tree */
        double sample_state[X_DIM];
        for (int i = 0; i < X_DIM; ++i) sample_state[i] = new_start_tree_node->x[i];
        kd_insert(start_kd_tree, sample_state, new_start_tree_node);
        
        n_p = n_p->parent;
        parent_p = new_start_tree_node;
      }

      n_p = sampled_node_for_start_tree;
      parent_p = sampled_node_for_goal_tree;
      while (n_p->parent != nullptr)
      {
        RRTNodePtr new_goal_tree_node;
        if (n_p->parent->parent == nullptr)
          new_goal_tree_node = end_tree_goal_node_;
        else
          new_goal_tree_node = goal_tree_[valid_goal_tree_node_nums_++]; 
        new_goal_tree_node->x = n_p->parent->x;
        new_goal_tree_node->parent = parent_p;
        new_goal_tree_node->cost_from_start = parent_p->cost_from_start + (n_p->cost_from_start - n_p->parent->cost_from_start);
        new_goal_tree_node->tau_from_start = parent_p->tau_from_start + n_p->tau_from_parent;
        new_goal_tree_node->tau_from_parent = n_p->tau_from_parent;
        new_goal_tree_node->n_order = 3;
        //do mirror by t = tau/2, that is, substitute t for (tau - t)
        // mirror_coeff(new_goal_tree_node->x_coeff, n_p->x_coeff, n_p->tau_from_parent);
        // mirror_coeff(new_goal_tree_node->y_coeff, n_p->y_coeff, n_p->tau_from_parent);
        // mirror_coeff(new_goal_tree_node->z_coeff, n_p->z_coeff, n_p->tau_from_parent);
        for (int i=0; i<4; ++i) 
        {
          new_goal_tree_node->x_coeff[i] = n_p->x_coeff[i];
          new_goal_tree_node->y_coeff[i] = n_p->y_coeff[i];
          new_goal_tree_node->z_coeff[i] = n_p->z_coeff[i];
        }

        parent_p->children.push_back(new_goal_tree_node);
        /* add the node to goal_kd_tree */
        double sample_state[X_DIM];
        for (int i = 0; i < X_DIM; ++i) sample_state[i] = new_goal_tree_node->x[i];
        kd_insert(goal_kd_tree, sample_state, new_goal_tree_node);
        
        n_p = n_p->parent;
        parent_p = new_goal_tree_node;
      }

      if (first_time_find_goal) 
      {
        first_goal_found_time = ros::Time::now();
        first_time_find_goal = false;
        RRTNodePtr node = goal_node_;
        while (node) 
        {
          //new a RRTNodePtr instead of use ptr in start_tree_ because as sampling goes on and rewire happens, 
          //parent, children and other attributes of start_tree_[i] may change.
          RRTNodePtr n_p = new RRTNode;
          n_p->children = node->children;
          n_p->cost_from_start = node->cost_from_start;
          n_p->parent = node->parent;
          n_p->x = node->x;
          n_p->tau_from_parent = node->tau_from_parent;
          n_p->tau_from_start = node->tau_from_start;
          n_p->n_order = node->n_order;
          for (int i=0; i<4; ++i) 
          {
            n_p->x_coeff[i] = node->x_coeff[i];
            n_p->y_coeff[i] = node->y_coeff[i];
            n_p->z_coeff[i] = node->z_coeff[i];
          }
          // path_vector[0] is goal node, path_vector[n-1] is start node
          path_on_first_search_.push_back(n_p); 
          node = node->parent;
        }
      }
      if (stop_after_first_traj_found_)
        break; //stop searching after first time find the goal?
    }
    else 
    {
      continue;
    }/* end of connect two trees */
  }/* end of sample once */
  
  // vis.visualizeAllTreeNode(start_node_, occ_map_->getLocalTime());
  if (goal_found) 
  {
    final_goal_found_time = ros::Time::now();
    fillPath(goal_node_, path_);  
    ros::Time t_p = ros::Time::now();
    patching(path_, patched_path_, u_init);
    ROS_ERROR("patch use time: %lf", (ros::Time::now()-t_p).toSec());
    /* for traj vis */
    vis_x.clear();
    vis_u.clear();
    getVisStateAndControl(path_on_first_search_, &vis_x, &vis_u);
    vis.visualizeFirstTraj(vis_x, vis_u, occ_map_->getLocalTime());
    vis_x.clear();
    vis_u.clear();
    getVisStateAndControl(path_, &vis_x, &vis_u);
    vis.visualizeATraj(vis_x, vis_u, occ_map_->getLocalTime());
    
    // vis_x.clear();
    // vis_u.clear();
    // findAllStatesAndControlInAllTrunks(start_node_, &vis_x, &vis_u);
    // vis.visualizeATraj(vis_x, vis_u, occ_map_->getLocalTime());
    
//     ros::Time t_c = ros::Time::now();
//     vector<Vector3d> cover_grid;
//     getVisTrajCovering(path_, cover_grid);
//     cout << "[===========]: " << (ros::Time::now()-t_c).toSec() << endl;
//     vis.visualizeTrajCovering(cover_grid, resolution_, occ_map_->getLocalTime());
    
    double first_traj_len(0.0), first_traj_duration(0.0), first_traj_ctrl_cost(0.0);
    double final_traj_len(0.0), final_traj_duration(0.0), final_traj_ctrl_cost(0.0);
    int first_traj_seg_nums(0), final_traj_seg_nums(0);
    getTrajAttributes(path_on_first_search_, first_traj_duration, first_traj_ctrl_cost, first_traj_len, first_traj_seg_nums);
    getTrajAttributes(path_, final_traj_duration, final_traj_ctrl_cost, final_traj_len, final_traj_seg_nums);
    double first_traj_use_time = first_goal_found_time.toSec()-rrt_start_time.toSec();
    double final_traj_use_time = final_goal_found_time.toSec()-rrt_start_time.toSec();
    // ROS_INFO_STREAM("[KRRT-connect]: total sample times: " << idx);
    // ROS_INFO_STREAM("[KRRT-connect]: valid sample times: " << valid_sample_nums);
    // ROS_INFO_STREAM("[KRRT-connect]: valid tree node nums: " << valid_start_tree_node_nums_);
    // ROS_INFO_STREAM("[KRRT-connect]: [front-end first path]: " << endl 
    //      << "    -   seg nums: " << first_traj_seg_nums << endl 
    //      << "    -   time: " << first_traj_use_time << endl 
    //      << "    -   ctrl cost: " << first_traj_ctrl_cost << endl
    //      << "    -   traj duration: " << first_traj_duration << endl 
    //      << "    -   path length: " << first_traj_len);
        
    // ROS_INFO_STREAM("[KRRT-connect]: [front-end final path]: " << endl 
    //      << "    -   seg nums: " << final_traj_seg_nums << endl 
    //      << "    -   time: " << final_traj_use_time << endl 
    //      << "    -   ctrl cost: " << final_traj_ctrl_cost << endl
    //      << "    -   traj duration: " << final_traj_duration << endl 
    //      << "    -   path length: " << final_traj_len);
    return 1;
  } 
  else if (valid_start_tree_node_nums_ == n)
  {
    // ROS_INFO_STREAM("[KRRT-connect]: NOT CONNECTED TO GOAL after " << n << " nodes added to rrt-tree");
    // ROS_INFO_STREAM("[KRRT-connect total sample times: " << idx);
    // ROS_INFO_STREAM("[KRRT-connect]: valid sample times: " << valid_sample_nums);
    // ROS_INFO_STREAM("[KRRT-connect]: valid tree node nums: " << valid_start_tree_node_nums_);
    return 0;
  }
  else if ((ros::Time::now() - rrt_start_time).toSec() >= search_time_)
  {
    // ROS_INFO_STREAM("[KRRT-connect]: NOT CONNECTED TO GOAL after " << (ros::Time::now() - rrt_start_time).toSec() << " seconds");
    // ROS_INFO_STREAM("[KRRT-connect]: total sample times: " << idx);
    // ROS_INFO_STREAM("[KRRT-connect]: valid sample times: " << valid_sample_nums);
    // ROS_INFO_STREAM("[KRRT-connect]: valid tree node nums: " << valid_start_tree_node_nums_);
    return 0;
  }
}

void KRRTPlanner::mirror_coeff(double *new_coeff, double *ori_coeff, double t)
{
  double c3 = ori_coeff[0];
  double c2 = ori_coeff[1];
  double c1 = ori_coeff[2];
  double c0 = ori_coeff[3];
  new_coeff[0] = -c3;
  new_coeff[1] = 3*t*c3 + c2;
  new_coeff[2] = -3*t*t*c3 - 2*t*c2 - c1;
  new_coeff[3] = t*t*t*c3 + t*t*c2 + t*c1 + c0;
}

void KRRTPlanner::calc_forward_reachable_bounds(const State& init_state, 
                                          const double& radius, BOUNDS& bounds) 
/*
//my calculation of bounds
{
#ifdef TIMING
    ros::Time s = ros::Time::now();
#endif
    bounds.resize(X_DIM);

    double x0_0 = init_state[0];
    double x0_1 = init_state[1];
    double x0_2 = init_state[2];
    double x0_3 = init_state[3];
    double x0_4 = init_state[4];
    double x0_5 = init_state[5];
    bounds[0].first = x0_0;
    bounds[0].second = x0_0;
    bounds[1].first = x0_1;
    bounds[1].second = x0_1;
    bounds[2].first = x0_2;
    bounds[2].second = x0_2;
    bounds[3].first = x0_3;
    bounds[3].second = x0_3;
    bounds[4].first = x0_4;
    bounds[4].second = x0_4;
    bounds[5].first = x0_5;
    bounds[5].second = x0_5;

    //calculate x0 x1 x2 bounds
    int segments = 10; 
    for (int i=0; i<=segments; ++i) {
        double tau = (double)i*radius/(double)segments; //tau = i/segments * radius
        double t1 = tau*tau*tau;
        double t2 = radius-tau;
        double t3 = t1*t2/3.0/rou_;
        double squred_diag_M_0_1_2 = sqrt(t3);
        double x_ba_tau_0 = x0_0+c_[0]*tau+x0_3*tau+(c_[3]*tau*tau)/2;
        double x_ba_tau_1 = x0_1+c_[1]*tau+x0_4*tau+(c_[4]*tau*tau)/2;
        double x_ba_tau_2 = x0_2+c_[2]*tau+x0_5*tau+(c_[5]*tau*tau)/2;
        bounds[0].first =  min(bounds[0].first, x_ba_tau_0-squred_diag_M_0_1_2);
        bounds[0].second = max(bounds[0].second, x_ba_tau_0+squred_diag_M_0_1_2);
        bounds[1].first =  min(bounds[1].first, x_ba_tau_1-squred_diag_M_0_1_2);
        bounds[1].second = max(bounds[1].second, x_ba_tau_1+squred_diag_M_0_1_2);
        bounds[2].first =  min(bounds[2].first, x_ba_tau_2-squred_diag_M_0_1_2);
        bounds[2].second = max(bounds[2].second, x_ba_tau_2+squred_diag_M_0_1_2);
    }
    
    //calculate x3 x4 x5 bounds
    for (int i=0; i<=segments; ++i) {
        double tau = (double)i*radius/(double)segments; //tau = i/segments * radius
        double t1 = tau*(radius-tau)/rou_;
        double squred_diag_M_3_4_5 = sqrt(t1);
        double x_ba_tau_3 = x0_3+c_[3]*tau;
        double x_ba_tau_4 = x0_4+c_[4]*tau;
        double x_ba_tau_5 = x0_5+c_[5]*tau;
        bounds[3].first =  min(bounds[3].first, x_ba_tau_3-squred_diag_M_3_4_5);
        bounds[3].second = max(bounds[3].second, x_ba_tau_3+squred_diag_M_3_4_5);
        bounds[4].first =  min(bounds[4].first, x_ba_tau_4-squred_diag_M_3_4_5);
        bounds[4].second = max(bounds[4].second, x_ba_tau_4+squred_diag_M_3_4_5);
        bounds[5].first =  min(bounds[5].first, x_ba_tau_5-squred_diag_M_3_4_5);
        bounds[5].second = max(bounds[5].second, x_ba_tau_5+squred_diag_M_3_4_5);
    }
    
#ifdef TIMING
    ros::Time e = ros::Time::now();
    cal_forward_bound_time += e.toSec()-s.toSec();
    cal_forward_bound_nums++;
#endif
    
#ifdef DEBUG
    ROS_INFO("init state: %lf, %lf, %lf, %lf, %lf, %lf, radius: %lf", x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, radius);
    for (int i=0; i<X_DIM; ++i) {
        ROS_INFO("forward bounds[%d]: %lf, %lf", i, bounds[i].first, bounds[i].second);
    }
#endif
}
*/
//the author's calculation of bounds
{
    complex<double> im(0,1);
    double x0_0 = init_state[0];
    double x0_1 = init_state[1];
    double x0_2 = init_state[2];
    double x0_3 = init_state[3];
    double x0_4 = init_state[4];
    double x0_5 = init_state[5];
    bounds.resize(X_DIM);

    // Calculate x1 min
    {
        complex<double> t1 = x0_3*x0_3;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/4.0;
        complex<double> t25 = 4.0*(t1/4.0-t4/16.0)/t18;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = t19-t25+t26;
        complex<double> t29 = sqrt(3.0);
        complex<double> t31 = t27*t27;
        complex<double> t34 = sqrt((t26-t19+t25)*t31*t27);
        complex<double> t37 = x0_0+t27*x0_3-t29*t34/3.0;

        bounds[0].first = t37.real();
    }

    {
        complex<double> t1 = x0_3*x0_3;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t31 = t18/4.0+4.0*t24;
        complex<double> t33 = -t19+t25+t26+(1.0/2.0*im)*t27*t31;
        complex<double> t38 = t33*t33;
        complex<double> t41 = sqrt((t26+t19-t25+(-1.0/2.0*im)*t27*t31)*t38*t33);
        complex<double> t44 = x0_0+t33*x0_3-t27*t41/3.0;

        bounds[0].first = min(bounds[0].first, t44.real());
    }

    {
        complex<double> t1 = x0_3*x0_3;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t31 = t18/4.0+4.0*t24;
        complex<double> t33 = -t19+t25+t26+(-1.0/2.0*im)*t27*t31;
        complex<double> t38 = t33*t33;
        complex<double> t41 = sqrt((t26+t19-t25+(1.0/2.0*im)*t27*t31)*t38*t33);
        complex<double> t44 = x0_0+t33*x0_3-t27*t41/3.0;

        bounds[0].first = min(bounds[0].first, t44.real());
    }

    // Calculate x1 max
    {
        complex<double> t1 = x0_3*x0_3;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/4.0;
        complex<double> t25 = 4.0*(t1/4.0-t4/16.0)/t18;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = t19-t25+t26;
        complex<double> t29 = sqrt(3.0);
        complex<double> t31 = t27*t27;
        complex<double> t34 = sqrt((t26-t19+t25)*t31*t27);
        complex<double> t37 = x0_0+t27*x0_3+t29*t34/3.0;

        bounds[0].second = t37.real();
    }

    {
        complex<double> t1 = x0_3*x0_3;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t31 = t18/4.0+4.0*t24;
        complex<double> t33 = -t19+t25+t26+(1.0/2.0*im)*t27*t31;
        complex<double> t38 = t33*t33;
        complex<double> t41 = sqrt((t26+t19-t25+(-1.0/2.0*im)*t27*t31)*t38*t33);
        complex<double> t44 = x0_0+t33*x0_3+t27*t41/3.0;

        bounds[0].second = max(bounds[0].second, t44.real());
    }

    {
        complex<double> t1 = x0_3*x0_3;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t31 = t18/4.0+4.0*t24;
        complex<double> t33 = -t19+t25+t26+(-1.0/2.0*im)*t27*t31;
        complex<double> t38 = t33*t33;
        complex<double> t41 = sqrt((t26+t19-t25+(1.0/2.0*im)*t27*t31)*t38*t33);
        complex<double> t44 = x0_0+t33*x0_3+t27*t41/3.0;

        bounds[0].second = max(bounds[0].second, t44.real());
    }

    // Calculate x2 min
    {
        complex<double> t1 = x0_4*x0_4;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/4.0;
        complex<double> t25 = 4.0*(t1/4.0-t4/16.0)/t18;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = t19-t25+t26;
        complex<double> t29 = sqrt(3.0);
        complex<double> t31 = t27*t27;
        complex<double> t34 = sqrt((t26-t19+t25)*t31*t27);
        complex<double> t37 = x0_1+t27*x0_4-t29*t34/3.0;

        bounds[1].first = t37.real();
    }

    {
        complex<double> t1 = x0_4*x0_4;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t31 = t18/4.0+4.0*t24;
        complex<double> t33 = -t19+t25+t26+(1.0/2.0*im)*t27*t31;
        complex<double> t38 = t33*t33;
        complex<double> t41 = sqrt((t26+t19-t25+(-1.0/2.0*im)*t27*t31)*t38*t33);
        complex<double> t44 = x0_1+t33*x0_4-t27*t41/3.0;

        bounds[1].first = min(bounds[1].first, t44.real());
    }

    {
        complex<double> t1 = x0_4*x0_4;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t31 = t18/4.0+4.0*t24;
        complex<double> t33 = -t19+t25+t26+(-1.0/2.0*im)*t27*t31;
        complex<double> t38 = t33*t33;
        complex<double> t41 = sqrt((t26+t19-t25+(1.0/2.0*im)*t27*t31)*t38*t33);
        complex<double> t44 = x0_1+t33*x0_4-t27*t41/3.0;

        bounds[1].first = min(bounds[1].first, t44.real());
    }

    // Calculate x2 max
    {
        complex<double> t1 = x0_4*x0_4;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/4.0;
        complex<double> t25 = 4.0*(t1/4.0-t4/16.0)/t18;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = t19-t25+t26;
        complex<double> t29 = sqrt(3.0);
        complex<double> t31 = t27*t27;
        complex<double> t34 = sqrt((t26-t19+t25)*t31*t27);
        complex<double> t37 = x0_1+t27*x0_4+t29*t34/3.0;

        bounds[1].second = t37.real();
    }

    {
        complex<double> t1 = x0_4*x0_4;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t31 = t18/4.0+4.0*t24;
        complex<double> t33 = -t19+t25+t26+(1.0/2.0*im)*t27*t31;
        complex<double> t38 = t33*t33;
        complex<double> t41 = sqrt((t26+t19-t25+(-1.0/2.0*im)*t27*t31)*t38*t33);
        complex<double> t44 = x0_1+t33*x0_4+t27*t41/3.0;

        bounds[1].second = max(bounds[1].second, t44.real());
    }

    {
        complex<double> t1 = x0_4*x0_4;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t31 = t18/4.0+4.0*t24;
        complex<double> t33 = -t19+t25+t26+(-1.0/2.0*im)*t27*t31;
        complex<double> t38 = t33*t33;
        complex<double> t41 = sqrt((t26+t19-t25+(1.0/2.0*im)*t27*t31)*t38*t33);
        complex<double> t44 = x0_1+t33*x0_4+t27*t41/3.0;

        bounds[1].second = max(bounds[1].second, t44.real());
    }
    
        // Calculate x3 min
    {
        complex<double> t1 = x0_5*x0_5;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/4.0;
        complex<double> t25 = 4.0*(t1/4.0-t4/16.0)/t18;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = t19-t25+t26;
        complex<double> t29 = sqrt(3.0);
        complex<double> t31 = t27*t27;
        complex<double> t34 = sqrt((t26-t19+t25)*t31*t27);
        complex<double> t37 = x0_2+t27*x0_5-t29*t34/3.0;

        bounds[2].first = t37.real();
    }

    {
        complex<double> t1 = x0_5*x0_5;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t31 = t18/4.0+4.0*t24;
        complex<double> t33 = -t19+t25+t26+(1.0/2.0*im)*t27*t31;
        complex<double> t38 = t33*t33;
        complex<double> t41 = sqrt((t26+t19-t25+(-1.0/2.0*im)*t27*t31)*t38*t33);
        complex<double> t44 = x0_2+t33*x0_5-t27*t41/3.0;

        bounds[2].first = min(bounds[2].first, t44.real());
    }

    {
        complex<double> t1 = x0_5*x0_5;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t31 = t18/4.0+4.0*t24;
        complex<double> t33 = -t19+t25+t26+(-1.0/2.0*im)*t27*t31;
        complex<double> t38 = t33*t33;
        complex<double> t41 = sqrt((t26+t19-t25+(1.0/2.0*im)*t27*t31)*t38*t33);
        complex<double> t44 = x0_2+t33*x0_5-t27*t41/3.0;

        bounds[2].first = min(bounds[2].first, t44.real());
    }

    // Calculate x3 max
    {
        complex<double> t1 = x0_5*x0_5;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/4.0;
        complex<double> t25 = 4.0*(t1/4.0-t4/16.0)/t18;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = t19-t25+t26;
        complex<double> t29 = sqrt(3.0);
        complex<double> t31 = t27*t27;
        complex<double> t34 = sqrt((t26-t19+t25)*t31*t27);
        complex<double> t37 = x0_2+t27*x0_5+t29*t34/3.0;

        bounds[2].second = t37.real();
    }

    {
        complex<double> t1 = x0_5*x0_5;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t31 = t18/4.0+4.0*t24;
        complex<double> t33 = -t19+t25+t26+(1.0/2.0*im)*t27*t31;
        complex<double> t38 = t33*t33;
        complex<double> t41 = sqrt((t26+t19-t25+(-1.0/2.0*im)*t27*t31)*t38*t33);
        complex<double> t44 = x0_2+t33*x0_5+t27*t41/3.0;

        bounds[2].second = max(bounds[2].second, t44.real());
    }

    {
        complex<double> t1 = x0_5*x0_5;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(12.0*radius*t1-t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t31 = t18/4.0+4.0*t24;
        complex<double> t33 = -t19+t25+t26+(-1.0/2.0*im)*t27*t31;
        complex<double> t38 = t33*t33;
        complex<double> t41 = sqrt((t26+t19-t25+(1.0/2.0*im)*t27*t31)*t38*t33);
        complex<double> t44 = x0_2+t33*x0_5+t27*t41/3.0;

        bounds[2].second = max(bounds[2].second, t44.real());
    }
    
    // Calculate min/max bounds for x4
    {
        double t1 = sqrt(4.0);
        double t2 = radius*radius;
        double t3 = sqrt(t2);

        bounds[3].first = x0_3-t1*t3/4.0;
        bounds[3].second = x0_3+t1*t3/4.0;
    }

    // Calculate min/max bounds for x5
    {
        double t1 = sqrt(4.0);
        double t2 = radius*radius;
        double t3 = sqrt(t2);

        bounds[4].first = x0_4-t1*t3/4.0;
        bounds[4].second = x0_4+t1*t3/4.0;
    }
    
    // Calculate min/max bounds for x6
    {
        double t1 = sqrt(4.0);
        double t2 = radius*radius;
        double t3 = sqrt(t2);

        bounds[5].first = x0_5-t1*t3/4.0;
        bounds[5].second = x0_5+t1*t3/4.0;
    }
#ifdef DEBUG
    ROS_INFO("init state: %lf, %lf, %lf, %lf, %lf, %lf, radius: %lf", x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, radius);
    for (int i=0; i<X_DIM; ++i) {
        ROS_INFO("forward bounds[%d]: %lf, %lf", i, bounds[i].first, bounds[i].second);
    }
#endif
}


inline void KRRTPlanner::calc_backward_reachable_bounds(const State& init_state, 
                                          const double& radius, BOUNDS& bounds)
/*
//my calculation of bounds
{
#ifdef TIMING
    ros::Time s = ros::Time::now();
#endif
    bounds.resize(X_DIM);

    double x1_0 = init_state[0];
    double x1_1 = init_state[1];
    double x1_2 = init_state[2];
    double x1_3 = init_state[3];
    double x1_4 = init_state[4];
    double x1_5 = init_state[5];
    double x0_3 = x1_3;
    double x0_4 = x1_4;
    double x0_5 = x1_5;
    double x0_0 = x1_0-(c_[0]*radius+x0_3*radius+(c_[3]*radius*radius)/2);
    double x0_1 = x1_1-(c_[1]*radius+x0_4*radius+(c_[4]*radius*radius)/2);
    double x0_2 = x1_2-(c_[2]*radius+x0_5*radius+(c_[5]*radius*radius)/2);
    
    bounds[0].first = x0_0;
    bounds[0].second = x0_0;
    bounds[1].first = x0_1;
    bounds[1].second = x0_1;
    bounds[2].first = x0_2;
    bounds[2].second = x0_2;
    bounds[3].first = x0_3;
    bounds[3].second = x0_3;
    bounds[4].first = x0_4;
    bounds[4].second = x0_4;
    bounds[5].first = x0_5;
    bounds[5].second = x0_5;

    //calculate x0 x1 x2 bounds
    int segments = 10; 
    for (int i=0; i<=segments; ++i) {
        double tau = (double)i*radius/(double)segments; //tau = i/segments * radius
        double t1 = tau*tau*tau;
        double t2 = radius-tau;
        double t3 = t1*t2/3.0/rou_; //actually 1.0 should be rou_ in my calculation
        double squred_diag_M_0_1_2 = sqrt(t3);
        double x_ba_tau_0 = x0_0+c_[0]*tau+x0_3*tau+(c_[3]*tau*tau)/2;
        double x_ba_tau_1 = x0_1+c_[1]*tau+x0_4*tau+(c_[4]*tau*tau)/2;
        double x_ba_tau_2 = x0_2+c_[2]*tau+x0_5*tau+(c_[5]*tau*tau)/2;
        bounds[0].first =  min(bounds[0].first, x_ba_tau_0-squred_diag_M_0_1_2);
        bounds[0].second = max(bounds[0].second, x_ba_tau_0+squred_diag_M_0_1_2);
        bounds[1].first =  min(bounds[1].first, x_ba_tau_1-squred_diag_M_0_1_2);
        bounds[1].second = max(bounds[1].second, x_ba_tau_1+squred_diag_M_0_1_2);
        bounds[2].first =  min(bounds[2].first, x_ba_tau_2-squred_diag_M_0_1_2);
        bounds[2].second = max(bounds[2].second, x_ba_tau_2+squred_diag_M_0_1_2);
    }
    
    //calculate x3 x4 x5 bounds
    for (int i=0; i<=segments; ++i) {
        double tau = (double)i*radius/(double)segments; //tau = i/segments * radius
        double t1 = tau*(radius-tau)/rou_; //actually 1.0 should be rou_ in my calculation
        double squred_diag_M_3_4_5 = sqrt(t1);
        double x_ba_tau_3 = x0_3+c_[3]*tau;
        double x_ba_tau_4 = x0_4+c_[4]*tau;
        double x_ba_tau_5 = x0_5+c_[5]*tau;
        bounds[3].first =  min(bounds[3].first, x_ba_tau_3-squred_diag_M_3_4_5);
        bounds[3].second = max(bounds[3].second, x_ba_tau_3+squred_diag_M_3_4_5);
        bounds[4].first =  min(bounds[4].first, x_ba_tau_4-squred_diag_M_3_4_5);
        bounds[4].second = max(bounds[4].second, x_ba_tau_4+squred_diag_M_3_4_5);
        bounds[5].first =  min(bounds[5].first, x_ba_tau_5-squred_diag_M_3_4_5);
        bounds[5].second = max(bounds[5].second, x_ba_tau_5+squred_diag_M_3_4_5);
    }

#ifdef TIMING
    ros::Time e = ros::Time::now();
    cal_backward_bound_time += e.toSec()-s.toSec();
    cal_backward_bound_nums++;
#endif
    
#ifdef DEBUG
    ROS_INFO("init state: %lf, %lf, %lf, %lf, %lf, %lf, radius: %lf", x1_0, x1_1, x1_2, x1_3, x1_4, x1_5, radius);
    for (int i=0; i<X_DIM; ++i) {
        ROS_INFO("backward bounds[%d]: %lf, %lf", i, bounds[i].first, bounds[i].second);
    }
#endif
}
*/
//the author's calculation of bounds
{
    complex<double> im(0,1);
    double x0_0 = init_state[0];
    double x0_1 = init_state[1];
    double x0_2 = init_state[2];
    double x0_3 = init_state[3];
    double x0_4 = init_state[4];
    double x0_5 = init_state[5];
    bounds.resize(X_DIM);
    
    // Calculate x1 bounds
    {
        double t1 = x0_3*x0_3;
        double t4 = radius*radius;
        double t6 = t1*t1;
        double t11 = t4*t4;
        double t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        double t18 = pow(-12.0*radius*t1+t4*radius+2.0*t15,0.3333333333333333);
        double t19 = t18/4.0;
        double t25 = 4.0*(t1/4.0-t4/16.0)/t18;
        double t26 = radius/2.0;
        double t27 = t19-t25-t26;
        double t29 = t27*t27;
        double t34 = sqrt(-3.0*t29*t27*(t26+t19-t25));
        double t36_min = x0_0+t27*x0_3-t34/3.0;
        double t36_max = x0_0+t27*x0_3+t34/3.0;

        bounds[0].first = t36_min;
        bounds[0].second = t36_max;
    }

    {
        complex<double> t1 = x0_3*x0_3;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(-12.0*radius*t1+t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t32 = (1.0/2.0*im)*t27*(t18/4.0+4.0*t24);
        complex<double> t33 = -t19+t25-t26+t32;
        complex<double> t35 = t33*t33;
        complex<double> t40 = sqrt(-3.0*t35*t33*(t26-t19+t25+t32));
        complex<double> t42_min = x0_0+t33*x0_3-t40/3.0;
        complex<double> t42_max = x0_0+t33*x0_3+t40/3.0;

        bounds[0].first = min(bounds[0].first, t42_min.real());
        bounds[0].second = max(bounds[0].second, t42_max.real());
    }

    {
        complex<double> t1 = x0_3*x0_3;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(-12.0*radius*t1+t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t32 = (-1.0/2.0*im)*t27*(t18/4.0+4.0*t24);
        complex<double> t33 = -t19+t25-t26+t32;
        complex<double> t35 = t33*t33;
        complex<double> t40 = sqrt(-3.0*t35*t33*(t26-t19+t25+t32));
        complex<double> t42_min = x0_0+t33*x0_3-t40/3.0;
        complex<double> t42_max = x0_0+t33*x0_3+t40/3.0;

        bounds[0].first = min(bounds[0].first, t42_min.real());
        bounds[0].second = max(bounds[0].second, t42_max.real());
    }

    // Calculate x2 bounds
    {
        complex<double> t1 = x0_4*x0_4;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(-12.0*radius*t1+t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/4.0;
        complex<double> t25 = 4.0*(t1/4.0-t4/16.0)/t18;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = t19-t25-t26;
        complex<double> t29 = t27*t27;
        complex<double> t34 = sqrt(-3.0*t29*t27*(t26+t19-t25));
        complex<double> t36_min = x0_1+t27*x0_4-t34/3.0;
        complex<double> t36_max = x0_1+t27*x0_4+t34/3.0;

        bounds[1].first = t36_min.real();
        bounds[1].second = t36_max.real();
    }

    {
        complex<double> t1 = x0_4*x0_4;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(-12.0*radius*t1+t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t32 = (1.0/2.0*im)*t27*(t18/4.0+4.0*t24);
        complex<double> t33 = -t19+t25-t26+t32;
        complex<double> t35 = t33*t33;
        complex<double> t40 = sqrt(-3.0*t35*t33*(t26-t19+t25+t32));
        complex<double> t42_min = x0_1+t33*x0_4-t40/3.0;
        complex<double> t42_max = x0_1+t33*x0_4+t40/3.0;

        bounds[1].first = min(bounds[1].first, t42_min.real());
        bounds[1].second = max(bounds[1].second, t42_max.real());
    }

    {
        complex<double> t1 = x0_4*x0_4;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(-12.0*radius*t1+t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t32 = (-1.0/2.0*im)*t27*(t18/4.0+4.0*t24);
        complex<double> t33 = -t19+t25-t26+t32;
        complex<double> t35 = t33*t33;
        complex<double> t40 = sqrt(-3.0*t35*t33*(t26-t19+t25+t32));
        complex<double> t42_min = x0_1+t33*x0_4-t40/3.0;
        complex<double> t42_max = x0_1+t33*x0_4+t40/3.0;

        bounds[1].first = min(bounds[1].first, t42_min.real());
        bounds[1].second = max(bounds[1].second, t42_max.real());
    }

        // Calculate x3 bounds
    {
        complex<double> t1 = x0_5*x0_5;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(-12.0*radius*t1+t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/4.0;
        complex<double> t25 = 4.0*(t1/4.0-t4/16.0)/t18;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = t19-t25-t26;
        complex<double> t29 = t27*t27;
        complex<double> t34 = sqrt(-3.0*t29*t27*(t26+t19-t25));
        complex<double> t36_min = x0_2+t27*x0_5-t34/3.0;
        complex<double> t36_max = x0_2+t27*x0_5+t34/3.0;

        bounds[2].first = t36_min.real();
        bounds[2].second = t36_max.real();
    }

    {
        complex<double> t1 = x0_5*x0_5;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(-12.0*radius*t1+t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t32 = (1.0/2.0*im)*t27*(t18/4.0+4.0*t24);
        complex<double> t33 = -t19+t25-t26+t32;
        complex<double> t35 = t33*t33;
        complex<double> t40 = sqrt(-3.0*t35*t33*(t26-t19+t25+t32));
        complex<double> t42_min = x0_2+t33*x0_5-t40/3.0;
        complex<double> t42_max = x0_2+t33*x0_5+t40/3.0;

        bounds[2].first = min(bounds[2].first, t42_min.real());
        bounds[2].second = max(bounds[2].second, t42_max.real());
    }

    {
        complex<double> t1 = x0_5*x0_5;
        complex<double> t4 = radius*radius;
        complex<double> t6 = t1*t1;
        complex<double> t11 = t4*t4;
        complex<double> t15 = sqrt(16.0*t6*t1+24.0*t6*t4-3.0*t1*t11);
        complex<double> t18 = pow(-12.0*radius*t1+t4*radius+2.0*t15,0.3333333333333333);
        complex<double> t19 = t18/8.0;
        complex<double> t24 = (t1/4.0-t4/16.0)/t18;
        complex<double> t25 = 2.0*t24;
        complex<double> t26 = radius/2.0;
        complex<double> t27 = sqrt(3.0);
        complex<double> t32 = (-1.0/2.0*im)*t27*(t18/4.0+4.0*t24);
        complex<double> t33 = -t19+t25-t26+t32;
        complex<double> t35 = t33*t33;
        complex<double> t40 = sqrt(-3.0*t35*t33*(t26-t19+t25+t32));
        complex<double> t42_min = x0_2+t33*x0_5-t40/3.0;
        complex<double> t42_max = x0_2+t33*x0_5+t40/3.0;

        bounds[2].first = min(bounds[2].first, t42_min.real());
        bounds[2].second = max(bounds[2].second, t42_max.real());
    }
    
    // Calculate x4 bounds
    {
        double t1 = sqrt(4.0);
        double t2 = radius*radius;
        double t3 = sqrt(t2);
        double t6_min = x0_3-t1*t3/4.0;
        double t6_max = x0_3+t1*t3/4.0;

        bounds[3].first = t6_min;
        bounds[3].second = t6_max;
    }

    // Calculate x5 bounds
    {
        double t1 = sqrt(4.0);
        double t2 = radius*radius;
        double t3 = sqrt(t2);
        double t6_min = x0_4-t1*t3/4.0;
        double t6_max = x0_4+t1*t3/4.0;

        bounds[4].first = t6_min;
        bounds[4].second = t6_max;
    }
    
    // Calculate x6 bounds
    {
        double t1 = sqrt(4.0);
        double t2 = radius*radius;
        double t3 = sqrt(t2);
        double t6_min = x0_5-t1*t3/4.0;
        double t6_max = x0_5+t1*t3/4.0;

        bounds[5].first = t6_min;
        bounds[5].second = t6_max;
    }
#ifdef DEBUG
    ROS_INFO("init state: %lf, %lf, %lf, %lf, %lf, %lf, radius: %lf", x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, radius);
    for (int i=0; i<X_DIM; ++i) {
        ROS_INFO("backward bounds[%d]: %lf, %lf", i, bounds[i].first, bounds[i].second);
    }
#endif
}

inline double KRRTPlanner::dist(const State& x0, const State& x1)
{
  Vector3d p_diff(x0.head(3) - x1.head(3));
  return p_diff.norm();
}

inline bool KRRTPlanner::isClose(const State& x0, const State& x1)
{
  Vector3d p_diff(x0.head(3) - x1.head(3));
  Vector3d v0(x0.segment(3,3)), v1(x1.segment(3,3));
  double angle = v0.dot(v1)/v0.norm()/v1.norm();
  if (p_diff.norm() <= 2 && 1 >= angle && angle > 0.2)
    return true;
  else
    return false;    
}

inline double KRRTPlanner::applyHeuristics(const State& x_init, 
                                           const State& x_final)
{
  double cost = 0.0;
  double x_time = abs(x_final[0]-x_init[0])/vx_max_;
  double y_time = abs(x_final[1]-x_init[1])/vy_max_;
  double z_time = abs(x_final[2]-x_init[2])/vz_max_;
  cost = max(max(x_time, y_time), z_time);
  return cost;
}

inline bool KRRTPlanner::HtfConnect(const State& x0, const State& x1, 
                                 double radius, double& cost, double& tau, 
                                 double *x_coeff, double *y_coeff, 
                                 double *z_coeff)
{
  if (checkCost(x0, x1, radius, cost, x_coeff, y_coeff, z_coeff, tau)) {
    //ROS_INFO_STREAM("x0: " << x0.transpose() << ", x1: " << x1.transpose() << ", cost: " << cost);
    if (checkPath(x0, x1, x_coeff, y_coeff, z_coeff, tau)) {
      return true;
    } 
    else {
      //ROS_INFO_STREAM("Failed checkPath");
    }
  } 
  else {
    //ROS_INFO_STREAM("Failed checkCost");
  }
  return false;
}

inline bool KRRTPlanner::checkCost(const State& x0, const State& x1, 
                                   double radius, double& cost, double *x_coeff, 
                                 double *y_coeff, double *z_coeff, double& tau)
{
  if (applyHeuristics(x0, x1) > radius) return false;

  //calculate for tau*
  //DYNAMICS == DOUBLE_INTEGRATOR_3D when c=[0;0;0;0;0;0]
  double p[POLY_DEGREE + 1];
  p[0] = 1;
  p[1] = 0;
  p[2] =(- 4.0*x0[3]*x0[3] - 4.0*x0[3]*x1[3] - 4.0*x1[3]*x1[3] 
         - 4.0*x0[4]*x0[4] - 4.0*x0[4]*x1[4] - 4.0*x1[4]*x1[4] 
         - 4.0*x0[5]*x0[5] - 4.0*x0[5]*x1[5] - 4.0*x1[5]*x1[5]) * rou_;
  p[3] =(- 24.0*x0[0]*x0[3] - 24.0*x0[0]*x1[3] + 24.0*x1[0]*x0[3] 
         + 24.0*x1[0]*x1[3] - 24.0*x0[1]*x0[4] - 24.0*x0[1]*x1[4] 
         + 24.0*x1[1]*x0[4] + 24.0*x1[1]*x1[4] - 24.0*x0[2]*x0[5] 
         - 24.0*x0[2]*x1[5] + 24.0*x1[2]*x0[5] + 24.0*x1[2]*x1[5]) * rou_;
  p[4] =(- 36.0*x0[0]*x0[0] + 72.0*x0[0]*x1[0] - 36.0*x1[0]*x1[0] 
         - 36.0*x0[1]*x0[1] + 72.0*x0[1]*x1[1] - 36.0*x1[1]*x1[1] 
         - 36.0*x0[2]*x0[2] + 72.0*x0[2]*x1[2] - 36.0*x1[2]*x1[2]) * rou_;
          
  std::vector<double> roots = quartic(p[0], p[1], p[2], p[3], p[4]);
  
  //calculate for cost[tau*] when c=[0;0;0;0;0;0]
  bool result = false;
  tau = radius;
  cost = radius;

  for (size_t i = 0; i < roots.size();++i) {
    if (roots[i] <= 0.0 || roots[i] >= cost) {
      //tau* which is real number larger than 0, less than radius
      continue;
    }
    double t1 = x0[0] - x1[0];
    double t2 = x0[1] - x1[1];
    double t3 = x0[2] - x1[2];
    double t4 = x0[3] + x1[3];
    double t5 = x0[4] + x1[4];
    double t6 = x0[5] + x1[5];
    double t7 = t1*t1 + t2*t2 + t3*t3;
    double t8 = t4*t4 + t5*t5 + t6*t6 - x0[3]*x1[3] - x0[4]*x1[4] - x0[5]*x1[5];
    double t9 = t1*t4 + t2*t5 + t3*t6;
    
    double current = roots[i] + rou_*(t7*12/roots[i]/roots[i]/roots[i] 
                                    + t8*4/roots[i] + t9*12/roots[i]/roots[i]);
    if (current < cost) {
      tau = roots[i];
      cost = current;
      result = true;
    }
  }
  
  if (result == true) {
    double t2 = tau*tau;
    double t3 = t2*tau;
    x_coeff[0] = (2.0*(x0[0]-x1[0])+tau*(x0[3]+x1[3]))/t3;
    x_coeff[1] = -(3.0*(x0[0]-x1[0])+tau*(2*x0[3]+x1[3]))/t2;
    x_coeff[2] = x0[3];
    x_coeff[3] = x0[0];
    y_coeff[0] = (2.0*(x0[1]-x1[1])+tau*(x0[4]+x1[4]))/t3;
    y_coeff[1] = -(3.0*(x0[1]-x1[1])+tau*(2*x0[4]+x1[4]))/t2;
    y_coeff[2] = x0[4];
    y_coeff[3] = x0[1];
    z_coeff[0] = (2.0*(x0[2]-x1[2])+tau*(x0[5]+x1[5]))/t3;
    z_coeff[1] = -(3.0*(x0[2]-x1[2])+tau*(2*x0[5]+x1[5]))/t2;
    z_coeff[2] = x0[5];
    z_coeff[3] = x0[2];
  }

  return result;
}

inline bool KRRTPlanner::checkPath(const State& x0, const State& x1, 
                                   double *x_coeff, double *y_coeff, 
                                   double *z_coeff, double tau)
{
  double vx_min, vx_max, vy_min, vy_max, vz_min, vz_max;
  double ax_min, ax_max, ay_min, ay_max, az_min, az_max;
  
  double x_root_a = -x_coeff[1]/3.0/x_coeff[0];
  double y_root_a = -y_coeff[1]/3.0/y_coeff[0];
  double z_root_a = -z_coeff[1]/3.0/z_coeff[0];
  
  if (x_root_a > 0 && x_root_a < tau) {
    if (x_coeff[0] >= 0) {
      ax_min = 2.0*x_coeff[1];
      ax_max = 2.0*x_coeff[1] + 6.0*x_coeff[0]*tau;
      vx_min = 3*x_coeff[0]*x_root_a*x_root_a + 2*x_coeff[1]*x_root_a + x_coeff[2];
      vx_max = max(x_coeff[2], 3*x_coeff[0]*tau*tau + 2*x_coeff[1]*tau + x_coeff[2]);
    }
    else {
      ax_max = 2.0*x_coeff[1];
      ax_min = 2.0*x_coeff[1] + 6.0*x_coeff[0]*tau;
      vx_max = 3*x_coeff[0]*x_root_a*x_root_a + 2*x_coeff[1]*x_root_a + x_coeff[2];
      vx_min = min(x_coeff[2], 3*x_coeff[0]*tau*tau + 2*x_coeff[1]*tau + x_coeff[2]);
    }
  }
  else if (x_root_a <= 0) {
    if (x_coeff[0] >= 0) {
      ax_min = 2.0*x_coeff[1];
      ax_max = 2.0*x_coeff[1] + 6.0*x_coeff[0]*tau;
      vx_min = x_coeff[2];
      vx_max = 3*x_coeff[0]*tau*tau + 2*x_coeff[1]*tau + x_coeff[2];
    } 
    else {
      ax_max = 2.0*x_coeff[1];
      ax_min = 2.0*x_coeff[1] + 6.0*x_coeff[0]*tau;
      vx_max = x_coeff[2];
      vx_min = 3*x_coeff[0]*tau*tau + 2*x_coeff[1]*tau + x_coeff[2];
    }
  }
  else {
    if (x_coeff[0] >= 0) {
      ax_min = 2.0*x_coeff[1];
      ax_max = 2.0*x_coeff[1] + 6.0*x_coeff[0]*tau;
      vx_max = x_coeff[2];
      vx_min = 3*x_coeff[0]*tau*tau + 2*x_coeff[1]*tau + x_coeff[2];
    } 
    else {
      ax_max = 2.0*x_coeff[1];
      ax_min = 2.0*x_coeff[1] + 6.0*x_coeff[0]*tau;
      vx_min = x_coeff[2];
      vx_max = 3*x_coeff[0]*tau*tau + 2*x_coeff[1]*tau + x_coeff[2];
    }
  }
  
  if (y_root_a > 0 && y_root_a < tau) {
    if (y_coeff[0] >= 0) {
      ay_min = 2.0*y_coeff[1];
      ay_max = 2.0*y_coeff[1] + 6.0*y_coeff[0]*tau;
      vy_min = 3*y_coeff[0]*y_root_a*y_root_a + 2*y_coeff[1]*y_root_a + y_coeff[2];
      vy_max = max(y_coeff[2], 3*y_coeff[0]*tau*tau + 2*y_coeff[1]*tau + y_coeff[2]);
    }
    else {
      ay_max = 2.0*y_coeff[1];
      ay_min = 2.0*y_coeff[1] + 6.0*y_coeff[0]*tau;
      vy_max = 3*y_coeff[0]*y_root_a*y_root_a + 2*y_coeff[1]*y_root_a + y_coeff[2];
      vy_min = min(y_coeff[2], 3*y_coeff[0]*tau*tau + 2*y_coeff[1]*tau + y_coeff[2]);
    }
  }
  else if (y_root_a <= 0) {
    if (y_coeff[0] >= 0) {
      ay_min = 2.0*y_coeff[1];
      ay_max = 2.0*y_coeff[1] + 6.0*y_coeff[0]*tau;
      vy_min = y_coeff[2];
      vy_max = 3*y_coeff[0]*tau*tau + 2*y_coeff[1]*tau + y_coeff[2];
    } 
    else {
      ay_max = 2.0*y_coeff[1];
      ay_min = 2.0*y_coeff[1] + 6.0*y_coeff[0]*tau;
      vy_max = y_coeff[2];
      vy_min = 3*y_coeff[0]*tau*tau + 2*y_coeff[1]*tau + y_coeff[2];
    }
  }
  else {
    if (y_coeff[0] >= 0) {
      ay_min = 2.0*y_coeff[1];
      ay_max = 2.0*y_coeff[1] + 6.0*y_coeff[0]*tau;
      vy_max = y_coeff[2];
      vy_min = 3*y_coeff[0]*tau*tau + 2*y_coeff[1]*tau + y_coeff[2];
    } 
    else {
      ay_max = 2.0*y_coeff[1];
      ay_min = 2.0*y_coeff[1] + 6.0*y_coeff[0]*tau;
      vy_min = y_coeff[2];
      vy_max = 3*y_coeff[0]*tau*tau + 2*y_coeff[1]*tau + y_coeff[2];
    }
  }
  
  if (z_root_a > 0 && z_root_a < tau) {
    if (z_coeff[0] >= 0) {
      az_min = 2.0*z_coeff[1];
      az_max = 2.0*z_coeff[1] + 6.0*z_coeff[0]*tau;
      vz_min = 3*z_coeff[0]*z_root_a*z_root_a + 2*z_coeff[1]*z_root_a + z_coeff[2];
      vz_max = max(z_coeff[2], 3*z_coeff[0]*tau*tau + 2*z_coeff[1]*tau + z_coeff[2]);
    }
    else {
      az_max = 2.0*z_coeff[1];
      az_min = 2.0*z_coeff[1] + 6.0*z_coeff[0]*tau;
      vz_max = 3*z_coeff[0]*z_root_a*z_root_a + 2*z_coeff[1]*z_root_a + z_coeff[2];
      vz_min = min(z_coeff[2], 3*z_coeff[0]*tau*tau + 2*z_coeff[1]*tau + z_coeff[2]);
    }
  }
  else if (z_root_a <= 0) {
    if (z_coeff[0] >= 0) {
      az_min = 2.0*z_coeff[1];
      az_max = 2.0*z_coeff[1] + 6.0*z_coeff[0]*tau;
      vz_min = z_coeff[2];
      vz_max = 3*z_coeff[0]*tau*tau + 2*z_coeff[1]*tau + z_coeff[2];
    } 
    else {
      az_max = 2.0*z_coeff[1];
      az_min = 2.0*z_coeff[1] + 6.0*z_coeff[0]*tau;
      vz_max = z_coeff[2];
      vz_min = 3*z_coeff[0]*tau*tau + 2*z_coeff[1]*tau + z_coeff[2];
    }
  }
  else {
    if (z_coeff[0] >= 0) {
      az_min = 2.0*z_coeff[1];
      az_max = 2.0*z_coeff[1] + 6.0*z_coeff[0]*tau;
      vz_max = z_coeff[2];
      vz_min = 3*z_coeff[0]*tau*tau + 2*z_coeff[1]*tau + z_coeff[2];
    } 
    else {
      az_max = 2.0*z_coeff[1];
      az_min = 2.0*z_coeff[1] + 6.0*z_coeff[0]*tau;
      vz_min = z_coeff[2];
      vz_max = 3*z_coeff[0]*tau*tau + 2*z_coeff[1]*tau + z_coeff[2];
    }
  }
  
  if (vx_min < vx_min_) {
    //ROS_INFO("vx(%lf) < vx_min_(%lf)", vx_min, vx_min_);
    return false;
  }
  if (vx_max > vx_max_) {
    //ROS_INFO("vx(%lf) > vx_max_(%lf)", vx_max, vx_max_); 
    return false;
  }
  if (vy_min < vy_min_) {
    //ROS_INFO("vy(%lf) < vy_min_(%lf)", vy_min, vy_min_);
    return false;
  }
  if (vy_max > vy_max_) {
    //ROS_INFO("vy(%lf) > vx_max_(%lf)", vy_max, vy_max_);
    return false;
  }
  if (vz_min < vz_min_) {
    //ROS_INFO("vz(%lf) < vz_min_(%lf)", vz_min, vz_min_);
    return false;
  }
  if (vz_max > vz_max_) {
    //ROS_INFO("vz(%lf) > vz_max_(%lf)", vz_max, vz_max_);
    return false;
  }
  
  if (ax_min < ax_min_) {
    //ROS_INFO("ax(%lf) < ax_min_(%lf)", ax_min, ax_min_);
    return false;
  }
  if (ax_max > ax_max_) {
    //ROS_INFO("ax(%lf) > ax_max_(%lf)", ax_max, ax_max_);
    return false;
  }
  if (ay_min < ay_min_) {
    //ROS_INFO("ay(%lf) < ay_min_(%lf)", ay_min, ay_min_);
    return false;
  }
  if (ay_max > ay_max_) {
    //ROS_INFO("ay(%lf) > ay_max_(%lf)", ay_max, ay_max_);
    return false;
  }
  if (az_min < az_min_) {
    //ROS_INFO("az(%lf) < az_min_(%lf)", az_min, az_min_);
    return false;
  }
  if (az_max > az_max_) {
     //ROS_INFO("az(%lf) > az_max_(%lf)", az_max, az_max_);
    return false;
  }

  // double t;
  // size_t numPoints = ceil(tau / deltaT_);
  // size_t step = 1;
  // while (step < numPoints) step *= 2;
  // double actual_deltaT = tau/numPoints;
  // for ( ; step > 1; step /= 2) {
  //   for (size_t i = step / 2; i < numPoints; i += step) {
  //     t = actual_deltaT*i;
  //     Eigen::Vector3d pos, vel, acc;
  //     calPVAFromCoeff(pos, vel, acc, x_coeff, y_coeff, z_coeff, t, 3);
  //     vector<Vector3d> line_grids;
  //     getCheckPos(pos, vel, acc, line_grids, hor_safe_radius_, ver_safe_radius_);
  //     for (const auto& grid : line_grids)
  //     {
  //       if (occ_map_->getVoxelState(grid) != 0) {
  //         //cout <<"collision: "<<grid.transpose()<<endl;
  //         return false;
  //       }
  //     }
  //   }
  // }

  int numPoints = floor(tau / deltaT_);
  double actual_deltaT = tau/numPoints;
  int order = 3;
  for (int j = 0; j < numPoints; ++j) 
  {
    double t = actual_deltaT * j;
    Eigen::Vector3d pos, vel, acc;
    calPVAFromCoeff(pos, vel, acc, x_coeff, y_coeff, z_coeff, t, order);
    vector<Vector3d> line_grids;
    getCheckPos(pos, vel, acc, line_grids, hor_safe_radius_, ver_safe_radius_);
    for (const auto& grid : line_grids)
    {
      if (occ_map_->getVoxelState(grid) != 0) 
      {
        return false;
      }
    }
  }
  
  return true;
}

inline bool KRRTPlanner::checkVelAcc(const State& x0, const State& x1, 
                                   double *x_coeff, double *y_coeff, 
                                   double *z_coeff, double tau)
{
  double vx_min, vx_max, vy_min, vy_max, vz_min, vz_max;
  double ax_min, ax_max, ay_min, ay_max, az_min, az_max;
  
  double x_root_a = -x_coeff[1]/3.0/x_coeff[0];
  double y_root_a = -y_coeff[1]/3.0/y_coeff[0];
  double z_root_a = -z_coeff[1]/3.0/z_coeff[0];
  
  if (x_root_a > 0 && x_root_a < tau) {
    if (x_coeff[0] >= 0) {
      ax_min = 2.0*x_coeff[1];
      ax_max = 2.0*x_coeff[1] + 6.0*x_coeff[0]*tau;
      vx_min = 3*x_coeff[0]*x_root_a*x_root_a + 2*x_coeff[1]*x_root_a + x_coeff[2];
      vx_max = max(x_coeff[2], 3*x_coeff[0]*tau*tau + 2*x_coeff[1]*tau + x_coeff[2]);
    }
    else {
      ax_max = 2.0*x_coeff[1];
      ax_min = 2.0*x_coeff[1] + 6.0*x_coeff[0]*tau;
      vx_max = 3*x_coeff[0]*x_root_a*x_root_a + 2*x_coeff[1]*x_root_a + x_coeff[2];
      vx_min = min(x_coeff[2], 3*x_coeff[0]*tau*tau + 2*x_coeff[1]*tau + x_coeff[2]);
    }
  }
  else if (x_root_a <= 0) {
    if (x_coeff[0] >= 0) {
      ax_min = 2.0*x_coeff[1];
      ax_max = 2.0*x_coeff[1] + 6.0*x_coeff[0]*tau;
      vx_min = x_coeff[2];
      vx_max = 3*x_coeff[0]*tau*tau + 2*x_coeff[1]*tau + x_coeff[2];
    } 
    else {
      ax_max = 2.0*x_coeff[1];
      ax_min = 2.0*x_coeff[1] + 6.0*x_coeff[0]*tau;
      vx_max = x_coeff[2];
      vx_min = 3*x_coeff[0]*tau*tau + 2*x_coeff[1]*tau + x_coeff[2];
    }
  }
  else {
    if (x_coeff[0] >= 0) {
      ax_min = 2.0*x_coeff[1];
      ax_max = 2.0*x_coeff[1] + 6.0*x_coeff[0]*tau;
      vx_max = x_coeff[2];
      vx_min = 3*x_coeff[0]*tau*tau + 2*x_coeff[1]*tau + x_coeff[2];
    } 
    else {
      ax_max = 2.0*x_coeff[1];
      ax_min = 2.0*x_coeff[1] + 6.0*x_coeff[0]*tau;
      vx_min = x_coeff[2];
      vx_max = 3*x_coeff[0]*tau*tau + 2*x_coeff[1]*tau + x_coeff[2];
    }
  }
  
  if (y_root_a > 0 && y_root_a < tau) {
    if (y_coeff[0] >= 0) {
      ay_min = 2.0*y_coeff[1];
      ay_max = 2.0*y_coeff[1] + 6.0*y_coeff[0]*tau;
      vy_min = 3*y_coeff[0]*y_root_a*y_root_a + 2*y_coeff[1]*y_root_a + y_coeff[2];
      vy_max = max(y_coeff[2], 3*y_coeff[0]*tau*tau + 2*y_coeff[1]*tau + y_coeff[2]);
    }
    else {
      ay_max = 2.0*y_coeff[1];
      ay_min = 2.0*y_coeff[1] + 6.0*y_coeff[0]*tau;
      vy_max = 3*y_coeff[0]*y_root_a*y_root_a + 2*y_coeff[1]*y_root_a + y_coeff[2];
      vy_min = min(y_coeff[2], 3*y_coeff[0]*tau*tau + 2*y_coeff[1]*tau + y_coeff[2]);
    }
  }
  else if (y_root_a <= 0) {
    if (y_coeff[0] >= 0) {
      ay_min = 2.0*y_coeff[1];
      ay_max = 2.0*y_coeff[1] + 6.0*y_coeff[0]*tau;
      vy_min = y_coeff[2];
      vy_max = 3*y_coeff[0]*tau*tau + 2*y_coeff[1]*tau + y_coeff[2];
    } 
    else {
      ay_max = 2.0*y_coeff[1];
      ay_min = 2.0*y_coeff[1] + 6.0*y_coeff[0]*tau;
      vy_max = y_coeff[2];
      vy_min = 3*y_coeff[0]*tau*tau + 2*y_coeff[1]*tau + y_coeff[2];
    }
  }
  else {
    if (y_coeff[0] >= 0) {
      ay_min = 2.0*y_coeff[1];
      ay_max = 2.0*y_coeff[1] + 6.0*y_coeff[0]*tau;
      vy_max = y_coeff[2];
      vy_min = 3*y_coeff[0]*tau*tau + 2*y_coeff[1]*tau + y_coeff[2];
    } 
    else {
      ay_max = 2.0*y_coeff[1];
      ay_min = 2.0*y_coeff[1] + 6.0*y_coeff[0]*tau;
      vy_min = y_coeff[2];
      vy_max = 3*y_coeff[0]*tau*tau + 2*y_coeff[1]*tau + y_coeff[2];
    }
  }
  
  if (z_root_a > 0 && z_root_a < tau) {
    if (z_coeff[0] >= 0) {
      az_min = 2.0*z_coeff[1];
      az_max = 2.0*z_coeff[1] + 6.0*z_coeff[0]*tau;
      vz_min = 3*z_coeff[0]*z_root_a*z_root_a + 2*z_coeff[1]*z_root_a + z_coeff[2];
      vz_max = max(z_coeff[2], 3*z_coeff[0]*tau*tau + 2*z_coeff[1]*tau + z_coeff[2]);
    }
    else {
      az_max = 2.0*z_coeff[1];
      az_min = 2.0*z_coeff[1] + 6.0*z_coeff[0]*tau;
      vz_max = 3*z_coeff[0]*z_root_a*z_root_a + 2*z_coeff[1]*z_root_a + z_coeff[2];
      vz_min = min(z_coeff[2], 3*z_coeff[0]*tau*tau + 2*z_coeff[1]*tau + z_coeff[2]);
    }
  }
  else if (z_root_a <= 0) {
    if (z_coeff[0] >= 0) {
      az_min = 2.0*z_coeff[1];
      az_max = 2.0*z_coeff[1] + 6.0*z_coeff[0]*tau;
      vz_min = z_coeff[2];
      vz_max = 3*z_coeff[0]*tau*tau + 2*z_coeff[1]*tau + z_coeff[2];
    } 
    else {
      az_max = 2.0*z_coeff[1];
      az_min = 2.0*z_coeff[1] + 6.0*z_coeff[0]*tau;
      vz_max = z_coeff[2];
      vz_min = 3*z_coeff[0]*tau*tau + 2*z_coeff[1]*tau + z_coeff[2];
    }
  }
  else {
    if (z_coeff[0] >= 0) {
      az_min = 2.0*z_coeff[1];
      az_max = 2.0*z_coeff[1] + 6.0*z_coeff[0]*tau;
      vz_max = z_coeff[2];
      vz_min = 3*z_coeff[0]*tau*tau + 2*z_coeff[1]*tau + z_coeff[2];
    } 
    else {
      az_max = 2.0*z_coeff[1];
      az_min = 2.0*z_coeff[1] + 6.0*z_coeff[0]*tau;
      vz_min = z_coeff[2];
      vz_max = 3*z_coeff[0]*tau*tau + 2*z_coeff[1]*tau + z_coeff[2];
    }
  }
  
  if (vx_min < vx_min_) {
    return false;
  }
  if (vx_max > vx_max_) {
    return false;
  }
  if (vy_min < vy_min_) {
    return false;
  }
  if (vy_max > vy_max_) {
    return false;
  }
  if (vz_min < vz_min_) {
    return false;
  }
  if (vz_max > vz_max_) {
    return false;
  }
  
  if (ax_min < ax_min_) {
    return false;
  }
  if (ax_max > ax_max_) {
    return false;
  }
  if (ay_min < ay_min_) {
    return false;
  }
  if (ay_max > ay_max_) {
    return false;
  }
  if (az_min < az_min_) {
    return false;
  }
  if (az_max > az_max_) {
    return false;
  }
  return true;
}

inline void KRRTPlanner::computeCostAndTime(const State& x0, const State& x1, double& cost, double& tau)
{
    Eigen::Matrix<double,X_DIM,1> x1diffbarx0, d;

    //calculate for tau*
    //DYNAMICS == DOUBLE_INTEGRATOR_3D when c=[0;0;0;0;0;0]
    double p[POLY_DEGREE + 1];
    p[0] = 1;
    p[1] = 0;
    p[2] =(- 4.0*x0[3]*x0[3] - 4.0*x0[3]*x1[3] - 4.0*x1[3]*x1[3] 
           - 4.0*x0[4]*x0[4] - 4.0*x0[4]*x1[4] - 4.0*x1[4]*x1[4] 
           - 4.0*x0[5]*x0[5] - 4.0*x0[5]*x1[5] - 4.0*x1[5]*x1[5]) * rou_;
    p[3] =(- 24.0*x0[0]*x0[3] - 24.0*x0[0]*x1[3] + 24.0*x1[0]*x0[3] 
           + 24.0*x1[0]*x1[3] - 24.0*x0[1]*x0[4] - 24.0*x0[1]*x1[4] 
           + 24.0*x1[1]*x0[4] + 24.0*x1[1]*x1[4] - 24.0*x0[2]*x0[5] 
           - 24.0*x0[2]*x1[5] + 24.0*x1[2]*x0[5] + 24.0*x1[2]*x1[5]) * rou_;
    p[4] =(- 36.0*x0[0]*x0[0] + 72.0*x0[0]*x1[0] - 36.0*x1[0]*x1[0] 
           - 36.0*x0[1]*x0[1] + 72.0*x0[1]*x1[1] - 36.0*x1[1]*x1[1] 
           - 36.0*x0[2]*x0[2] + 72.0*x0[2]*x1[2] - 36.0*x1[2]*x1[2]) * rou_;
    
    std::vector<double> roots = quartic(p[0], p[1], p[2], p[3], p[4]);

    //calculate for cost[tau*] when c=[0;0;0;0;0;0]
    cost = DBL_MAX;
    tau = DBL_MAX;
    for (size_t i = 0; i < roots.size();++i) {
        if (!((roots[i] > 0.0) && (roots[i] < cost))) {
            //tau* which is real number larger than 0, less than radius
            continue;
        }
        //DYNAMICS == DOUBLE_INTEGRATOR_2D
        double t7 = roots[i]*roots[i];
        double t10 = rou_/t7/roots[i];
        double t14 = 1/t7*rou_;
        double t26 = rou_/roots[i];
        x1diffbarx0[0] = x1[0]-x0[0]-roots[i]*x0[3];
        x1diffbarx0[1] = x1[1]-x0[1]-roots[i]*x0[4];
        x1diffbarx0[2] = x1[2]-x0[2]-roots[i]*x0[5];
        x1diffbarx0[3] = x1[3]-x0[3];
        x1diffbarx0[4] = x1[4]-x0[4];
        x1diffbarx0[5] = x1[5]-x0[5];
        d[0] = 12.0*t10*x1diffbarx0[0]-6.0*t14*x1diffbarx0[3];
        d[1] = 12.0*t10*x1diffbarx0[1]-6.0*t14*x1diffbarx0[4];
        d[2] = 12.0*t10*x1diffbarx0[2]-6.0*t14*x1diffbarx0[5];
        d[3] = -6.0*t14*x1diffbarx0[0]+4.0*t26*x1diffbarx0[3];
        d[4] = -6.0*t14*x1diffbarx0[1]+4.0*t26*x1diffbarx0[4];
        d[5] = -6.0*t14*x1diffbarx0[2]+4.0*t26*x1diffbarx0[5];
        
        double current = roots[i] + x1diffbarx0[0]*d[0] + x1diffbarx0[1]*d[1]
                                  + x1diffbarx0[2]*d[2] + x1diffbarx0[3]*d[3]
                                  + x1diffbarx0[4]*d[4] + x1diffbarx0[5]*d[5];
                                      
        if ((roots[i] > 0.0) && (current < cost)) {
            tau = roots[i];
            cost = current;
        }
    }
}

inline void KRRTPlanner::calculateStateAndControl(const State& x0, const State& x1, 
                                           const double tau, const State& d_tau, 
                                           double t, State& x, Control& u)
{
    ros::Time s = ros::Time::now();
    Eigen::Matrix<double,2*X_DIM,1> chi;
//     Eigen::Matrix<double,U_DIM,U_DIM> R;
//     R.setZero();
//     for (int i=0; i<U_DIM; ++i) {
//         R(i,i) = rou_;
//     }
//     Eigen::Matrix<double,X_DIM,U_DIM> B;
//     B.setZero();
//     B(3,0)=1;
//     B(4,1)=1;
//     B(5,2)=1;
    //DYNAMICS == DOUBLE_INTEGRATOR_2D
    double t1 = t-tau;
    double t3 = tau*tau;
    double t7 = t*t;
    double t12 = 1/rou_;
    double t13 = (t3*tau-3.0*t3*t+3.0*tau*t7-t7*t)*t12;
    double t19 = (t3-2.0*tau*t+t7)*t12;
    double t31 = -t1*t12;
    chi[0] = x1[0]+t1*x1[3]+t13*d_tau[0]/6.0+t19*d_tau[3]/2.0;
    chi[1] = x1[1]+t1*x1[4]+t13*d_tau[1]/6.0+t19*d_tau[4]/2.0;
    chi[2] = x1[2]+t1*x1[5]+t13*d_tau[2]/6.0+t19*d_tau[5]/2.0;
    chi[3] = x1[3]-t19*d_tau[0]/2.0-t31*d_tau[3];
    chi[4] = x1[4]-t19*d_tau[1]/2.0-t31*d_tau[4];
    chi[5] = x1[5]-t19*d_tau[2]/2.0-t31*d_tau[5];
    chi[6] = d_tau[0];
    chi[7] = d_tau[1];
    chi[8] = d_tau[2];
    chi[9] = -t1*d_tau[0]+d_tau[3];
    chi[10] = -t1*d_tau[1]+d_tau[4];
    chi[11] = -t1*d_tau[2]+d_tau[5];

//     x = chi.block<X_DIM,1>(0,0);
//     u = R.inverse()*B.transpose()*chi.block<X_DIM,1>(X_DIM,0);
    
    for (int i=0; i<X_DIM; ++i) {
        x(i) = chi(i);
    }
    u[0] = chi[9]/rou_;
    u[1] = chi[10]/rou_;
    u[2] = chi[11]/rou_;
    
    ros::Time e = ros::Time::now();
    calculate_state_control_time += e.toSec()-s.toSec();
    calculate_state_control_nums++;
}

inline bool KRRTPlanner::validateStateAndControl(const State& x, const Control& u)
{
    Vector3d pos, vel, acc;
    pos = x.head(3);
    vel = x.tail(3);
    acc = u.head(3);
    
    if (!validatePosSurround(pos, vel, acc)) {
      return false;
    }
    if (!validateVel(vel)) {
      return false;
    }
    if (!validateAcc(acc)) {
      return false;
    }
    return true;
}

inline bool KRRTPlanner::validatePosSurround(const Vector3d& pos, 
                                             const Vector3d& vel, 
                                             const Vector3d& acc)
{
  vector<Vector3d> surrounding_grids;
  getCheckPos(pos, vel, acc, surrounding_grids, hor_safe_radius_, ver_safe_radius_);
  for (const auto& grid : surrounding_grids)
  {
    if (occ_map_->getVoxelState(grid) != 0) {
      //cout <<"collision: "<<grid.transpose()<<endl;
      return false;
    }
  }
  return true;
}

inline bool KRRTPlanner::validatePosSurround(const Vector3d& pos)
{
  int x_size = ceil(copter_diag_len_/2.0/resolution_);
  int y_size = ceil(copter_diag_len_/2.0/resolution_);
  int z_size = ceil(replan_ver_safe_radius_/resolution_);
  Vector3d grid(pos);
  for (int i = -x_size; i <= x_size; ++i)
    for (int j = -y_size; j <= y_size; ++j)
      for (int k = -z_size; k <= z_size; ++k)
      {
        grid = pos + Vector3d(i,j,k) * resolution_;
        if (occ_map_->getVoxelState(grid) != 0) {
          //cout <<"collision: "<<grid.transpose()<<endl;
          return false;
        }
      }
  return true;
}

inline bool KRRTPlanner::validateVel(const Vector3d& vel)
{
    if (vel(0) < vx_min_ || vel(0) > vx_max_) {
//         ROS_ERROR("vel x");
        return false;
    }
    if (vel(1) < vy_min_ || vel(1) > vy_max_) {
//         ROS_ERROR("vel y");
        return false;
    }
    if (vel(2) < vz_min_ || vel(2) > vz_max_) {
//         ROS_ERROR("vel z");
        return false;
    }
    return true;
}

inline bool KRRTPlanner::validateAcc(const Vector3d& acc)
{
    if (acc(0) < ax_min_ || acc(0) > ax_max_) {
//         ROS_ERROR("acc x");
        return false;
    }
    if (acc(1) < ay_min_ || acc(1) > ay_max_) {
//         ROS_ERROR("acc y");
        return false;
    }
    if (acc(2) < az_min_ || acc(2) > az_max_) {
//         ROS_ERROR("acc z");
        return false;
    }
    return true;
}

inline void KRRTPlanner::findAllSaC(vector< State >* vis_x, 
                                    vector< Control >* vis_u)
{
  for (int i=0; i<valid_start_tree_node_nums_; ++i) {
    for (const auto leafptr : start_tree_[i]->children) {
      getVisPoints(leafptr, vis_x, vis_u);
    }
  }
}

inline void KRRTPlanner::findAllStatesAndControlInAllTrunks(RRTNodePtr root, 
                                                       vector< State >* vis_x, 
                                                       vector< Control >* vis_u)
{
  if (root == nullptr) 
    return;
  //whatever dfs or bfs
  RRTNode* node = root;
  std::queue<RRTNode*> Q;
  Q.push(node);
  while (!Q.empty()) 
  {
    node = Q.front();
    Q.pop();
    for (const auto& leafptr : node->children) 
    {
      double cost, tau, a_dT;
//       computeBestCost(node->x, leafptr->x, cost, tau, vis_x, vis_u);
      getVisPoints(leafptr, vis_x, vis_u);
      Q.push(leafptr);
    }
  }
}

inline void KRRTPlanner::getVisPoints(RRTNodePtr x1, vector< State >* vis_x, vector< Control >* vis_u)
{
  int cur_order = x1->n_order;
  double d_t = 0.015;
  int n = floor(x1->tau_from_parent / d_t);
  for (int i=0; i<=n; ++i) {
    double t1 = d_t*i;
    Eigen::Vector3d pos, vel, acc;
    calPVAFromCoeff(pos, vel, acc, x1->x_coeff, x1->y_coeff, x1->z_coeff, t1, cur_order);
    State x;
    Control u;
    x << pos(0), pos(1), pos(2), vel(0), vel(1), vel(2);
    u << acc(0), acc(1), acc(2);
    vis_x->push_back(x);
    vis_u->push_back(u);
  }
}

bool KRRTPlanner::computeBestCost(const Eigen::VectorXd& x0, const Eigen::VectorXd& x1, 
                                  vector<pair<State, State>>& segs, 
                                  double& cost, double& tau, 
                         vector<State>* vis_x, vector<Control>* vis_u, 
                         double *x_coeff, double *y_coeff, double *z_coeff)
{
  double p[5];
  p[0] = 35 + rou_*(3*x0[6]*x0[6] + 3*x0[7]*x0[7] + 3*x0[8]*x0[8] 
                    + x0[6]*x1[6] + 3*x1[6]*x1[6] + x0[7]*x1[7]
                    + 3*x1[7]*x1[7] + x0[8]*x1[8] + 3*x1[8]*x1[8]);
  p[1] = 0;
  p[2] = -6*rou_*(32*x0[3]*x0[3] + 32*x0[4]*x0[4] + 32*x0[5]*x0[5]
                + 5*x0[0]*x0[6] + 5*x0[1]*x0[7] + 5*x0[2]*x0[8] 
                - 5*x0[6]*x1[0] - 5*x0[7]*x1[1] - 5*x0[8]*x1[2] 
                + 36*x0[3]*x1[3] + 32*x1[3]*x1[3] + 36*x0[4]*x1[4] 
                + 32*x1[4]*x1[4] + 36*x0[5]*x1[5] + 32*x1[5]*x1[5] 
                - 5*x0[0]*x1[6] + 5*x1[0]*x1[6] - 5*x0[1]*x1[7] 
                + 5*x1[1]*x1[7] - 5*x0[2]*x1[8] + 5*x1[2]*x1[8]);
  p[3] = -1200*rou_*(x0[2]*x0[5] - x0[3]*x1[0] - x0[4]*x1[1] - x0[5]*x1[2] 
                   - x1[0]*x1[3] + x0[0]*(x0[3] + x1[3]) - x1[1]*x1[4] 
                   + x0[1]*(x0[4] + x1[4]) + x0[2]*x1[5] - x1[2]*x1[5]);
  p[4] = -1800*rou_*(x0[0]*x0[0] + x0[1]*x0[1] + x0[2]*x0[2] 
                 - 2*x0[0]*x1[0] + x1[0]*x1[0] - 2*x0[1]*x1[1] 
                   + x1[1]*x1[1] - 2*x0[2]*x1[2] + x1[2]*x1[2]);
          

  std::vector<double> roots = quartic(p[0], p[1], p[2], p[3], p[4]);
  //calculate for cost[tau*] when c=[0;0;0;0;0;0]
  bool result = false;
  tau = DBL_MAX;
  cost = DBL_MAX;
  
  for (size_t i = 0; i < roots.size();++i) {
    if (roots[i] <= 1e-3 || roots[i] > cost) {
      continue;
    }
    
    double current = (1/(35*roots[i]*roots[i]*roots[i]))
    *(600*rou_*((x0[0] - x1[0])*(x0[0] - x1[0]) 
              + (x0[1] - x1[1])*(x0[1] - x1[1]) 
              + (x0[2] - x1[2])*(x0[2] - x1[2]))
    + 600*rou_*roots[i]*((x0[0] - x1[0])*(x0[3] + x1[3]) 
                       + (x0[1] - x1[1])*(x0[4] + x1[4]) 
                       + (x0[2] - x1[2])*(x0[5] + x1[5]))
    + 6*rou_*roots[i]*roots[i]*(32*x0[3]*x0[3] + 32*x0[4]*x0[4] + 36*x0[3]*x1[3] 
                              + 36*x0[4]*x1[4] + 32*(x1[3]*x1[3] + x1[4]*x1[4]) 
                              + 4*(8*x0[5]*x0[5] + 9*x0[5]*x1[5] + 8*x1[5]*x1[5])
                              + 5*(x0[0] - x1[0])*(x0[6] - x1[6]) 
                              + 5*(x0[1] - x1[1])*(x0[7] - x1[7]) 
                              + 5*(x0[2] - x1[2])*(x0[8] - x1[8]))
    + 2*rou_*roots[i]*roots[i]*roots[i]*(11*x0[3]*x0[6] + 11*x0[4]*x0[7] 
                                       + 11*x0[5]*x0[8] + 4*x0[6]*x1[3] 
                                       + 4*x0[7]*x1[4] + 4*x0[8]*x1[5] 
                                       - 4*x0[3]*x1[6] - 11*x1[3]*x1[6] 
                                       - 4*x0[4]*x1[7] - 11*x1[4]*x1[7] 
                                       - 4*x0[5]*x1[8] - 11*x1[5]*x1[8])
    + roots[i]*roots[i]*roots[i]*roots[i]*(35 + rou_*(3*x0[6]*x0[6] + 3*x0[7]*x0[7] 
                                                      + x0[6]*x1[6] + x0[7]*x1[7] 
                                                      + 3*(x0[8]*x0[8] + x1[6]*x1[6] + x1[7]*x1[7])
                                                      + x0[8]*x1[8] + 3*x1[8]*x1[8])));
    
    if (current < cost) {
      tau = roots[i];
      cost = current;
      result = true;
    }
  }
  // ROS_INFO_STREAM("best tau: " << tau << ", best cost: " << cost);
  
  if (result == true) {
    double t2 = tau*tau;
    double t3 = tau*t2;
    double t4 = tau*t3;
    double t5 = tau*t4;
    x_coeff[0] = -(12*x0[0] + 6*tau*x0[3] + t2*x0[6] - 12*x1[0] + 6*tau*x1[3] - t2*x1[6])/(2*t5); 
    y_coeff[0] = -(12*x0[1] + 6*tau*x0[4] + t2*x0[7] - 12*x1[1] + 6*tau*x1[4] - t2*x1[7])/(2*t5); 
    z_coeff[0] = -(12*x0[2] + 6*tau*x0[5] + t2*x0[8] - 12*x1[2] + 6*tau*x1[5] - t2*x1[8])/(2*t5);
    x_coeff[1] = -(-30*x0[0] - 16*tau*x0[3] - 3*t2*x0[6] + 30*x1[0] - 14*tau*x1[3] + 2*t2*x1[6])/(2*t4);
    y_coeff[1] = -(-30*x0[1] - 16*tau*x0[4] - 3*t2*x0[7] + 30*x1[1] - 14*tau*x1[4] + 2*t2*x1[7])/(2*t4);
    z_coeff[1] = -(-30*x0[2] - 16*tau*x0[5] - 3*t2*x0[8] + 30*x1[2] - 14*tau*x1[5] + 2*t2*x1[8])/(2*t4); 
    x_coeff[2] = -(20*x0[0] + 12*tau*x0[3] + 3*t2*x0[6] - 20*x1[0] + 8*tau*x1[3] - t2*x1[6])/(2*t3); 
    y_coeff[2] = -(20*x0[1] + 12*tau*x0[4] + 3*t2*x0[7] - 20*x1[1] + 8*tau*x1[4] - t2*x1[7])/(2*t3); 
    z_coeff[2] = -(20*x0[2] + 12*tau*x0[5] + 3*t2*x0[8] - 20*x1[2] + 8*tau*x1[5] - t2*x1[8])/(2*t3); 
    x_coeff[3] = x0[6]/2; 
    y_coeff[3] = x0[7]/2; 
    z_coeff[3] = x0[8]/2;
    x_coeff[4] = x0[3]; 
    y_coeff[4] = x0[4];
    z_coeff[4] = x0[5]; 
    x_coeff[5] = x0[0]; 
    y_coeff[5] = x0[1]; 
    z_coeff[5] = x0[2];
  }
  else 
    return false;

  double t = 0;
  double bound = deltaT_;
  size_t numPoints = ceil(tau / bound);
  double actual_deltaT = tau/numPoints;
    
  vector<State> xs;
  vector<Control> us;
  State x;
  Control u;
  State last_x= x0.head(6);
  pair<State, State> seg;
  bool is_valid = true;
  for (int i=0; t<=tau; ++i) {
    t = actual_deltaT*i;
    Eigen::Vector3d pos, vel, acc;
    calPVAFromCoeff(pos, vel, acc, x_coeff, y_coeff, z_coeff, t, 5);
    x.head(3) = pos;
    x.tail(3) = vel;
    u = acc;
    if (result)      
    {
      if (!validateVel(vel) || !validateAcc(acc))
        result = false;
    }
    if (is_valid && !validatePosSurround(pos, vel, acc)) {
      result = false;
      is_valid = false;
      seg.first = last_x;
    } 
    else if (!is_valid && validatePosSurround(pos, vel, acc)) {
      is_valid = true;
      seg.second = x;
      segs.push_back(seg);
    }
    last_x = x;
    if (vis_x) 
      vis_x->push_back(x);
    if (vis_u) 
      vis_u->push_back(u);
  }
  return result;
}

/*
bool KRRTPlanner::computeBestCost(const State& x0, const State& x1, 
                                  vector<pair<State, State>>& segs, 
                                  double& cost, double& tau, 
                         vector<State>* vis_x, vector<Control>* vis_u, 
                         double *x_coeff, double *y_coeff, double *z_coeff)
{
    Eigen::Matrix<double,X_DIM,1> x1diffbarx0, d;
    State d_tau = State::Zero();
    
    //calculate for tau*
    //DYNAMICS == DOUBLE_INTEGRATOR_3D when c=[0;0;0;0;0;0]
    double p[POLY_DEGREE + 1];
    p[0] = 1;
    p[1] = 0;
    p[2] =(- 4.0*x0[3]*x0[3] - 4.0*x0[3]*x1[3] - 4.0*x1[3]*x1[3] 
           - 4.0*x0[4]*x0[4] - 4.0*x0[4]*x1[4] - 4.0*x1[4]*x1[4] 
           - 4.0*x0[5]*x0[5] - 4.0*x0[5]*x1[5] - 4.0*x1[5]*x1[5]) * rou_;
    p[3] =(- 24.0*x0[0]*x0[3] - 24.0*x0[0]*x1[3] + 24.0*x1[0]*x0[3] 
           + 24.0*x1[0]*x1[3] - 24.0*x0[1]*x0[4] - 24.0*x0[1]*x1[4] 
           + 24.0*x1[1]*x0[4] + 24.0*x1[1]*x1[4] - 24.0*x0[2]*x0[5] 
           - 24.0*x0[2]*x1[5] + 24.0*x1[2]*x0[5] + 24.0*x1[2]*x1[5]) * rou_;
    p[4] =(- 36.0*x0[0]*x0[0] + 72.0*x0[0]*x1[0] - 36.0*x1[0]*x1[0] 
           - 36.0*x0[1]*x0[1] + 72.0*x0[1]*x1[1] - 36.0*x1[1]*x1[1] 
           - 36.0*x0[2]*x0[2] + 72.0*x0[2]*x1[2] - 36.0*x1[2]*x1[2]) * rou_;
           
    std::vector<double> roots = quartic(p[0], p[1], p[2], p[3], p[4]);

    cost = DBL_MAX;
    tau = DBL_MAX;
    bool result = false;
    for (size_t i = 0; i < roots.size();++i) {
        if (!((roots[i] > 0.0) && (roots[i] < cost))) {
            //tau* which is real number larger than 0, less than radius
            continue;
        }
        //DYNAMICS == DOUBLE_INTEGRATOR_2D
        double t7 = roots[i]*roots[i];
        double t10 = rou_/t7/roots[i];
        double t14 = 1/t7*rou_;
        double t26 = rou_/roots[i];
        x1diffbarx0[0] = x1[0]-x0[0]-roots[i]*x0[3];
        x1diffbarx0[1] = x1[1]-x0[1]-roots[i]*x0[4];
        x1diffbarx0[2] = x1[2]-x0[2]-roots[i]*x0[5];
        x1diffbarx0[3] = x1[3]-x0[3];
        x1diffbarx0[4] = x1[4]-x0[4];
        x1diffbarx0[5] = x1[5]-x0[5];
        d[0] = 12.0*t10*x1diffbarx0[0]-6.0*t14*x1diffbarx0[3];
        d[1] = 12.0*t10*x1diffbarx0[1]-6.0*t14*x1diffbarx0[4];
        d[2] = 12.0*t10*x1diffbarx0[2]-6.0*t14*x1diffbarx0[5];
        d[3] = -6.0*t14*x1diffbarx0[0]+4.0*t26*x1diffbarx0[3];
        d[4] = -6.0*t14*x1diffbarx0[1]+4.0*t26*x1diffbarx0[4];
        d[5] = -6.0*t14*x1diffbarx0[2]+4.0*t26*x1diffbarx0[5];
        
//         double current = roots[i] + (x1diffbarx0.transpose()*d).trace();
        double current = roots[i] + x1diffbarx0[0]*d[0] + x1diffbarx0[1]*d[1]
                                      + x1diffbarx0[2]*d[2] + x1diffbarx0[3]*d[3]
                                      + x1diffbarx0[4]*d[4] + x1diffbarx0[5]*d[5];
                                      
        if ((roots[i] > 0.0) && (current < cost)) {
            cost = current;
            tau = roots[i];
            d_tau = d;
            result = true;
        }
    }
    
    if (result == true) {
        double t2 = tau*tau;
        double t3 = t2*tau;
        x_coeff[0] = (2.0*(x0[0]-x1[0])+tau*(x0[3]+x1[3]))/t3;
        x_coeff[1] = -(3.0*(x0[0]-x1[0])+tau*(2*x0[3]+x1[3]))/t2;
        x_coeff[2] = x0[3];
        x_coeff[3] = x0[0];
        y_coeff[0] = (2.0*(x0[1]-x1[1])+tau*(x0[4]+x1[4]))/t3;
        y_coeff[1] = -(3.0*(x0[1]-x1[1])+tau*(2*x0[4]+x1[4]))/t2;
        y_coeff[2] = x0[4];
        y_coeff[3] = x0[1];
        z_coeff[0] = (2.0*(x0[2]-x1[2])+tau*(x0[5]+x1[5]))/t3;
        z_coeff[1] = -(3.0*(x0[2]-x1[2])+tau*(2*x0[5]+x1[5]))/t2;
        z_coeff[2] = x0[5];
        z_coeff[3] = x0[2];
        result = checkVelAcc(x0, x1, x_coeff, y_coeff, z_coeff, tau);
    } 
    
    double t = 0;
    double bound = deltaT_;
    size_t numPoints = ceil(tau / bound);
    double actual_deltaT = tau/numPoints;
    
    vector<State> xs;
    vector<Control> us;
    State last_x= x0;
    pair<State, State> seg;
    bool is_valid = true;
    for (int i=0; t<=tau; ++i) {
        t = actual_deltaT*i;
        State x;
        Control u;
        calculateStateAndControl(x0, x1, tau, d_tau, t, x, u);
        if (is_valid && !validatePosSurround(x.head(3), x.tail(3), u)) {
            is_valid = false;
            seg.first = last_x;
        } 
        else if (!is_valid && validatePosSurround(x.head(3), x.tail(3), u)) {
            is_valid = true;
            seg.second = x;
            segs.push_back(seg);
        }
        last_x = x;
        if (vis_x) 
            vis_x->push_back(x);
        if (vis_u) 
            vis_u->push_back(u);
    }
    return result;
}
*/

void KRRTPlanner::getVisStateAndControl(RRTNodePtrVector node_ptr_vector, 
                                           vector< State >* vis_x, 
                                           vector< Control >* vis_u)
{
  for (int i=0; i<node_ptr_vector.size()-1; ++i)
//   for (const auto& n_p : node_ptr_vector)
  {
    RRTNodePtr n_p = node_ptr_vector[i];
//     cout << "patched traj node tau_from_p: " << n_p->tau_from_parent << endl;
//     cout << "patched traj node order: " << n_p->n_order << endl;
    getVisPoints(n_p, vis_x, vis_u);
  }
}

void KRRTPlanner::getVisTrajCovering(RRTNodePtrVector node_ptr_vector, 
                                        vector<Vector3d>& cover_grids)
{
  cover_grids.clear();
  for (int i=0; i<node_ptr_vector.size()-1; ++i)
  {
    RRTNodePtr n_p = node_ptr_vector[i];
    double t;
    size_t numPoints = ceil(n_p->tau_from_parent / deltaT_);
    size_t step = 1;
    while (step < numPoints) step *= 2;
    double actual_deltaT = n_p->tau_from_parent/numPoints;
  
    for ( ; step > 1; step /= 2) {
      for (size_t i = step / 2; i < numPoints; i += step) {
        t = actual_deltaT*i;
        Eigen::Vector3d pos, vel, acc;
        calPVAFromCoeff(pos, vel, acc, n_p->x_coeff, n_p->y_coeff, n_p->z_coeff, t, n_p->n_order);
        vector<Vector3d> line_grids;
        getCheckPos(pos, vel, acc, line_grids, hor_safe_radius_, ver_safe_radius_);
//        cover_grids.push_back(line_grids.front());
//        cover_grids.push_back(line_grids.back());
        for (const auto& p : line_grids)
        {
          cover_grids.push_back(p);
        }
      
      }
    }
  }
}

void KRRTPlanner::getVisTrajCovering(const RRTNodeVector& node_ptr_vector, 
                                        vector<Vector3d>& cover_grids)
{
  cover_grids.clear();
  double d_t(0.01);
  for (int i=0; i<node_ptr_vector.size()-1; ++i)
  {
    RRTNode n_p = node_ptr_vector[i];
    double t;
    size_t numPoints = ceil(n_p.tau_from_parent / d_t);
    size_t step = 1;
    while (step < numPoints) step *= 2;
    double actual_deltaT = n_p.tau_from_parent/numPoints;
  
    for ( ; step > 1; step /= 2) {
      for (size_t i = step / 2; i < numPoints; i += step) {
        t = actual_deltaT*i;
        Eigen::Vector3d pos, vel, acc;
        calPVAFromCoeff(pos, vel, acc, n_p.x_coeff, n_p.y_coeff, n_p.z_coeff, t, n_p.n_order);
        vector<Vector3d> line_grids;
        getCheckPos(pos, vel, acc, line_grids, hor_safe_radius_, ver_safe_radius_);
//        cover_grids.push_back(line_grids.front());
//        cover_grids.push_back(line_grids.back());
        for (const auto& p : line_grids)
        {
          cover_grids.push_back(p);
        }
      
      }
    }
  }
}

void KRRTPlanner::getCheckPos(const Vector3d& pos, const Vector3d& vel, 
                              const Vector3d& acc, vector<Vector3d>& grids, 
                              double hor_radius, double ver_radius)
{
  Eigen::Vector3d cw_edge_pos, ccw_edge_pos;
  Eigen::Vector2d vel_hor, cw_radius_vec;
  cw_edge_pos[2] = pos[2];
  ccw_edge_pos[2] = pos[2];
  vel_hor[0] = vel[0];
  vel_hor[1] = vel[1];
  double v_hor_norm = vel_hor.norm();
  if (v_hor_norm < 1e-4)
  {
    vel_hor[0] = 1;
    vel_hor[1] = 1;
  }
  cw_radius_vec = rotate90Clockwise2d(vel_hor);
  cw_radius_vec = cw_radius_vec.normalized() * hor_radius;
  cw_edge_pos.head(2) = pos.head(2) + cw_radius_vec;
  ccw_edge_pos.head(2) = pos.head(2) - cw_radius_vec;
  //add horizontal vox;
  getlineGrids(cw_edge_pos, ccw_edge_pos, grids);
  Eigen::Vector3d vertical_up(pos), vertical_down(pos);
  vertical_up(2) += ver_radius;
  vertical_down(2) -= ver_radius; 
  //add veltical vox;
  getlineGrids(vertical_up, vertical_down, grids);
}

void KRRTPlanner::getlineGrids(const Vector3d& s_p, const Vector3d& e_p, 
                               vector<Vector3d>& grids)
{
  RayCaster raycaster;
  Eigen::Vector3d ray_pt;
  Eigen::Vector3d start = s_p/resolution_, end = e_p/resolution_;
  bool need_ray = raycaster.setInput(start, end);
  if (need_ray)
  {
    while (raycaster.step(ray_pt))
    {
      Eigen::Vector3d tmp = (ray_pt) * resolution_;
      tmp[0] += resolution_/2.0;
      tmp[1] += resolution_/2.0;
      tmp[2] += resolution_/2.0;
      grids.push_back(tmp);
    }
  }
  
  //check end
  Eigen::Vector3d end_idx;
  end_idx[0] = std::floor(end.x());
  end_idx[1] = std::floor(end.y());
  end_idx[2] = std::floor(end.z());

  ray_pt[0] = (double)end_idx[0];
  ray_pt[1] = (double)end_idx[1];
  ray_pt[2] = (double)end_idx[2];
  Eigen::Vector3d tmp = (ray_pt) * resolution_;
  tmp[0] += resolution_/2.0;
  tmp[1] += resolution_/2.0;
  tmp[2] += resolution_/2.0;    
  grids.push_back(tmp);
  
}

double KRRTPlanner::getTrajLength(RRTNodePtrVector node_ptr_vector, double d_t)
{
  double length = 0.0;
  
  for (int i=0; i<node_ptr_vector.size()-1; ++i)
  {
    RRTNodePtr n_p = node_ptr_vector[i];
    
    int cur_order = n_p->n_order;
    int n = floor(n_p->tau_from_parent / d_t);
    for (int i=0; i<n; ++i) 
    {
      double t1 = d_t*i;
      Eigen::Vector3d vel;
      vel[0] = calVelFromCoeff(t1, n_p->x_coeff, cur_order);
      vel[1] = calVelFromCoeff(t1, n_p->y_coeff, cur_order);
      vel[2] = calVelFromCoeff(t1, n_p->z_coeff, cur_order);
      length += vel.norm() * d_t;
    }
  }
  return length;
}

void KRRTPlanner::getTrajAttributes(const RRTNodeVector& node_vector, double& traj_duration, double& ctrl_cost, double& jerk_itg, double& traj_length, int& seg_nums)
{
  traj_length = 0.0;
  ctrl_cost = 0.0;
  jerk_itg = 0.0;
  traj_duration = 0.0;
  double d_t = 0.015;
  seg_nums = node_vector.size() - 1;
  
  // std::ofstream oFile;
  // oFile.open("/home/kyle/ros_ws/kinorrtstar/benchmark/krrt-back-acc(t).csv", ios::out | ios::app);   
  
  for (int i=node_vector.size()-2; i>=0; --i)
  {
    traj_duration += node_vector[i].tau_from_parent;
    
    int cur_order = node_vector[i].n_order;
    int n = floor(node_vector[i].tau_from_parent / d_t);
    for (int j=0; j<n; ++j) 
    {
      double t1 = d_t*j;
      Eigen::Vector3d vel, acc, jerk;
      vel[0] = calVelFromCoeff(t1, node_vector[i].x_coeff, cur_order);
      vel[1] = calVelFromCoeff(t1, node_vector[i].y_coeff, cur_order);
      vel[2] = calVelFromCoeff(t1, node_vector[i].z_coeff, cur_order);
      acc[0] = calAccFromCoeff(t1, node_vector[i].x_coeff, cur_order);
      acc[1] = calAccFromCoeff(t1, node_vector[i].y_coeff, cur_order);
      acc[2] = calAccFromCoeff(t1, node_vector[i].z_coeff, cur_order);
      jerk[0] = calJerkFromCoeff(t1, node_vector[i].x_coeff, cur_order);
      jerk[1] = calJerkFromCoeff(t1, node_vector[i].y_coeff, cur_order);
      jerk[2] = calJerkFromCoeff(t1, node_vector[i].z_coeff, cur_order);
//       cout << "vel: " << vel.norm() << endl;
//       cout << "acc: " << acc.norm() << endl;
//       cout << "jerk: " << jerk.norm() << endl;
      // oFile << acc.norm() << "," << vel.norm() << endl;
      traj_length += vel.norm() * d_t;
      ctrl_cost += acc.dot(acc) * d_t;
      jerk_itg += jerk.dot(jerk) * d_t;
    }
  }
  // oFile.close();
}

void KRRTPlanner::getTrajAttributes(const RRTNodePtrVector& node_ptr_vector, double& traj_duration, double& ctrl_cost, double& traj_length, int& seg_nums)
{
  traj_length = 0.0;
  ctrl_cost = 0.0;
  traj_duration = 0.0;
  double d_t = 0.03;
  seg_nums = node_ptr_vector.size() - 1;
  
  for (int i=0; i<node_ptr_vector.size()-1; ++i)
  {
    RRTNodePtr n_p = node_ptr_vector[i];
    traj_duration += n_p->tau_from_parent;
    
    int cur_order = n_p->n_order;
    int n = floor(n_p->tau_from_parent / d_t);
    for (int j=0; j<n; ++j) 
    {
      double t1 = d_t*j;
      Eigen::Vector3d vel, acc;
      vel[0] = calVelFromCoeff(t1, n_p->x_coeff, cur_order);
      vel[1] = calVelFromCoeff(t1, n_p->y_coeff, cur_order);
      vel[2] = calVelFromCoeff(t1, n_p->z_coeff, cur_order);
      acc[0] = calAccFromCoeff(t1, n_p->x_coeff, cur_order);
      acc[1] = calAccFromCoeff(t1, n_p->y_coeff, cur_order);
      acc[2] = calAccFromCoeff(t1, n_p->z_coeff, cur_order);
      traj_length += vel.norm() * d_t;
      ctrl_cost += acc.dot(acc) * d_t;
    }
  }
}

void KRRTPlanner::getStateAndControl(RRTNodeVector node_vector, 
                                     vector< State >* vis_x, 
                                     vector< Control >* vis_u)
{      
  // std::ofstream oFile;
  // oFile.open("/home/dji/yhk/krrt_ws/krrt-back-acc(t).csv", ios::out | ios::app);   
  for (int k=0; k<node_vector.size()-1; ++k)
//   for (const RRTNode& n_p : node_vector)
  {
    int cur_order = node_vector[k].n_order;
    double d_t = 0.015;
    int n = floor(node_vector[k].tau_from_parent / d_t);
    for (int i=0; i<=n; ++i) {
      double t1 = d_t*i;
      Eigen::Vector3d pos, vel, acc;
      calPVAFromCoeff(pos, vel, acc, node_vector[k].x_coeff, node_vector[k].y_coeff, node_vector[k].z_coeff, t1, cur_order);
      State x;
      Control u;
      // oFile << acc.norm() << "," << vel.norm() << endl;
      x << pos(0), pos(1), pos(2), vel(0), vel(1), vel(2);
      u << acc(0), acc(1), acc(2);
      vis_x->push_back(x);
      vis_u->push_back(u);
    }
  }
  // oFile.close();
}

void KRRTPlanner::fillPath(RRTNodePtr goal_leaf, RRTNodePtrVector& path)
{
  path.clear();
  RRTNodePtr node = goal_leaf;
  while (node) 
  {
    path.push_back(node);
    traj_duration_ += node->tau_from_parent;
    node = node->parent;
  }
}

int KRRTPlanner::getPath(RRTNodeVector& path_node_g2s, int path_type)
{
  path_node_g2s.clear();
  
  if (path_type == 1)
  {
    int n = path_.size();
    RRTNode node;
    for (int i=0; i<n; ++i) 
    {
      node.cost_from_start = path_[i]->cost_from_start;
      node.tau_from_parent = path_[i]->tau_from_parent;
      node.tau_from_start = path_[i]->tau_from_start;
      node.n_order = path_[i]->n_order;
      for (int j=0; j<=path_[i]->n_order; ++j) 
      {
        node.x_coeff[j] = path_[i]->x_coeff[j];
        node.y_coeff[j] = path_[i]->y_coeff[j];
        node.z_coeff[j] = path_[i]->z_coeff[j];
      }
      path_node_g2s.push_back(node);
    }
    return n;
  }
  else if (path_type == 2)
  {
    int n = patched_path_.size();
    RRTNode node;
    for (int i=0; i<n; ++i) 
    {
      node.cost_from_start = patched_path_[i]->cost_from_start;
      node.tau_from_parent = patched_path_[i]->tau_from_parent;
      node.tau_from_start = patched_path_[i]->tau_from_start;
      node.n_order = patched_path_[i]->n_order;
      for (int j=0; j<=patched_path_[i]->n_order; ++j) 
      {
        node.x_coeff[j] = patched_path_[i]->x_coeff[j];
        node.y_coeff[j] = patched_path_[i]->y_coeff[j];
        node.z_coeff[j] = patched_path_[i]->z_coeff[j];
      }
      path_node_g2s.push_back(node);
    }
    return n;
  }
}

int KRRTPlanner::getBypass(double time, RRTNodeVector& path_node_g2s)
{
  path_node_g2s.clear();
  int idx = 0;
  double t = 0;
  for (int i=2; i<valid_start_tree_node_nums_; ++i) 
  {
    double dt = start_tree_[i]->tau_from_start - time;
    if (dt > t)
    {
      t = dt;
      idx = i;
      if (dt > 1)
        break;
    }
  }  
  if (idx == 0)
    return 0;
  
  ROS_WARN("idx: %d, bypass duration: %lf", idx, start_tree_[idx]->tau_from_start);
  RRTNodePtr node_ptr = start_tree_[idx];
  RRTNode node;
  while (node_ptr) 
  {
    node.cost_from_start = node_ptr->cost_from_start;
    node.tau_from_parent = node_ptr->tau_from_parent;
    node.tau_from_start = node_ptr->tau_from_start;
    node.n_order = node_ptr->n_order;
    for (int j=0; j<4; ++j) 
    {
      node.x_coeff[j] = node_ptr->x_coeff[j];
      node.y_coeff[j] = node_ptr->y_coeff[j];
      node.z_coeff[j] = node_ptr->z_coeff[j];
    }
    path_node_g2s.push_back(node);
    node_ptr = node_ptr->parent;
  }
  
  vector<State> vis_x;
  vector<Control> vis_u;
  vis_x.clear();
  vis_u.clear();
  findTrajBackwardFromGoal(start_tree_[idx], &vis_x, &vis_u);
  vis.visualizeBypassTraj(vis_x, vis_u, occ_map_->getLocalTime());
    
  return path_node_g2s.size();
}

void KRRTPlanner::findTrajBackwardFromGoal(RRTNodePtr goal_leaf, 
                                           vector< State >* vis_x, 
                                           vector< Control >* vis_u)
{
  if (goal_leaf == nullptr) 
    return;
    
  RRTNodePtr node = goal_leaf;
  while (node->parent != nullptr) 
  {
    double cost, tau, a_dT;
//     computeBestCost(node->parent->x, node->x, cost, tau, vis_x, vis_u);
    getVisPoints(node, vis_x, vis_u);
    node = node->parent;
  }
}

int KRRTPlanner::getTree(RRTNodePtrVector& tree)
{
  tree = start_tree_;
  return valid_start_tree_node_nums_;
}

double KRRTPlanner::getTrajDura()
{
  return traj_duration_;
}

// for replan usage
bool KRRTPlanner::updateColideInfo(const RRTNodeVector& node_vector, double check_start_time, double check_dura, Eigen::Vector3d& collide_pos, double& t_safe_dura)
{
  t_safe_dura = traj_duration_ - check_start_time;
  size_t n = node_vector.size();
  if (n < 2)
  {
    return false;
  } 
  if (check_start_time >= traj_duration_)
  {
    //ROS_ERROR("check time more than traj duration!");
    return false;
  }
  int idx = n-2;
  double t_start_in_traj = check_start_time;
  for (idx=n-2; idx>=0; --idx)
  {
    check_start_time -= node_vector[idx].tau_from_parent;
    if (check_start_time < 0) 
    {
      check_start_time += node_vector[idx].tau_from_parent;
      break;
    }
  } 
  double t_start_in_seg = check_start_time;
//   cout << "check_start_time in one seg: " << check_start_time << endl;
//   cout << "idx: " << idx << " vector size: " << n << endl;
//   cout << "seg time: " << node_vector[idx].tau_from_parent << endl;
//   cout << "seg order: " << node_vector[idx].n_order << endl;
  Eigen::Vector3d last_uncollide_pos;
  int order = node_vector[idx].n_order;
  for (double t_in_seg = t_start_in_seg; t_in_seg < node_vector[idx].tau_from_parent; t_in_seg += 0.03) 
  {
    // only check check_dura seconds from check_start_time
    if ((t_in_seg - t_start_in_seg) >= check_dura)
      return false;
    Eigen::Vector3d pos, vel, acc;
    calPVAFromCoeff(pos, vel, acc, node_vector[idx].x_coeff, node_vector[idx].y_coeff, node_vector[idx].z_coeff, t_in_seg, order);
    vector<Vector3d> line_grids;
    getCheckPos(pos, vel, acc, line_grids, replan_hor_safe_radius_, replan_ver_safe_radius_);
    for (const auto& grid : line_grids)
    {
      if (occ_map_->getVoxelState(grid) != 0) {
        // ROS_INFO_STREAM("Future collide pos: " << pos.transpose()); 
        collide_pos = last_uncollide_pos;
        t_safe_dura = t_in_seg - t_start_in_seg;
        vis.visualizeTrajCovering(line_grids, resolution_, occ_map_->getLocalTime());
        return true;
      }
    }
    last_uncollide_pos = pos;
  }
  
  if (idx == 0)
    return false;
  
  t_safe_dura = 0;
  t_safe_dura += node_vector[idx].tau_from_parent - t_start_in_seg;
  for (int i=idx-1; i>=0; --i) 
  {
    double d_t = 0.03;
    int n_check_point = floor(node_vector[i].tau_from_parent / d_t);
    int order = node_vector[i].n_order;
    for (int j=0; j<n_check_point; ++j) 
    {
      double t_in_seg = d_t*j;
      if ((t_safe_dura + t_in_seg) >= check_dura)
        return false;
      Eigen::Vector3d pos, vel, acc;
      calPVAFromCoeff(pos, vel, acc, node_vector[i].x_coeff, node_vector[i].y_coeff, node_vector[i].z_coeff, t_in_seg, order);
      vector<Vector3d> line_grids;
      getCheckPos(pos, vel, acc, line_grids, replan_hor_safe_radius_, replan_ver_safe_radius_);
      for (const auto& grid : line_grids)
      {
        if (occ_map_->getVoxelState(grid) != 0) {
          // ROS_INFO_STREAM("Future collide pos: " << pos.transpose()); 
          collide_pos = last_uncollide_pos;
          t_safe_dura += t_in_seg;
          vis.visualizeTrajCovering(line_grids, resolution_, occ_map_->getLocalTime());
          return true;
        }
      }
      last_uncollide_pos = pos;
    }
    t_safe_dura += node_vector[i].tau_from_parent;
  }
  return false;
}

bool KRRTPlanner::checkOptimizedTraj(const RRTNodeVector& node_vector)
{
  size_t n = node_vector.size();
  if (n < 2)
  {
    return false;
  }
  
  for (int i=n-2; i>=0; --i) 
  {
    double d_t = 0.01;
    int n_check_point = floor(node_vector[i].tau_from_parent / d_t);
    int order = node_vector[i].n_order;
    for (int j=0; j<n_check_point; ++j) 
    {
      double t1 = d_t*j;
      Eigen::Vector3d pos, vel, acc;
      calPVAFromCoeff(pos, vel, acc, node_vector[i].x_coeff, node_vector[i].y_coeff, node_vector[i].z_coeff, t1, order);
      vector<Vector3d> line_grids;
      getCheckPos(pos, vel, acc, line_grids, replan_hor_safe_radius_+ resolution_, replan_ver_safe_radius_+ resolution_);
      for (const auto& grid : line_grids)
      {
        if (occ_map_->getVoxelState(grid) != 0) {
          // ROS_INFO_STREAM("Optimization violates Pos constrain, t: " << t1 << " pos: " << pos.transpose()); 
          return true;
        }
      }
      if (!validateVel(vel))
      {
        // ROS_INFO_STREAM("Optimization violates Vel constrain, t1: " << t1 << " pos: " << pos.transpose() << " , vel: " << vel.transpose()); 
        return true;
      }
      if (!validateAcc(acc))
      {
        // ROS_INFO_STREAM("Optimization violates Acc constrain, t1: " << t1 << " pos: " << pos.transpose() << " , acc: " << acc.transpose()); 
        return true;
      }
    }
  }
  return false;
}

void KRRTPlanner::patching(const RRTNodePtrVector& path, 
                           RRTNodePtrVector& patched_path, 
                           const Vector3d& u_init)
{
  patched_path.clear();
  int n = path.size();
  ROS_WARN("raw path seg num: %d", n - 1);
  if (n < 2) 
  {
    return;
  }
  
  double last_percent = 0;
  for (int i=0; i<n-2; ++i)
  {
    bool patch_one_hole = false;
    for (int j=2; j<=4; ++j) 
    {
      double percent = j/10.0;
      Eigen::VectorXd end_state = getPatchState(path[i], percent);
      Eigen::VectorXd start_state = getPatchState(path[i+1], 1-percent);
      // ROS_INFO_STREAM("start: " << start_state.transpose());
      // ROS_INFO_STREAM("end  : " << end_state.transpose());
      double patch_traj_time, patch_traj_cost;
      double x_coeff[6], y_coeff[6], z_coeff[6];
      bool patched = patchTwoState(start_state, end_state, patch_traj_cost, patch_traj_time, x_coeff, y_coeff, z_coeff);
      if (patched)
      {
        if (i == 0)
        {
          //try patch end seg to make end acc 0 when rest
          double best_cost, best_tau;
          vector<pair<State, State>> segs;
          double x_coeff[6], y_coeff[6], z_coeff[6];
          Vector3d p0, p1, v0, v1, a0, a1;
          calPVAFromCoeff(p0, v0, a0, path[0]->x_coeff, path[0]->y_coeff, path[0]->z_coeff, path[0]->tau_from_parent*percent, path[0]->n_order);
          calPVAFromCoeff(p1, v1, a1, path[0]->x_coeff, path[0]->y_coeff, path[0]->z_coeff, path[0]->tau_from_parent, path[0]->n_order);
          VectorXd x_u_init = VectorXd::Zero(9);
          x_u_init.head(3) = p0;
          x_u_init.segment(3, 3) = v0;
          x_u_init.tail(3) = a0;
          VectorXd x_u_final = VectorXd::Zero(9);
          x_u_final.head(3) = p1;
          x_u_final.segment(3, 3) = Eigen::Vector3d(0.0,0.0,0.0);
          x_u_final.tail(3) = Eigen::Vector3d(0.0,0.0,0.0);
          if (computeBestCost(x_u_init, x_u_final, segs, best_cost, best_tau, nullptr, nullptr, x_coeff, y_coeff, z_coeff))
          {
            // ROS_WARN("end patched"); 
            RRTNodePtr n_p_5order = new RRTNode;
            n_p_5order->n_order = 5;
            n_p_5order->tau_from_parent = best_tau;
            for (int i=0; i<6; ++i) 
            {
              n_p_5order->x_coeff[i] = x_coeff[i];
              n_p_5order->y_coeff[i] = y_coeff[i];
              n_p_5order->z_coeff[i] = z_coeff[i];
            }
            last_percent = 0.0;
            patched_path.push_back(n_p_5order);
          }  
          else
          {
            // ROS_ERROR("end not patched");
            RRTNodePtr n_p_3order = new RRTNode;
            cutTraj(n_p_3order, path[i], path[i]->tau_from_parent*percent, path[i]->tau_from_parent*(1-last_percent));
            last_percent = 0.0;
            patched_path.push_back(n_p_3order);
          }
        }
        else
        {
          RRTNodePtr n_p_3order = new RRTNode;
          cutTraj(n_p_3order, path[i], path[i]->tau_from_parent*percent, path[i]->tau_from_parent*(1-last_percent));
          last_percent = 0.0;
          patched_path.push_back(n_p_3order);
        }
        RRTNodePtr n_p_5order = new RRTNode;
        n_p_5order->n_order = 5;
        n_p_5order->tau_from_parent = patch_traj_time;
        for (int i=0; i<6; ++i) 
        {
          n_p_5order->x_coeff[i] = x_coeff[i];
          n_p_5order->y_coeff[i] = y_coeff[i];
          n_p_5order->z_coeff[i] = z_coeff[i];
        }
        patched_path.push_back(n_p_5order);
        patch_one_hole = true;
        last_percent = percent;
        break;
      }
    }
    if (!patch_one_hole) 
    {
      // ROS_WARN("form goal to start, %d hole not patched", i+1);
      RRTNodePtr n_p_3order = new RRTNode;
      cutTraj(n_p_3order, path[i], 0.0, path[i]->tau_from_parent*(1-last_percent));
      last_percent = 0.0;
      patched_path.push_back(n_p_3order);
    }
    else
    {
      // ROS_ERROR("from goal to start, %d hole patched", i+1);
    }
  }
  
  //try patch start seg to make start acc consistant when replan
  double best_cost, best_tau;
  vector<pair<State, State>> segs;
  double x_coeff[6], y_coeff[6], z_coeff[6];
  Vector3d p0, p1, v0, v1, a0, a1;
  calPVAFromCoeff(p0, v0, a0, path[n-2]->x_coeff, path[n-2]->y_coeff, path[n-2]->z_coeff, 0.0, path[n-2]->n_order);
  calPVAFromCoeff(p1, v1, a1, path[n-2]->x_coeff, path[n-2]->y_coeff, path[n-2]->z_coeff, path[n-2]->tau_from_parent*(1-last_percent), path[n-2]->n_order);
  VectorXd x_u_init = VectorXd::Zero(9);
  x_u_init.head(3) = p0;
  x_u_init.segment(3, 3) = v0;
  x_u_init.tail(3) = u_init;
  VectorXd x_u_final = VectorXd::Zero(9);
  x_u_final.head(3) = p1;
  x_u_final.segment(3, 3) = v1;
  x_u_final.tail(3) = a1;
  if (computeBestCost(x_u_init, x_u_final, segs, best_cost, best_tau, nullptr, nullptr, x_coeff, y_coeff, z_coeff))
  {
    ROS_WARN("start patched"); 
    RRTNodePtr n_p_5order = new RRTNode;
    n_p_5order->n_order = 5;
    n_p_5order->tau_from_parent = best_tau;
    for (int i=0; i<6; ++i) 
    {
      n_p_5order->x_coeff[i] = x_coeff[i];
      n_p_5order->y_coeff[i] = y_coeff[i];
      n_p_5order->z_coeff[i] = z_coeff[i];
    }
    patched_path.push_back(n_p_5order);
    RRTNodePtr start_node = new RRTNode;
    start_node->n_order = 0;
    patched_path.push_back(start_node);
  }  
  else
  {
    // ROS_ERROR("start not patched");
    RRTNodePtr n_p_3order = new RRTNode;
    cutTraj(n_p_3order, path[n-2], 0.0, path[n-2]->tau_from_parent*(1-last_percent));
    last_percent = 0.0;
    patched_path.push_back(n_p_3order);
    RRTNodePtr start_node = new RRTNode;
    start_node->n_order = 0;
    patched_path.push_back(start_node);
  }
}

Eigen::VectorXd KRRTPlanner::getPatchState(const RRTNodePtr& node_ptr, double percent)
{
  double t = node_ptr->tau_from_parent * percent;
  // ROS_INFO("tau: %lf, percent: %lf", node_ptr->tau_from_parent, percent);
  Eigen::Vector3d pos, vel, acc;
  double t1 = t;
  double t2 = t1*t1;
  double t3 = t2*t1;
  pos[0] = node_ptr->x_coeff[0]*t3 + node_ptr->x_coeff[1]*t2 + node_ptr->x_coeff[2]*t1 + node_ptr->x_coeff[3];
  pos[1] = node_ptr->y_coeff[0]*t3 + node_ptr->y_coeff[1]*t2 + node_ptr->y_coeff[2]*t1 + node_ptr->y_coeff[3];
  pos[2] = node_ptr->z_coeff[0]*t3 + node_ptr->z_coeff[1]*t2 + node_ptr->z_coeff[2]*t1 + node_ptr->z_coeff[3];
  vel[0] = 3*node_ptr->x_coeff[0]*t2 + 2*node_ptr->x_coeff[1]*t1 + node_ptr->x_coeff[2];
  vel[1] = 3*node_ptr->y_coeff[0]*t2 + 2*node_ptr->y_coeff[1]*t1 + node_ptr->y_coeff[2];
  vel[2] = 3*node_ptr->z_coeff[0]*t2 + 2*node_ptr->z_coeff[1]*t1 + node_ptr->z_coeff[2];
  acc[0] = 6*node_ptr->x_coeff[0]*t1 + 2*node_ptr->x_coeff[1];
  acc[1] = 6*node_ptr->y_coeff[0]*t1 + 2*node_ptr->y_coeff[1];
  acc[2] = 6*node_ptr->z_coeff[0]*t1 + 2*node_ptr->z_coeff[1];
  Eigen::VectorXd s(9);
  s << pos, vel, acc;
  return s;
}

bool KRRTPlanner::patchTwoState(const Eigen::VectorXd& x0, const Eigen::VectorXd& x1, 
                              double& cost, double& tau, 
                              double *x_coeff, double *y_coeff, double *z_coeff)
{
  double p[5];
  p[0] = 35 + rou_*(3*x0[6]*x0[6] + 3*x0[7]*x0[7] + 3*x0[8]*x0[8] 
                    + x0[6]*x1[6] + 3*x1[6]*x1[6] + x0[7]*x1[7]
                    + 3*x1[7]*x1[7] + x0[8]*x1[8] + 3*x1[8]*x1[8]);
  p[1] = 0;
  p[2] = -6*rou_*(32*x0[3]*x0[3] + 32*x0[4]*x0[4] + 32*x0[5]*x0[5]
                + 5*x0[0]*x0[6] + 5*x0[1]*x0[7] + 5*x0[2]*x0[8] 
                - 5*x0[6]*x1[0] - 5*x0[7]*x1[1] - 5*x0[8]*x1[2] 
                + 36*x0[3]*x1[3] + 32*x1[3]*x1[3] + 36*x0[4]*x1[4] 
                + 32*x1[4]*x1[4] + 36*x0[5]*x1[5] + 32*x1[5]*x1[5] 
                - 5*x0[0]*x1[6] + 5*x1[0]*x1[6] - 5*x0[1]*x1[7] 
                + 5*x1[1]*x1[7] - 5*x0[2]*x1[8] + 5*x1[2]*x1[8]);
  p[3] = -1200*rou_*(x0[2]*x0[5] - x0[3]*x1[0] - x0[4]*x1[1] - x0[5]*x1[2] 
                   - x1[0]*x1[3] + x0[0]*(x0[3] + x1[3]) - x1[1]*x1[4] 
                   + x0[1]*(x0[4] + x1[4]) + x0[2]*x1[5] - x1[2]*x1[5]);
  p[4] = -1800*rou_*(x0[0]*x0[0] + x0[1]*x0[1] + x0[2]*x0[2] 
                 - 2*x0[0]*x1[0] + x1[0]*x1[0] - 2*x0[1]*x1[1] 
                   + x1[1]*x1[1] - 2*x0[2]*x1[2] + x1[2]*x1[2]);
          
//   Eigen::Vector3d accsIni, accsFin, velsIni, velsFin, posIni, posFin;
//   accsIni << x0[6], x0[7], x0[8];
//   accsFin << x1[6], x1[7], x1[8];
//   velsIni << x0[3], x0[4], x0[5];
//   velsFin << x1[3], x1[4], x1[5];
//   posIni << x0[0], x0[1], x0[2];
//   posFin << x1[0], x1[1], x1[2];
//   Eigen::VectorXd coeffsGradT(5);        
//   coeffsGradT(0) = 35.0 * 1.0/rou_ + (3.0 * accsIni.array().square() + accsIni.array() * accsFin.array() + 3.0 * accsFin.array().square()).sum();
//   coeffsGradT(1) = 0.0;        
//   coeffsGradT(2) = -6.0 * (32.0 * velsIni.array().square() + 36.0 * velsIni.array() * velsFin.array() + 32.0 * velsFin.array().square() + 5.0 * (accsIni.array() - accsFin.array()) * (posIni.array() - posFin.array())).sum();        
//   coeffsGradT(3) = -1200.0 * ((velsIni.array() + velsFin.array()) * (posIni.array() - posFin.array())).sum();        
//   coeffsGradT(4) = -1800.0 * (posIni.array() - posFin.array()).square().sum();

  std::vector<double> roots = quartic(p[0], p[1], p[2], p[3], p[4]);
//   std::vector<double> roots = quartic(coeffsGradT(0), coeffsGradT(1), coeffsGradT(2), coeffsGradT(3), coeffsGradT(4));

  //calculate for cost[tau*] when c=[0;0;0;0;0;0]
  bool result = false;
  tau = DBL_MAX;
  cost = DBL_MAX;
  
  for (size_t i = 0; i < roots.size();++i) {
    if (roots[i] <= 1e-3 || roots[i] > cost) {
      continue;
    }
    
    double current = (1/(35*roots[i]*roots[i]*roots[i]))
    *(600*rou_*((x0[0] - x1[0])*(x0[0] - x1[0]) 
              + (x0[1] - x1[1])*(x0[1] - x1[1]) 
              + (x0[2] - x1[2])*(x0[2] - x1[2]))
    + 600*rou_*roots[i]*((x0[0] - x1[0])*(x0[3] + x1[3]) 
                       + (x0[1] - x1[1])*(x0[4] + x1[4]) 
                       + (x0[2] - x1[2])*(x0[5] + x1[5]))
    + 6*rou_*roots[i]*roots[i]*(32*x0[3]*x0[3] + 32*x0[4]*x0[4] + 36*x0[3]*x1[3] 
                              + 36*x0[4]*x1[4] + 32*(x1[3]*x1[3] + x1[4]*x1[4]) 
                              + 4*(8*x0[5]*x0[5] + 9*x0[5]*x1[5] + 8*x1[5]*x1[5])
                              + 5*(x0[0] - x1[0])*(x0[6] - x1[6]) 
                              + 5*(x0[1] - x1[1])*(x0[7] - x1[7]) 
                              + 5*(x0[2] - x1[2])*(x0[8] - x1[8]))
    + 2*rou_*roots[i]*roots[i]*roots[i]*(11*x0[3]*x0[6] + 11*x0[4]*x0[7] 
                                       + 11*x0[5]*x0[8] + 4*x0[6]*x1[3] 
                                       + 4*x0[7]*x1[4] + 4*x0[8]*x1[5] 
                                       - 4*x0[3]*x1[6] - 11*x1[3]*x1[6] 
                                       - 4*x0[4]*x1[7] - 11*x1[4]*x1[7] 
                                       - 4*x0[5]*x1[8] - 11*x1[5]*x1[8])
    + roots[i]*roots[i]*roots[i]*roots[i]*(35 + rou_*(3*x0[6]*x0[6] + 3*x0[7]*x0[7] 
                                                      + x0[6]*x1[6] + x0[7]*x1[7] 
                                                      + 3*(x0[8]*x0[8] + x1[6]*x1[6] + x1[7]*x1[7])
                                                      + x0[8]*x1[8] + 3*x1[8]*x1[8])));
    
    if (current < cost) {
      tau = roots[i];
      cost = current;
      result = true;
    }
  }
  
  if (result == true) {
    double t2 = tau*tau;
    double t3 = tau*t2;
    double t4 = tau*t3;
    double t5 = tau*t4;
    
    x_coeff[0] = -(12*x0[0] + 6*tau*x0[3] + t2*x0[6] - 12*x1[0] + 6*tau*x1[3] - t2*x1[6])/(2*t5); 
    y_coeff[0] = -(12*x0[1] + 6*tau*x0[4] + t2*x0[7] - 12*x1[1] + 6*tau*x1[4] - t2*x1[7])/(2*t5); 
    z_coeff[0] = -(12*x0[2] + 6*tau*x0[5] + t2*x0[8] - 12*x1[2] + 6*tau*x1[5] - t2*x1[8])/(2*t5);
    x_coeff[1] = -(-30*x0[0] - 16*tau*x0[3] - 3*t2*x0[6] + 30*x1[0] - 14*tau*x1[3] + 2*t2*x1[6])/(2*t4);
    y_coeff[1] = -(-30*x0[1] - 16*tau*x0[4] - 3*t2*x0[7] + 30*x1[1] - 14*tau*x1[4] + 2*t2*x1[7])/(2*t4);
    z_coeff[1] = -(-30*x0[2] - 16*tau*x0[5] - 3*t2*x0[8] + 30*x1[2] - 14*tau*x1[5] + 2*t2*x1[8])/(2*t4); 
    x_coeff[2] = -(20*x0[0] + 12*tau*x0[3] + 3*t2*x0[6] - 20*x1[0] + 8*tau*x1[3] - t2*x1[6])/(2*t3); 
    y_coeff[2] = -(20*x0[1] + 12*tau*x0[4] + 3*t2*x0[7] - 20*x1[1] + 8*tau*x1[4] - t2*x1[7])/(2*t3); 
    z_coeff[2] = -(20*x0[2] + 12*tau*x0[5] + 3*t2*x0[8] - 20*x1[2] + 8*tau*x1[5] - t2*x1[8])/(2*t3); 
    x_coeff[3] = x0[6]/2; 
    y_coeff[3] = x0[7]/2; 
    z_coeff[3] = x0[8]/2;
    x_coeff[4] = x0[3]; 
    y_coeff[4] = x0[4];
    z_coeff[4] = x0[5]; 
    x_coeff[5] = x0[0]; 
    y_coeff[5] = x0[1]; 
    z_coeff[5] = x0[2];
    
    result = checkPath(x_coeff, y_coeff, z_coeff, tau, 5);
  }

  return result;
}

bool KRRTPlanner::checkPath(double *x_coeff, double *y_coeff, double *z_coeff, double t, int order)
{
  double d_t = 0.03;
  int n = floor(t / d_t);
  for (int i=0; i<n; ++i) {
    double t1 = d_t*i;
    Eigen::Vector3d pos, vel, acc;
    calPVAFromCoeff(pos, vel, acc, x_coeff, y_coeff, z_coeff, t1, order);
    State x;
    Control u;
    x << pos(0), pos(1), pos(2), vel(0), vel(1), vel(2);
    u << acc(0), acc(1), acc(2);
    
    if(!validateStateAndControl(x, u))
      return false;
  }
  return true;
}

void KRRTPlanner::cutTraj(RRTNodePtr& new_traj, const RRTNodePtr& ori_traj, double ori_t_s, double ori_t_e)
{
  double tau = ori_t_e - ori_t_s;
  if (tau <= 0)
  {
    ROS_ERROR("wrong cut");
    return;
  }
  
  int order = ori_traj->n_order;
  if (order == 3)
  {
    State x0, x1;
    x0[0] = calPosFromCoeff(ori_t_s, ori_traj->x_coeff, order);
    x0[1] = calPosFromCoeff(ori_t_s, ori_traj->y_coeff, order);
    x0[2] = calPosFromCoeff(ori_t_s, ori_traj->z_coeff, order);
    x0[3] = calVelFromCoeff(ori_t_s, ori_traj->x_coeff, order);
    x0[4] = calVelFromCoeff(ori_t_s, ori_traj->y_coeff, order);
    x0[5] = calVelFromCoeff(ori_t_s, ori_traj->z_coeff, order);
        
    x1[0] = calPosFromCoeff(ori_t_e, ori_traj->x_coeff, order);
    x1[1] = calPosFromCoeff(ori_t_e, ori_traj->y_coeff, order);
    x1[2] = calPosFromCoeff(ori_t_e, ori_traj->z_coeff, order);
    x1[3] = calVelFromCoeff(ori_t_e, ori_traj->x_coeff, order);
    x1[4] = calVelFromCoeff(ori_t_e, ori_traj->y_coeff, order);
    x1[5] = calVelFromCoeff(ori_t_e, ori_traj->z_coeff, order);
    
    double t2 = tau*tau;
    double t3 = t2*tau;
    new_traj->x_coeff[0] = (2.0*(x0[0]-x1[0])+tau*(x0[3]+x1[3]))/t3;
    new_traj->x_coeff[1] = -(3.0*(x0[0]-x1[0])+tau*(2*x0[3]+x1[3]))/t2;
    new_traj->x_coeff[2] = x0[3];
    new_traj->x_coeff[3] = x0[0];
    new_traj->y_coeff[0] = (2.0*(x0[1]-x1[1])+tau*(x0[4]+x1[4]))/t3;
    new_traj->y_coeff[1] = -(3.0*(x0[1]-x1[1])+tau*(2*x0[4]+x1[4]))/t2;
    new_traj->y_coeff[2] = x0[4];
    new_traj->y_coeff[3] = x0[1];
    new_traj->z_coeff[0] = (2.0*(x0[2]-x1[2])+tau*(x0[5]+x1[5]))/t3;
    new_traj->z_coeff[1] = -(3.0*(x0[2]-x1[2])+tau*(2*x0[5]+x1[5]))/t2;
    new_traj->z_coeff[2] = x0[5];
    new_traj->z_coeff[3] = x0[2];
    
    new_traj->tau_from_parent = tau;
    new_traj->n_order = order;
  }
  else if (order == 5)
  {
    Vector3d p0, p1, v0, v1, a0, a1;
    calPVAFromCoeff(p0, v0, a0, ori_traj->x_coeff, ori_traj->y_coeff, ori_traj->z_coeff, ori_t_s, order);
    calPVAFromCoeff(p1, v1, a1, ori_traj->x_coeff, ori_traj->y_coeff, ori_traj->z_coeff, ori_t_e, order);
    
    double t2 = tau*tau;
    double t3 = t2*tau;
    double t4 = t3*tau;
    double t5 = t2*t3;
    new_traj->x_coeff[0] = -(12*p0[0] - 12*p1[0] + 6*tau*v0[0] + 6*tau*v1[0] + a0[0]*t2 - a1[0]*t2)/(2*t5);
    new_traj->x_coeff[1] = (30*p0[0] - 30*p1[0] + 16*tau*v0[0] + 14*tau*v1[0] + 3*a0[0]*t2 - 2*a1[0]*t2)/(2*t4);
    new_traj->x_coeff[2] = -(20*p0[0] - 20*p1[0] + 12*tau*v0[0] + 8*tau*v1[0] + 3*a0[0]*t2 - a1[0]*t2)/(2*t3);
    new_traj->x_coeff[3] = a0[0]/2;
    new_traj->x_coeff[4] = v0[0];
    new_traj->x_coeff[5] = p0[0];
    new_traj->y_coeff[0] = -(12*p0[1] - 12*p1[1] + 6*tau*v0[1] + 6*tau*v1[1] + a0[1]*t2 - a1[1]*t2)/(2*t5);
    new_traj->y_coeff[1] = (30*p0[1] - 30*p1[1] + 16*tau*v0[1] + 14*tau*v1[1] + 3*a0[1]*t2 - 2*a1[1]*t2)/(2*t4);
    new_traj->y_coeff[2] = -(20*p0[1] - 20*p1[1] + 12*tau*v0[1] + 8*tau*v1[1] + 3*a0[1]*t2 - a1[1]*t2)/(2*t3);
    new_traj->y_coeff[3] = a0[1]/2;
    new_traj->y_coeff[4] = v0[1];
    new_traj->y_coeff[5] = p0[1];
    new_traj->z_coeff[0] = -(12*p0[2] - 12*p1[2] + 6*tau*v0[2] + 6*tau*v1[2] + a0[2]*t2 - a1[2]*t2)/(2*t5);
    new_traj->z_coeff[1] = (30*p0[2] - 30*p1[2] + 16*tau*v0[2] + 14*tau*v1[2] + 3*a0[2]*t2 - 2*a1[2]*t2)/(2*t4);
    new_traj->z_coeff[2] = -(20*p0[2] - 20*p1[2] + 12*tau*v0[2] + 8*tau*v1[2] + 3*a0[2]*t2 - a1[2]*t2)/(2*t3);
    new_traj->z_coeff[3] = a0[2]/2;
    new_traj->z_coeff[4] = v0[2];
    new_traj->z_coeff[5] = p0[2];
    
    new_traj->tau_from_parent = tau;
    new_traj->n_order = order;
  }
  else 
  {
    ROS_ERROR("wrong poly order");
  }
}

// void KRRTPlanner::getOptiSegs(double seg_time, vector<Eigen::Vector3d>& way_points, vector<double>& seg_times,
//                               vector<Eigen::Vector3d>& vels, vector<Eigen::Vector3d>& accs)
// {
//   int patched_path_node_num = patched_path_.size();
//   vector<double> stair_time;
//   double sum_time = 0.0;
//   for (int i=patched_path_node_num-2; i>=0; --i) 
//   {
//     sum_time += patched_path_[i]->tau_from_parent;
//     stair_time.push_back(sum_time);
//   }
//   
//   //start and middle way points, along with seg times
//   double t_now = 0.0;
//   for ( ; t_now<stair_time[patched_path_node_num-2]; t_now+=seg_time) 
//   {
//     int idx = 0;
//     while (t_now > stair_time[idx]) 
//       idx++;
//     double t_in_seg = t_now;
//     if (idx > 0)
//       t_in_seg = t_now - stair_time[idx-1];
//     int patch_idx = patched_path_node_num-2-idx;
//     Eigen::Vector3d way_point(0.0, 0.0, 0.0), vel(0.0, 0.0, 0.0), acc(0.0, 0.0, 0.0);
//     int order = patched_path_[patch_idx]->n_order;
//     calPVAFromCoeff(way_point, vel, acc, patched_path_[patch_idx]->x_coeff, 
//                     patched_path_[patch_idx]->y_coeff, patched_path_[patch_idx]->z_coeff, t_in_seg, order);
//     way_points.push_back(way_point);
//     vels.push_back(vel);
//     accs.push_back(acc);
//     if (t_now > 0)
//       seg_times.push_back(seg_time);
//   }
//   
//   //end way point, along with seg time
//   way_points.push_back(goal_node_->x.head(3));
//   vels.push_back(goal_node_->x.tail(3));
//   accs.push_back(Eigen::Vector3d(0,0,0));
//   seg_times.push_back(stair_time[patched_path_node_num-2] - t_now + seg_time);
//   
// }

void KRRTPlanner::getOptiSegs(vector<Eigen::Vector3d>& way_points, Eigen::VectorXd& seg_times,
                              vector<Eigen::Vector3d>& vels, vector<Eigen::Vector3d>& accs, Eigen::MatrixXd& coeff)
{
  int n = path_.size();
  coeff = Eigen::MatrixXd::Zero((n - 1) * 6, 3);
  seg_times.resize(n - 1);
  
  for (int i=n-2; i>=0; --i)
  {
    Eigen::Vector3d way_point(0.0, 0.0, 0.0), vel(0.0, 0.0, 0.0), acc(0.0, 0.0, 0.0);
    int patch_idx = i;
    int order = path_[patch_idx]->n_order;
    calPVAFromCoeff(way_point, vel, acc, path_[patch_idx]->x_coeff, 
                    path_[patch_idx]->y_coeff, path_[patch_idx]->z_coeff, 0.0, order);
    for(int j = 0; j <= order; j++)
    {
      coeff((n - 2 - i) * 6 + j, 0) = path_[patch_idx]->x_coeff[order-j];
      coeff((n - 2 - i) * 6 + j, 1) = path_[patch_idx]->y_coeff[order-j];
      coeff((n - 2 - i) * 6 + j, 2) = path_[patch_idx]->z_coeff[order-j];
    }
    way_points.push_back(way_point);
    vels.push_back(vel);
    // when start to move from hovering, the acc of the very first point 
    // of the first seg should acctually be 0, but according to the a(t) polynomial like calculating above, it's not.
    if (patch_idx == n-2 && vel.norm() < 1e-4)
      accs.push_back(Eigen::Vector3d(0.0,0.0,0.0));
    else
      accs.push_back(acc);
    seg_times[n - 2 - i] = path_[patch_idx]->tau_from_parent;
  }
  way_points.push_back(goal_node_->x.head(3));
  vels.push_back(goal_node_->x.tail(3));
  accs.push_back(Eigen::Vector3d(0.0,0.0,0.0));
}

void KRRTPlanner::adjustOptiSegs(vector< Vector3d >& way_points, 
                                 Eigen::VectorXd& seg_times, 
                                 vector< Vector3d >& vels, 
                                 vector< Vector3d >& accs, 
                                 Eigen::MatrixXd& coeff, 
                                 set<int>& fixed_pos_idx,
                                 vector<double>& collide_times,
                                 vector<Vector3d>& collide_pos,
                                 vector<int>& collide_seg_idx)
{
  //TODO add assert collide info size equality.
  int n_c_s = collide_seg_idx.size();
  if (n_c_s == 0)
    return;
  vector<int> inbetween_collide_inx;
  for (int i = 0; i< n_c_s; ++i)
  {
    int former_seg = seg_times.rows()-collide_seg_idx[i]-1;
    if (collide_times[i] < 0.1)
    {
      inbetween_collide_inx.push_back(i);
      if (fixed_pos_idx.find(former_seg - 1) != fixed_pos_idx.end())
      {
        fixed_pos_idx.insert(former_seg - 1);
      }
    } 
    else if (seg_times(former_seg) - collide_times[i] < 0.1)
    {
      inbetween_collide_inx.push_back(i);
      if (fixed_pos_idx.find(former_seg) != fixed_pos_idx.end())
      {
        fixed_pos_idx.insert(former_seg);
      }
    }
  }
  int shift = 0;
  for (int i=0; i<inbetween_collide_inx.size(); ++i)
  {
    vector<double>::iterator it_t = collide_times.begin();
    collide_times.erase(it_t + inbetween_collide_inx[i] - shift);
    vector<int>::iterator it_i = collide_seg_idx.begin();
    collide_seg_idx.erase(it_i + inbetween_collide_inx[i] - shift);
    vector<Vector3d>::iterator it_p = collide_pos.begin();
    collide_pos.erase(it_p + inbetween_collide_inx[i] - shift);
    shift++;
  }
  n_c_s = collide_seg_idx.size();
  if (n_c_s == 0)
    return;
  for (int i = 0; i< n_c_s; ++i)
  {
    double t_in_traj = 0.0;
    int former_seg = 0;
    for (; former_seg < seg_times.rows()-collide_seg_idx[i]-1; former_seg++) {
      t_in_traj += seg_times(former_seg);
    }
    t_in_traj += collide_times[i];
    Vector3d pos_ori, vel_ori, acc_ori;
    calPVAFromTraj(pos_ori, vel_ori, acc_ori, t_in_traj, path_);
    //TODO check shifted pos collision
    Vector3d curr_colli_pos = collide_pos[i];
    Vector3d shift_pos = pos_ori + (pos_ori - curr_colli_pos)/((pos_ori - curr_colli_pos)).norm() * 0.2;
    shift_pos = getShiftedPos(pos_ori, curr_colli_pos - pos_ori);
//     ROS_INFO_STREAM("t_in_traj: " << t_in_traj);
//     ROS_INFO_STREAM("pos_ori: " << pos_ori.transpose());
//     ROS_INFO_STREAM("curr_colli_pos: " << curr_colli_pos.transpose());
//     ROS_INFO_STREAM("shift_pos: " << shift_pos.transpose());
    vector<Vector3d>::iterator it;
    int last_ite_path_seg_num = seg_times.rows();
    int last_ite_path_idx = collide_seg_idx[i];
    it = way_points.begin();
    way_points.insert(it + last_ite_path_seg_num - last_ite_path_idx + i, shift_pos);
    it = vels.begin();
    vels.insert(it + last_ite_path_seg_num - last_ite_path_idx + i, vel_ori);
    it = accs.begin();
    accs.insert(it + last_ite_path_seg_num - last_ite_path_idx + i, acc_ori);
  }
  
  Eigen::VectorXd ori_seg_times = seg_times;
  Eigen::MatrixXd ori_coeff = coeff;
  int ori_seg_num = ori_seg_times.rows();
  seg_times.resize(ori_seg_num + n_c_s);
  coeff = Eigen::MatrixXd::Zero((ori_seg_num + n_c_s) * 6, 3);
  int new_seg_idx = 0, ori_seg_idx = 0;
  RRTNodePtr cut_node = new RRTNode;
  RRTNodePtr ori_node = new RRTNode;
  while(new_seg_idx < ori_seg_num - 1 - collide_seg_idx[0])
  {
    coeff.block(new_seg_idx * 6, 0, 6, 3) = ori_coeff.block(ori_seg_idx * 6, 0, 6, 3);
    seg_times(new_seg_idx++) = ori_seg_times(ori_seg_idx++);
  }
  
  for (int i = 0; i < n_c_s - 1; ++i)
  {
    ori_node->n_order = 5;
    for(int ord = 0; ord <= 5; ord++)
    {
      ori_node->x_coeff[5 - ord] = ori_coeff(ori_seg_idx * 6 + ord, 0);
      ori_node->y_coeff[5 - ord] = ori_coeff(ori_seg_idx * 6 + ord, 1);
      ori_node->z_coeff[5 - ord] = ori_coeff(ori_seg_idx * 6 + ord, 2);
    }
//     cout <<"cut time: 0.0, " << collide_times[i] << endl;
    cutTraj(cut_node, ori_node, 0.0, collide_times[i]);
    for(int ord = 0; ord <= 5; ord++)
    {
      coeff(new_seg_idx * 6 + ord, 0) = cut_node->x_coeff[5 - ord];
      coeff(new_seg_idx * 6 + ord, 1) = cut_node->y_coeff[5 - ord];
      coeff(new_seg_idx * 6 + ord, 2) = cut_node->z_coeff[5 - ord];
    }
    for (auto it=fixed_pos_idx.begin(); it!=fixed_pos_idx.end(); ++it)
    {
      if (*it >= new_seg_idx)
      {
        fixed_pos_idx.erase(*it);
        fixed_pos_idx.insert((*it) + 1);
      }
    }
    fixed_pos_idx.insert(new_seg_idx);
    seg_times(new_seg_idx++) = collide_times[i];
    
//     cout <<"cut time: " << collide_times[i] << ", " << ori_seg_times(ori_seg_idx) << endl;
    cutTraj(cut_node, ori_node, collide_times[i], ori_seg_times(ori_seg_idx));
    for(int ord = 0; ord <= 5; ord++)
    {
      coeff(new_seg_idx * 6 + ord, 0) = cut_node->x_coeff[5 - ord];
      coeff(new_seg_idx * 6 + ord, 1) = cut_node->y_coeff[5 - ord];
      coeff(new_seg_idx * 6 + ord, 2) = cut_node->z_coeff[5 - ord];
    }
    seg_times(new_seg_idx++) = ori_seg_times(ori_seg_idx++) - collide_times[i];
    
    int cur_colli_idx = collide_seg_idx[i];
    int next_colli_idx = collide_seg_idx[i + 1];
    for (int j = cur_colli_idx - 1; j > next_colli_idx; --j)
    {
      coeff.block(new_seg_idx * 6, 0, 6, 3) = ori_coeff.block(ori_seg_idx * 6, 0, 6, 3);
      seg_times(new_seg_idx++) = ori_seg_times(ori_seg_idx++);
    }
  }
  
  ori_node->n_order = 5;
  for(int ord = 0; ord <= 5; ord++)
  {
    ori_node->x_coeff[5 - ord] = ori_coeff(ori_seg_idx * 6 + ord, 0);
    ori_node->y_coeff[5 - ord] = ori_coeff(ori_seg_idx * 6 + ord, 1);
    ori_node->z_coeff[5 - ord] = ori_coeff(ori_seg_idx * 6 + ord, 2);
  }
//   cout <<"cut time: 0.0, " << collide_times[n_c_s - 1] << endl;
  cutTraj(cut_node, ori_node, 0.0, collide_times[n_c_s - 1]);
  for(int ord = 0; ord <= 5; ord++)
  {
    coeff(new_seg_idx * 6 + ord, 0) = cut_node->x_coeff[5 - ord];
    coeff(new_seg_idx * 6 + ord, 1) = cut_node->y_coeff[5 - ord];
    coeff(new_seg_idx * 6 + ord, 2) = cut_node->z_coeff[5 - ord];
  }
  for (auto it=fixed_pos_idx.begin(); it!=fixed_pos_idx.end(); ++it)
  {
    if (*it >= new_seg_idx)
    {
      fixed_pos_idx.erase(*it);
      fixed_pos_idx.insert((*it) + 1);
    }
  }
  fixed_pos_idx.insert(new_seg_idx);
  seg_times(new_seg_idx++) = collide_times[n_c_s - 1];
  
//   cout <<"cut time: " << collide_times[n_c_s - 1] << ", " << ori_seg_times(ori_seg_idx) << endl;
  cutTraj(cut_node, ori_node, collide_times[n_c_s - 1], ori_seg_times(ori_seg_idx));
  for(int ord = 0; ord <= 5; ord++)
  {
    coeff(new_seg_idx * 6 + ord, 0) = cut_node->x_coeff[5 - ord];
    coeff(new_seg_idx * 6 + ord, 1) = cut_node->y_coeff[5 - ord];
    coeff(new_seg_idx * 6 + ord, 2) = cut_node->z_coeff[5 - ord];
  }
  seg_times(new_seg_idx++) = ori_seg_times(ori_seg_idx++) - collide_times[n_c_s - 1];
  
  while(new_seg_idx < ori_seg_num + n_c_s)
  {
    coeff.block(new_seg_idx * 6, 0, 6, 3) = ori_coeff.block(ori_seg_idx * 6, 0, 6, 3);
    seg_times(new_seg_idx++) = ori_seg_times(ori_seg_idx++);
  }
  delete cut_node;
  delete ori_node;
}

void KRRTPlanner::getDenseWp(vector<Vector3d>& wps, vector<Vector3d>& grads, double dt)
{
  int seg_num = path_.size() - 1;
  for (int i = seg_num - 1; i >= 0; --i)
  {
    double t = 0.0;
    double dura = path_[i]->tau_from_parent;
    int order = path_[i]->n_order;
    Vector3d pos, vel, acc, grad;
    for (; t < dura; t += dt)
    {
      calPVAFromCoeff(pos, vel, acc, path_[i]->x_coeff, path_[i]->y_coeff, path_[i]->z_coeff, t, order);
      wps.push_back(pos);
      Eigen::Vector2d vel_hor, cw_radius_vec;
      grad[2] = 0.0;
      vel_hor[0] = vel[0];
      vel_hor[1] = vel[1];
      if (vel_hor.norm() < 1e-4)
      {
        vel_hor[0] = 1;
        vel_hor[1] = 1;
      }
      cw_radius_vec = rotate90Clockwise2d(vel_hor);
      grad[0] = cw_radius_vec[0];
      grad[1] = cw_radius_vec[1];
      grads.push_back(grad);
    }
  }
}

Vector3d KRRTPlanner::getShiftedPos(Vector3d ori_p, Vector3d grad)
{
  Vector3d shifetd_p(ori_p);
  RayCaster raycaster;
  Vector3d start, end, ray_pt;
  Vector3d center, l_end, r_end;
  center = ori_p;
  grad.normalize();
  l_end = center + grad * clear_radius_;
  start = center / resolution_;
  end = l_end / resolution_;
  raycaster.setInput(start, end);
  bool l_clear = true;
  double len_l_e2o = 0;
  Vector3d l_free_p = l_end;
  while (raycaster.step(ray_pt))
  {
    Eigen::Vector3d tmp = (ray_pt) * resolution_;
    tmp += Vector3d(resolution_/2.0, resolution_/2.0, resolution_/2.0);
//       cout << "left-tmp: " << tmp.transpose() << "occ: " << occ_map_->getVoxelState(tmp) << endl;
    if (occ_map_->getVoxelState(tmp) != 0)
    {
      l_free_p = tmp;
      len_l_e2o = (tmp - l_end).norm();
      l_clear = false;
      break;
    }
  }
  
  r_end = center - grad * (clear_radius_ + len_l_e2o);
  double len_r_e2o = 0;
  Vector3d r_free_p = r_end;
  end = r_end / resolution_;
  raycaster.setInput(start, end);
  bool r_clear = true;
  while (raycaster.step(ray_pt))
  {
    Eigen::Vector3d tmp = (ray_pt) * resolution_;
    tmp += Vector3d(resolution_/2.0, resolution_/2.0, resolution_/2.0);
//       cout << "right-tmp: " << tmp.transpose() << "occ: " << occ_map_->getVoxelState(tmp) << endl;
    if (occ_map_->getVoxelState(tmp) != 0)
    {
      len_r_e2o = (tmp - r_end).norm();
      r_free_p = tmp;
      r_clear = false;
      break;
    }
  }
  
  if (!l_clear)
  {
    shifetd_p = (l_free_p + r_free_p) /2;
//       cout << "left not clear, wp: " << i << ": " << wp_[i].transpose() << endl;
  }
  else if (!r_clear)
  {
    Vector3d ll_end =  center + grad * (clear_radius_ + len_r_e2o);
    Vector3d ll_free_p = ll_end;
    start = l_end / resolution_;
    end = ll_end / resolution_;
    raycaster.setInput(start, end);
    while (raycaster.step(ray_pt))
    {
      Eigen::Vector3d tmp = (ray_pt) * resolution_;
      tmp += Vector3d(resolution_/2.0, resolution_/2.0, resolution_/2.0);
      if (occ_map_->getVoxelState(tmp) != 0)
      {
        ll_free_p = tmp;
        break;
      }
    }
    shifetd_p = (ll_free_p + r_free_p) /2;
//       cout << "left&right not clear, wp: " << i << ": " << wp_[i].transpose() << endl;
  }
  return shifetd_p;
}

bool KRRTPlanner::getCollisions(const RRTNodeVector& path_node_g2s, 
                                vector< double >& collide_times, 
                                vector< Vector3d >& collide_pos, 
                                vector< int >& collide_seg_idx)
{
  collide_times.clear();
  collide_pos.clear();
  collide_seg_idx.clear();
  bool res = false;
  double dt(0.01);
  int seg_num = path_node_g2s.size() - 1;
  for (int i = seg_num - 1; i >= 0; --i)
  {
    bool curr_seg_collide(false);
    double t = 0.0;
    double dura = path_node_g2s[i].tau_from_parent;
    int order = path_node_g2s[i].n_order;
    Vector3d pos, vel, acc;
    for (; t < dura; t += dt)
    {
      calPVAFromCoeff(pos, vel, acc, path_node_g2s[i].x_coeff, path_node_g2s[i].y_coeff, path_node_g2s[i].z_coeff, t, order);
      vector<Vector3d> line_grids;
      getCheckPos(pos, vel, acc, line_grids, hor_safe_radius_, ver_safe_radius_);
      for (const auto& grid : line_grids)
      {
        if (occ_map_->getVoxelState(grid) != 0) 
        {
//           if (t > 0.1 && dura - t > 0.1)
//           {
            collide_seg_idx.push_back(i);
            collide_pos.push_back(pos);
            collide_times.push_back(t);
            curr_seg_collide = true;
            res = true;
//           }
          break;
        }
      }
      if (curr_seg_collide)
        break;
    }
  }
  return res;
}

void KRRTPlanner::calPVAFromCoeff(Vector3d& pos, Vector3d& vel, Vector3d& acc, 
                                         const double* x_coeff, const double* y_coeff, const double* z_coeff, 
                                         double t, const int& order)
{
  pos[0] = calPosFromCoeff(t, x_coeff, order);
  pos[1] = calPosFromCoeff(t, y_coeff, order);
  pos[2] = calPosFromCoeff(t, z_coeff, order);
  vel[0] = calVelFromCoeff(t, x_coeff, order);
  vel[1] = calVelFromCoeff(t, y_coeff, order);
  vel[2] = calVelFromCoeff(t, z_coeff, order);
  acc[0] = calAccFromCoeff(t, x_coeff, order);
  acc[1] = calAccFromCoeff(t, y_coeff, order);
  acc[2] = calAccFromCoeff(t, z_coeff, order);
}

inline void KRRTPlanner::calPVAFromTraj(Vector3d& pos, Vector3d& vel, Vector3d& acc, 
                                        double t, const RRTNodePtrVector& path)
{
  int seg_num = path.size() - 1;
  if (seg_num < 1)
  {
    ROS_WARN("not a path");
    return;
  }
  int seg_idx = seg_num - 1;
  double t_remain = t;
  for (; seg_idx >= 0; --seg_idx)
  {
    t_remain -= path[seg_idx]->tau_from_parent;
    if (t_remain <= 0) 
    {
      t_remain += path[seg_idx]->tau_from_parent;
      break;
    }
  }
  if (seg_idx < 0 && t_remain > path[0]->tau_from_parent + 1e-3)
  {
    ROS_ERROR("t > traj duration");
    return;
  }
  int order = path[seg_idx]->n_order;
  pos[0] = calPosFromCoeff(t_remain, path[seg_idx]->x_coeff, order);
  pos[1] = calPosFromCoeff(t_remain, path[seg_idx]->y_coeff, order);
  pos[2] = calPosFromCoeff(t_remain, path[seg_idx]->z_coeff, order);
  vel[0] = calVelFromCoeff(t_remain, path[seg_idx]->x_coeff, order);
  vel[1] = calVelFromCoeff(t_remain, path[seg_idx]->y_coeff, order);
  vel[2] = calVelFromCoeff(t_remain, path[seg_idx]->z_coeff, order);
  acc[0] = calAccFromCoeff(t_remain, path[seg_idx]->x_coeff, order);
  acc[1] = calAccFromCoeff(t_remain, path[seg_idx]->y_coeff, order);
  acc[2] = calAccFromCoeff(t_remain, path[seg_idx]->z_coeff, order);
}

inline double KRRTPlanner::calPosFromCoeff(double t, const double* coeff, const int& order)
{
  double f = coeff[0];
  for (int i=0; i<order; ++i) 
  {
    f = t * f + coeff[i+1];
  }
  return f;
}

inline double KRRTPlanner::calVelFromCoeff(double t, const double* coeff, const int& order)
{
  double f = order * coeff[0];
  for (int i=0; i<order-1; ++i) 
  {
    f = t * f + (order - i - 1) * coeff[i+1];
  }
  return f;
}

inline double KRRTPlanner::calAccFromCoeff(double t, const double* coeff, const int& order)
{
  double f = order * (order-1) * coeff[0];
  for (int i=0; i<order - 2; ++i) 
  {
    f = t * f + (order - i - 1) * (order - i - 2) * coeff[i+1];
  }
  return f;
}

inline double KRRTPlanner::calJerkFromCoeff(double t, const double* coeff, const int& order)
{
  double f = order * (order-1) * (order-2) * coeff[0];
  for (int i=0; i<order - 3; ++i) 
  {
    f = t * f + (order - i - 1) * (order - i - 2) * (order - i - 3) * coeff[i+1];
  }
  return f;
}

// Provides solutions to the equation a*x^4 + b*x^3 + c*x^2 + d*x + e = 0 using
// Ferrari's method to reduce to problem to a depressed cubic.
vector<double> KRRTPlanner::quartic(double a, double b, double c, double d, double e)
{
  vector<double> dts;

  double a3 = b / a;
  double a2 = c / a;
  double a1 = d / a;
  double a0 = e / a;

  vector<double> ys = cubic(1, -a2, a1 * a3 - 4 * a0, 4 * a2 * a0 - a1 * a1 - a3 * a3 * a0);
  double y1 = ys.front();
  double r = a3 * a3 / 4 - a2 + y1;
  if (r < 0)
    return dts;

  double R = sqrt(r);
  double D, E;
  if (R != 0)
  {
    D = sqrt(0.75 * a3 * a3 - R * R - 2 * a2 + 0.25 * (4 * a3 * a2 - 8 * a1 - a3 * a3 * a3) / R);
    E = sqrt(0.75 * a3 * a3 - R * R - 2 * a2 - 0.25 * (4 * a3 * a2 - 8 * a1 - a3 * a3 * a3) / R);
  }
  else
  {
    D = sqrt(0.75 * a3 * a3 - 2 * a2 + 2 * sqrt(y1 * y1 - 4 * a0));
    E = sqrt(0.75 * a3 * a3 - 2 * a2 - 2 * sqrt(y1 * y1 - 4 * a0));
  }

  if (!std::isnan(D))
  {
    dts.push_back(-a3 / 4 + R / 2 + D / 2);
    dts.push_back(-a3 / 4 + R / 2 - D / 2);
  }
  if (!std::isnan(E))
  {
    dts.push_back(-a3 / 4 - R / 2 + E / 2);
    dts.push_back(-a3 / 4 - R / 2 - E / 2);
  }

  return dts;
}

vector<double> KRRTPlanner::cubic(double a, double b, double c, double d)
{
  vector<double> dts;

  double a2 = b / a;
  double a1 = c / a;
  double a0 = d / a;

  double Q = (3 * a1 - a2 * a2) / 9;
  double R = (9 * a1 * a2 - 27 * a0 - 2 * a2 * a2 * a2) / 54;
  double D = Q * Q * Q + R * R;
  if (D > 0)
  {
    double S = std::cbrt(R + sqrt(D));
    double T = std::cbrt(R - sqrt(D));
    dts.push_back(-a2 / 3 + (S + T));
    return dts;
  }
  else if (D == 0)
  {
    double S = std::cbrt(R);
    dts.push_back(-a2 / 3 + S + S);
    dts.push_back(-a2 / 3 - S);
    return dts;
  }
  else
  {
    double theta = acos(R / sqrt(-Q * Q * Q));
    dts.push_back(2 * sqrt(-Q) * cos(theta / 3) - a2 / 3);
    dts.push_back(2 * sqrt(-Q) * cos((theta + 2 * M_PI) / 3) - a2 / 3);
    dts.push_back(2 * sqrt(-Q) * cos((theta + 4 * M_PI) / 3) - a2 / 3);
    return dts;
  }
}

inline Vector3d KRRTPlanner::calPointByDir(const Vector3d& init_p, 
                                           const Vector3d& dir, double len)
{
  return init_p + dir/dir.norm()*len;
}


/* for kino s star evaluate */
Eigen::MatrixXd KRRTPlanner::getSamples(double& ts, int& K)
{
  RRTNodePtr node = path_.front();
  double T_sum = node->tau_from_start;
  cout << "final time:" << T_sum << endl;

  /* ---------- init for sampling ---------- */
  K = floor(T_sum / ts);
  ts = T_sum / (K + 1);
  
  Eigen::VectorXd sx(K + 2), sy(K + 2), sz(K + 2);
  int sample_num = 0;

  double t;
  t = node->tau_from_parent;

  for (double ti = T_sum; ti > -1e-5; ti -= ts)
  {
    sx(sample_num) = calPosFromCoeff(t, node->x_coeff, node->n_order);
    sy(sample_num) = calPosFromCoeff(t, node->y_coeff, node->n_order);
    sz(sample_num) = calPosFromCoeff(t, node->z_coeff, node->n_order);
    ++sample_num;

    t -= ts;
    // cout << "t: " << t << ", t acc: " << T_accumulate << endl;
    if (t < -1e-5 && node->parent != NULL)
    {
      node = node->parent;
      t += node->tau_from_parent;
    }
  }
  
  /* ---------- return samples ---------- */
  Eigen::MatrixXd samples(3, K + 5);
  samples.block(0, 0, 1, K + 2) = sx.reverse().transpose();
  samples.block(1, 0, 1, K + 2) = sy.reverse().transpose();
  samples.block(2, 0, 1, K + 2) = sz.reverse().transpose();
  samples.col(K + 2) = start_node_->x.tail(3);
  samples.col(K + 3) = goal_node_->x.tail(3);
  samples.col(K + 4) = Eigen::Vector3d(0.0,0.0,0.0);

  return samples;
}

double KRRTPlanner::PVHeu(const State& x_init, const State& x_goal)
{
  // Eigen::Matrix4d T_NED2map = occ_map_->getTNED2map();
  double p_heu = (x_init.head(3) - x_goal.head(3)).norm();
  Eigen::Vector3d v_init = x_init.tail(3), v_goal = x_goal.tail(3);
  double v_heu = v_init.dot(v_goal); 
  double heu = p_heu - v_heu;
  // cout << "init s: " << (T_NED2map.block<3,3>(0,0) * x_init.head(3) + T_NED2map.block<3,1>(0,3)).transpose() << endl;
  // cout << "gaol s: " << (T_NED2map.block<3,3>(0,0) * x_goal.head(3) + T_NED2map.block<3,1>(0,3)).transpose() << endl;
  // cout << "p_heu: " << p_heu << ", v_heu: " << v_heu << ", heu: " << heu << endl;
  
  return heu; 
  // return p_heu;
}

} //namespace fast_planner
