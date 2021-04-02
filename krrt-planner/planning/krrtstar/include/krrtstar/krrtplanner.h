#ifndef _KRRTPLANNER_H_
#define _KRRTPLANNER_H_

#include "utils.h"
#include "kdtree.h"
#include "visualize_rviz.h"
#include "occ_grid/occ_map.h"
#include "topo_prm.h"

#include <random>
#include <vector>
#include <stack>

//#define DEBUG
#define TIMING

using std::vector;
using std::stack;
using Eigen::Vector3d;
using Eigen::Vector2d;
using Eigen::Vector3i;
using Eigen::Vector2i;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::Matrix2d;

namespace fast_planner
{
class KRRTPlanner{
public:
  KRRTPlanner();
  KRRTPlanner(ros::NodeHandle& nh);
  ~KRRTPlanner();
  
  // api
  int plan(Vector3d start_pos, Vector3d start_vel, Vector3d start_acc,  
           Vector3d end_pos, Vector3d end_vel, Vector3d end_acc, 
           double search_time, bool bidirection);
  int getPath(RRTNodeVector& path_node_g2s, int path_type = 1);
  int getBypass(double time, RRTNodeVector& path_node_g2s);
  int getTree(RRTNodePtrVector& tree);
  void init(ros::NodeHandle& nh);
  void setEnv(const OccMap::Ptr& env);
  void setTopoFinder(const TopologyPRM::Ptr& topo_prm);
  /*
   * check curruent traj pos colide with curruent updated occ info, 
   * t is the time in second from curr traj start time.
   */
  double getTrajDura();
  bool updateColideInfo(const RRTNodeVector& node_vector, double check_start_time, double check_dura, Eigen::Vector3d& collide_pos, double& t_safe_dura);
  bool checkOptimizedTraj(const RRTNodeVector& node_vector);
//   void getOptiSegs(double seg_time, vector<Eigen::Vector3d>& way_points, vector<double>& seg_times,
//                               vector<Eigen::Vector3d>& vels, vector<Eigen::Vector3d>& accs);
  void getOptiSegs(vector<Eigen::Vector3d>& way_points, Eigen::VectorXd& seg_times,
                   vector<Eigen::Vector3d>& vels, vector<Eigen::Vector3d>& accs, 
                   Eigen::MatrixXd& coeff);
  void adjustOptiSegs(vector<Eigen::Vector3d>& way_points, Eigen::VectorXd& seg_times,
                      vector<Eigen::Vector3d>& vels, vector<Eigen::Vector3d>& accs, 
                      Eigen::MatrixXd& coeff, set<int>& fixed_pos_idx,
                      vector<double>& collide_times,
                      vector<Vector3d>& collide_pos,
                      vector< int >& collide_seg_idx);
  void getDenseWp(vector<Vector3d>& wps, vector<Vector3d>& grads, double dt);
  void getStateAndControl(RRTNodeVector node_vector, vector<State>* vis_x, 
                             vector<Control>* vis_u);
  void calPVAFromCoeff(Vector3d& pos, Vector3d& vel, Vector3d& acc, 
                              const double* x_coeff, const double* y_coeff, 
                              const double* z_coeff, 
                              double t, const int& order);
  inline void calPVAFromTraj(Vector3d& pos, Vector3d& vel, Vector3d& acc, 
                                        double t, const RRTNodePtrVector& path);
  
  //evaluate
  void getTrajAttributes(const RRTNodeVector& node_vector, double& traj_duration, double& ctrl_cost, double& jerk_itg, double& traj_length, int& seg_nums);
  void getTrajAttributes(const RRTNodePtrVector& node_ptr_vector, double& traj_duration, double& ctrl_cost, double& traj_length, int& seg_nums);

  typedef shared_ptr<KRRTPlanner> KRRTPlannerPtr;
    
private:
  void reset();
  void findSamplingSpace(const State& x_init, const State& x_goal, 
                          const vector<pair<State, State>>& segs,
                          vector<pair<Vector3d, Vector3d>>& all_corners);
  void setupRandomSampling(const State& x_init, const State& x_goal, 
                          const vector<pair<Vector3d, Vector3d>>& all_corners,
                          vector<Vector3d>& unit_tracks,
                          vector<Vector3d>& p_head,
                          vector<Vector3d>& tracks, 
                          vector<Vector3d>& rotated_unit_tracks);
  void setupRandomSampling(const Vector3d& x0, const Vector3d& x1, 
                          const vector<vector<Vector3d>>& paths,
                          vector<Vector3d>& unit_tracks,
                          vector<Vector3d>& p_head,
                          vector<Vector3d>& tracks, 
                          vector<Vector3d>& rotated_unit_tracks);
  bool samplingOnce(int i, State& rand_state,
                    const vector<Vector3d>& unit_tracks,
                    const vector<Vector3d>& p_head,
                    const vector<Vector3d>& tracks, 
                    const vector<Vector3d>& rotated_unit_tracks);
  int rrtStar(const State& x_init, const State& x_final, 
               const Vector3d& u_init, const Vector3d& u_final, int n, 
               double radius, const bool rewire, const float epsilon);
  int rrtStarConnect(const State& x_init, const State& x_final, 
            const Vector3d& u_init, const Vector3d& u_final, int n, 
            double radius, const bool rewire);
  void mirror_coeff(double *new_coeff, double *ori_coeff, double t);
  void calc_backward_reachable_bounds(const State& init_state, 
                                      const double& radius, BOUNDS& bounds);
  void calc_forward_reachable_bounds(const State& init_state, 
                                      const double& radius, BOUNDS& bounds);
  double applyHeuristics(const State& x_init, const State& x_final);
  double dist(const State& x0, const State& x1);
  bool isClose(const State& x0, const State& x1);
  double PVHeu(const State& x_init, const State& x_goal);
  void getWayPointFromTraj(double time, Eigen::Vector3d pos){};
  Vector3d getShiftedPos(Vector3d ori_p, Vector3d grad);
  
  /*
    * fixed final state, free final time, use Gremian
    */
  bool connect(const State &x0, const State& x1, const double radius, 
                double& cost, double& tau, double* actual_deltaT, 
                vector<State>* vis_x, vector<Control>* vis_u);
  bool checkCost(const State& x0, const State& x1, double radius, 
                  double& cost, double& tau, State& d_tau);
  bool checkPath(const State& x0, const State& x1, const double tau, 
                  const State& d_tau, double *actual_deltaT, 
                  vector<State>* vis_x, vector<Control>* vis_u);
  void computeCostAndTime(const State& x0, const State& x1, 
                          double& cost, double& tau);
  void calculateStateAndControl(const State& x0, const State& x1,
                                const double tau, const State& d_tau,
                                double t, State& x, Control& u);
  bool validateStateAndControl(const State& x, const Control& u);
  
  /*
   * use Qingjiushao's method to calculate the value of f(t) = p0 + p1*t^1 + ... + pn*t^n
   * coeff[i]: pn, pn-1, pn-2, ..., p1, p0
   * order: n
   */
  inline double calPosFromCoeff(double t, const double* coeff, const int& order);
  inline double calVelFromCoeff(double t, const double* coeff, const int& order);
  inline double calAccFromCoeff(double t, const double* coeff, const int& order);
  inline double calJerkFromCoeff(double t, const double* coeff, const int& order);
  
  /*
    * fixed final state, free final time, use calculus of variations
    */
  bool HtfConnect(const State& x0, const State& x1, double radius, double& cost, 
                double& tau, double *x_coeff, double *y_coeff, double *z_coeff);
  bool checkCost(const State& x0, const State& x1, double radius, 
                  double& cost, double *x_coeff, double *y_coeff, 
                  double *z_coeff, double& tau);
  bool checkPath(const State& x0, const State& x1, double *x_coeff, 
                  double *y_coeff, double *z_coeff, double tau);
  bool checkVelAcc(const State& x0, const State& x1, double *x_coeff, 
                   double *y_coeff, double *z_coeff, double tau);
  
  void fillPath(RRTNodePtr goal_leaf, RRTNodePtrVector& path);
  vector<double> cubic(double a, double b, double c, double d);
  vector<double> quartic(double a, double b, double c, double d, double e);
  /*
    * given a point 'init_p', return the point 'len' away from it in 
    * direction 'dir'.
    */
  Vector3d calPointByDir(const Vector3d& init_p, 
                                const Vector3d& dir, double len);
  void rotateClockwise2d(double theta, Vector2d& v) {
      Matrix2d r_m;
      r_m << cos(theta), sin(theta), 
            -sin(theta), cos(theta); 
      v = r_m*v;
  };
  void rotateClockwise3d(double theta, Vector3d& v) {
      Matrix3d r_m;
      r_m << cos(theta),-sin(theta), 0,
              sin(theta), cos(theta), 0,
              0, 0, 1;
      v = r_m*v;
  };
  Vector3d rotate90Clockwise3d(const Vector3d& v) {
      Matrix3d r_m;
      r_m << 0, 1, 0,
          -1, 0, 0,
          0, 0, 1;
      return r_m*v;
  };
  Vector3d rotate90AntiClockwise3d(const Vector3d& v) {
      Matrix3d r_m;
      r_m << 0,-1, 0,
          1, 0, 0,
          0, 0, 1;
      return r_m*v;
  };
  Vector2d rotate90Clockwise2d(const Vector2d& v) {
      Matrix2d r_m;
      r_m << 0, 1,
          -1, 0;
      return r_m*v;
  };
  Vector2d rotate90AntiClockwise2d(const Vector2d& v) {
      Matrix2d r_m;
      r_m << 0,-1,
          1, 0;
      return r_m*v;
  };
  
  //for vis&test
  bool computeBestCost(const VectorXd& x0, const VectorXd& x1, 
                       vector<pair<State, State>>& segs,
        double& cost, double& tau, vector<State>* vis_x, vector<Control>* vis_u, 
        double *x_coeff, double *y_coeff, double *z_coeff);
  void findAllStatesAndControlInAllTrunks(RRTNodePtr root, vector<State>* vis_x, 
                                          vector<Control>* vis_u);
  void findAllSaC(vector<State>* vis_x, vector<Control>* vis_u);
  void findTrajBackwardFromGoal(RRTNodePtr goal_leaf, vector<State>* vis_x, 
                                vector<Control>* vis_u);
  void getVisStateAndControl(RRTNodePtrVector node_ptr_vector, vector<State>* vis_x, 
                             vector<Control>* vis_u);
  void getVisTrajCovering(RRTNodePtrVector node_ptr_vector, 
                                        vector<Vector3d>& cover_grids);
  void getlineGrids(const Vector3d& s_p, const Vector3d& e_p, vector<Vector3d>& grids);
  void getCheckPos(const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, 
                   vector<Vector3d>& grids, double hor_radius, double ver_radius);
  double getTrajLength(RRTNodePtrVector node_ptr_vector, double d_t);
  bool validatePosSurround(const Vector3d& pos, const Vector3d& vel, const Vector3d& acc);
  bool validatePosSurround(const Vector3d& pos);
  bool validateVel(const Vector3d& vel);
  bool validateAcc(const Vector3d& acc);
  void getVisPoints(RRTNodePtr x1, vector< State >* vis_x, 
                    vector< Control >* vis_u);
  
  //for patch
  void patching(const RRTNodePtrVector& path, RRTNodePtrVector& patched_path, 
                          const Vector3d& u_init);
  bool patchTwoState(const Eigen::VectorXd& x0, const Eigen::VectorXd& x1, 
                     double& cost, double& tau, 
                     double *x_coeff, double *y_coeff, double *z_coeff);
  void cutTraj(RRTNodePtr& new_traj, const RRTNodePtr& ori_traj, double ori_t_s, double ori_t_e);
  Eigen::VectorXd getPatchState(const RRTNodePtr& node_ptr, double percent);
  bool checkPath(double *x_coeff, double *y_coeff, double *z_coeff, double t, int order);
  
  //random sampling
  mt19937_64 gen_;
  uniform_real_distribution<double> px_rand_, py_rand_, pz_rand_,
                                    vx_rand_, vy_rand_, vz_rand_;
                                    
  uniform_real_distribution<double> pos_mean_rand_, seg_rand_;
  normal_distribution<double> pos_hor_rand_;
  normal_distribution<double> pos_ver_rand_;
  normal_distribution<double> vel_hor_dir_rand_;
  normal_distribution<double> vel_ver_dir_rand_;
  normal_distribution<double> vel_mag_rand_;
  bool dim_three_;
  
  VisualRviz vis;
  double traj_duration_;
  RRTNodePtrVector start_tree_, goal_tree_; //pre allocated in Constructor
  std::vector<State> orphans_;
  RRTNodePtrVector path_; //initialized when finishing search, goal is path_[0], start is path_[n]
  RRTNodePtrVector path_on_first_search_; //allocated when first path found
  RRTNodePtrVector patched_path_; //initialized when patching
  double search_time_;
  int tree_node_nums_, valid_start_tree_node_nums_, valid_goal_tree_node_nums_, orphan_nums_;
  int path_node_nums_;
  RRTNodePtr start_node_, goal_node_, close_goal_node_, end_tree_goal_node_, end_tree_start_node_;
  bool allow_orphan_, allow_close_goal_, stop_after_first_traj_found_;
  vector<Vector3d> skeleton_;
  
  
  //params
  const double deltaT_ = 0.01;
  double radius_cost_between_two_states_;
  double rou_;
  double v_mag_sample_;
  double px_min_, px_max_, py_min_, py_max_, pz_min_, pz_max_;
  double vx_min_, vx_max_, vy_min_, vy_max_, vz_min_, vz_max_;
  double ax_min_, ax_max_, ay_min_, ay_max_, az_min_, az_max_;
  double c_[X_DIM];
  double resolution_;
  
  //environment
  fast_planner::OccMap::Ptr occ_map_;

  //topo finder
  TopologyPRM::Ptr topo_prm_;
  
public:
  double hor_safe_radius_, ver_safe_radius_;
  double replan_hor_safe_radius_, replan_ver_safe_radius_;
  double copter_diag_len_;
  double clear_radius_;
    //for kino a star evaluate
  Eigen::MatrixXd getSamples(double& ts, int& K);
  bool getCollisions(const RRTNodeVector& path_node_g2s, vector< double >& collide_times, 
                     vector< Vector3d >& collide_pos, vector< int >& collide_seg_idx);
  void getVisTrajCovering(const RRTNodeVector& node_ptr_vector, 
                                        vector<Vector3d>& cover_grids);
    
}; //class KRRTPlanner

} ///namespace fast_planner

#endif //_KRRTPLANNER_H_
