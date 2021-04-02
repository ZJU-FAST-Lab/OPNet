#ifndef _TRAJ_OPTIMIZER_H_
#define _TRAJ_OPTIMIZER_H_

#include <Eigen/Eigen>
#include <ros/ros.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace fast_planner 
{
class TrajOptimizer
{
public:
  TrajOptimizer(ros::NodeHandle &node);
  TrajOptimizer();
  void getCoefficient(Eigen::MatrixXd &coeff);
  void getSegmentTime(Eigen::VectorXd &T);
  void setWayPointsAndTime(const vector<Eigen::Vector3d> &way_points,
                           const vector<Eigen::Vector3d> &vel, 
                           const vector<Eigen::Vector3d> &acc, 
                           const Eigen::VectorXd &time, 
                           const Eigen::MatrixXd &coeff);
  void setWayPointsAndTime(const vector<Eigen::Vector3d>& way_points, 
                           const vector<Eigen::Vector3d>& vel, 
                           const vector<Eigen::Vector3d>& acc, 
                           const Eigen::VectorXd &time);
  void tryQPCloseForm(double percent_of_close, double percent_of_acc);
  void tryQPCloseForm();
  typedef shared_ptr<TrajOptimizer> Ptr;
private:
  /** coefficient of polynomials*/
  Eigen::MatrixXd coeff_;  
  Eigen::MatrixXd coeff0_;
  
  /** way points info, from start point to end point **/
  std::set<int> fixed_pos_;
  Eigen::MatrixXd path_;
  Eigen::MatrixXd vel_way_points_; 
  Eigen::MatrixXd acc_way_points_; 
  Eigen::VectorXd time_;
  size_t m_; //segments number
  size_t n_; //fixed_pos number (exclude start and goal)
  
  /** important matrix and  variables*/
  Eigen::MatrixXd Ct_;
  Eigen::MatrixXd Z_, Zp_;
  Eigen::MatrixXd A_inv_multiply_Ct_;
  
  Eigen::MatrixXd A_;
  Eigen::MatrixXd A_inv_;
  Eigen::MatrixXd Q_smooth_, Q_close_, Q_acc_;
  Eigen::MatrixXd C_;
  Eigen::MatrixXd L_;
  Eigen::MatrixXd R_;
  Eigen::MatrixXd Rff_;
  Eigen::MatrixXd Rpp_;
  Eigen::MatrixXd Rpf_;
  Eigen::MatrixXd Rfp_;
  Eigen::VectorXd Dx_, Dy_, Dz_;
  
  Eigen::MatrixXd V_;
  Eigen::MatrixXd Df_;
  Eigen::MatrixXd Dp_;
  int num_dp;
  int num_df;
  int num_point;

  inline void getPositionFromCoeff(Eigen::Vector3d &pos, const Eigen::MatrixXd &coeff, const int &index,
                            const double &time) const;
  inline void getVelocityFromCoeff(Eigen::Vector3d &vel, const Eigen::MatrixXd &coeff, const int &index,
                            const double &time) const;
  inline void getAccelerationFromCoeff(Eigen::Vector3d &acc, const Eigen::MatrixXd &coeff,
                                const int &index, const double &time) const;
  void calMatrixA();
  void calMatrixCandMatrixZ(int type);
  void calMatrixC();
  void calMatrixQ_smooth(int type);
  void calMatrixQ_close();
  void calMatrixQ_acc_consistent();
  double w_smooth_, w_close_, w_acc_;
  enum QSmoothType
  {
    MINIMUM_ACC, 
    MINIMUM_JERK, 
    MINIMUM_SNAP
  };
  enum CType
  {
    MIDDLE_P_V_CONSISTANT,
    MIDDLE_P_V_A_CONSISTANT
  };
};

}

#endif