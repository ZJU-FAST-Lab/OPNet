#include "poly_opt/traj_optimizer.h"

namespace fast_planner 
{
TrajOptimizer::TrajOptimizer(ros::NodeHandle &node)
{
  node.param("optimization/w_smooth", w_smooth_, 1.0);  
  node.param("optimization/w_close", w_close_, 1.0);
  node.param("optimization/w_acc", w_acc_, 1.0);
}

void TrajOptimizer::setWayPointsAndTime(const std::vector<Eigen::Vector3d>& way_points, 
                                            const std::vector<Eigen::Vector3d>& vel, 
                                            const std::vector<Eigen::Vector3d>& acc, 
                                            const Eigen::VectorXd &time, 
                                            const Eigen::MatrixXd &coeff)
{
  //TODO add assert to check size equavalence
  this->path_ = Eigen::MatrixXd::Zero(way_points.size(), 3);
  this->vel_way_points_ = Eigen::MatrixXd::Zero(vel.size(), 3);
  this->acc_way_points_ = Eigen::MatrixXd::Zero(acc.size(), 3);
  for(int i = 0; i < way_points.size(); ++i)
  {
    path_.row(i) = way_points[i].transpose();
    vel_way_points_.row(i) = vel[i].transpose();
    acc_way_points_.row(i) = acc[i].transpose();
  }
  this->time_ = time;
  this->m_ = time.size();
  this->calMatrixA();
  this->calMatrixQ_smooth(MINIMUM_JERK);
  this->calMatrixQ_close();
  this->calMatrixQ_acc_consistent();
  this->coeff_.resize(m_, 6*3);
  this->coeff0_ = coeff;
  this->calMatrixCandMatrixZ(MIDDLE_P_V_CONSISTANT);
}

void TrajOptimizer::setWayPointsAndTime(const std::vector<Eigen::Vector3d>& way_points, 
                                        const std::vector<Eigen::Vector3d>& vel, 
                                        const std::vector<Eigen::Vector3d>& acc, 
                                        const Eigen::VectorXd &time)
{
  //TODO add assert to check size equavalence
  this->path_ = Eigen::MatrixXd::Zero(way_points.size(), 3);
  this->vel_way_points_ = Eigen::MatrixXd::Zero(vel.size(), 3);
  this->acc_way_points_ = Eigen::MatrixXd::Zero(acc.size(), 3);
  for(int i = 0; i < way_points.size(); ++i)
  {
    path_.row(i) = way_points[i].transpose();
    vel_way_points_.row(i) = vel[i].transpose();
    acc_way_points_.row(i) = acc[i].transpose();
  }
  this->time_ = time;
  this->m_ = time.size();
  this->calMatrixA();
  this->calMatrixQ_smooth(MINIMUM_JERK);
  this->coeff_.resize(m_, 6*3);
  this->calMatrixC();
}

// void TrajOptimizer::setWayPointsAndTime(const std::vector<Eigen::Vector3d>& way_points, 
//                                             const std::vector<Eigen::Vector3d>& vel, 
//                                             const std::vector<Eigen::Vector3d>& acc, 
//                                             const Eigen::VectorXd &time, 
//                                             const Eigen::MatrixXd &coeff, 
//                                             const std::set<int> &fixed_pos)
// {
//   //TODO add assert to check size equavalence
//   this->path_ = Eigen::MatrixXd::Zero(way_points.size(), 3);
//   this->vel_way_points_ = Eigen::MatrixXd::Zero(vel.size(), 3);
//   this->acc_way_points_ = Eigen::MatrixXd::Zero(acc.size(), 3);
//   for(int i = 0; i < way_points.size(); ++i)
//   {
//     path_.row(i) = way_points[i].transpose();
//     vel_way_points_.row(i) = vel[i].transpose();
//     acc_way_points_.row(i) = acc[i].transpose();
//   }
//   this->fixed_pos_ = fixed_pos;
//   this->n_ = fixed_pos.size();
//   this->time_ = time;
//   this->m_ = time.size();
//   this->calMatrixA();
//   this->calMatrixQ_smooth(MINIMUM_JERK);
//   this->calMatrixQ_close();
//   this->calMatrixQ_acc_consistent();
  
//   this->coeff_.resize(m_, 6*3);
//   this->coeff0_ = coeff;
//   this->calMatrixCandMatrixZ(MIDDLE_P_V_CONSISTANT);
// }


// fixed start&end pos, vel, acc and middle positions
// void TrajOptimizer::tryQPCloseForm()
// {
//   /*   Produce the dereivatives in X, Y and Z axis, end vel and acc are all zeroes  */
//   Eigen::VectorXd Dx = Eigen::VectorXd::Zero(m_ * 6);
//   Eigen::VectorXd Dy = Eigen::VectorXd::Zero(m_ * 6);
//   Eigen::VectorXd Dz = Eigen::VectorXd::Zero(m_ * 6);
// //   cout <<"11111111111111" << endl;
//   for(int k = 1; k < (m_ + 1); k++ ){
//     Dx((k-1)*6) = path_(k - 1, 0); Dx((k-1)*6 + 1) = path_(k, 0); 
//     Dy((k-1)*6) = path_(k - 1, 1); Dy((k-1)*6 + 1) = path_(k, 1); 
//     Dz((k-1)*6) = path_(k - 1, 2); Dz((k-1)*6 + 1) = path_(k, 2); 
//     
//     if( k == 1){
//       Dx((k-1)*6 + 2) = vel_way_points_(k - 1, 0);
//       Dy((k-1)*6 + 2) = vel_way_points_(k - 1, 1); 
//       Dz((k-1)*6 + 2) = vel_way_points_(k - 1, 2);
// 
//       Dx((k-1)*6 + 4) = acc_way_points_(k - 1, 0);
//       Dy((k-1)*6 + 4) = acc_way_points_(k - 1, 1); 
//       Dz((k-1)*6 + 4) = acc_way_points_(k - 1, 2);
//     }
//   }   
// //   Dx(0) = path_(0, 0); Dx((m_-1)*6) = path_(m_-1, 0); 
// //   Dy(0) = path_(0, 1); Dy((m_-1)*6) = path_(m_-1, 1); 
// //   Dz(0) = path_(0, 2); Dz((m_-1)*6) = path_(m_-1, 2);
// //   Dx(2) = vel_way_points_(0, 0);
// //   Dy(2) = vel_way_points_(0, 1); 
// //   Dz(2) = vel_way_points_(0, 2);
// //   Dx(4) = acc_way_points_(0, 0);
// //   Dy(4) = acc_way_points_(0, 1); 
// //   Dz(4) = acc_way_points_(0, 2);
//     
// //   cout <<"22222222222222" << endl;
// //   cout <<m_ << endl;
//   // generating minumum snap curve
//   int num_f = 2 * m_ + 4; //3 + 3 + (m_ - 1) * 2 = 2m + 4
// //   int num_f = m_ + 5; //3 + 3 + (m_ - 1) * 1 = m_ + 5
//   int num_p = 2 * m_ - 2; //(m_ - 1) * 2 = 2m - 2
//   int num_d = 6 * m_;
//   Eigen::MatrixXd Ct; // The transpose of selection matrix C
// //   Eigen::MatrixXd C;  // The selection matrix C
//   Ct = Eigen::MatrixXd::Zero(num_d, num_f + num_p); 
//   Ct( 0, 0 ) = 1; Ct( 2, 1 ) = 1; Ct( 4, 2 ) = 1; // stack the start point
//   Ct( 1, 3 ) = 1; Ct( 3, 2 * m_ + 4 ) = 1; Ct( 5, 2 * m_ + 5 ) = 1; 
//   Ct(6 * (m_ - 1) + 0, 2 * m_ + 0) = 1; 
//   Ct(6 * (m_ - 1) + 1, 2 * m_ + 1) = 1; // Stack the end point
//   Ct(6 * (m_ - 1) + 2, 4 * m_ + 0) = 1;
//   Ct(6 * (m_ - 1) + 3, 2 * m_ + 2) = 1; // Stack the end point
//   Ct(6 * (m_ - 1) + 4, 4 * m_ + 1) = 1;
//   Ct(6 * (m_ - 1) + 5, 2 * m_ + 3) = 1; // Stack the end point
//   for(int j = 2; j < m_; j ++ ){
//     Ct( 6 * (j - 1) + 0, 2 + 2 * (j - 1) + 0 ) = 1;
//     Ct( 6 * (j - 1) + 1, 2 + 2 * (j - 1) + 1 ) = 1;
//     Ct( 6 * (j - 1) + 2, 2 * m_ + 4 + 2 * (j - 2) + 0 ) = 1;
//     Ct( 6 * (j - 1) + 3, 2 * m_ + 4 + 2 * (j - 1) + 0 ) = 1;
//     Ct( 6 * (j - 1) + 4, 2 * m_ + 4 + 2 * (j - 2) + 1 ) = 1;
//     Ct( 6 * (j - 1) + 5, 2 * m_ + 4 + 2 * (j - 1) + 1 ) = 1;
//   }
//   
// //   C = Ct.transpose();
//   Eigen::VectorXd Dx1 = Ct.transpose() * Dx;
//   Eigen::VectorXd Dy1 = Ct.transpose() * Dy;
//   Eigen::VectorXd Dz1 = Ct.transpose() * Dz;
// //   Eigen::VectorXd Dxf(2 * m_ + 4), Dyf(2 * m_ + 4), Dzf(2 * m_ + 4);
// //   Dxf = Dx1.segment( 0, 2 * m_ + 4 );
// //   Dyf = Dy1.segment( 0, 2 * m_ + 4 );
// //   Dzf = Dz1.segment( 0, 2 * m_ + 4 );
// //   cout <<"666666666666666" << endl;
//   Eigen::MatrixXd A_inv_multiply_Ct = A_inv_ * Ct;
//   Eigen::MatrixXd R = A_inv_multiply_Ct.transpose() * Q_smooth_ * A_inv_multiply_Ct;
// //   Eigen::MatrixXd Rff(2 * m_ + 4, 2 * m_ + 4);
//   Eigen::MatrixXd Rfp(2 * m_ + 4, 2 * m_ - 2);
// //   Eigen::MatrixXd Rpf(2 * m_ - 2, 2 * m_ + 4);
//   Eigen::MatrixXd Rpp(2 * m_ - 2, 2 * m_ - 2);
// //   Rff = R.block(0, 0, 2 * m_ + 4, 2 * m_ + 4);
//   Rfp = R.block(0, 2 * m_ + 4, 2 * m_ + 4, 2 * m_ - 2);
// //   Rpf = R.block(2 * m_ + 4, 0,         2 * m_ - 2, 2 * m_ + 4);
//   Rpp = R.block(2 * m_ + 4, 2 * m_ + 4, 2 * m_ - 2, 2 * m_ - 2);
// //   cout <<"777777777777777" << endl;
// //   Eigen::VectorXd Dxp(2 * m_ - 2), Dyp(2 * m_ - 2), Dzp(2 * m_ - 2);
//   Eigen::MatrixXd Rpp_inv = Rpp.inverse();
// //   Dxp = - (Rpp_inv * Rfp.transpose()) * Dx1.segment( 0, 2 * m_ + 4 );
// //   Dyp = - (Rpp_inv * Rfp.transpose()) * Dy1.segment( 0, 2 * m_ + 4 );
// //   Dzp = - (Rpp_inv * Rfp.transpose()) * Dz1.segment( 0, 2 * m_ + 4 );
// //   cout <<"88888888888888" << endl;
//   Eigen::MatrixXd neg_Rpp_inv_multiply_Rfp_tran = - Rpp_inv * Rfp.transpose();
//   Dx1.segment(2 * m_ + 4, 2 * m_ - 2) = neg_Rpp_inv_multiply_Rfp_tran * Dx1.segment( 0, 2 * m_ + 4 );
//   Dy1.segment(2 * m_ + 4, 2 * m_ - 2) = neg_Rpp_inv_multiply_Rfp_tran * Dy1.segment( 0, 2 * m_ + 4 );
//   Dz1.segment(2 * m_ + 4, 2 * m_ - 2) = neg_Rpp_inv_multiply_Rfp_tran * Dz1.segment( 0, 2 * m_ + 4 );
// //   cout <<"9999999999999" << endl;
//   Eigen::VectorXd Px = A_inv_multiply_Ct * Dx1;
//   Eigen::VectorXd Py = A_inv_multiply_Ct * Dy1;
//   Eigen::VectorXd Pz = A_inv_multiply_Ct * Dz1;
//   
// //   Dx_ = Ct * Dx1;
// //   Dy_ = Ct * Dy1;
// //   Dz_ = Ct * Dz1;
// //   cout <<"10101010101010" << endl;
//   for(int i = 0; i < m_; i ++){
//     coeff_.block(i, 0,  1, 6) = Px.segment( i * 6, 6 ).transpose();
//     coeff_.block(i, 6,  1, 6) = Py.segment( i * 6, 6 ).transpose();
//     coeff_.block(i, 12, 1, 6) = Pz.segment( i * 6, 6 ).transpose();
//   }
// }

// fixed start&end pos, vel, acc 
// void GradTrajOptimizer::tryQPCloseForm()
// {
//   /*   Produce the dereivatives in X, Y and Z axis, end vel and acc are all zeroes  */
//   Eigen::VectorXd Dx = Eigen::VectorXd::Zero(m_ * 6);
//   Eigen::VectorXd Dy = Eigen::VectorXd::Zero(m_ * 6);
//   Eigen::VectorXd Dz = Eigen::VectorXd::Zero(m_ * 6);
// 
//   Dx(0) = path_(0, 0); Dx((m_-1)*6 + 1) = path_(m_, 0); 
//   Dy(0) = path_(0, 1); Dy((m_-1)*6 + 1) = path_(m_, 1); 
//   Dz(0) = path_(0, 2); Dz((m_-1)*6 + 1) = path_(m_, 2);
//   Dx(2) = vel_way_points_(0, 0);
//   Dy(2) = vel_way_points_(0, 1); 
//   Dz(2) = vel_way_points_(0, 2);
//   Dx(4) = acc_way_points_(0, 0);
//   Dy(4) = acc_way_points_(0, 1); 
//   Dz(4) = acc_way_points_(0, 2);
//     
//   int num_f = 6;          // 3 + 3 : only start and target has fixed derivatives   
//   int num_p = 3 * m_ - 3;  // All other derivatives are free   
//   int num_d = 6 * m_;
//   MatrixXd Ct;    
//   Ct = MatrixXd::Zero(num_d, num_f + num_p); 
//   Ct( 0, 0 ) = 1; Ct( 2, 1 ) = 1; Ct( 4, 2 ) = 1;  // Stack the start point
//   Ct( 1, 6 ) = 1; Ct( 3, 7 ) = 1; Ct( 5, 8 ) = 1; 
// 
//   Ct(6 * (m_ - 1) + 0, 3 * m_ + 0 ) = 1; 
//   Ct(6 * (m_ - 1) + 2, 3 * m_ + 1 ) = 1;
//   Ct(6 * (m_ - 1) + 4, 3 * m_ + 2 ) = 1;
// 
//   Ct(6 * (m_ - 1) + 1, 3) = 1; // Stack the end point
//   Ct(6 * (m_ - 1) + 3, 4) = 1;
//   Ct(6 * (m_ - 1) + 5, 5) = 1;
// 
//   for(int j = 2; j < m_; j ++ ){
//     Ct( 6 * (j - 1) + 0, 6 + 3 * (j - 2) + 0 ) = 1;
//     Ct( 6 * (j - 1) + 1, 6 + 3 * (j - 1) + 0 ) = 1;
//     Ct( 6 * (j - 1) + 2, 6 + 3 * (j - 2) + 1 ) = 1;
//     Ct( 6 * (j - 1) + 3, 6 + 3 * (j - 1) + 1 ) = 1;
//     Ct( 6 * (j - 1) + 4, 6 + 3 * (j - 2) + 2 ) = 1;
//     Ct( 6 * (j - 1) + 5, 6 + 3 * (j - 1) + 2 ) = 1;
//   }
// //   C = Ct.transpose();
//   Eigen::VectorXd Dx1 = Ct.transpose() * Dx;
//   Eigen::VectorXd Dy1 = Ct.transpose() * Dy;
//   Eigen::VectorXd Dz1 = Ct.transpose() * Dz;
//   Eigen::MatrixXd A_inv_multiply_Ct = A_inv_ * Ct;
//   Eigen::MatrixXd R = A_inv_multiply_Ct.transpose() * Q_smooth_ * A_inv_multiply_Ct;
//   Eigen::MatrixXd Rfp(6, 3 * m_ - 3);
//   Eigen::MatrixXd Rpp(3 * m_ - 3, 3 * m_ - 3);
//   Rfp = R.block(0, 6, 6, 3 * m_ - 3);
//   Rpp = R.block(6, 6, 3 * m_ - 3, 3 * m_ - 3);
//   Eigen::MatrixXd Rpp_inv = Rpp.inverse();
//   Eigen::MatrixXd neg_Rpp_inv_multiply_Rfp_tran = - Rpp_inv * Rfp.transpose();
//   Dx1.segment(6, 3 * m_ - 3) = neg_Rpp_inv_multiply_Rfp_tran * Dx1.segment( 0, 6 );
//   Dy1.segment(6, 3 * m_ - 3) = neg_Rpp_inv_multiply_Rfp_tran * Dy1.segment( 0, 6 );
//   Dz1.segment(6, 3 * m_ - 3) = neg_Rpp_inv_multiply_Rfp_tran * Dz1.segment( 0, 6 );
//   Eigen::VectorXd Px = A_inv_multiply_Ct * Dx1;
//   Eigen::VectorXd Py = A_inv_multiply_Ct * Dy1;
//   Eigen::VectorXd Pz = A_inv_multiply_Ct * Dz1;
//   
//   for(int i = 0; i < m_; i ++){
//     coeff_.block(i, 0,  1, 6) = Px.segment( i * 6, 6 ).transpose();
//     coeff_.block(i, 6,  1, 6) = Py.segment( i * 6, 6 ).transpose();
//     coeff_.block(i, 12, 1, 6) = Pz.segment( i * 6, 6 ).transpose();
//   }
// }

//middle p\v consistent
void TrajOptimizer::tryQPCloseForm(double percent_of_close, double percent_of_acc)
{
  double weight_smooth = 100.0 - percent_of_close - percent_of_acc;
  double weight_close = percent_of_close;
  double weight_acc = percent_of_acc;
//   cout << weight_smooth*Q_smooth_ + weight_close*Q_close_ + weight_acc*Q_acc_ << endl << endl;
  Eigen::VectorXd Dx1 = Eigen::VectorXd::Zero(4 * m_ + 2);
  Eigen::VectorXd Dy1 = Eigen::VectorXd::Zero(4 * m_ + 2);
  Eigen::VectorXd Dz1 = Eigen::VectorXd::Zero(4 * m_ + 2);
  Dx1(0) = path_(0, 0);  Dx1(1) = vel_way_points_(0, 0);  Dx1(2) =  acc_way_points_(0, 0); 
  Dx1(3) = path_(m_, 0); Dx1(4) = vel_way_points_(m_, 0); Dx1(5) =  acc_way_points_(m_, 0); 
  Dy1(0) = path_(0, 1);  Dy1(1) = vel_way_points_(0, 1);  Dy1(2) =  acc_way_points_(0, 1);
  Dy1(3) = path_(m_, 1); Dy1(4) = vel_way_points_(m_, 1); Dy1(5) =  acc_way_points_(m_, 1); 
  Dz1(0) = path_(0, 2);  Dz1(1) = vel_way_points_(0, 2);  Dz1(2) =  acc_way_points_(0, 2);
  Dz1(3) = path_(m_, 2); Dz1(4) = vel_way_points_(m_, 2); Dz1(5) =  acc_way_points_(m_, 2); 
  
  Eigen::MatrixXd R = A_inv_multiply_Ct_.transpose() 
                      * (weight_smooth*Q_smooth_ + weight_close*Q_close_ + weight_acc*Q_acc_) 
                      * A_inv_multiply_Ct_;
  Eigen::MatrixXd Rpf(4 * m_ - 4, 6);
  Eigen::MatrixXd Rpp(4 * m_ - 4, 4 * m_ - 4);
  Rpf = R.block(6, 0, 4 * m_ - 4, 6);
  Rpp = R.block(6, 6, 4 * m_ - 4, 4 * m_ - 4);
  
  Eigen::MatrixXd Rpp_inv = Rpp.inverse();
  Eigen::MatrixXd neg_Rpp_inv_multiply_Rfp_tran = - Rpp_inv * Rpf;
  Eigen::MatrixXd Rpp_inv_multiply_Zp = weight_close * Rpp_inv * Zp_;
  Dx1.segment(6, 4 * m_ - 4) = Rpp_inv_multiply_Zp.col(0) + neg_Rpp_inv_multiply_Rfp_tran * Dx1.segment( 0, 6 );
  Dy1.segment(6, 4 * m_ - 4) = Rpp_inv_multiply_Zp.col(1) + neg_Rpp_inv_multiply_Rfp_tran * Dy1.segment( 0, 6 );
  Dz1.segment(6, 4 * m_ - 4) = Rpp_inv_multiply_Zp.col(2) + neg_Rpp_inv_multiply_Rfp_tran * Dz1.segment( 0, 6 );
  Eigen::VectorXd Px = A_inv_multiply_Ct_ * Dx1;
  Eigen::VectorXd Py = A_inv_multiply_Ct_ * Dy1;
  Eigen::VectorXd Pz = A_inv_multiply_Ct_ * Dz1;
  
  for(int i = 0; i < m_; i ++) {
    coeff_.block(i, 0,  1, 6) = Px.segment( i * 6, 6 ).transpose();
    coeff_.block(i, 6,  1, 6) = Py.segment( i * 6, 6 ).transpose();
    coeff_.block(i, 12, 1, 6) = Pz.segment( i * 6, 6 ).transpose();
  }
}

// middle p\v\a consistent, middle p fixed, middle v\a free
void TrajOptimizer::tryQPCloseForm()
{
  Eigen::VectorXd Dx1 = Eigen::VectorXd::Zero(3 * m_ + 3);
  Eigen::VectorXd Dy1 = Eigen::VectorXd::Zero(3 * m_ + 3);
  Eigen::VectorXd Dz1 = Eigen::VectorXd::Zero(3 * m_ + 3);
  Dx1(0) = path_(0, 0);  Dx1(1) = vel_way_points_(0, 0);  Dx1(2) =  acc_way_points_(0, 0); 
  Dx1(3) = path_(m_, 0); Dx1(4) = vel_way_points_(m_, 0); Dx1(5) =  acc_way_points_(m_, 0); 
  Dy1(0) = path_(0, 1);  Dy1(1) = vel_way_points_(0, 1);  Dy1(2) =  acc_way_points_(0, 1);
  Dy1(3) = path_(m_, 1); Dy1(4) = vel_way_points_(m_, 1); Dy1(5) =  acc_way_points_(m_, 1); 
  Dz1(0) = path_(0, 2);  Dz1(1) = vel_way_points_(0, 2);  Dz1(2) =  acc_way_points_(0, 2);
  Dz1(3) = path_(m_, 2); Dz1(4) = vel_way_points_(m_, 2); Dz1(5) =  acc_way_points_(m_, 2); 
  for (int i = 0; i < m_ - 1; ++i)
  {
    Dx1(6 + i) = path_(i + 1, 0);
    Dy1(6 + i) = path_(i + 1, 1);
    Dz1(6 + i) = path_(i + 1, 2);
  }
  Eigen::MatrixXd R = A_inv_multiply_Ct_.transpose() * Q_smooth_ * A_inv_multiply_Ct_;
  Eigen::MatrixXd Rpf(2 * m_ - 2, 5 + m_);
  Eigen::MatrixXd Rpp(2 * m_ - 2, 2 * m_ - 2);
  Rpf = R.block(5 + m_, 0, 2 * m_ - 2, 5 + m_);
  Rpp = R.block(5 + m_, 5 + m_, 2 * m_ - 2, 2 * m_ - 2);
  
  Eigen::MatrixXd Rpp_inv = Rpp.inverse();
  Eigen::MatrixXd neg_Rpp_inv_multiply_Rfp_tran = - Rpp_inv * Rpf;
  Dx1.segment(5 + m_, 2 * m_ - 2) = neg_Rpp_inv_multiply_Rfp_tran * Dx1.segment( 0, 5 + m_ );
  Dy1.segment(5 + m_, 2 * m_ - 2) = neg_Rpp_inv_multiply_Rfp_tran * Dy1.segment( 0, 5 + m_ );
  Dz1.segment(5 + m_, 2 * m_ - 2) = neg_Rpp_inv_multiply_Rfp_tran * Dz1.segment( 0, 5 + m_ );
  Eigen::VectorXd Px = A_inv_multiply_Ct_ * Dx1;
  Eigen::VectorXd Py = A_inv_multiply_Ct_ * Dy1;
  Eigen::VectorXd Pz = A_inv_multiply_Ct_ * Dz1;
  for(int i = 0; i < m_; i ++) 
  {
    coeff_.block(i, 0,  1, 6) = Px.segment( i * 6, 6 ).transpose();
    coeff_.block(i, 6,  1, 6) = Py.segment( i * 6, 6 ).transpose();
    coeff_.block(i, 12, 1, 6) = Pz.segment( i * 6, 6 ).transpose();
  }
}

// middle p\v\a consistent
// void TrajOptimizer::tryQPCloseForm(double percent_of_close, double percent_of_acc)
// {
//   double weight_smooth = 100.0 - percent_of_close;
//   double weight_close = percent_of_close;
  
//   weight_smooth = 0.02;
//   weight_close = 1;
  
//   Eigen::VectorXd Dx1 = Eigen::VectorXd::Zero(3 * m_ + 3);
//   Eigen::VectorXd Dy1 = Eigen::VectorXd::Zero(3 * m_ + 3);
//   Eigen::VectorXd Dz1 = Eigen::VectorXd::Zero(3 * m_ + 3);
//   Dx1(0) = path_(0, 0);  Dx1(1) = vel_way_points_(0, 0);  Dx1(2) =  acc_way_points_(0, 0); 
//   Dx1(3) = path_(m_, 0); Dx1(4) = vel_way_points_(m_, 0); Dx1(5) =  acc_way_points_(m_, 0); 
//   Dy1(0) = path_(0, 1);  Dy1(1) = vel_way_points_(0, 1);  Dy1(2) =  acc_way_points_(0, 1);
//   Dy1(3) = path_(m_, 1); Dy1(4) = vel_way_points_(m_, 1); Dy1(5) =  acc_way_points_(m_, 1); 
//   Dz1(0) = path_(0, 2);  Dz1(1) = vel_way_points_(0, 2);  Dz1(2) =  acc_way_points_(0, 2);
//   Dz1(3) = path_(m_, 2); Dz1(4) = vel_way_points_(m_, 2); Dz1(5) =  acc_way_points_(m_, 2); 
//   int num = 0;
//   for (std::set<int>::iterator it = fixed_pos_.begin(); it != fixed_pos_.end(); ++it)
//   {
//     Dx1(6 + num) = path_((*it) + 1, 0);
//     Dy1(6 + num) = path_((*it) + 1, 1);
//     Dz1(6 + num) = path_((*it) + 1, 2);
//     ++num;
//   }
  
// //   Eigen::MatrixXd A_inv_multiply_Ct = A_inv_ * Ct_;
//   Eigen::MatrixXd R = A_inv_multiply_Ct_.transpose() * (weight_smooth*Q_smooth_ + weight_close*Q_close_) * A_inv_multiply_Ct_;
//   Eigen::MatrixXd Rpf(3 * m_ - 3 - n_, 6 + n_);
//   Eigen::MatrixXd Rpp(3 * m_ - 3 - n_, 3 * m_ - 3 - n_);
//   Rpf = R.block(6 + n_, 0, 3 * m_ - 3 - n_, 6 + n_);
//   Rpp = R.block(6 + n_, 6 + n_, 3 * m_ - 3 - n_, 3 * m_ - 3 - n_);
  
//   Eigen::MatrixXd Rpp_inv = Rpp.inverse();
//   Eigen::MatrixXd neg_Rpp_inv_multiply_Rfp_tran = - Rpp_inv * Rpf;
//   Eigen::MatrixXd Rpp_inv_multiply_Zp = weight_close * Rpp_inv * Zp_;
//   Dx1.segment(6 + n_, 3 * m_ - 3 - n_) = Rpp_inv_multiply_Zp.col(0) + neg_Rpp_inv_multiply_Rfp_tran * Dx1.segment( 0, 6 + n_ );
//   Dy1.segment(6 + n_, 3 * m_ - 3 - n_) = Rpp_inv_multiply_Zp.col(1) + neg_Rpp_inv_multiply_Rfp_tran * Dy1.segment( 0, 6 + n_ );
//   Dz1.segment(6 + n_, 3 * m_ - 3 - n_) = Rpp_inv_multiply_Zp.col(2) + neg_Rpp_inv_multiply_Rfp_tran * Dz1.segment( 0, 6 + n_ );
//   Eigen::VectorXd Px = A_inv_multiply_Ct_ * Dx1;
//   Eigen::VectorXd Py = A_inv_multiply_Ct_ * Dy1;
//   Eigen::VectorXd Pz = A_inv_multiply_Ct_ * Dz1;
  
//   for(int i = 0; i < m_; i ++) 
//   {
//     coeff_.block(i, 0,  1, 6) = Px.segment( i * 6, 6 ).transpose();
//     coeff_.block(i, 6,  1, 6) = Py.segment( i * 6, 6 ).transpose();
//     coeff_.block(i, 12, 1, 6) = Pz.segment( i * 6, 6 ).transpose();
//   }
// }

void TrajOptimizer::calMatrixCandMatrixZ(int type)
{
  if (type == MIDDLE_P_V_CONSISTANT) /* a inconsistant */
  {
    int num_f = 6;          // 3 + 3 : only start and target has fixed derivatives   
    int num_p = 4 * m_ - 4;  // 4 * (m - 1)
    int num_d = 6 * m_;
  
    Ct_ = MatrixXd::Zero(num_d, num_f + num_p); 
    Ct_( 0, 0 ) = 1; Ct_( 2, 1 ) = 1; Ct_( 4, 2 ) = 1;  // Stack the start point
    Ct_( 1, 6 ) = 1; Ct_( 3, 7 ) = 1; Ct_( 5, 8 ) = 1; 

    Ct_(6 * (m_ - 1) + 0, 4 * (m_ - 2) + 6 + 0 ) = 1; 
    Ct_(6 * (m_ - 1) + 2, 4 * (m_ - 2) + 6 + 1 ) = 1;
    Ct_(6 * (m_ - 1) + 4, 4 * (m_ - 2) + 6 + 3 ) = 1;

    Ct_(6 * (m_ - 1) + 1, 3) = 1; // Stack the end point
    Ct_(6 * (m_ - 1) + 3, 4) = 1;
    Ct_(6 * (m_ - 1) + 5, 5) = 1;

    for(int j = 2; j < m_; j ++ ){
      Ct_( 6 * (j - 1) + 0, 6 + 4 * (j - 2) + 0 ) = 1;
      Ct_( 6 * (j - 1) + 1, 6 + 4 * (j - 1) + 0 ) = 1;
      Ct_( 6 * (j - 1) + 2, 6 + 4 * (j - 2) + 1 ) = 1;
      Ct_( 6 * (j - 1) + 3, 6 + 4 * (j - 1) + 1 ) = 1;
      Ct_( 6 * (j - 1) + 4, 6 + 4 * (j - 2) + 3 ) = 1;
      Ct_( 6 * (j - 1) + 5, 6 + 4 * (j - 1) + 2 ) = 1;
    }
    
    A_inv_multiply_Ct_ = A_inv_ * Ct_;
    Z_ = A_inv_multiply_Ct_.transpose() * Q_close_ * coeff0_;
    Zp_ = Z_.block(6, 0, 4 * m_ - 4, 3);
  }
  else if (type == MIDDLE_P_V_A_CONSISTANT) /* p\v\a consistant */
  {
    int num_f = 6;          // 3 + 3 : only start and target has fixed derivatives   
    int num_p = 3 * m_ - 3;  // 3 * (m - 1)
    int num_d = 6 * m_;
  
    Ct_ = MatrixXd::Zero(num_d, num_f + num_p); 
    Ct_( 0, 0 ) = 1; Ct_( 2, 1 ) = 1; Ct_( 4, 2 ) = 1;  // Stack the start point
    Ct_( 1, 6 ) = 1; Ct_( 3, 7 ) = 1; Ct_( 5, 8 ) = 1; 

    Ct_(6 * (m_ - 1) + 0, 3 * m_ + 0 ) = 1; // Stack the end point
    Ct_(6 * (m_ - 1) + 2, 3 * m_ + 1 ) = 1;
    Ct_(6 * (m_ - 1) + 4, 3 * m_ + 2 ) = 1;
    Ct_(6 * (m_ - 1) + 1, 3) = 1;
    Ct_(6 * (m_ - 1) + 3, 4) = 1;
    Ct_(6 * (m_ - 1) + 5, 5) = 1;

    for(int j = 2; j < m_; j ++ ){
      Ct_( 6 * (j - 1) + 0, 6 + 3 * (j - 2) + 0 ) = 1;
      Ct_( 6 * (j - 1) + 1, 6 + 3 * (j - 1) + 0 ) = 1;
      Ct_( 6 * (j - 1) + 2, 6 + 3 * (j - 2) + 1 ) = 1;
      Ct_( 6 * (j - 1) + 3, 6 + 3 * (j - 1) + 1 ) = 1;
      Ct_( 6 * (j - 1) + 4, 6 + 3 * (j - 2) + 2 ) = 1;
      Ct_( 6 * (j - 1) + 5, 6 + 3 * (j - 1) + 2 ) = 1;
    }

    A_inv_multiply_Ct_ = A_inv_ * Ct_;
    Z_ = A_inv_multiply_Ct_.transpose() * Q_close_ * coeff0_;
    Zp_ = Z_.block(6, 0, 3 * m_ - 3, 3);
  }
}

// middle p fixed, p\v\a consistant
void TrajOptimizer::calMatrixC()
{
  int num_f = 5 + m_;          // 3 + 3 + m_-1: start and target has fixed derivatives, middle p fixed   
  int num_p = 2 * m_ - 2;  // 2 * (m - 1)
  int num_d = 6 * m_;

  Ct_ = MatrixXd::Zero(num_d, num_f + num_p); 
  
  Ct_(0, 0) = 1; Ct_(2, 1) = 1; Ct_(4, 2) = 1;
  Ct_(1, 6) = 1; Ct_(3, 6 + (m_ - 1)) = 1; Ct_(5, 6 + 2 * (m_ - 1)) = 1;
  Ct_(6*(m_-1), 6 + (m_ - 1) - 1) = 1; 
  Ct_(6*(m_-1)+2, 6 + 2 * (m_ - 1) - 1) = 1; 
  Ct_(6*(m_-1)+4, 6 + 3 * (m_ - 1) - 1) = 1; 
  Ct_(6*(m_-1)+1, 3) = 1; Ct_(6*(m_-1)+3, 4) = 1; Ct_(6*(m_-1)+5, 5) = 1; 
  for (int i = 1; i < m_ - 1; ++i)
  {
    Ct_(6 * i, 6 + i - 1) = 1; 
    Ct_(6 * i + 1, 6 + i) = 1; 
    Ct_(6 * i + 2, 6 + (m_ - 1) + i - 1) = 1; 
    Ct_(6 * i + 3, 6 + (m_ - 1) + i) = 1;
    Ct_(6 * i + 4, 6 + 2 * (m_ - 1) + i - 1) = 1;
    Ct_(6 * i + 5, 6 + 2 * (m_ - 1) + i) = 1;
  }  


  // MatrixXd Ca = MatrixXd::Zero(num_d, num_f + num_p); 
  // MatrixXd Cd = MatrixXd::Zero(num_f + num_p, num_f + num_p); 
  // for (int i = 0; i < m_; ++i)
  // {
  //   Ca(6 * i + 0, 3 * i + 0) = 1;
  //   Ca(6 * i + 1, 3 * i + 1) = 1;
  //   Ca(6 * i + 2, 3 * i + 2) = 1;
  //   Ca(6 * i + 3, 3 * i + 3) = 1;
  //   Ca(6 * i + 4, 3 * i + 4) = 1;
  //   Ca(6 * i + 5, 3 * i + 5) = 1;
  // }
  
  // Cd(0, 0) = 1; Cd(1, 1) = 1; Cd(2, 2) = 1;
  // Cd(3 * m_ + 0, 3) = 1;
  // Cd(3 * m_ + 1, 4) = 1;
  // Cd(3 * m_ + 2, 5) = 1;
  
  // int j = 1;
  // for (int i = 0; i < m_-1 ; ++i)
  // {
  //   Cd(3 * i + 3, 5 + j) = 1;
  //   j++;
  // }
  
  // int k = 0;
  // for (int i = 0; i < m_ - 1; ++i)
  // {
  //   Cd(3 * i + 3 + 1, 5 + m_ + k) = 1;
  //   Cd(3 * i + 3 + 2, 5 + m_ + 1 + k) = 1;
  //   k += 2;
  // }
  // Ct_ = Ca * Cd;

  A_inv_multiply_Ct_ = A_inv_ * Ct_;
}

/* some middle points fixed */
// void TrajOptimizer::calMatrixCandMatrixZ()
// {
//   int num_f = 6 + n_;          // 3 + 3 : only start and target has fixed derivatives   
//   int num_p = 3 * m_ - 3 - n_;  // 3 * (m - 1)
//   int num_d = 6 * m_;
 
//   MatrixXd Ca = MatrixXd::Zero(num_d, num_f + num_p); 
//   MatrixXd Cd = MatrixXd::Zero(num_f + num_p, num_f + num_p); 
//   for (int i = 0; i < m_; ++i)
//   {
//     Ca(6 * i + 0, 3 * i + 0) = 1;
//     Ca(6 * i + 1, 3 * i + 1) = 1;
//     Ca(6 * i + 2, 3 * i + 2) = 1;
//     Ca(6 * i + 3, 3 * i + 3) = 1;
//     Ca(6 * i + 4, 3 * i + 4) = 1;
//     Ca(6 * i + 5, 3 * i + 5) = 1;
//   }
  
//   Cd(0, 0) = 1; Cd(1, 1) = 1; Cd(2, 2) = 1;
//   Cd(3 * m_ + 0, 3) = 1;
//   Cd(3 * m_ + 1, 4) = 1;
//   Cd(3 * m_ + 2, 5) = 1;
  
//   int j = 1;
//   for (std::set<int>::iterator it = fixed_pos_.begin(); it != fixed_pos_.end(); ++it)
//   {
//     Cd(3 * (*it) + 3, 5 + j) = 1;
//     j++;
//   }
  
//   int k = 0;
//   for (int i = 0; i < m_ - 1; ++i)
//   {
//     if (fixed_pos_.find(i) == fixed_pos_.end())
//     {
//       Cd(3 * i + 3, 5 + n_ + 1 + k) = 1;
//       Cd(3 * i + 3 + 1, 5 + n_ + 2 + k) = 1;
//       Cd(3 * i + 3 + 2, 5 + n_ + 3 + k) = 1;
//       k += 3;
//     }
//     else 
//     {
//       Cd(3 * i + 3 + 1, 5 + n_ + 1 + k) = 1;
//       Cd(3 * i + 3 + 2, 5 + n_ + 2 + k) = 1;
//       k += 2;
//     }
//   }
  
// //   cout << "Ca: \n" << Ca << endl;
// //   cout << "Cd: \n" << Cd << endl;
  
//   Ct_ = Ca * Cd;
//   A_inv_multiply_Ct_ = A_inv_ * Ct_;
//   Z_ = A_inv_multiply_Ct_.transpose() * Q_close_ * coeff0_;
//   Zp_ = Z_.block(6 + n_, 0, 3 * m_ - 3 - n_, 3);
// }


/*   Produce Mapping Matrix A_ and its inverse A_inv_ */
void TrajOptimizer::calMatrixA()
{
  A_ = Eigen::MatrixXd::Zero(m_ * 6, m_ * 6);
  
  const static auto Factorial = [](int x)
  {
    int fac = 1;
    for(int i = x; i > 0; i--)
      fac = fac * i;
    return fac;
  };
  
  Eigen::MatrixXd Ab;
  for(int k = 0; k < m_; k++)
  {
    Ab = Eigen::MatrixXd::Zero(6, 6);
    for(int i = 0; i < 3; i++)
    {
      Ab(2 * i, i) = Factorial(i);
      for(int j = i; j < 6; j++)
        Ab(2 * i + 1, j ) = Factorial(j) / Factorial( j - i ) * pow(time_(k), j - i );
    }
    A_.block(k * 6, k * 6, 6, 6) = Ab;    
  }
  
  A_inv_ = A_.inverse();
}

/*   Produce the smoothness cost Hessian   */
void TrajOptimizer::calMatrixQ_smooth(int type)
{
  Q_smooth_ = Eigen::MatrixXd::Zero(m_ * 6, m_ * 6);
  
  if (type == MINIMUM_ACC)
  {
    for(int k = 0; k < m_; k ++)
      for(int i = 2; i < 6; i ++)
        for(int j = i; j < 6; j ++) 
        {
          Q_smooth_( k*6 + i, k*6 + j ) = i * (i - 1) * j * (j - 1) / (i + j - 3) * pow(time_(k), (i + j - 3) );
          Q_smooth_( k*6 + j, k*6 + i ) = Q_smooth_( k*6 + i, k*6 + j );
        }
  }
  else if (type == MINIMUM_JERK)
  {
    for(int k = 0; k < m_; k ++)
      for(int i = 3; i < 6; i ++)
        for(int j = i; j < 6; j ++) 
        {
          Q_smooth_( k*6 + i, k*6 + j ) = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * pow(time_(k), (i + j - 5) );
          Q_smooth_( k*6 + j, k*6 + i ) = Q_smooth_( k*6 + i, k*6 + j );
        }
  }
  else if (type == MINIMUM_SNAP)
  {
    for(int k = 0; k < m_; k ++)
      for(int i = 4; i < 6; i ++)
        for(int j = i; j < 6; j ++) 
        {
          Q_smooth_( k*6 + i, k*6 + j ) = i * (i - 1) * (i - 2) * (i - 3) * j * (j - 1) * (j - 2) * (i - 3) / (i + j - 7) * pow(time_(k), (i + j - 7) );
          Q_smooth_( k*6 + j, k*6 + i ) = Q_smooth_( k*6 + i, k*6 + j );
        }
  }
  else 
  {
    cout << "[OPT]: unfedined Q_smooth type" << endl;
  }
}

/*   Produce the closeness cost Hessian matrix   */
void TrajOptimizer::calMatrixQ_close()
{
  Q_close_ = Eigen::MatrixXd::Zero(m_ * 6, m_ * 6);
  
  for(int k = 0; k < m_; k ++)
    for(int i = 0; i < 6; i ++)
      for(int j = i; j < 6; j ++) 
      {
        Q_close_( k*6 + i, k*6 + j ) = pow(time_(k), (i + j + 1)) / (i + j + 1);
        Q_close_( k*6 + j, k*6 + i ) = Q_close_( k*6 + i, k*6 + j );
      }
}

void TrajOptimizer::calMatrixQ_acc_consistent()
{
  Q_acc_ = Eigen::MatrixXd::Zero(m_ * 6, m_ * 6);
  
  for (int k = 0; k < m_ - 1; k++) 
  {
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m_ * 6, 1);
    T(k * 6 + 2, 0) = 2;
    T(k * 6 + 3, 0) = 6 * time_(k);
    T(k * 6 + 4, 0) = 12 * time_(k) * time_(k);
    T(k * 6 + 5, 0) = 20 * time_(k) * time_(k) * time_(k);
    T((k + 1) * 6 + 2, 0) = -2;
    Q_acc_ += T * T.transpose();
  }
}

void TrajOptimizer::getCoefficient(Eigen::MatrixXd &coe)
{
  coe = this->coeff_;
}

void TrajOptimizer::getSegmentTime(Eigen::VectorXd &T)
{
  T = this->time_;
}

// get position from coefficient
inline void TrajOptimizer::getPositionFromCoeff(Eigen::Vector3d &pos, const Eigen::MatrixXd &coeff,
                                             const int &index, const double &time) const
{
  int s = index;
  double t = time;
  float x = coeff(s, 0) + coeff(s, 1) * t + coeff(s, 2) * pow(t, 2) + coeff(s, 3) * pow(t, 3) +
            coeff(s, 4) * pow(t, 4) + coeff(s, 5) * pow(t, 5);
  float y = coeff(s, 6) + coeff(s, 7) * t + coeff(s, 8) * pow(t, 2) + coeff(s, 9) * pow(t, 3) +
            coeff(s, 10) * pow(t, 4) + coeff(s, 11) * pow(t, 5);
  float z = coeff(s, 12) + coeff(s, 13) * t + coeff(s, 14) * pow(t, 2) + coeff(s, 15) * pow(t, 3) +
            coeff(s, 16) * pow(t, 4) + coeff(s, 17) * pow(t, 5);

  pos(0) = x;
  pos(1) = y;
  pos(2) = z;
}

// get velocity from cofficient
inline void TrajOptimizer::getVelocityFromCoeff(Eigen::Vector3d &vel, const Eigen::MatrixXd &coeff,
                                             const int &index, const double &time) const
{
  int s = index;
  double t = time;
  float vx = coeff(s, 1) + 2 * coeff(s, 2) * pow(t, 1) + 3 * coeff(s, 3) * pow(t, 2) +
             4 * coeff(s, 4) * pow(t, 3) + 5 * coeff(s, 5) * pow(t, 4);
  float vy = coeff(s, 7) + 2 * coeff(s, 8) * pow(t, 1) + 3 * coeff(s, 9) * pow(t, 2) +
             4 * coeff(s, 10) * pow(t, 3) + 5 * coeff(s, 11) * pow(t, 4);
  float vz = coeff(s, 13) + 2 * coeff(s, 14) * pow(t, 1) + 3 * coeff(s, 15) * pow(t, 2) +
             4 * coeff(s, 16) * pow(t, 3) + 5 * coeff(s, 17) * pow(t, 4);

  vel(0) = vx;
  vel(1) = vy;
  vel(2) = vz;
}

// get acceleration from coefficient
inline void TrajOptimizer::getAccelerationFromCoeff(Eigen::Vector3d &acc, const Eigen::MatrixXd &coeff,
                                                 const int &index, const double &time) const
{
  int s = index;
  double t = time;
  float ax = 2 * coeff(s, 2) + 6 * coeff(s, 3) * pow(t, 1) + 12 * coeff(s, 4) * pow(t, 2) +
             20 * coeff(s, 5) * pow(t, 3);
  float ay = 2 * coeff(s, 8) + 6 * coeff(s, 9) * pow(t, 1) + 12 * coeff(s, 10) * pow(t, 2) +
             20 * coeff(s, 11) * pow(t, 3);
  float az = 2 * coeff(s, 14) + 6 * coeff(s, 15) * pow(t, 1) + 12 * coeff(s, 16) * pow(t, 2) +
             20 * coeff(s, 17) * pow(t, 3);

  acc(0) = ax;
  acc(1) = ay;
  acc(2) = az;
}


}
