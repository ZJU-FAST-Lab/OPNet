#ifndef _UTILS_H_
#define _UTILS_H_

#include <Eigen/Eigen>

#define X_DIM 6
#define U_DIM 3
#define EPSILON 0.1 //probability of sampling towards the goal
#define POLY_DEGREE 4
using std::vector;
using std::pair;
using std::list;

typedef Eigen::Matrix<double,X_DIM,1> State;
typedef Eigen::Matrix<double,U_DIM,1> Control;

typedef pair<double, double> BOUND;
typedef vector< BOUND > BOUNDS;

struct RRTNode {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  RRTNode* parent;
  State x;
  double cost_from_start;
  double tau_from_parent;
  double tau_from_start;
  uint32_t n_order;
  double x_coeff[6];
  double y_coeff[6];
  double z_coeff[6];
  list<RRTNode*> children;
  RRTNode(){};
};

typedef RRTNode* RRTNodePtr;
typedef vector<RRTNodePtr, Eigen::aligned_allocator<RRTNodePtr>> RRTNodePtrVector;
typedef vector<RRTNode, Eigen::aligned_allocator<RRTNode>> RRTNodeVector;
typedef pair<double, RRTNodePtr> CostNodePtrPair;


#endif //_UTILS_H_