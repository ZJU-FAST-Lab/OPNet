#include "poly_opt/waypoint_opt.h"

namespace fast_planner 
{
  
wpOptimizer::wpOptimizer(ros::NodeHandle& nh)
{
  nh.param("wpo/resolution", resolution_, 0.1);
  nh.param("wpo/clear_radius", clear_radius_, 1.0);
}

wpOptimizer::~wpOptimizer()
{
}

void wpOptimizer::optimize()
{
  RayCaster raycaster;
  Vector3d start, end, ray_pt;
  Vector3d center, l_end, r_end;
  for (int i = 0; i < wp_num_; ++i)
  {
//     cout << "wp: " << i << ": " << wp_[i].transpose() << endl;
//     cout << "grad: " << i << ": " << grad_[i].transpose() << endl;
    center = wp_[i];
    grad_[i].normalize();
    l_end = center + grad_[i] * clear_radius_;
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
//       cout << "left-tmp: " << tmp.transpose() << "occ: " << env_ptr_->getVoxelState(tmp) << endl;
      if (env_ptr_->getVoxelState(tmp) != 0)
      {
        l_free_p = tmp;
        len_l_e2o = (tmp - l_end).norm();
        l_clear = false;
        break;
      }
    }
    
    r_end = center - grad_[i] * (clear_radius_ + len_l_e2o);
    double len_r_e2o = 0;
    Vector3d r_free_p = r_end;
    end = r_end / resolution_;
    raycaster.setInput(start, end);
    bool r_clear = true;
    while (raycaster.step(ray_pt))
    {
      Eigen::Vector3d tmp = (ray_pt) * resolution_;
      tmp += Vector3d(resolution_/2.0, resolution_/2.0, resolution_/2.0);
//       cout << "right-tmp: " << tmp.transpose() << "occ: " << env_ptr_->getVoxelState(tmp) << endl;
      if (env_ptr_->getVoxelState(tmp) != 0)
      {
        len_r_e2o = (tmp - r_end).norm();
        r_free_p = tmp;
        r_clear = false;
        break;
      }
    }
    
    if (!l_clear)
    {
      wp_[i] = (l_free_p + r_free_p) /2;
//       cout << "left not clear, wp: " << i << ": " << wp_[i].transpose() << endl;
      continue;
    }
    else if (!r_clear)
    {
      Vector3d ll_end =  center + grad_[i] * (clear_radius_ + len_r_e2o);
      Vector3d ll_free_p = ll_end;
      start = l_end / resolution_;
      end = ll_end / resolution_;
      raycaster.setInput(start, end);
      while (raycaster.step(ray_pt))
      {
        Eigen::Vector3d tmp = (ray_pt) * resolution_;
        tmp += Vector3d(resolution_/2.0, resolution_/2.0, resolution_/2.0);
        if (env_ptr_->getVoxelState(tmp) != 0)
        {
          ll_free_p = tmp;
          break;
        }
      }
      wp_[i] = (ll_free_p + r_free_p) /2;
//       cout << "left&right not clear, wp: " << i << ": " << wp_[i].transpose() << endl;
      continue;
    }
  }
}

}