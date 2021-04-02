#include <vis_utils/planning_visualization.h>

using std::cout;
using std::endl;
namespace fast_planner
{
PlanningVisualization::PlanningVisualization(ros::NodeHandle& nh)
{
  node = nh;

  traj_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/trajectory", 10);
  pubs_.push_back(traj_pub_);

  topo_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/topo_path", 10);
  pubs_.push_back(topo_pub_);

  predict_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/prediction", 10);
  pubs_.push_back(predict_pub_);
  
  ref_traj_pos_point_pub_ = node.advertise<visualization_msgs::Marker>("ref_traj_pos", 1);
  ref_traj_vel_vec_pub_ = node.advertise<visualization_msgs::Marker>("ref_traj_vel", 1);
  ref_traj_acc_vec_pub_ = node.advertise<visualization_msgs::Marker>("ref_traj_acc", 1);

  opti_traj_pos_point_pub_ = node.advertise<visualization_msgs::Marker>("opti_traj_pos", 1);
  opti_traj_vel_vec_pub_ = node.advertise<visualization_msgs::Marker>("opti_traj_vel", 1);
  opti_traj_acc_vec_pub_ = node.advertise<visualization_msgs::Marker>("opti_traj_acc", 1);
  
  bspl_opti_traj_pos_point_pub_ = node.advertise<visualization_msgs::Marker>("bspl_opti_traj_pos", 1);
  bspl_opti_traj_vel_vec_pub_ = node.advertise<visualization_msgs::Marker>("bspl_opti_traj_vel", 1);
  bspl_opti_traj_acc_vec_pub_ = node.advertise<visualization_msgs::Marker>("bspl_opti_traj_acc", 1);
  
  voxel_pub_ = node.advertise<visualization_msgs::Marker>("collisions", 1);
  cover_pub_ = node.advertise<visualization_msgs::Marker>("optimized_traj_cover", 1);
  
  last_graph_num_ = 0;
  last_path_num_ = 0;
  last_bspline_num_ = 0;
  last_guide_num_ = 0;
}

void PlanningVisualization::displaySphereList(const vector<Eigen::Vector3d>& list, double resolution,
                                              Eigen::Vector4d color, int id, int pub_id)
{
  visualization_msgs::Marker mk;
  mk.header.frame_id = "map";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::SPHERE_LIST;
  mk.action = visualization_msgs::Marker::DELETE;
  mk.id = id;
  pubs_[pub_id].publish(mk);

  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;
  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);
  mk.scale.x = resolution;
  mk.scale.y = resolution;
  mk.scale.z = resolution;
  geometry_msgs::Point pt;
  for (int i = 0; i < int(list.size()); i++)
  {
    pt.x = list[i](0);
    pt.y = list[i](1);
    pt.z = list[i](2);
    mk.points.push_back(pt);
  }
  pubs_[pub_id].publish(mk);
  ros::Duration(0.001).sleep();
}

void PlanningVisualization::displayCubeList(const vector<Eigen::Vector3d>& list, double resolution,
                                            Eigen::Vector4d color, int id, int pub_id)
{
  visualization_msgs::Marker mk;
  mk.header.frame_id = "map";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::CUBE_LIST;
  mk.action = visualization_msgs::Marker::DELETE;
  mk.id = id;
  pubs_[pub_id].publish(mk);

  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;
  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);
  mk.scale.x = resolution;
  mk.scale.y = resolution;
  mk.scale.z = resolution;
  geometry_msgs::Point pt;
  for (int i = 0; i < int(list.size()); i++)
  {
    pt.x = list[i](0);
    pt.y = list[i](1);
    pt.z = list[i](2);
    mk.points.push_back(pt);
  }
  pubs_[pub_id].publish(mk);

  ros::Duration(0.001).sleep();
}

void PlanningVisualization::displayLineList(const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2,
                                            double line_width, Eigen::Vector4d color, int id, int pub_id)
{
  visualization_msgs::Marker mk;
  mk.header.frame_id = "map";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::LINE_LIST;
  mk.action = visualization_msgs::Marker::DELETE;
  mk.id = id;
  pubs_[pub_id].publish(mk);

  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;
  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);
  mk.scale.x = line_width;

  geometry_msgs::Point pt;

  for (int i = 0; i < int(list1.size()); ++i)
  {
    pt.x = list1[i](0);
    pt.y = list1[i](1);
    pt.z = list1[i](2);

    mk.points.push_back(pt);

    pt.x = list2[i](0);
    pt.y = list2[i](1);
    pt.z = list2[i](2);

    mk.points.push_back(pt);
  }

  pubs_[pub_id].publish(mk);

  ros::Duration(0.001).sleep();
}


void PlanningVisualization::drawSelectTopoPaths(vector<vector<vector<Eigen::Vector3d>>>& paths, double line_width)
{
  Eigen::Vector4d color1(1, 1, 1, 1);
  for (int i = 0; i < last_path_num_; ++i)
  {
    vector<Eigen::Vector3d> empty;
    displayLineList(empty, empty, line_width, color1, SELECT_PATH + i % 100, 1);
    displaySphereList(empty, line_width, color1, PATH + i % 100, 1);
  }

  last_path_num_ = 0;

  int total_path_num = 0;
  for (int i = 0; i < paths.size(); ++i)
    for (vector<vector<Eigen::Vector3d>>::iterator it = paths[i].begin(); it != paths[i].end(); ++it)
    {
      total_path_num++;
    }

  for (int i = 0; i < paths.size(); ++i)
  {
    for (vector<vector<Eigen::Vector3d>>::iterator it = paths[i].begin(); it != paths[i].end(); ++it)
    {
      // draw one path
      vector<Eigen::Vector3d> edge_pt1, edge_pt2;
      for (int j = 0; j < (*it).size() - 1; ++j)
      {
        edge_pt1.push_back((*it)[j]);
        edge_pt2.push_back((*it)[j + 1]);
      }
      // path_num % 3 == 0 ? color1 : (path_num % 3 == 1 ? color2 : color3)
      displayLineList(edge_pt1, edge_pt2, line_width, getColor(double(last_path_num_) / (total_path_num)),
                      SELECT_PATH + last_path_num_ % 100, 1);
      ++last_path_num_;
    }
  }
}

void PlanningVisualization::drawFilteredTopoPaths(vector<vector<vector<Eigen::Vector3d>>>& guides, double size)
{
  Eigen::Vector4d color1(1, 1, 1, 1);
  for (int i = 0; i < last_guide_num_; ++i)
  {
    vector<Eigen::Vector3d> empty;
    displayLineList(empty, empty, size, color1, FILTERED_PATH + i % 100, 1);
  }

  last_guide_num_ = 0;

  int total_guide_num = 0;
  for (int i = 0; i < guides.size(); ++i)
    for (vector<vector<Eigen::Vector3d>>::iterator it = guides[i].begin(); it != guides[i].end(); ++it)
    {
      total_guide_num++;
    }

  for (int i = 0; i < guides.size(); ++i)
  {
    for (vector<vector<Eigen::Vector3d>>::iterator it = guides[i].begin(); it != guides[i].end(); ++it)
    {
      // draw one path
      vector<Eigen::Vector3d> edge_pt1, edge_pt2;
      for (int j = 0; j < (*it).size() - 1; ++j)
      {
        edge_pt1.push_back((*it)[j]);
        edge_pt2.push_back((*it)[j + 1]);
      }
      displayLineList(edge_pt1, edge_pt2, size, getColor(double(last_guide_num_) / (total_guide_num), 0.2),
                      FILTERED_PATH + last_guide_num_ % 100, 1);
      ++last_guide_num_;
    }
  }
}

void PlanningVisualization::drawGoal(Eigen::Vector3d goal, double resolution, Eigen::Vector4d color, int id)
{
  vector<Eigen::Vector3d> goal_vec = { goal };

  displaySphereList(goal_vec, resolution, color, GOAL + id % 100);
}

void PlanningVisualization::drawPath(const vector<Eigen::Vector3d>& path, double resolution, Eigen::Vector4d color,
                                     int id)
{
  displayCubeList(path, resolution, color, PATH + id % 100);
}

void PlanningVisualization::visualizeOptiTraj(const std::vector<Eigen::Matrix<double,6,1>>& x, const std::vector<Eigen::Matrix<double,3,1>>& u, ros::Time local_time) 
{
    if (x.empty() || u.empty())
        return;
    
    visualization_msgs::Marker pos_point, vel_vec, acc_vec;
    geometry_msgs::Point p, a;
    //x and u are of same size;
    for (int i=0; i<x.size(); ++i) {
        p.x = x[i](0,0);
        p.y = x[i](1,0);
        p.z = x[i](2,0);
        a.x = x[i](0,0);
        a.y = x[i](1,0);
        a.z = x[i](2,0);
        pos_point.points.push_back(p);
                
        vel_vec.points.push_back(p);
        p.x += x[i](3,0)/5.0;
        p.y += x[i](4,0)/5.0;
        p.z += x[i](5,0)/5.0;
        vel_vec.points.push_back(p);
        
        acc_vec.points.push_back(a);
        a.x += u[i](0,0)/1.0;
        a.y += u[i](1,0)/1.0;
        a.z += u[i](2,0)/1.0;
        acc_vec.points.push_back(a);
    }
    
    pos_point.header.frame_id = "map";
    pos_point.header.stamp = local_time;
    pos_point.ns = "traj";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 101;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.1;
    pos_point.scale.y = 0.1;
    pos_point.scale.z = 0.1;
    pos_point.color.r = 0.118;
    pos_point.color.g = 0.58;
    pos_point.color.b = 1; // blue
    pos_point.color.a = 1.0;
    
    vel_vec.header.frame_id = "map";
    vel_vec.header.stamp = local_time;
    vel_vec.ns = "traj";
    vel_vec.action = visualization_msgs::Marker::ADD;
    vel_vec.lifetime = ros::Duration(0);
    vel_vec.pose.orientation.w = 1.0;
    vel_vec.id = 201;
    vel_vec.type = visualization_msgs::Marker::LINE_LIST;
    vel_vec.scale.x = 0.03;
    vel_vec.color.r = 1.0f; 
    vel_vec.color.g = 0.5f;
    vel_vec.color.a = 1.0;
    
    acc_vec.header.frame_id = "map";
    acc_vec.header.stamp = local_time;
    acc_vec.ns = "traj";
    acc_vec.action = visualization_msgs::Marker::ADD;
    acc_vec.lifetime = ros::Duration(0);
    acc_vec.pose.orientation.w = 1.0;
    acc_vec.id = 301;
    acc_vec.type = visualization_msgs::Marker::LINE_LIST;
    acc_vec.scale.x = 0.03;
    acc_vec.color.r = 1;
    acc_vec.color.g = 0.55f;
    acc_vec.color.a = 0.5;
    
    opti_traj_pos_point_pub_.publish(pos_point);
    opti_traj_vel_vec_pub_.publish(vel_vec);
    opti_traj_acc_vec_pub_.publish(acc_vec);
}

void PlanningVisualization::visualizeRefTraj(const std::vector<Eigen::Matrix<double,6,1>>& x, const std::vector<Eigen::Matrix<double,3,1>>& u, ros::Time local_time) 
{
    if (x.empty() || u.empty())
        return;
    
    visualization_msgs::Marker pos_point, vel_vec, acc_vec;
    geometry_msgs::Point p, a;
    //x and u are of same size;
    for (int i=0; i<x.size(); ++i) {
        p.x = x[i](0,0);
        p.y = x[i](1,0);
        p.z = x[i](2,0);
        a.x = x[i](0,0);
        a.y = x[i](1,0);
        a.z = x[i](2,0);
        pos_point.points.push_back(p);
                
        vel_vec.points.push_back(p);
        p.x += x[i](3,0)/5.0;
        p.y += x[i](4,0)/5.0;
        p.z += x[i](5,0)/5.0;
        vel_vec.points.push_back(p);
        
        acc_vec.points.push_back(a);
        a.x += u[i](0,0)/1.0;
        a.y += u[i](1,0)/1.0;
        a.z += u[i](2,0)/1.0;
        acc_vec.points.push_back(a);
    }
    
    pos_point.header.frame_id = "map";
    pos_point.header.stamp = local_time;
    pos_point.ns = "ref_traj";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 101;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.03;
    pos_point.scale.y = 0.03;
    pos_point.scale.z = 0.03;
    pos_point.color.r = 0.118;
    pos_point.color.g = 0.58;
    pos_point.color.b = 1; // blue
    pos_point.color.a = 1.0;
    
    vel_vec.header.frame_id = "map";
    vel_vec.header.stamp = local_time;
    vel_vec.ns = "ref_traj";
    vel_vec.action = visualization_msgs::Marker::ADD;
    vel_vec.lifetime = ros::Duration(0);
    vel_vec.pose.orientation.w = 1.0;
    vel_vec.id = 201;
    vel_vec.type = visualization_msgs::Marker::LINE_LIST;
    vel_vec.scale.x = 0.03;
    vel_vec.color.r = 1.0f; 
    vel_vec.color.g = 0.5f;
    vel_vec.color.a = 1.0;
    
    acc_vec.header.frame_id = "map";
    acc_vec.header.stamp = local_time;
    acc_vec.ns = "ref_traj";
    acc_vec.action = visualization_msgs::Marker::ADD;
    acc_vec.lifetime = ros::Duration(0);
    acc_vec.pose.orientation.w = 1.0;
    acc_vec.id = 301;
    acc_vec.type = visualization_msgs::Marker::LINE_LIST;
    acc_vec.scale.x = 0.03;
    acc_vec.color.r = 1;
    acc_vec.color.g = 0.55f;
    acc_vec.color.a = 0.5;
    
    ref_traj_pos_point_pub_.publish(pos_point);
    ref_traj_vel_vec_pub_.publish(vel_vec);
    ref_traj_acc_vec_pub_.publish(acc_vec);
}

void PlanningVisualization::visualizeBsplOptiTraj(const std::vector<Eigen::Matrix<double,6,1>>& x, const std::vector<Eigen::Matrix<double,3,1>>& u, ros::Time local_time) 
{
    if (x.empty() || u.empty())
        return;
    
    visualization_msgs::Marker pos_point, vel_vec, acc_vec;
    geometry_msgs::Point p, a;
    //x and u are of same size;
    for (int i=0; i<x.size(); ++i) {
        p.x = x[i](0,0);
        p.y = x[i](1,0);
        p.z = x[i](2,0);
        a.x = x[i](0,0);
        a.y = x[i](1,0);
        a.z = x[i](2,0);
        pos_point.points.push_back(p);
                
        vel_vec.points.push_back(p);
        p.x += x[i](3,0)/5.0;
        p.y += x[i](4,0)/5.0;
        p.z += x[i](5,0)/5.0;
        vel_vec.points.push_back(p);
        
        acc_vec.points.push_back(a);
        a.x += u[i](0,0)/1.0;
        a.y += u[i](1,0)/1.0;
        a.z += u[i](2,0)/1.0;
        acc_vec.points.push_back(a);
    }
    
    pos_point.header.frame_id = "map";
    pos_point.header.stamp = local_time;
    pos_point.ns = "traj";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 101;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.15;
    pos_point.scale.y = 0.15;
    pos_point.scale.z = 0.15;
    pos_point.color.r = 0.1;
    pos_point.color.g = 0.55;
    pos_point.color.b = 1;
    pos_point.color.a = 1.0;
    
    vel_vec.header.frame_id = "map";
    vel_vec.header.stamp = local_time;
    vel_vec.ns = "traj";
    vel_vec.action = visualization_msgs::Marker::ADD;
    vel_vec.lifetime = ros::Duration(0);
    vel_vec.pose.orientation.w = 1.0;
    vel_vec.id = 201;
    vel_vec.type = visualization_msgs::Marker::LINE_LIST;
    vel_vec.scale.x = 0.03;
    vel_vec.color.r = 1.0f; 
    vel_vec.color.g = 0.5f;
    vel_vec.color.a = 1.0;
    
    acc_vec.header.frame_id = "map";
    acc_vec.header.stamp = local_time;
    acc_vec.ns = "traj";
    acc_vec.action = visualization_msgs::Marker::ADD;
    acc_vec.lifetime = ros::Duration(0);
    acc_vec.pose.orientation.w = 1.0;
    acc_vec.id = 301;
    acc_vec.type = visualization_msgs::Marker::LINE_LIST;
    acc_vec.scale.x = 0.03;
    acc_vec.color.r = 0.1;
    acc_vec.color.g = 0.55;
    acc_vec.color.b = 1;
    acc_vec.color.a = 0.5;
    
    bspl_opti_traj_pos_point_pub_.publish(pos_point);
    bspl_opti_traj_vel_vec_pub_.publish(vel_vec);
    bspl_opti_traj_acc_vec_pub_.publish(acc_vec);
}

void PlanningVisualization::visualizeCollide(vector< Eigen::Vector3d > positions, ros::Time local_time)
{
  visualization_msgs::Marker pos_point;
  geometry_msgs::Point p;
  //x and u are of same size;
  for (int i=0; i<positions.size(); ++i) {
    p.x = positions[i][0];
    p.y = positions[i][1];
    p.z = positions[i][2];
    pos_point.points.push_back(p);
  }
  
  pos_point.header.frame_id = "map";
  pos_point.header.stamp = local_time;
  pos_point.ns = "colli";
  pos_point.action = visualization_msgs::Marker::ADD;
  pos_point.lifetime = ros::Duration(0);
  pos_point.pose.orientation.w = 1.0;
  pos_point.id = 1011;
  pos_point.type = visualization_msgs::Marker::POINTS;
  pos_point.scale.x = 0.15;
  pos_point.scale.y = 0.15;
  pos_point.scale.z = 0.15;
  pos_point.color.r = 1;
  pos_point.color.g = 0;
  pos_point.color.b = 0;
  pos_point.color.a = 1.0;
  
  voxel_pub_.publish(pos_point);
}

void PlanningVisualization::visualizeTrajCovering(const vector< Eigen::Vector3d >& covering_grids, double grid_len, ros::Time local_time)
{
    if (covering_grids.size() == 0)
        return;
    visualization_msgs::Marker cover_point;
    geometry_msgs::Point p;
    for (int i=0; i<covering_grids.size(); ++i) {
        p.x = covering_grids[i][0];
        p.y = covering_grids[i][1];
        p.z = covering_grids[i][2];
        cover_point.points.push_back(p);
    }
    
    cover_point.header.frame_id = "map";
    cover_point.header.stamp = local_time;
    cover_point.ns = "surface";
    cover_point.action = visualization_msgs::Marker::ADD;
    cover_point.lifetime = ros::Duration(0);
    cover_point.pose.orientation.w = 1.0;
    cover_point.pose.orientation.x = 0.0;
    cover_point.pose.orientation.y = 0.0;
    cover_point.pose.orientation.z = 0.0;
    cover_point.id = 182;
    cover_point.type = visualization_msgs::Marker::POINTS;
    cover_point.scale.x = grid_len;
    cover_point.scale.y = grid_len;
    cover_point.scale.z = grid_len;
    cover_point.color.r = 0.4;
    cover_point.color.g = 1;
    cover_point.color.b = 0.3;
    cover_point.color.a = 0.2;

    cover_pub_.publish(cover_point);
}

Eigen::Vector4d PlanningVisualization::getColor(double h, double alpha)
{
  if (h < 0.0 || h > 1.0)
  {
    std::cout << "h out of range" << std::endl;
    h = 0.0;
  }

  double lambda;
  Eigen::Vector4d color1, color2;

  if (h >= -1e-4 && h < 1.0 / 6)
  {
    lambda = (h - 0.0) * 6;
    color1 = Eigen::Vector4d(1, 0, 0, 1);
    color2 = Eigen::Vector4d(1, 0, 1, 1);
  }
  else if (h >= 1.0 / 6 && h < 2.0 / 6)
  {
    lambda = (h - 1.0 / 6) * 6;
    color1 = Eigen::Vector4d(1, 0, 1, 1);
    color2 = Eigen::Vector4d(0, 0, 1, 1);
  }
  else if (h >= 2.0 / 6 && h < 3.0 / 6)
  {
    lambda = (h - 2.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 0, 1, 1);
    color2 = Eigen::Vector4d(0, 1, 1, 1);
  }
  else if (h >= 3.0 / 6 && h < 4.0 / 6)
  {
    lambda = (h - 3.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 1, 1, 1);
    color2 = Eigen::Vector4d(0, 1, 0, 1);
  }
  else if (h >= 4.0 / 6 && h < 5.0 / 6)
  {
    lambda = (h - 4.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 1, 0, 1);
    color2 = Eigen::Vector4d(1, 1, 0, 1);
  }
  else if (h >= 5.0 / 6 && h <= 1.0 + 1e-4)
  {
    lambda = (h - 5.0 / 6) * 6;
    color1 = Eigen::Vector4d(1, 1, 0, 1);
    color2 = Eigen::Vector4d(1, 0, 0, 1);
  }

  Eigen::Vector4d fcolor = (1 - lambda) * color1 + lambda * color2;
  fcolor(3) = alpha;

  return fcolor;
}
// PlanningVisualization::
}  // namespace fast_planner
