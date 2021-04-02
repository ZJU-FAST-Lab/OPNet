#include "krrtstar/visualize_rviz.h"
#include <ros/ros.h>
#include <queue>
#include <string>

VisualRviz::VisualRviz(ros::NodeHandle nh): nh_(nh) 
{
    tree_node_pos_point_pub_ = nh_.advertise<visualization_msgs::Marker>("tree_node_pos_points", 1);
    tree_node_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("tree_node_vel_vecs", 1);
    rand_sample_pos_point_pub_ = nh_.advertise<visualization_msgs::Marker>("rand_sample_pos_points", 1);
    rand_sample_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("rand_sample_vel_vecs", 1);
    rand_sample_acc_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("rand_sample_acc_vecs", 1);
    tree_trunks_pub_ = nh_.advertise<visualization_msgs::Marker>("tree_trunks", 1);
    tree_traj_pos_point_pub_ = nh_.advertise<visualization_msgs::Marker>("tree_traj_pos", 1);
    tree_traj_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("tree_traj_vel", 1);
    tree_traj_acc_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("tree_traj_acc", 1);
    a_traj_pos_point_pub_ = nh_.advertise<visualization_msgs::Marker>("a_traj_pos", 1);
    a_traj_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("a_traj_vel", 1);
    a_traj_acc_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("a_traj_acc", 1);
    first_traj_pos_point_pub_ = nh_.advertise<visualization_msgs::Marker>("first_traj_pos", 1);
    first_traj_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("first_traj_vel", 1);
    first_traj_acc_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("first_traj_acc", 1);
    best_traj_pos_point_pub_ = nh_.advertise<visualization_msgs::Marker>("best_traj_pos", 1);
    best_traj_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("best_traj_vel", 1);
    best_traj_acc_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("best_traj_acc", 1);
    bypass_traj_pos_point_pub_ = nh_.advertise<visualization_msgs::Marker>("bypass_traj_pos", 1);
    bypass_traj_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("bypass_traj_vel", 1);
    bypass_traj_acc_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("bypass_traj_acc", 1);
    start_and_goal_pub_ = nh_.advertise<visualization_msgs::Marker>("start_and_goal", 1);
    skeleton_pub_ = nh_.advertise<visualization_msgs::Marker>("skeleton", 1);
    grad_pub_ = nh_.advertise<visualization_msgs::Marker>("grad", 1);
    topo_pub_ = nh_.advertise<visualization_msgs::Marker>("topo", 1);
    surface_pub_ = nh_.advertise<visualization_msgs::Marker>("traj_covering", 1);
    orphans_pos_pub_ = nh_.advertise<visualization_msgs::Marker>("orphan_pos", 1);
    orphans_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("orphan_vel", 1);
}

void VisualRviz::visualizeAllTreeNode(RRTNode* root, ros::Time local_time) 
{
    visualization_msgs::Marker pos_points, vel_vecs, trunks;
//     visualization_msgs::MarkerArray pos_point_ary, vel_vec_ary, acc_vec_ary;
    RRTNode* node = root;
    
    //layer traverse, bfs
    std::queue<RRTNode*> Q;
    Q.push(node);
    while (!Q.empty()) {
        node = Q.front();
        Q.pop();
        
        geometry_msgs::Point p;
        p.x = node->x[0];
        p.y = node->x[1];
        p.z = node->x[2];
        pos_points.points.push_back(p);
        vel_vecs.points.push_back(p);
        p.x += node->x[3]/5.0;
        p.y += node->x[4]/5.0;
        p.z += node->x[5]/5.0;
        vel_vecs.points.push_back(p);
    
        for (const auto& leafptr : node->children) {
            geometry_msgs::Point parent, leaf;
            parent.x = node->x[0];
            parent.y = node->x[1];
            parent.z = node->x[2];
            leaf.x = leafptr->x[0];
            leaf.y = leafptr->x[1];
            leaf.z = leafptr->x[2];
            trunks.points.push_back(parent);
            trunks.points.push_back(leaf);
            
            Q.push(leafptr);
        }
    }
    
    pos_points.header.frame_id = "map";
    pos_points.header.stamp = local_time;
    pos_points.ns = "tree";
    pos_points.action = visualization_msgs::Marker::ADD;
    pos_points.pose.orientation.w = 1.0;
    pos_points.id = 1;
    pos_points.type = visualization_msgs::Marker::POINTS;
    pos_points.scale.x = 0.1;
    pos_points.scale.y = 0.1;
    pos_points.scale.z = 0.1;
    pos_points.color.g = 0.8f;  //green points
    pos_points.color.a = 1.0;
    
    vel_vecs.header.frame_id = "map";
    vel_vecs.header.stamp = local_time;
    vel_vecs.ns = "tree";
    vel_vecs.action = visualization_msgs::Marker::ADD;
    vel_vecs.lifetime = ros::Duration(0);
    vel_vecs.pose.orientation.w = 1.0;
    vel_vecs.id = 2;
    vel_vecs.type = visualization_msgs::Marker::LINE_LIST;
    vel_vecs.scale.x = 0.03;
    vel_vecs.color.r = 1.0f; //red lines
    vel_vecs.color.a = 1.0;
    
    trunks.header.frame_id = "map";
    trunks.header.stamp = local_time;
    trunks.ns = "tree";
    trunks.action = visualization_msgs::Marker::ADD;
    trunks.lifetime = ros::Duration(0);
    trunks.pose.orientation.w = 1.0;
    trunks.id = 3;
    trunks.type = visualization_msgs::Marker::LINE_LIST;
    trunks.scale.x = 0.03;
    trunks.color.b = 2.0f; //blue paths
    trunks.color.g = 0.9f;
    trunks.color.a = 1.0;
    
    tree_node_pos_point_pub_.publish(pos_points);
    tree_node_vel_vec_pub_.publish(vel_vecs);
    tree_trunks_pub_.publish(trunks);
}

void VisualRviz::visualizeSampledState(const State& node, ros::Time local_time) 
{
    visualization_msgs::Marker pos_point, vel_vec, acc_vec;
    
    geometry_msgs::Point p;
    p.x = node[0];
    p.y = node[1];
    p.z = node[2];
    
    pos_point.points.push_back(p);
    pos_point.header.frame_id = "map";
    pos_point.header.stamp = local_time;
    pos_point.ns = "sample";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(10);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 10;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.05;
    pos_point.scale.y = 0.05;
    pos_point.scale.z = 0.05;
    pos_point.color.g = 0.8f;  //green point
    pos_point.color.a = 1.0;
    
    vel_vec.points.push_back(p);
    p.x += node[3];
    p.y += node[4];
    p.z += node[5];
    vel_vec.points.push_back(p);
    vel_vec.header.frame_id = "map";
    vel_vec.header.stamp = local_time;
    vel_vec.ns = "sample";
    vel_vec.action = visualization_msgs::Marker::ADD;
    vel_vec.lifetime = ros::Duration(0);
    vel_vec.pose.orientation.w = 1.0;
    vel_vec.id = 20;
    vel_vec.type = visualization_msgs::Marker::ARROW;
    vel_vec.scale.x = 0.03;
    vel_vec.color.r = 1.0f; //red line
    vel_vec.color.a = 1.0;
    
    rand_sample_pos_point_pub_.publish(pos_point);
    rand_sample_vel_vec_pub_.publish(vel_vec);
}

void VisualRviz::visualizeTreeTraj(const std::vector<State>& x, const std::vector<Control>& u, ros::Time local_time) 
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
    
    pos_point.header.frame_id = "NED";
    pos_point.header.stamp = local_time;
    pos_point.ns = "traj";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 134;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.03;
    pos_point.scale.y = 0.03;
    pos_point.scale.z = 0.03;
    pos_point.color.r = 0;
    pos_point.color.g = 0;
    pos_point.color.b = 1; // blue
    pos_point.color.a = 1.0;
    
    vel_vec.header.frame_id = "NED";
    vel_vec.header.stamp = local_time;
    vel_vec.ns = "traj";
    vel_vec.action = visualization_msgs::Marker::ADD;
    vel_vec.lifetime = ros::Duration(0);
    vel_vec.pose.orientation.w = 1.0;
    vel_vec.id = 200;
    vel_vec.type = visualization_msgs::Marker::LINE_LIST;
    vel_vec.scale.x = 0.03;
    vel_vec.color.r = 0;
    vel_vec.color.g = 1.0f;
    vel_vec.color.b = 1.0f; //light blue line
    vel_vec.color.a = 1.0;
    
    acc_vec.header.frame_id = "NED";
    acc_vec.header.stamp = local_time;
    acc_vec.ns = "traj";
    acc_vec.action = visualization_msgs::Marker::ADD;
    acc_vec.lifetime = ros::Duration(0);
    acc_vec.pose.orientation.w = 1.0;
    acc_vec.id = 300;
    acc_vec.type = visualization_msgs::Marker::LINE_LIST;
    acc_vec.scale.x = 0.03;
    acc_vec.color.r = 0.24f;
    acc_vec.color.g = 0.38f;
    acc_vec.color.b = 0.8;
    acc_vec.color.a = 0.4;
    
    tree_traj_pos_point_pub_.publish(pos_point);
    tree_traj_vel_vec_pub_.publish(vel_vec);
    tree_traj_acc_vec_pub_.publish(acc_vec);
}

void VisualRviz::visualizeATraj(const std::vector<State>& x, const std::vector<Control>& u, ros::Time local_time) 
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
    pos_point.id = 100;
    pos_point.type = visualization_msgs::Marker::POINTS;
//     pos_point.scale.x = 0.07;
//     pos_point.scale.y = 0.07;
//     pos_point.scale.z = 0.07;
//     pos_point.color.r = 0.24;
//     pos_point.color.g = 0.7;
//     pos_point.color.b = 0.46;//green
    pos_point.scale.x = 0.1;
    pos_point.scale.y = 0.1;
    pos_point.scale.z = 0.1;
    pos_point.color.r = 0;
    pos_point.color.g = 0;
    pos_point.color.b = 1; // blue
    pos_point.color.a = 1.0;
    
    vel_vec.header.frame_id = "map";
    vel_vec.header.stamp = local_time;
    vel_vec.ns = "traj";
    vel_vec.action = visualization_msgs::Marker::ADD;
    vel_vec.lifetime = ros::Duration(0);
    vel_vec.pose.orientation.w = 1.0;
    vel_vec.id = 200;
    vel_vec.type = visualization_msgs::Marker::LINE_LIST;
    vel_vec.scale.x = 0.03;
    vel_vec.color.r = 0;
    vel_vec.color.g = 1.0f;
    vel_vec.color.b = 1.0f; //light blue line
    vel_vec.color.a = 1.0;
    
    acc_vec.header.frame_id = "map";
    acc_vec.header.stamp = local_time;
    acc_vec.ns = "traj";
    acc_vec.action = visualization_msgs::Marker::ADD;
    acc_vec.lifetime = ros::Duration(0);
    acc_vec.pose.orientation.w = 1.0;
    acc_vec.id = 300;
    acc_vec.type = visualization_msgs::Marker::LINE_LIST;
    acc_vec.scale.x = 0.03;
    acc_vec.color.r = 0.24f;
    acc_vec.color.g = 0.38f;
    acc_vec.color.b = 0.8;
    acc_vec.color.a = 0.3;
    
    a_traj_pos_point_pub_.publish(pos_point);
    a_traj_vel_vec_pub_.publish(vel_vec);
    a_traj_acc_vec_pub_.publish(acc_vec);
}

void VisualRviz::visualizeBestTraj(const std::vector< State >& x, const std::vector< Control >& u, ros::Time local_time)
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
    pos_point.id = 100;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.1;
    pos_point.scale.y = 0.1;
    pos_point.scale.z = 0.1;
    pos_point.color.r = 0.8;
    pos_point.color.b = 0.2;
    pos_point.color.g = 0.2f; 
    pos_point.color.a = 1.0;
    
    vel_vec.header.frame_id = "map";
    vel_vec.header.stamp = local_time;
    vel_vec.ns = "traj";
    vel_vec.action = visualization_msgs::Marker::ADD;
    vel_vec.lifetime = ros::Duration(0);
    vel_vec.pose.orientation.w = 1.0;
    vel_vec.id = 200;
    vel_vec.type = visualization_msgs::Marker::LINE_LIST;
    vel_vec.scale.x = 0.03;
    vel_vec.color.r = 0;
    vel_vec.color.g = 1.0f;
    vel_vec.color.b = 1.0f; //light blue line
    vel_vec.color.a = 1.0;
    
    acc_vec.header.frame_id = "map";
    acc_vec.header.stamp = local_time;
    acc_vec.ns = "traj";
    acc_vec.action = visualization_msgs::Marker::ADD;
    acc_vec.lifetime = ros::Duration(0);
    acc_vec.pose.orientation.w = 1.0;
    acc_vec.id = 300;
    acc_vec.type = visualization_msgs::Marker::LINE_LIST;
    acc_vec.scale.x = 0.03;
    acc_vec.color.g = 0.85f;
    acc_vec.color.a = 0.3;
    
    best_traj_pos_point_pub_.publish(pos_point);
    best_traj_vel_vec_pub_.publish(vel_vec);
    best_traj_acc_vec_pub_.publish(acc_vec);
}

void VisualRviz::visualizeFirstTraj(const std::vector<State>& x, const std::vector<Control>& u, ros::Time local_time) 
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
    pos_point.id = 100;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.1;
    pos_point.scale.y = 0.1;
    pos_point.scale.z = 0.1; 
    pos_point.color.r = 0;
    pos_point.color.g = 0.8;
    pos_point.color.b = 1.0f;  //light blue point
    pos_point.color.a = 1.0;
    
    vel_vec.header.frame_id = "map";
    vel_vec.header.stamp = local_time;
    vel_vec.ns = "traj";
    vel_vec.action = visualization_msgs::Marker::ADD;
    vel_vec.lifetime = ros::Duration(0);
    vel_vec.pose.orientation.w = 1.0;
    vel_vec.id = 200;
    vel_vec.type = visualization_msgs::Marker::LINE_LIST;
    vel_vec.scale.x = 0.03;
    vel_vec.color.r = 0;
    vel_vec.color.g = 1.0f;
    vel_vec.color.b = 1.0f; //light blue line
    vel_vec.color.a = 1.0;
    
    acc_vec.header.frame_id = "map";
    acc_vec.header.stamp = local_time;
    acc_vec.ns = "traj";
    acc_vec.action = visualization_msgs::Marker::ADD;
    acc_vec.lifetime = ros::Duration(0);
    acc_vec.pose.orientation.w = 1.0;
    acc_vec.id = 300;
    acc_vec.type = visualization_msgs::Marker::LINE_LIST;
    acc_vec.scale.x = 0.03;
    acc_vec.color.r = 0; 
    acc_vec.color.g = 0.85f;
    acc_vec.color.b = 1;
    acc_vec.color.a = 0.3;
    
    first_traj_pos_point_pub_.publish(pos_point);
    first_traj_vel_vec_pub_.publish(vel_vec);
    first_traj_acc_vec_pub_.publish(acc_vec);
}

void VisualRviz::visualizeBypassTraj(const std::vector<State>& x, const std::vector<Control>& u, ros::Time local_time) 
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
    pos_point.id = 100;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.1;
    pos_point.scale.y = 0.1;
    pos_point.scale.z = 0.1;
    pos_point.color.b = 0.8f;  //blue point
    pos_point.color.a = 1.0;
    
    vel_vec.header.frame_id = "map";
    vel_vec.header.stamp = local_time;
    vel_vec.ns = "traj";
    vel_vec.action = visualization_msgs::Marker::ADD;
    vel_vec.lifetime = ros::Duration(0);
    vel_vec.pose.orientation.w = 1.0;
    vel_vec.id = 200;
    vel_vec.type = visualization_msgs::Marker::LINE_LIST;
    vel_vec.scale.x = 0.03;
    vel_vec.color.r = 0;
    vel_vec.color.g = 1.0f;
    vel_vec.color.b = 1.0f; //light blue line
    vel_vec.color.a = 1.0;
    
    acc_vec.header.frame_id = "map";
    acc_vec.header.stamp = local_time;
    acc_vec.ns = "traj";
    acc_vec.action = visualization_msgs::Marker::ADD;
    acc_vec.lifetime = ros::Duration(0);
    acc_vec.pose.orientation.w = 1.0;
    acc_vec.id = 300;
    acc_vec.type = visualization_msgs::Marker::LINE_LIST;
    acc_vec.scale.x = 0.03;
    acc_vec.color.r = 1.0f; //yellow line
    acc_vec.color.g = 0.85f;
    acc_vec.color.a = 0.5;
    
    bypass_traj_pos_point_pub_.publish(pos_point);
    bypass_traj_vel_vec_pub_.publish(vel_vec);
    bypass_traj_acc_vec_pub_.publish(acc_vec);
}

void VisualRviz::visualizeStartAndGoal(State start, State goal, ros::Time local_time)
{
    visualization_msgs::Marker pos_point;
    
    geometry_msgs::Point p;
    p.x = start[0];
    p.y = start[1];
    p.z = start[2];
    pos_point.points.push_back(p);
    p.x = goal[0];
    p.y = goal[1];
    p.z = goal[2];
    pos_point.points.push_back(p);
    
    pos_point.header.frame_id = "map";
    pos_point.header.stamp = local_time;
    pos_point.ns = "s_g";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 11;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.4;
    pos_point.scale.y = 0.4;
    pos_point.scale.z = 0.4;
    pos_point.color.r = 0.62;
    pos_point.color.g = 0.44;
    pos_point.color.b = 0.98;
    pos_point.color.a = 1.0;
    
    start_and_goal_pub_.publish(pos_point);
}

void VisualRviz::visualizeSkeleton(const std::vector<Eigen::Vector3d>& skeleton,
                                   const std::vector<Eigen::Vector3d>& grads, 
                                   const vector< double >& dists,
																	 ros::Time local_time)
{
    if (skeleton.size() != grads.size())
        return;
    visualization_msgs::Marker pos_point, grad;
    geometry_msgs::Point p;
    for (int i=0; i<skeleton.size(); ++i) {
        p.x = skeleton[i][0];
        p.y = skeleton[i][1];
        p.z = dists[i];
        pos_point.points.push_back(p);
        grad.points.push_back(p);   
        p.x += grads[i][0]/6.0;
        p.y += grads[i][1]/6.0;
        p.z += grads[i][2]/6.0;
        grad.points.push_back(p);
    }
    
    grad.header.frame_id = "map";
    grad.header.stamp = local_time;
    grad.ns = "grad";
    grad.action = visualization_msgs::Marker::ADD;
    grad.lifetime = ros::Duration(0);
    grad.pose.orientation.w = 1.0;
    grad.id = 112;
    grad.type = visualization_msgs::Marker::LINE_LIST;
    grad.scale.x = 0.015;
    grad.color.g = 0.5f;
    grad.color.b = 0.7f;
    grad.color.a = 0.8;
    
    pos_point.header.frame_id = "map";
    pos_point.header.stamp = local_time;
    pos_point.ns = "skeleton";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 113;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.015;
    pos_point.scale.y = 0.015;
    pos_point.scale.z = 0.015;
    pos_point.color.r = 1.0f;
    pos_point.color.b = 0.3f;
    pos_point.color.a = 1.0;

    grad_pub_.publish(grad);
    skeleton_pub_.publish(pos_point);
}

void VisualRviz::visualizeTrajCovering(const vector< Eigen::Vector3d >& covering_grids, double grid_len, ros::Time local_time)
{
    if (covering_grids.size() == 0)
        return;
    visualization_msgs::Marker surface_point;
    geometry_msgs::Point p;
    for (int i=0; i<covering_grids.size(); ++i) {
        p.x = covering_grids[i][0];
        p.y = covering_grids[i][1];
        p.z = covering_grids[i][2];
        surface_point.points.push_back(p);
    }
    
    surface_point.header.frame_id = "map";
    surface_point.header.stamp = local_time;
    surface_point.ns = "surface";
    surface_point.action = visualization_msgs::Marker::ADD;
    surface_point.lifetime = ros::Duration(0);
    surface_point.pose.orientation.w = 1.0;
    surface_point.pose.orientation.x = 0.0;
    surface_point.pose.orientation.y = 0.0;
    surface_point.pose.orientation.z = 0.0;
    surface_point.id = 182;
    surface_point.type = visualization_msgs::Marker::POINTS;
    surface_point.scale.x = grid_len;
    surface_point.scale.y = grid_len;
    surface_point.scale.z = grid_len;
    surface_point.color.r = 0.0f;
    surface_point.color.g = 0.0f;
    surface_point.color.b = 0.0f;
    surface_point.color.a = 0.9;

    surface_pub_.publish(surface_point);
}

void VisualRviz::visualizeTopo(const std::vector<Eigen::Vector3d>& p_head, 
                               const std::vector<Eigen::Vector3d>& tracks, 
															 ros::Time local_time)
{
    if (tracks.empty() || p_head.empty() || tracks.size()!= p_head.size())
        return;
    
    visualization_msgs::Marker topo;
    geometry_msgs::Point p;
    
    for (int i=0; i<tracks.size(); i++) {
        p.x = p_head[i][0];
        p.y = p_head[i][1];
        p.z = p_head[i][2];
        topo.points.push_back(p);
        p.x += tracks[i][0];
        p.y += tracks[i][1];
        p.z += tracks[i][2];
        topo.points.push_back(p);
    }
    
    topo.header.frame_id = "map";
    topo.header.stamp = local_time;
    topo.ns = "topo";
    topo.action = visualization_msgs::Marker::ADD;
    topo.lifetime = ros::Duration(0);
    topo.pose.orientation.w = 1.0;
    topo.id = 117;
    topo.type = visualization_msgs::Marker::LINE_LIST;
    topo.scale.x = 0.15;
    topo.color.g = 0.7f;
    topo.color.b = 0.2f;
    topo.color.a = 1.0;
    
    topo_pub_.publish(topo);
}

void VisualRviz::visualizeOrphans(const vector<State>& ophs, ros::Time local_time)
{
    visualization_msgs::Marker pos_point, vel_vec;
    geometry_msgs::Point p;
    
    for (int i=0; i<ophs.size(); i++) {
        p.x = ophs[i](0,0);
        p.y = ophs[i](1,0);
        p.z = ophs[i](2,0);
        pos_point.points.push_back(p);
        
        vel_vec.points.push_back(p);
        p.x += ophs[i](3,0)/10.0;
        p.y += ophs[i](4,0)/10.0;
        p.z += ophs[i](5,0)/10.0;
        vel_vec.points.push_back(p);
    }
    
    pos_point.header.frame_id = "map";
    pos_point.header.stamp = local_time;
    pos_point.ns = "orphan";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 43;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.1;
    pos_point.scale.y = 0.1;
    pos_point.scale.z = 0.1;
    pos_point.color.g = 1.0f;  //green point
    pos_point.color.a = 1.0;
    
    vel_vec.header.frame_id = "map";
    vel_vec.header.stamp = local_time;
    vel_vec.ns = "orphan";
    vel_vec.action = visualization_msgs::Marker::ADD;
    vel_vec.lifetime = ros::Duration(0);
    vel_vec.pose.orientation.w = 1.0;
    vel_vec.id = 244;
    vel_vec.type = visualization_msgs::Marker::LINE_LIST;
    vel_vec.scale.x = 0.1;
    vel_vec.scale.y = 0.1;
    vel_vec.scale.z = 0.1;
    vel_vec.color.r = 0.7f; //red line
    vel_vec.color.a = 1.0;
    
    orphans_pos_pub_.publish(pos_point);
    orphans_vel_vec_pub_.publish(vel_vec);
}

void VisualRviz::visualizeTopoPaths(const vector<vector<Eigen::Vector3d>>& paths, int id, Eigen::Vector4d color, ros::Time local_time)
{
  if (paths.size() == 0)
    return;

  visualization_msgs::Marker mk;
  mk.header.frame_id = "map";
  mk.header.stamp = local_time;
  mk.type = visualization_msgs::Marker::LINE_LIST;
  mk.ns = "topo";
  mk.id = id;
  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;
  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);
  mk.scale.x = 0.05;

  geometry_msgs::Point pt;
  for (const auto& path : paths)
  {
    int wp_num = path.size();
    for (int i = 0; i < wp_num - 1; ++i)
    {
      pt.x = path[i](0);
      pt.y = path[i](1);
      pt.z = path[i](2);
      mk.points.push_back(pt);
      pt.x = path[i + 1](0);
      pt.y = path[i + 1](1);
      pt.z = path[i + 1](2);
      mk.points.push_back(pt);
    }
  }

  topo_pub_.publish(mk);
}