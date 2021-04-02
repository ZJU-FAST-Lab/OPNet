#include "state_machine/fsm.h"
#include <ros/console.h>

namespace fast_planner
{
FSM::FSM()
{
}

FSM::~FSM()
{
}

void FSM::init(ros::NodeHandle &nh)
{
  env_ptr_.reset(new OccMap);
  env_ptr_->init(nh);

  topo_prm_.reset(new TopologyPRM);
  topo_prm_->setEnvironment(env_ptr_);
  topo_prm_->init(nh);

  krrt_planner_ptr_.reset(new KRRTPlanner(nh));
  krrt_planner_ptr_->init(nh);
  krrt_planner_ptr_->setEnv(env_ptr_);
  krrt_planner_ptr_->setTopoFinder(topo_prm_);

  optimizer_ptr_.reset(new TrajOptimizer(nh));

  vis_.reset(new PlanningVisualization(nh));

  qrcode_pose_sub_ = nh_.subscribe("/qrcode_detector/qrcode_position", 1, &FSM::qrcodeCallback, this);
  goal_sub_ = nh_.subscribe("/goal", 1, &FSM::goalCallback, this);
  final_goal_sub_ = nh_.subscribe("/final_goal", 1, &FSM::finalGoalCallback, this);
  traj_pub_ = nh_.advertise<quadrotor_msgs::PolynomialTrajectory>("planning/poly_traj", 10);
  execution_timer_ = nh_.createTimer(ros::Duration(0.01), &FSM::executionCallback, this); // 100Hz
  ref_traj_sub_ = nh_.subscribe("/ref_traj", 1, &FSM::refTrajCallback, this);
  track_err_trig_sub_ = nh_.subscribe("/trig/tracking_err", 1, &FSM::trackErrCallback, this);
  receding_horizon_timer_ = nh.createTimer(ros::Duration(0.5), &FSM::rcdHrzCallback, this);

  nh.param("fsm/use_optimization", use_optimization_, false);
  nh.param("fsm/conservative", conservative_, false);
  nh.param("fsm/replan", replan_, false);
  nh.param("fsm/replan_time", replan_time_, 0.02);
  nh.param("fsm/allow_track_err_replan", allow_track_err_replan_, false);
  nh.param("fsm/e_stop_time_margin", e_stop_time_margin_, 1.0);
  nh.param("fsm/bidirection", bidirection_, false);
  nh.param("occ_map/local_grid_size_y", sense_grid_size_, 64);
  nh.param("fsm/resolution", resolution_, 0.1);

  track_err_replan_ = false;
  new_goal_ = false;
  started_ = false;
  get_final_goal_ = false;
  last_goal_pos_ << 0.0, 0.0, 0.0;
  machine_state_ = INIT;
  cuur_traj_start_time_ = ros::Time::now();
  emergency_stop_pos_ << 0.0, 0.0, 0.0;
  pos_about_to_collide_ << 0.0, 0.0, 0.0;
  remain_safe_time_ = 0.0;
  traj_tree_node_num_ = 0;

  receive_ref_traj_ = false;

  random_device rd;
  gen_ = mt19937_64(rd());
  angle_rand_ = uniform_real_distribution<double>(0.0, M_PI * 2);
  radius_rand_ = uniform_real_distribution<double>(0.2, sense_grid_size_ * resolution_);
  height_rand_ = uniform_real_distribution<double>(-0.5, 0.5);
}

void FSM::refTrajCallback(const quadrotor_msgs::PolynomialTrajectory &traj)
{
  ROS_WARN("[FSM] rcv ref traj.");
  n_segment_ = traj.time.size();
  receive_ref_traj_ = true;

  ref_traj_g2s_.clear();
  for (int i = traj.time.size() - 1; i >= 0; --i)
  {
    RRTNode node;
    node.tau_from_parent = traj.time[i];
    node.n_order = traj.order[i];
    for (int j = 0; j <= traj.order[i]; ++j)
    {
      node.x_coeff[j] = traj.coef_x[i * (traj.order[i] + 1) + j];
      node.y_coeff[j] = traj.coef_y[i * (traj.order[i] + 1) + j];
      node.z_coeff[j] = traj.coef_z[i * (traj.order[i] + 1) + j];
    }
    ref_traj_g2s_.push_back(node);
  }
  //add start node
  RRTNode node;
  ref_traj_g2s_.push_back(node);
  vector<State> vis_x;
  vector<Control> vis_u;
  krrt_planner_ptr_->getStateAndControl(ref_traj_g2s_, &vis_x, &vis_u);
  vis_->visualizeRefTraj(vis_x, vis_u, env_ptr_->getLocalTime());

  idx_ = 0;
  t_in_ref_traj_ = ref_traj_g2s_[idx_].tau_from_parent;
  boost::shared_ptr<quadrotor_msgs::PositionCommand> first_goal;
  first_goal.reset(new quadrotor_msgs::PositionCommand());
  Eigen::VectorXd goal_state = getReplanStateFromPath(t_in_ref_traj_, ref_traj_g2s_);
  first_goal->position.x = goal_state[0];
  first_goal->position.y = goal_state[1];
  first_goal->position.z = goal_state[2];
  first_goal->velocity.x = goal_state[3];
  first_goal->velocity.y = goal_state[4];
  first_goal->velocity.z = goal_state[5];
  first_goal->acceleration.x = goal_state[6];
  first_goal->acceleration.y = goal_state[7];
  first_goal->acceleration.z = goal_state[8];
  goalCallback(first_goal);
}

void FSM::trackErrCallback(const std_msgs::Empty &msg)
{
  if (allow_track_err_replan_)
    track_err_replan_ = true;
}

// bool FSM::computeInterGoalPos(Eigen::Vector3d &inte_pos, const Eigen::Vector3d &curr_pos, const Eigen::Vector3d &goal, double range)
// {

// }

void FSM::finalGoalCallback(const geometry_msgs::PointStamped::ConstPtr &msg)
{
  final_goal_pos_ << msg->point.x, msg->point.y, msg->point.z;
  get_final_goal_ = true;
}

bool FSM::computeInterGoalPos(Eigen::Vector3d &inte_pos, const Eigen::Vector3d &curr_pos, const Eigen::Vector3d &goal, double range)
{
  if (conservative_){
    static Eigen::Vector3d last_sampled_goal;
    ROS_WARN("conservative_");
    Eigen::Vector3d dire = goal - curr_pos;
    double dis = dire.norm();
    Eigen::Vector3d local_goal(goal);
    if (dis <= range)
    {
      if (env_ptr_->getVoxelState(goal) == 0)
      {
        inte_pos = goal;
        return false;
      }
    }
    else
    {
      local_goal = curr_pos + dire.normalized() * range;
    }
    int step = ceil((local_goal - curr_pos).norm() / resolution_ / 2);
    Eigen::Vector3d delta_dire = -1 * dire.normalized() * resolution_ * 2;
    for (int i = 1; i <= step - 20; ++i)
    {
      local_goal += delta_dire;
      // ROS_INFO_STREAM("LOCAL GOAL: "<< local_goal.transpose());
      if (env_ptr_->getVoxelState(local_goal) == 0)
      {
        inte_pos = local_goal;
        return true;
      }
    }
    while (1)
    {
      // float min_cost = last_sampled_goal == Eigen::Vector3d(0,0,0) ? 10000 :
      //    (last_sampled_goal - curr_pos).norm() * 0.5 + (last_sampled_goal - goal).norm() + 0.0;
      float min_cost = 1000;
      bool found_valid_goal = false;
      for (int count = 0; count < 1000; count ++){
        Eigen::Vector3d current_point;
        double angle = angle_rand_(gen_);
        double radius = radius_rand_(gen_);
        double height = height_rand_(gen_);
        double y = radius * sin(angle);
        double x = radius * cos(angle);
        current_point[0] = curr_pos[0] + x;
        current_point[1] = curr_pos[1] + y;
        current_point[2] = curr_pos[2] + height;
        if (env_ptr_->getVoxelState(current_point) == 0)
        {
          float curent_cost = (current_point - curr_pos).norm() * 0.5 + (current_point - goal).norm();
          if (curent_cost < min_cost && (current_point - goal).norm() < (curr_pos - goal).norm() + 0.5)
          {
            min_cost = curent_cost;
            local_goal = current_point;
            found_valid_goal = true;
          }
        }
      }
      if (found_valid_goal)
      {
        inte_pos = local_goal;
        last_sampled_goal = local_goal;
        return true;
      }
    }
  }
  else{
    Eigen::Vector3d dire = goal - curr_pos;
    double dis = dire.norm();
    if (dis <= range)
    {
      inte_pos = goal;
      return false;
    }
    inte_pos = curr_pos + dire.normalized() * range;
    return true;
  }
}

void FSM::rcdHrzCallback(const ros::TimerEvent &event)
{
  if (!get_final_goal_)
    return;
  Eigen::Vector3d inter_goal_pos;
  double range = sense_grid_size_ * resolution_ * 1.0;
  // ROS_WARN_STREAM("final goal: " << final_goal_pos_.transpose() << endl
  //                                << "odom: " << env_ptr_->get_curr_posi() << endl
  //                                << "range" << range);
  if (!computeInterGoalPos(inter_goal_pos, env_ptr_->get_curr_posi(), final_goal_pos_, range))
    get_final_goal_ = false;

  boost::shared_ptr<quadrotor_msgs::PositionCommand> inter_goal;
  inter_goal.reset(new quadrotor_msgs::PositionCommand());
  inter_goal->position.x = inter_goal_pos[0];
  inter_goal->position.y = inter_goal_pos[1];
  inter_goal->position.z = inter_goal_pos[2];
  inter_goal->velocity.x = 0.0;
  inter_goal->velocity.y = 0.0;
  inter_goal->velocity.z = 0.0;
  inter_goal->acceleration.x = 0.0;
  inter_goal->acceleration.y = 0.0;
  inter_goal->acceleration.z = 0.0;
  goalCallback(inter_goal);
}

void FSM::qrcodeCallback(const geometry_msgs::PointStamped::ConstPtr &msg)
{
  Vector3d pos;
  pos << msg->point.x,
      msg->point.y,
      msg->point.z;
  if ((last_goal_pos_ - pos).norm() >= 2.0)
  {
    end_pos_ = pos;
    end_vel_.setZero();
    end_acc_.setZero();
    started_ = true;
    new_goal_ = true;
    last_goal_pos_ = end_pos_;
  }
}

void FSM::goalCallback(const quadrotor_msgs::PositionCommand::ConstPtr &goal_msg)
{
  end_pos_ << goal_msg->position.x,
      goal_msg->position.y,
      goal_msg->position.z;
  end_vel_ << goal_msg->velocity.x,
      goal_msg->velocity.y,
      goal_msg->velocity.z;
  end_acc_ << goal_msg->acceleration.x,
      goal_msg->acceleration.y,
      goal_msg->acceleration.z;
  started_ = true;
  new_goal_ = true;
  last_goal_pos_ = end_pos_;
}

void FSM::executionCallback(const ros::TimerEvent &event)
{
  static ros::Time start_follow_time, collision_detect_time;
  static int replan_state = 1;
  static int fsm_num = 0;
  static double bypass_dura(0.0);
  fsm_num++;
  if (fsm_num == 100)
  {
    printState();
    if (!env_ptr_->odomValid())
    {
      ROS_INFO("no odom.");
    }
    if (!env_ptr_->mapValid())
    {
      ROS_INFO("no map.");
    }
    if (!started_)
    {
      ROS_INFO("wait for goal in %lf but actual in %lf", event.current_expected.toSec(), event.current_real.toSec());
    }
    fsm_num = 0;
  }

  switch (machine_state_)
  {
  case INIT:
  {
    if (!env_ptr_->odomValid())
    {
      return;
    }
    if (!env_ptr_->mapValid())
    {
      return;
    }
    if (!started_)
    {
      return;
    }
    changeState(WAIT_GOAL);
    break;
  }

  case WAIT_GOAL:
  {
    if (!new_goal_)
    {
      return;
    }
    else
    {
      new_goal_ = false;
      changeState(GENERATE_TRAJ);
    }
    break;
  }

  case GENERATE_TRAJ:
  {
    start_pos_ = env_ptr_->get_curr_posi();
    start_vel_ = env_ptr_->get_curr_twist();
    //numerical problem
    if (start_vel_.norm() < 0.05)
    {
      start_vel_(0) = 0.0;
      start_vel_(1) = 0.0;
      start_vel_(2) = 0.0;
    }
    start_acc_.setZero();

    bool success = searchForTraj(start_pos_, start_vel_, start_acc_, end_pos_, end_vel_, end_acc_, replan_time_, bidirection_); //TODO what if it can not finish in 10ms?
    if (success)
    {
      if (use_optimization_)
      {
        ros::Time optimize_start_time = ros::Time::now();
        bool optimize_succ = optimize(path_node_g2s_, start_acc_);
        ros::Time optimize_end_time = ros::Time::now();
        // ROS_INFO_STREAM("seg num: " << path_node_g2s_.size() - 1 << ", optimize time: " << (optimize_end_time - optimize_start_time).toSec());
        vector<State> vis_x;
        vector<Control> vis_u;
        krrt_planner_ptr_->getStateAndControl(path_node_g2s_, &vis_x, &vis_u);
        vis_->visualizeOptiTraj(vis_x, vis_u, env_ptr_->getLocalTime());
      }
      else
      {
        krrt_planner_ptr_->getPath(path_node_g2s_, 2);
      }
      replan_state = 1;
      sendTrajToServer(path_node_g2s_);
      cuur_traj_start_time_ = ros::Time::now();
      new_goal_ = false;
      start_follow_time = ros::Time::now();
      changeState(FOLLOW_TRAJ);
    }
    //else
    //{
    //new_goal_ = false;
    //changeState(WAIT_GOAL);
    //}
    else if (receive_ref_traj_) //stay GENERATE_TRAJ, that is keep searching
    {
      t_in_ref_traj_ += 0.1;
      Eigen::VectorXd goal_state = getReplanStateFromPath(t_in_ref_traj_, ref_traj_g2s_);
      end_pos_ << goal_state[0], goal_state[1], goal_state[2];
      end_vel_ << goal_state[3], goal_state[4], goal_state[5];
      end_acc_ << goal_state[6], goal_state[7], goal_state[8];
    }
    break;
  }

  case FOLLOW_TRAJ:
  {
    double t_during_traj = (ros::Time::now() - cuur_traj_start_time_).toSec();
    if (reachGoal(0.5))
    {
      changeState(WAIT_GOAL);
    }
    else if (receive_ref_traj_ && reachGoal(3.5) && idx_ < n_segment_ - 1)
    {
      boost::shared_ptr<quadrotor_msgs::PositionCommand> goal;
      goal.reset(new quadrotor_msgs::PositionCommand());
      t_in_ref_traj_ += ref_traj_g2s_[idx_++].tau_from_parent;
      Eigen::VectorXd goal_state = getReplanStateFromPath(t_in_ref_traj_, ref_traj_g2s_);
      goal->position.x = goal_state[0];
      goal->position.y = goal_state[1];
      goal->position.z = goal_state[2];
      goal->velocity.x = goal_state[3];
      goal->velocity.y = goal_state[4];
      goal->velocity.z = goal_state[5];
      goal->acceleration.x = goal_state[6];
      goal->acceleration.y = goal_state[7];
      goal->acceleration.z = goal_state[8];
      remain_safe_time_ = krrt_planner_ptr_->getTrajDura() - t_during_traj;
      goalCallback(goal);
    }
    else if (new_goal_)
    {
      ROS_WARN("Replan because of new goal received");
      new_goal_ = false;
      remain_safe_time_ = krrt_planner_ptr_->getTrajDura() - t_during_traj;
      // ROS_INFO("t_during_traj: %lf", t_during_traj);
      // ROS_INFO("remain_safe_time: %lf", remain_safe_time_);
      collision_detect_time = ros::Time::now();
      changeState(REPLAN_TRAJ);
    }
    else if (replan_)
    {
      //replan because remaining traj may collide
      if (checkForReplan())
      {
        ROS_WARN("REPLAN because of future collision");
        collision_detect_time = ros::Time::now();
        changeState(REPLAN_TRAJ);
      }
      else if (track_err_replan_)
      {
        track_err_replan_ = false;
        ROS_WARN("REPLAN because of not tracking closely");
        remain_safe_time_ = krrt_planner_ptr_->getTrajDura() - t_during_traj;
        replan_state = 4;
        collision_detect_time = ros::Time::now();
        changeState(REPLAN_TRAJ);
      }
      else if (close_goal_traj_ && (krrt_planner_ptr_->getTrajDura() - t_during_traj) < 2)
      {
        close_goal_traj_ = false;
        remain_safe_time_ = krrt_planner_ptr_->getTrajDura() - t_during_traj;
        collision_detect_time = ros::Time::now();
        ROS_WARN("REPLAN cause t_remain is run out");
        changeState(REPLAN_TRAJ);
      }
    }
    //       else if ((ros::Time::now() - start_follow_time).toSec() > 2)
    //       {//replan because
    //         changeState(REFINE_REMAINING_TRAJ);
    //       }
    break;
  }

  case REPLAN_TRAJ:
  {
    ros::Time t_replan_start = ros::Time::now();
    //replan once
    double curr_remain_safe_time = remain_safe_time_ - (ros::Time::now() - collision_detect_time).toSec();
    double dt = max(0.0, min(replan_time_, curr_remain_safe_time));
    //start state is in dt (replan_front_time + optimization_time) second from current traj state
    double t_during_traj = (ros::Time::now() - cuur_traj_start_time_).toSec();
    Eigen::VectorXd start_state, e_end_state;
    if (replan_state == 1)
    {
      ROS_INFO_STREAM("state 1, replan from tracking traj");
      e_end_state = getReplanStateFromPath(t_during_traj, path_node_g2s_);
      start_state = getReplanStateFromPath(t_during_traj + dt, path_node_g2s_); //start in future state
    }
    else if (replan_state == 4)
    {
      ROS_INFO_STREAM("state 4, replan from curr state");
      e_end_state = getReplanStateFromPath(-1.0, path_node_g2s_);
      start_state = getReplanStateFromPath(-1.0, path_node_g2s_); //start in curr state
    }
    ROS_INFO_STREAM("replan start state: " << start_state.transpose());
    Eigen::Vector3d start_pos, start_vel, start_acc;
    start_pos = start_state.segment(0, 3);
    start_vel = start_state.segment(3, 3);
    start_acc = start_state.segment(6, 3);
    if (start_vel.norm() < 1e-4)
      start_vel = Vector3d(0.0, 0.0, 0.0);
    if (start_acc.norm() < 1e-4)
      start_acc = Vector3d(0.0, 0.0, 0.0);
    double front_time = dt;
    if (dt <= 0.005)
      front_time = replan_time_;
    bool success = searchForTraj(start_pos, start_vel, start_acc, end_pos_, end_vel_, end_acc_, front_time, bidirection_);
    //found a traj towards goal
    if (success)
    {
      // ROS_WARN("Replan front-end success");
      if (use_optimization_)
      {
        RRTNodeVector optimized_path_node;
        bool optimize_succ = optimize(optimized_path_node, start_acc);
        if (optimize_succ)
        {
          // ROS_WARN("Replan back-end success");
          double replan_duration = (ros::Time::now() - t_replan_start).toSec();
          if (replan_duration < dt)
          {
            // ROS_WARN("wait for it: %lf", dt - replan_duration);
            ros::Duration(dt - replan_duration).sleep();
          }
          replan_state = 1;
          path_node_g2s_ = optimized_path_node;
          sendTrajToServer(path_node_g2s_);
          cuur_traj_start_time_ = ros::Time::now();
          new_goal_ = false;
          start_follow_time = ros::Time::now();
          changeState(FOLLOW_TRAJ);
        }
        else
        {
          double curr_remain_safe_time = remain_safe_time_ - (ros::Time::now() - collision_detect_time).toSec();
          if (curr_remain_safe_time < e_stop_time_margin_ / 2.0)
          {
            ROS_WARN("Replan back-end fail, no time to try another time, use patched traj");
            double replan_duration = (ros::Time::now() - t_replan_start).toSec();
            if (replan_duration < dt)
            {
              ROS_WARN("wait for it: %lf", dt - replan_duration);
              ros::Duration(dt - replan_duration).sleep();
            }
            replan_state = 1;
            path_node_g2s_ = optimized_path_node;
            sendTrajToServer(path_node_g2s_);
            cuur_traj_start_time_ = ros::Time::now();
            new_goal_ = false;
            start_follow_time = ros::Time::now();
            changeState(FOLLOW_TRAJ);
          }
          else
          {
            double replan_duration = (ros::Time::now() - t_replan_start).toSec();
            if (replan_duration < dt)
            {
              ROS_WARN("wait for it: %lf", dt - replan_duration);
              ros::Duration(dt - replan_duration).sleep();
            }
            replan_state = 1;
            path_node_g2s_ = optimized_path_node;
            sendTrajToServer(path_node_g2s_);
            cuur_traj_start_time_ = ros::Time::now();
            new_goal_ = false;
            start_follow_time = ros::Time::now();
            changeState(FOLLOW_TRAJ);
            ROS_WARN("Replan back-end fail, use patched traj anyway"); //try replan another time, %lfs left for replan", curr_remain_safe_time);
          }
        }
        vector<State> vis_x;
        vector<Control> vis_u;
        krrt_planner_ptr_->getStateAndControl(optimized_path_node, &vis_x, &vis_u);
        vis_->visualizeOptiTraj(vis_x, vis_u, env_ptr_->getLocalTime());
      }
      else
      {
        krrt_planner_ptr_->getPath(path_node_g2s_, 2);
        double replan_duration = (ros::Time::now() - t_replan_start).toSec();
        if (replan_duration < dt)
        {
          ROS_WARN("wait for it: %lf", dt - replan_duration);
          ros::Duration(dt - replan_duration).sleep();
        }
        replan_state = 1;
        sendTrajToServer(path_node_g2s_);
        cuur_traj_start_time_ = ros::Time::now();
        new_goal_ = false;
        start_follow_time = ros::Time::now();
        changeState(FOLLOW_TRAJ);
      }
    }
    else
    {
      double curr_remain_safe_time = remain_safe_time_ - (ros::Time::now() - collision_detect_time).toSec();
      ROS_ERROR("Replan fail, %lf seconds to collide", curr_remain_safe_time);
      if (curr_remain_safe_time < e_stop_time_margin_)
      {
        sendEStopToServer();
        ROS_ERROR("ABOUT TO CRASH!! SERVER EMERGENCY STOP!!");
        changeState(GENERATE_TRAJ);
      }
      else
      {
        ROS_WARN("keep replanning");
      }
      break;
    }
    //after replan,
    if (receive_ref_traj_)
    {
      t_in_ref_traj_ += 0.2;
      Eigen::VectorXd goal_state = getReplanStateFromPath(t_in_ref_traj_, ref_traj_g2s_);
      end_pos_ << goal_state[0], goal_state[1], goal_state[2];
      end_vel_ << goal_state[3], goal_state[4], goal_state[5];
      end_acc_ << goal_state[6], goal_state[7], goal_state[8];
    }
    if (checkForReplan())
    {
      ROS_WARN("update future collision info");
      collision_detect_time = ros::Time::now();
    }
    else
    {
      ROS_WARN("future collision unchanged");
    }
    break;
  }

  case EMERGENCY_TRAJ:
  {

    break;
  }

  default:
    break;
  }
}

bool FSM::searchForTraj(Vector3d start_pos, Vector3d start_vel, Vector3d start_acc,
                        Vector3d end_pos, Vector3d end_vel, Vector3d end_acc,
                        double search_time, bool bidirection)
{
  int result(false);
  result = krrt_planner_ptr_->plan(start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, search_time, bidirection);
  if (result == 1)
  {
    close_goal_traj_ = false;
    return true;
  }
  else if (result == 2)
  {
    close_goal_traj_ = true;
    return true;
  }
  else
    return false;
}

bool FSM::optimize(RRTNodeVector &path_node_g2s, const Eigen::Vector3d &init_acc)
{
  bool result(false);
  ros::Time t1 = ros::Time::now();
  path_node_g2s.clear();
  vector<Eigen::Vector3d> way_points, vels, accs;
  Eigen::MatrixXd coeff0;
  Eigen::VectorXd seg_times;
  double seg_time = 1.0;

  //   krrt_planner_ptr_->getOptiSegs(seg_time, way_points, seg_times, vels, accs);
  krrt_planner_ptr_->getOptiSegs(way_points, seg_times, vels, accs, coeff0);

  if (seg_times.rows() == 1)
  { //one segment, no need to optimize
    krrt_planner_ptr_->getPath(path_node_g2s, 1);
    return true;
  }
  Eigen::VectorXd time;
  time.resize(seg_times.rows());
  for (int i = 0; i < seg_times.rows(); ++i)
  {
    time[i] = seg_times(i);
  }

  if (accs.size() > 0)
    accs.at(0) = init_acc;
  optimizer_ptr_->setWayPointsAndTime(way_points, vels, accs, time, coeff0);
  Eigen::MatrixXd coeff;

  RRTNodeVector last_valid_traj;
  double per_acc = 99.0;
  double per_high_close = (100.0 - per_acc) * 0.999;
  double per_low_close = 100.0 - per_acc - per_high_close;
  double per_close = per_high_close;
  bool adjust_per_acc = true;
  int devide_times = 0;
  double ttt = 0;
  for (;;)
  {
    path_node_g2s.clear();
    optimizer_ptr_->tryQPCloseForm(per_close, per_acc);
    devide_times++;
    // ROS_INFO("optimize for %d time(s), per_acc: %lf, per_close: %lf, per_smooth: %lf", devide_times, per_acc, per_close, 100.0 - per_acc - per_close);
    optimizer_ptr_->getCoefficient(coeff);
    for (int i = seg_times.rows() - 1; i >= 0; --i)
    {
      RRTNode node;
      node.tau_from_parent = seg_times(i);
      node.n_order = 5;
      for (int j = 0; j <= 5; ++j)
      {
        node.x_coeff[j] = coeff(i, 5 - j);
        node.y_coeff[j] = coeff(i, 5 - j + 6);
        node.z_coeff[j] = coeff(i, 5 - j + 12);
      }
      path_node_g2s.push_back(node);
    }
    //add start node
    RRTNode node;
    path_node_g2s.push_back(node);
    ros::Time t3 = ros::Time::now();
    bool violate = krrt_planner_ptr_->checkOptimizedTraj(path_node_g2s);
    ttt += (ros::Time::now() - t3).toSec();
    if (!violate)
    {
      result = true;
      // no collide
      last_valid_traj = path_node_g2s;
      adjust_per_acc = false;
      per_high_close = per_close;
      per_close = (per_low_close + per_close) / 2.0;
      if (devide_times >= 9)
      {
        ROS_WARN("optimization success");
        break;
      }
    }
    else
    {
      if (adjust_per_acc)
      {
        per_acc -= 10;
        per_high_close = (100.0 - per_acc) * 0.999;
        per_low_close = 100.0 - per_acc - per_high_close;
        per_close = per_high_close;
        if (per_acc <= 0)
        {
          per_acc = 1;
          per_high_close = (100.0 - per_acc) * 0.999;
          per_low_close = 100.0 - per_acc - per_high_close;
          per_close = per_high_close;
          adjust_per_acc = false;
        }
      }
      else
      {
        if (per_acc <= 1)
        {
          // ROS_ERROR("optimization failed, weight of acc_continuous can not be any less, use patched traj");
          krrt_planner_ptr_->getPath(path_node_g2s, 2);
          break;
        }
        per_low_close = per_close;
        per_close = (per_close + per_high_close) / 2.0;
      }
    }
    if (devide_times >= 19)
    {
      path_node_g2s = last_valid_traj;
      // ROS_INFO_STREAM("use last time optimized valid traj");
      break;
    }
  }

  // vector<Vector3d> cover_grid;
  // krrt_planner_ptr_->getVisTrajCovering(path_node_g2s, cover_grid);
  // vis_->visualizeTrajCovering(cover_grid, 0.05, env_ptr_->getLocalTime());

  // ROS_INFO_STREAM("check valid time: " << ttt << ", percentage of optimize time: " << ttt / (ros::Time::now() - t1).toSec());
  // ROS_INFO_STREAM("solve QP times: " << devide_times << ", optimize time: " << (ros::Time::now() - t1).toSec());
  return result;
}

void FSM::sendTrajToServer(const RRTNodeVector &path_nodes)
{
  static int traj_id = 0;
  int path_seg_num = path_nodes.size() - 1;
  if (path_seg_num < 1)
    return;
  quadrotor_msgs::PolynomialTrajectory traj;
  traj.trajectory_id = ++traj_id;
  traj.action = quadrotor_msgs::PolynomialTrajectory::ACTION_ADD;
  traj.num_segment = path_seg_num;
  for (int i = 0; i < path_seg_num; ++i)
  {
    traj.time.push_back(path_nodes[path_seg_num - i - 1].tau_from_parent);
    traj.order.push_back(path_nodes[path_seg_num - i - 1].n_order);
    for (size_t j = 0; j < traj.order[i] + 1; ++j)
    {
      traj.coef_x.push_back(path_nodes[path_seg_num - i - 1].x_coeff[j]);
      traj.coef_y.push_back(path_nodes[path_seg_num - i - 1].y_coeff[j]);
      traj.coef_z.push_back(path_nodes[path_seg_num - i - 1].z_coeff[j]);
    }
  }

  traj.header.frame_id = "map";
  traj.header.stamp = ros::Time::now();
  traj_pub_.publish(traj);
}

void FSM::sendEStopToServer()
{
  quadrotor_msgs::PolynomialTrajectory traj;
  traj.action = quadrotor_msgs::PolynomialTrajectory::ACTION_ABORT;

  traj.header.frame_id = "map";
  traj.header.stamp = ros::Time::now();
  traj_pub_.publish(traj);
}

bool FSM::reachGoal(double radius)
{
  Eigen::Vector3d pos_now = env_ptr_->get_curr_posi();
  if ((end_pos_ - pos_now).norm() < radius)
    return true;
  else
    return false;
}

bool FSM::checkForReplan()
{
  double t_during_traj = (ros::Time::now() - cuur_traj_start_time_).toSec();
  double check_dura = 1.5;
  if (krrt_planner_ptr_->updateColideInfo(path_node_g2s_, t_during_traj, check_dura, pos_about_to_collide_, remain_safe_time_))
  {
    ROS_INFO_STREAM("about to collide pos: " << pos_about_to_collide_.transpose() << ", remain safe time: " << remain_safe_time_);
    return true;
  }
  return false;
}

Eigen::VectorXd FSM::getReplanStateFromPath(double t, const RRTNodeVector &path_nodes)
{
  if (t < 0)
  {
    ROS_ERROR("not tracking well! use curr state as start state");
    Eigen::Vector3d pos(0.0, 0.0, 0.0), vel(0.0, 0.0, 0.0), acc(0.0, 0.0, 0.0), rpy(0.0, 0.0, 0.0);
    Eigen::Quaterniond q;
    pos = env_ptr_->get_curr_posi();
    vel = env_ptr_->get_curr_twist();
    acc = env_ptr_->get_curr_acc();
    Eigen::VectorXd start_state(9);
    start_state << pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], acc[0], acc[1], acc[2];
    return start_state;
  }

  size_t n = path_nodes.size();
  if (n < 2)
  {
    ROS_ERROR("path_nodes empty! use curr state(odom) as start state");
    Eigen::Vector3d pos(0.0, 0.0, 0.0), vel(0.0, 0.0, 0.0), acc(0.0, 0.0, 0.0), rpy(0.0, 0.0, 0.0);
    Eigen::Quaterniond q;
    pos = env_ptr_->get_curr_posi();
    vel = env_ptr_->get_curr_twist();
    acc = env_ptr_->get_curr_acc();
    Eigen::VectorXd start_state(9);
    start_state << pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], acc[0], acc[1], acc[2];
    return start_state;
  }

  int idx = n - 2;
  for (idx = n - 2; idx >= 0; --idx)
  {
    t -= path_nodes.at(idx).tau_from_parent;
    if (t <= 0)
    {
      t += path_nodes.at(idx).tau_from_parent;
      break;
    }
    else if (idx == 0)
    {
      //if comes here, then the input t is more than whole traj time
      t = path_nodes.at(idx).tau_from_parent;
      break;
    }
  }

  Eigen::Vector3d pos(0.0, 0.0, 0.0), vel(0.0, 0.0, 0.0), acc(0.0, 0.0, 0.0);
  double order = path_nodes.at(idx).n_order;
  krrt_planner_ptr_->calPVAFromCoeff(pos, vel, acc, path_nodes.at(idx).x_coeff,
                                     path_nodes.at(idx).y_coeff, path_nodes.at(idx).z_coeff, t, order);
  //TODO what if pos is not free??
  Eigen::VectorXd start_state(9);
  start_state << pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], acc[0], acc[1], acc[2];
  return start_state;
}

Eigen::Vector3d FSM::getRPY(const Eigen::Quaterniond &quat)
{
  double rotMat[3][3] = {(1 - 2 * (quat.y() * quat.y() + quat.z() * quat.z())), 2 * (quat.x() * quat.y() - quat.w() * quat.z()), 2 * (quat.x() * quat.z() + quat.w() * quat.y()),
                         2 * (quat.x() * quat.y() + quat.w() * quat.z()), (1 - 2 * (quat.x() * quat.x() + quat.z() * quat.z())), 2 * (quat.y() * quat.z() - quat.w() * quat.x()),
                         2 * (quat.x() * quat.z() - quat.w() * quat.y()), 2 * (quat.y() * quat.z() + quat.w() * quat.x()), (1 - 2 * (quat.x() * quat.x() + quat.y() * quat.y()))};

  double yaw, pitch, roll;
  if (rotMat[2][0] != 1 && rotMat[2][0] != -1)
  {
    double yaw1, yaw2, pitch1, pitch2, roll1, roll2;
    pitch1 = -asin(rotMat[2][0]);
    pitch2 = M_PI - pitch1;
    double cos_pitch1 = cos(pitch1);
    double cos_pitch2 = cos(pitch2);
    roll1 = atan2(rotMat[2][1] / cos_pitch1, rotMat[2][2] / cos_pitch1);
    roll2 = atan2(rotMat[2][1] / cos_pitch2, rotMat[2][2] / cos_pitch2);
    yaw1 = atan2(rotMat[1][0] / cos_pitch1, rotMat[0][0] / cos_pitch1);
    yaw2 = atan2(rotMat[1][0] / cos_pitch2, rotMat[0][0] / cos_pitch2);
    if (fabs(pitch1) <= fabs(pitch2))
    {
      yaw = yaw1;
      pitch = pitch1;
      roll = roll1;
    }
    else
    {
      yaw = yaw2;
      pitch = pitch2;
      roll = roll2;
    }
  }
  else if (rotMat[2][0] == 1)
  {
    yaw = 0;
    pitch = M_PI / 2;
    roll = yaw + atan2(rotMat[0][1], rotMat[0][2]);
  }
  else
  {
    yaw = 0;
    pitch = -M_PI / 2;
    roll = -yaw + atan2(-rotMat[0][1], -rotMat[0][2]);
  }

  return Eigen::Vector3d(roll, pitch, yaw);
}

Eigen::Vector3d FSM::getAccFromRPY(const Eigen::Vector3d &rpy)
{
  Eigen::Vector3d acc(0.0, 0.0, 0.0);
  return acc;
}

void FSM::printState()
{
  string state_str[6] = {"INIT", "WAIT_GOAL", "GENERATE_TRAJ", "FOLLOW_TRAJ", "REPLAN_TRAJ", "EMERGENCY_TRAJ"};
  ROS_INFO_STREAM("[FSM]: state: " << state_str[int(machine_state_)]);
}

void FSM::changeState(FSM::MACHINE_STATE new_state)
{
  string state_str[6] = {"INIT", "WAIT_GOAL", "GENERATE_TRAJ", "FOLLOW_TRAJ", "REPLAN_TRAJ", "EMERGENCY_TRAJ"};
  ROS_INFO_STREAM("[FSM]: change from " << state_str[int(machine_state_)] << " to " << state_str[int(new_state)]);
  machine_state_ = new_state;
}

} // namespace fast_planner
