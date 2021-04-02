#include "occ_grid/occ_map.h"
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
//#include <tf/transform_datatypes.h>
//#include <torch/torch.h>
//for img debug
#include <opencv2/opencv.hpp>

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

bool _has_pred = false;

// template <typename T>
// struct TrtDestroyer
// {
//     void operator()(T* t) { t->destroy(); }
// };

// template <typename T> using TrtUniquePtr = std::shared_ptr<T, TrtDestroyer<T> >;

class TrtModel
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    TrtModel(): mEngine(nullptr)
    {}
    //! build engine
    bool build();
    bool infer(std::vector<float>& input, std::vector<float>& output);
    void initParam(const samplesCommon::OnnxSampleParams& params);
    void saveEngine(const std::string fileName);

private:
    samplesCommon::OnnxSampleParams mParams; 
    nvinfer1::Dims mInputDims; 
    nvinfer1::Dims mOutputDims; 
    int mNumber{0};        
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; 
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);
    bool processInput(const samplesCommon::BufferManager& buffers, const std::vector<float>& input);
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
    bool loadEngine();
};

void TrtModel::initParam(const samplesCommon::OnnxSampleParams& params)
{
  mParams = params;
}

void TrtModel::saveEngine(std::string fileName ="./engine.trt")
{
  if (mEngine) {
    nvinfer1::IHostMemory * data = mEngine->serialize();
    std::ofstream file;
    file.open(fileName, std::ios::binary | std::ios::out);
    if (!file.is_open()) {
      std::cout << "read create engine file" << fileName << " failed" << std::endl;
      return;
    }

    file.write((const char *)data->data(), data->size());
    file.close();
  }
};

bool TrtModel::loadEngine()
{
    std::vector<char> trtModelStream;
    size_t size{0};
    std::ifstream file(mParams.onnxFileName, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream.resize(size);
        file.read(trtModelStream.data(), size);
        file.close();
    }else{
        cout << "Error loading engine file: " << mParams.onnxFileName << std::endl;
        return false;
    }

    IRuntime* infer = nvinfer1::createInferRuntime(gLogger);
    // if (mParams.dlaCore >= 0)
    // {
    //     infer->setDLACore(mParams.dlaCore);
    // }
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr), samplesCommon::InferDeleter());
    return true;
};

bool TrtModel::build()
{
  string suffixStr = mParams.onnxFileName.substr(mParams.onnxFileName.find_last_of('.') + 1);
  if (suffixStr == "onnx"){
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);     
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    cout << "inputsize: " << mInputDims.d[1] << ", "<< mInputDims.d[2] << ", " << mInputDims.d[3]  <<endl;
    cout << "outputsize: " << mOutputDims.d[1] << ", "<< mOutputDims.d[2] << ", " << mOutputDims.d[3]  <<endl;
    return true;
  }

  else if(suffixStr == "trt"){
    ROS_WARN("BUILDING FROM TRT ENGINE");
    if (!loadEngine()) return false;
    mInputDims = mEngine->getBindingDimensions(
        mEngine->getBindingIndex(mParams.inputTensorNames[0].c_str()));
    mOutputDims = mEngine->getBindingDimensions(
        mEngine->getBindingIndex(mParams.outputTensorNames[0].c_str()));
    cout << "inputsize: " << mInputDims.d[1] << ", "<< mInputDims.d[2] << ", " << mInputDims.d[3]  <<endl;
    cout << "outputsize: " << mOutputDims.d[1] << ", "<< mOutputDims.d[2] << ", " << mOutputDims.d[3]  <<endl;

    return true;
  }

  ROS_WARN("model path not ended with .onnx or .trt");
  return false;
}

bool TrtModel::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(
        mParams.onnxFileName.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    // auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(256_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    // samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

bool TrtModel::infer(std::vector<float>& input, std::vector<float>& output)
{
    // Create RAII buffer manager object
    // ROS_WARN("infer start");

    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers, input))
    {
      ROS_WARN("processInput FIAL");
      return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
      ROS_WARN("executeV2 FIAL");
      return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    output.reserve(int(mOutputDims.d[1] * mOutputDims.d[2] * mOutputDims.d[3]));
    float* xxx = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    for (int i = 0; i < mOutputDims.d[1] * mOutputDims.d[2] * mOutputDims.d[3]; i ++){
      output.push_back(xxx[i]);
    }

    return true;
}

bool TrtModel::processInput(const samplesCommon::BufferManager& buffers, const std::vector<float>& input)
{
    const int inputX = mInputDims.d[1];
    const int inputY = mInputDims.d[2];
    const int inputZ = mInputDims.d[3];

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));

    for (int i = 0; i < inputX * inputY * inputZ; i++)
    {
        hostDataBuffer[i] = input[i];
    }

    return true;
}


samplesCommon::OnnxSampleParams initializeSampleParams(const std::string model_path)
{
    samplesCommon::OnnxSampleParams params;
    params.onnxFileName = model_path;
    params.inputTensorNames.push_back("input1");
    params.batchSize = 1;
    params.outputTensorNames.push_back("output1");
    params.dlaCore = -1;
    params.int8 = false;
    params.fp16 = false;
    return params;
}

TrtModel my_model_;

namespace fast_planner
{
void OccMap::resetBuffer(Eigen::Vector3d min_pos, Eigen::Vector3d max_pos)
{
  min_pos(0) = max(min_pos(0), min_range_(0));
  min_pos(1) = max(min_pos(1), min_range_(1));
  min_pos(2) = max(min_pos(2), min_range_(2));

  max_pos(0) = min(max_pos(0), max_range_(0));
  max_pos(1) = min(max_pos(1), max_range_(1));
  max_pos(2) = min(max_pos(2), max_range_(2));

  Eigen::Vector3i min_id, max_id;

  posToIndex(min_pos, min_id);
  posToIndex(max_pos - Eigen::Vector3d(resolution_ / 2, resolution_ / 2, resolution_ / 2), max_id);

  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z)
      {
        occupancy_buffer_[x * grid_size_(1) * grid_size_(2) + y * grid_size_(2) + z] = clamp_min_log_;
      } 
}

void OccMap::publishLocalOccCallback(const ros::TimerEvent& E)
{
  if (! have_odom_) return;
  // ROS_WARN("1");
  // TODO: use pred resault to update pred_map
  ros::Time t_s = ros::Time::now();

  // ocp_msgs::PredictPCL srv;
  Eigen::Vector3i center_id, edge_id, node_id;
  Eigen::Vector3d node_pos, edge_pos;
  std::vector <int> idx_ctns_list;
  posToIndex(curr_posi_, center_id);
  // ROS_INFO_STREAM("current loc: " << curr_posi_(0) << curr_posi_(1) << curr_posi_(2));
  // CHANGE!
  edge_id = center_id - local_grid_size_ / 2; 
  edge_id(0) = floor((-2.0 - origin_(0)) * resolution_inv_);
  // edge_id(0) = center_id(0) - local_grid_size_(0) / 4;
  edge_id(1) = center_id(1) - local_grid_size_(1) / 4;
  edge_id(2) = floor((-0.6 - origin_(2)) * resolution_inv_); // floor z: -.30
  indexToPos(edge_id, edge_pos);
  // ROS_INFO_STREAM("edge_pos: " << edge_pos(0) << edge_pos(1) << edge_pos(2));

  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointCloud<pcl::PointXYZ> added_cloud;

  local_occupancy_buffer_.clear();
  // STORE POSITION OF ALL POINTS
  local_idx_buffer_.clear();

  int num_unknown, num_occ, num_free;
  num_unknown = 0;
  num_occ = 0;
  num_free = 0;
  for (int x = 0; x < local_grid_size_(0); ++x)
    for (int y = 0; y < local_grid_size_(1); ++y)
      for (int z = 0; z < local_grid_size_(2); ++z)
      {
        node_id(0) = x + edge_id(0);
        node_id(1) = y + edge_id(1);
        node_id(2) = z + edge_id(2);
        indexToPos(node_id, node_pos);
        int idx_ctns = node_id(0) * grid_size_(1) * grid_size_(2) + node_id(1) * grid_size_(2) + node_id(2);
        // int local_idx = x * grid_size_(1) * grid_size_(2) + y * grid_size_(2) + z;

        // not in map
        if (!isInMap(node_pos))
        {
          local_occupancy_buffer_.push_back(-1);
          local_idx_buffer_.push_back(Eigen::Vector3d(0,0,0));
          idx_ctns_list.push_back(INT_MIN);
          continue;
        }
        else
        {
          idx_ctns_list.push_back(idx_ctns);
          local_idx_buffer_.push_back(node_pos);
          // unknown
          if (known_buffer_[idx_ctns] < known_threshold_)
          {
            local_occupancy_buffer_.push_back(-1);
            num_unknown += 1;
          }
          // free
          else if (occupancy_buffer_[idx_ctns] < min_occupancy_log_)
          {
            local_occupancy_buffer_.push_back(0);
            num_free += 1;
          }
          // occ
          else
          {
            local_occupancy_buffer_.push_back(1);
            num_occ += 1;
          }
        }
      }

  // ROS_INFO_STREAM("num_unknown: " << num_unknown);
  // ROS_INFO_STREAM("num_free: " << num_free);
  // ROS_INFO_STREAM("num_occ: " << num_occ);

  // srv.request.dim_x = local_grid_size_(0);
  // srv.request.dim_y = local_grid_size_(1);
  // srv.request.dim_z = local_grid_size_(2);
  // ROS_INFO_STREAM("preprocess pcl uses: " << (ros::Time::now() - t_s).toSec());
  t_s = ros::Time::now();

  std::vector<float> trt_output;
  bool srv_success = false;
  if (use_pred_)
  {
    srv_success = my_model_.infer(local_occupancy_buffer_, trt_output);
  }
  // ROS_INFO_STREAM("inference uses: " << (ros::Time::now() - t_s).toSec());
  // t_s = ros::Time::now();

  // int out_size = *(trt_output - 8);
  // cout << "test size: "<<  trt_output.size() <<endl;
  // cout << "local_idx_buffer_ size: "<<  local_idx_buffer_.size() <<endl;
  // cout << "local_idx_buffer_ size: "<<  trt_output.size() <<endl;
  // cout << "out size: "<< out_size;
  // if (out_size != local_idx_buffer_.size()){
  //   ROS_WARN(" OUTPUT SIZE MISMATCH");
  //   srv_success = false;
  // }
  // ROS_INFO_STREAM("predict pcl uses: " << (ros::Time::now() - t_s).toSec());
  // for (int it = 0; it < int(local_idx_buffer_.size()); it ++)
  // {
  //   srv.request.input.push_back(local_occupancy_buffer_[it]);
  // }

  // ROS_INFO_STREAM("collect pcl uses: " << (ros::Time::now() - t_s).toSec());

  // bool srv_success = false;
  // if (pred_client_.call(srv))
  // {
  //   // ROS_INFO("occmap: pred service success");
  //   srv_success = true;
  //   _has_pred = true;
  // }
  // else
  // {
  //   srv_success = false;
  //   // ROS_ERROR("Failed to call service PREDICTPCL");
  //   // return;
  // }
  // // ROS_INFO_STREAM("predict pcl uses: " << (ros::Time::now() - t_s).toSec());
  // t_s = ros::Time::now();

  // process resault
  for (int it = 0; it < int(local_idx_buffer_.size()); it ++)
  {
    // not in map
    if (local_idx_buffer_[it](0) == 0 && local_idx_buffer_[it](1) == 0 && local_idx_buffer_[it](2) == 0)
      continue;
    // not in map
    if (srv_success && use_pred_)
    {
      if (trt_output[it] > pred_occ_thresord_)
        pred_occupancy_buffer_[idx_ctns_list[it]] = clamp_max_log_;
      else
        pred_occupancy_buffer_[idx_ctns_list[it]] = clamp_min_log_;
    }
    
    // do not show grids too high
    if (local_idx_buffer_[it](2)>=1.4)
      continue;
      
    if (local_occupancy_buffer_[it] == 1)
    {
      // pcl::PointXYZ point(local_idx_buffer_[it](0), local_idx_buffer_[it](1), local_idx_buffer_[it](2));
      pcl::PointXYZI point;
      point.x = local_idx_buffer_[it](0);
      point.y = local_idx_buffer_[it](1);
      point.z = local_idx_buffer_[it](2);
      point.intensity = 255.0;
      cloud.push_back(point);
/*      pcl::PointXYZ point1(local_idx_buffer_[it](0), local_idx_buffer_[it](1), local_idx_buffer_[it](2));
      added_cloud.push_back(point1);*/
    }
    else if (vis_unknown_ && local_occupancy_buffer_[it] == -1 && _has_pred)
    {
      pcl::PointXYZI point;
      point.x = local_idx_buffer_[it](0);
      point.y = local_idx_buffer_[it](1);
      point.z = local_idx_buffer_[it](2);
      point.intensity = 50.0;
      cloud.push_back(point);
    }    
    // pred is occ
    else if (!srv_success)
      continue;
    else if (use_pred_) 
    {
      if (pred_occupancy_buffer_[idx_ctns_list[it]] > 0.5) // && known_buffer_[idx_ctns_list[it]] < known_threshold_
      {
        pcl::PointXYZ point(local_idx_buffer_[it](0), local_idx_buffer_[it](1), local_idx_buffer_[it](2));
        added_cloud.push_back(point);
      }
    }
  }

  // for vis
  cloud.width = (int)cloud.points.size();
  cloud.height = 1;    //height=1 implies this is not an "ordered" point cloud
  added_cloud.width = (int)added_cloud.points.size();
  added_cloud.height = 1;    //height=1 implies this is not an "ordered" point cloud

  // Convert the cloud to ROS message
  sensor_msgs::PointCloud2 output, added_output;
  pcl::toROSMsg(cloud, output);
  pcl::toROSMsg(added_cloud, added_output);

  output.header.frame_id = "map";
  added_output.header.frame_id = "map";

  local_occ_pub_.publish(output);  
  added_occ_pub_.publish(added_output);

  // ROS_INFO_STREAM("postprocess uses: " << (ros::Time::now() - t_s).toSec());
  // t_s = ros::Time::now();
  return;
}

bool OccMap::isInMap(Eigen::Vector3d pos)
{
  if (pos(0) < min_range_(0) || pos(1) < min_range_(1) || pos(2) < min_range_(2))
  {
    return false;
  }

  if (pos(0) > max_range_(0) || pos(1) > max_range_(1) || pos(2) > max_range_(2))
  {
    return false;
  }

  return true;
}

void OccMap::posToIndex(Eigen::Vector3d pos, Eigen::Vector3i& id)
{
  for (int i = 0; i < 3; ++i)
    id(i) = floor((pos(i) - origin_(i)) * resolution_inv_);
}

void OccMap::indexToPos(Eigen::Vector3i id, Eigen::Vector3d& pos)
{
  pos = origin_;
  for (int i = 0; i < 3; ++i)
    pos(i) += (id(i) + 0.5) * resolution_;
}

void OccMap::setOccupancy(Eigen::Vector3d pos)
{
  if (!isInMap(pos))
    return;

  Eigen::Vector3i id;
  posToIndex(pos, id);

  occupancy_buffer_[id(0) * grid_size_(1) * grid_size_(2) + id(1) * grid_size_(2) + id(2)] = clamp_max_log_;
}

int OccMap::getVoxelState(Eigen::Vector3d pos)
{
  if (!isInMap(pos))
    return -1;

  if (pos[2] > 1.5)
    return 1;

  static Eigen::Vector3i id;
  posToIndex(pos, id);
  // because of property of raycasting, need to move around to clear unknown grids
  float distance = (pos - curr_posi_).norm();
  
  int idx = id(0) * grid_size_(1) * grid_size_(2) + id(1) * grid_size_(2) + id(2);
  // delay for several seconds otherwise the world will be full of known
  if ((!unknown_as_free_) && _has_pred)
    if (known_buffer_[idx] < known_threshold_)
      if (distance < 2.0)
        return 1;

  if (occupancy_buffer_[idx] > min_occupancy_log_)
    return 1;

  // if farer than local radius, return free
  // if (pos(0) - raycast_posi_(0) > local_grid_size_(0)  / 1.3 * resolution_ 
  //     || pos(1) - raycast_posi_(1) > local_grid_size_(1)  / 2.5 * resolution_)
  //   return 0;

  if (use_pred_for_collision_)
    if (pred_occupancy_buffer_[idx] > 0.5)
      return 1;

  return 0;
}

int OccMap::getVoxelState(Eigen::Vector3i id)
{
  if (id(0) < 0 || id(0) >= grid_size_(0) || id(1) < 0 || id(1) >= grid_size_(1) || id(2) < 0 || id(2) >= grid_size_(2))
    return -1;

  Eigen::Vector3d pos;
  indexToPos(id, pos);
  if (!isInMap(pos))
    return -1;


  if (pos[2] > 1.5)
    return 1;
  
  int idx = id(0) * grid_size_(1) * grid_size_(2) + id(1) * grid_size_(2) + id(2);
  float distance = (pos - curr_posi_).norm();
  // delay for several seconds otherwise the world will be full of known
  if ((!unknown_as_free_) && _has_pred)
    if (known_buffer_[idx] < known_threshold_)
      if (distance < 2.0)
        return 1;

  if (occupancy_buffer_[idx] > min_occupancy_log_)
    return 1;

  // // if farer than local radius, return free
  // if (pos(0) - raycast_posi_(0) > local_grid_size_(0)  / 2.5 * resolution_ 
  //     || pos(1) - raycast_posi_(1) > local_grid_size_(1)  / 2.5 * resolution_)
  //   return 0;

  if (use_pred_for_collision_)
    if (pred_occupancy_buffer_[idx] > 0.5)
      return 1;

  return 0;
}

int OccMap::getOriginalVoxelState(Eigen::Vector3d pos)
{
  if (!isInMap(pos))
    return -1;

  Eigen::Vector3i id;
  posToIndex(pos, id);
  int idx = id(0) * grid_size_(1) * grid_size_(2) + id(1) * grid_size_(2) + id(2);

  if (occupancy_buffer_[idx] > min_occupancy_log_)
    return 1;
  return 0;
}

bool OccMap::collisionCheck()
{
  for (int x = -2; x <= 2; x++)
  {
    for (int y = -2; y <= 2; y++)
    {
      for (int z = -1; z <= 1; z++){
        Eigen::Vector3d check_point = curr_posi_;
        check_point[0] += x * 0.05;
        check_point[1] += y * 0.05;
        check_point[2] += z * 0.03;
        if (getOriginalVoxelState(check_point) == 1) return false;
      }
    }
  }
  return true;
}

void OccMap::publishCollisionNum(const ros::TimerEvent& e)
{
  if (!_has_pred) return;
  if (!collisionCheck()){
    ROS_WARN("OCC GRID: CRASHING!");
    collisions_ ++;
  }
  std_msgs::Int8 msg;
  msg.data = collisions_;
  collision_pub_.publish(msg);
}

void OccMap::pubPointCloudFromDepth(const std_msgs::Header& header, 
                                    const cv::Mat& depth_img, 
                                    const Eigen::Matrix3d& intrinsic_K, 
                                    const string& camera_name)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::PointXYZ point; 

  for (int v = 0; v < depth_img.rows; v++)
    for (int u = 0; u < depth_img.cols; u++)
    {
      double depth = depth_img.at<u_int16_t>(v, u) / depth_scale_;
      Eigen::Vector3d uvd = Eigen::Vector3d(u, v, 1.0) * depth;
      Eigen::Vector3d xyz = intrinsic_K.inverse() * uvd;
      point.x = xyz(0);
      point.y = xyz(1);
      point.z = xyz(2);
      cloud.points.push_back(point);
    }

  cloud.width = (int)cloud.points.size();
  cloud.height = 1;    //height=1 implies this is not an "ordered" point cloud

  // Convert the cloud to ROS message
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(cloud, output);
  output.header = header;
  output.header.frame_id = camera_name;
  origin_pcl_pub_.publish(output);
}

void OccMap::globalOccVisCallback(const ros::TimerEvent& e)
{
  //for vis
  history_view_cloud_ptr_->points.clear();
  history_pred_cloud_ptr_->points.clear();
  for (int x = 0; x < grid_size_[0]; ++x)
    for (int y = 0; y < grid_size_[1]; ++y)
      for (int z = 0; z < grid_size_[2]; ++z)
      {
        //cout << "p(): " << occupancy_buffer_[x * grid_size_(1) * grid_size_(2) + y * grid_size_(2) + z] << endl;
        if (occupancy_buffer_[x * grid_size_(1) * grid_size_(2) + y * grid_size_(2) + z] > min_occupancy_log_)
        {
          Eigen::Vector3i idx(x,y,z);
          Eigen::Vector3d pos;
          indexToPos(idx, pos);
          pcl::PointXYZ pc(pos[0], pos[1], pos[2]);
          history_view_cloud_ptr_->points.push_back(pc);
        }
        if (use_pred_)
        {
          if (pred_occupancy_buffer_[x * grid_size_(1) * grid_size_(2) + y * grid_size_(2) + z] > 0.5)
          {
            Eigen::Vector3i idx(x,y,z);
            Eigen::Vector3d pos;
            indexToPos(idx, pos);
            pcl::PointXYZ pc(pos[0], pos[1], pos[2]);
            history_pred_cloud_ptr_->points.push_back(pc);
          }   
        }
      }
  history_view_cloud_ptr_->width = history_view_cloud_ptr_->points.size();
  history_view_cloud_ptr_->height = 1;
  history_view_cloud_ptr_->is_dense = true;
  history_view_cloud_ptr_->header.frame_id = "map";
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*history_view_cloud_ptr_, cloud_msg);
  hist_view_cloud_pub_.publish(cloud_msg);
  if (use_pred_)
  {
    history_pred_cloud_ptr_->width = history_pred_cloud_ptr_->points.size();
    history_pred_cloud_ptr_->height = 1;
    history_pred_cloud_ptr_->is_dense = true;
    history_pred_cloud_ptr_->header.frame_id = "map";
    sensor_msgs::PointCloud2 cloud_msg2;
    pcl::toROSMsg(*history_pred_cloud_ptr_, cloud_msg2);
    hist_pred_cloud_pub_.publish(cloud_msg2);    
  }
}

void OccMap::localOccVisCallback(const ros::TimerEvent& e)
{
  //for vis
  // ros::Time t_s = ros::Time::now();
  curr_view_cloud_ptr_->points.clear();
  Eigen::Vector3i min_id, max_id;
  posToIndex(local_range_min_, min_id);
  posToIndex(local_range_max_, max_id);
// 	ROS_INFO_STREAM("local_range_min_: " << local_range_min_.transpose());
// 	ROS_INFO_STREAM("local_range_max_: " << local_range_max_.transpose());
// 	ROS_INFO_STREAM("min_id: " << min_id.transpose());
// 	ROS_INFO_STREAM("max_id: " << max_id.transpose());
	
  min_id(0) = max(0, min_id(0));
  min_id(1) = max(0, min_id(1));
  min_id(2) = max(0, min_id(2));
  max_id(0) = min(grid_size_[0], max_id(0));
  max_id(1) = min(grid_size_[1], max_id(1));
  max_id(2) = min(grid_size_[2], max_id(2));
  for (int x = min_id(0); x < max_id(0); ++x)
    for (int y = min_id(1); y < max_id(1); ++y)
      for (int z = min_id(2); z < max_id(2); ++z)
      {
        if (occupancy_buffer_[x * grid_size_(1) * grid_size_(2) + y * grid_size_(2) + z] > min_occupancy_log_)
        {
          Eigen::Vector3i idx(x,y,z);
          Eigen::Vector3d pos;
          indexToPos(idx, pos);
					// ROS_INFO_STREAM("occupied idx: " << idx.transpose());
					// ROS_INFO_STREAM("occupied pos: " << pos.transpose());
          pcl::PointXYZ pc(pos[0], pos[1], pos[2]);
          curr_view_cloud_ptr_->points.push_back(pc);
        }
      }
    
  ROS_INFO_STREAM("curr_view_cloud_ptr_: " << curr_view_cloud_ptr_->points.size());
  
  curr_view_cloud_ptr_->width = curr_view_cloud_ptr_->points.size();
  curr_view_cloud_ptr_->height = 1;
  curr_view_cloud_ptr_->is_dense = true;
  curr_view_cloud_ptr_->header.frame_id = "map";
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*curr_view_cloud_ptr_, cloud_msg);
  curr_view_cloud_pub_.publish(cloud_msg);

  // ROS_INFO_STREAM("local vis once uses: " << (ros::Time::now() - t_s).toSec());
}


void OccMap::depthOdomCallback(const sensor_msgs::ImageConstPtr& depth_msg, 
                              const nav_msgs::OdometryConstPtr& odom, 
                              const Eigen::Matrix4d& T_ic, 
                              Eigen::Matrix4d& last_T_wc, 
                              cv::Mat& last_depth_image, 
                              const string& camera_name)
{
	// ros::Time t1 = ros::Time::now();
  { //TF map^ T ego
    static tf2_ros::TransformBroadcaster br_map_ego;
    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = depth_msg->header.stamp;
    transformStamped.header.frame_id = "map";
    transformStamped.child_frame_id = "ego";
    transformStamped.transform.translation.x = odom->pose.pose.position.x;
    transformStamped.transform.translation.y = odom->pose.pose.position.y;
    transformStamped.transform.translation.z = odom->pose.pose.position.z;
    transformStamped.transform.rotation.x = odom->pose.pose.orientation.x;
    transformStamped.transform.rotation.y = odom->pose.pose.orientation.y;
    transformStamped.transform.rotation.z = odom->pose.pose.orientation.z;
    transformStamped.transform.rotation.w = odom->pose.pose.orientation.w;
    br_map_ego.sendTransform(transformStamped);
  }

  { //TF imu^ T _camera_i
    static tf2_ros::StaticTransformBroadcaster static_broadcaster;
    geometry_msgs::TransformStamped static_transformStamped;
    static_transformStamped.header.stamp = depth_msg->header.stamp;
    static_transformStamped.header.frame_id = "ego";
    static_transformStamped.child_frame_id = camera_name;
    static_transformStamped.transform.translation.x = 0;
    static_transformStamped.transform.translation.y = 0;
    static_transformStamped.transform.translation.z = 0;
    Eigen::Quaterniond quat(T_ic.block<3,3>(0,0));
    static_transformStamped.transform.rotation.x = quat.x();
    static_transformStamped.transform.rotation.y = quat.y();
    static_transformStamped.transform.rotation.z = quat.z();
    static_transformStamped.transform.rotation.w = quat.w();
    static_broadcaster.sendTransform(static_transformStamped);
  }

  /* ---------- get pose ---------- */
  // w, x, y, z -> q0, q1, q2, q3
  Eigen::Matrix3d R_wi = Eigen::Quaterniond(odom->pose.pose.orientation.w, 
                                            odom->pose.pose.orientation.x, 
                                            odom->pose.pose.orientation.y, 
                                            odom->pose.pose.orientation.z).toRotationMatrix();
  Eigen::Matrix4d T_wi;
  T_wi.setZero();
  T_wi(0, 3) = odom->pose.pose.position.x;
  T_wi(1, 3) = odom->pose.pose.position.y;
  T_wi(2, 3) = odom->pose.pose.position.z;
  T_wi(3, 3) = 1.0;
  T_wi.block<3,3>(0,0) = R_wi;
  Eigen::Matrix4d T_wc = T_wi * T_ic;
  Eigen::Vector3d t_wc = T_wc.block<3,1>(0,3);
  local_range_min_ = t_wc - sensor_range_;
	local_range_max_ = t_wc + sensor_range_;

  /* ---------- get depth image ---------- */
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(depth_msg, depth_msg->encoding);
  if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
  {
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, depth_scale_);
  }
  cv_ptr->image.copyTo(depth_image_);

  if (show_raw_depth_)
  {
    pubPointCloudFromDepth(depth_msg->header, depth_image_, K_depth_, camera_name);
  }

  proj_points_cnt_ = 0;
  projectDepthImage(K_depth_, T_wc, depth_image_, last_T_wc, last_depth_image, depth_msg->header.stamp);
  raycastProcess(t_wc);

  local_map_valid_ = true;
  latest_odom_time_ = odom->header.stamp;
  curr_posi_[0] = odom->pose.pose.position.x;
  curr_posi_[1] = odom->pose.pose.position.y;
  curr_posi_[2] = odom->pose.pose.position.z;
  curr_twist_[0] = odom->twist.twist.linear.x;
  curr_twist_[1] = odom->twist.twist.linear.y;
  curr_twist_[2] = odom->twist.twist.linear.z;
  curr_q_.w() = odom->pose.pose.orientation.w;
  curr_q_.x() = odom->pose.pose.orientation.x;
  curr_q_.y() = odom->pose.pose.orientation.y;
  curr_q_.z() = odom->pose.pose.orientation.z;
  have_odom_ = true;
}

void OccMap::pclOdomCallback(const ocp_msgs::OdomPclConstPtr& odom_pcl)
{
  if (!have_odom_) return;
	ros::Time t1 = ros::Time::now();

  /* ---------- raycast pcl ---------- */

  raycast_posi_[0] = odom_pcl->odom.pose.pose.position.x;
  raycast_posi_[1] = odom_pcl->odom.pose.pose.position.y;
  raycast_posi_[2] = odom_pcl->odom.pose.pose.position.z;
  local_range_min_ = raycast_posi_ - sensor_range_;
	local_range_max_ = raycast_posi_ + sensor_range_;
  
  proj_points_cnt_ = 0;
  raycastPclProcess(raycast_posi_, odom_pcl->pointcloud);
  // ROS_INFO_STREAM("raycast_posi_: " << raycast_posi_[0] << raycast_posi_[1]);
  // ROS_INFO_STREAM("curr_posi_: " << curr_posi_[0] << curr_posi_[1]);

  local_map_valid_ = true;
  if (raycast_fisttime_ && have_odom_){
    init_odom_ = curr_posi_;
    raycast_fisttime_ = false;
    // _has_pred = true;
  }
  if (!_has_pred)
  {
    if ((raycast_posi_ - init_odom_).norm() > 0.3)
    {
      _has_pred = true;
      // have_odom_ = true;
    }
  }

  // ROS_INFO_STREAM("pclOdomCallback uses: " << (ros::Time::now() - t1).toSec());
  //TODO: collision checker
}

void OccMap::projectDepthImage(const Eigen::Matrix3d& K, 
                               const Eigen::Matrix4d& T_wc, const cv::Mat& depth_image, 
                               Eigen::Matrix4d& last_T_wc, cv::Mat& last_depth_image, ros::Time r_s)
{
  int cols = depth_image.cols;
  int rows = depth_image.rows;
	pcl::PointCloud<pcl::PointXYZRGB> cloud;
  pcl::PointXYZRGB point; //colored point clouds also have RGB values
  
  double depth;
  //   ROS_WARN("project for one rcved image");
  if (!use_shift_filter_)
  {
    for (int v = depth_filter_margin_; v < rows - depth_filter_margin_; v += skip_pixel_)
    {
      for (int u = depth_filter_margin_; u < cols - depth_filter_margin_; u += skip_pixel_)
      {
        depth = depth_image.at<uint16_t>(v, u) / depth_scale_;
				if (isnan(depth) || isinf(depth))
					continue;
			  if (depth < depth_filter_mindist_)
          continue;
				Eigen::Vector3d proj_pt_NED, proj_pt_cam;
        proj_pt_cam(0) = (u - K(0,2)) * depth / K(0,0);
        proj_pt_cam(1) = (v - K(1,2)) * depth / K(1,1);
        proj_pt_cam(2) = depth;
				proj_pt_NED = T_wc.block<3,3>(0,0) * proj_pt_cam + T_wc.block<3,1>(0,3);
        proj_points_[proj_points_cnt_++] = proj_pt_NED;
        // cout << "pt in cam: " << proj_pt_cam.transpose() << ", depth: " << depth << endl;
        // cout << "pt in map: " << proj_pt_NED.transpose() << ", depth: " << depth << endl;
        if (show_filter_proj_depth_)
	      {
          point.x = proj_pt_NED[0];
          point.y = proj_pt_NED[1];
          point.z = proj_pt_NED[2];
          point.r = 255;
          point.g = 0;
          point.b = 0;
          cloud.points.push_back(point);
        }
      }
    }
  }
  /* ---------- use depth filter ---------- */
  else
  {
    if (!has_first_depth_)
      has_first_depth_ = true;
    else
    {
      Eigen::Vector3d pt_cur, pt_NED, pt_reproj;

      for (int v = depth_filter_margin_; v < rows - depth_filter_margin_; v += skip_pixel_)
      {
        for (int u = depth_filter_margin_; u < cols - depth_filter_margin_; u += skip_pixel_)
        {
          depth = depth_image.at<uint16_t>(v, u) / depth_scale_;
					if (isnan(depth) || isinf(depth))
						continue;
					// points with depth > depth_filter_maxdist_ or < depth_filter_mindist_ are not trusted 
					if (depth < depth_filter_mindist_)
            continue;
					pt_cur(0) = (u - K(0,2)) * depth / K(0,0);
          pt_cur(1) = (v - K(1,2)) * depth / K(1,1);
          pt_cur(2) = depth;
					
          // check consistency
					pt_NED = T_wc.block<3,3>(0,0) * pt_cur + T_wc.block<3,1>(0,3);
          pt_reproj = last_T_wc.block<3,3>(0,0).inverse() * (pt_NED - last_T_wc.block<3,1>(0,3));
          double uu = pt_reproj.x() * K(0,0) / pt_reproj.z() + K(0,2);
          double vv = pt_reproj.y() * K(1,1) / pt_reproj.z() + K(1,2);
          if (uu >= 0 && uu < cols && vv >= 0 && vv < rows)
          {
            double drift_dis = fabs(last_depth_image.at<uint16_t>((int)vv, (int)uu) / depth_scale_ - pt_reproj.z());
            //cout << "drift dis: " << drift_dis << endl;
            if (drift_dis < depth_filter_tolerance_)
            {
              proj_points_[proj_points_cnt_++] = pt_NED;
              if (show_filter_proj_depth_)
            	{
                point.x = pt_NED[0];
                point.y = pt_NED[1];
                point.z = pt_NED[2];
                point.r = 255;
                point.g = 0;
                point.b = 0;
                cloud.points.push_back(point);
              }
            }
          }
          else
          {
						// new point
            proj_points_[proj_points_cnt_++] = pt_NED;
            if (show_filter_proj_depth_)
          	{
              point.x = pt_NED[0];
              point.y = pt_NED[1];
              point.z = pt_NED[2];
              point.r = 255;
              point.g = 0;
              point.b = 0;
              cloud.points.push_back(point);
            }
          }
        }
      }
    }
    /* ---------- maintain last ---------- */
    last_T_wc = T_wc;
    last_depth_image = depth_image;
  }


  if (show_filter_proj_depth_)
	{
    cloud.width = (int)cloud.points.size();
    cloud.height = 1;    //height=1 implies this is not an "ordered" point cloud
    // Convert the cloud to ROS message
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(cloud, output);
    output.header.stamp = r_s;
    output.header.frame_id = "map";
    projected_pc_pub_.publish(output);
  }
}

void OccMap::raycastProcess(const Eigen::Vector3d& t_wc)
{
  if (proj_points_cnt_ == 0)
    return;

  // raycast_num_ = (raycast_num_ + 1) % 100000;
  raycast_num_ += 1;

  //   ROS_INFO_STREAM("proj_points_ size: " << proj_points_cnt_);

  /* ---------- iterate projected points ---------- */
  int set_cache_idx;
  for (int i = 0; i < proj_points_cnt_; ++i)
  {
    /* ---------- occupancy of ray end ---------- */
    Eigen::Vector3d pt_w = proj_points_[i];
    if (!isInMap(pt_w))
      continue;
		
    double length = (pt_w - t_wc).norm();
  // 		ROS_INFO_STREAM("len: " << length);
    if (length < min_ray_length_)
      continue;
    else if (length > max_ray_length_)
    {
      pt_w = (pt_w - t_wc) / length * max_ray_length_ + t_wc;
      set_cache_idx = setCacheOccupancy(pt_w, 0);
    }
    else
      set_cache_idx = setCacheOccupancy(pt_w, 1);

    /* ---------- raycast will ignore close end ray ---------- */
    if (set_cache_idx != INVALID_IDX)
    {
      if (cache_rayend_[set_cache_idx] == raycast_num_)
      {
        continue;
      }
      else
        cache_rayend_[set_cache_idx] = raycast_num_;
    }

    //ray casting backwards from point in world frame to camera pos, 
    //the backwards way skips the overlap grids in each ray end by recording cache_traverse_.
    bool need_ray = raycaster_.setInput(pt_w / resolution_, t_wc / resolution_); //(ray start, ray end)
    if (!need_ray)
      continue;
    Eigen::Vector3d half = Eigen::Vector3d(0.5, 0.5, 0.5);
    Eigen::Vector3d ray_pt;
    if (!raycaster_.step(ray_pt)) // skip the ray start point since it's the projected point.
      continue;
    while (raycaster_.step(ray_pt))
    {
      Eigen::Vector3d tmp = (ray_pt + half) * resolution_;
      set_cache_idx = setCacheOccupancy(tmp, 0);
      if (set_cache_idx != INVALID_IDX)
      {
        //skip overlap grids in each ray
        if (cache_traverse_[set_cache_idx] == raycast_num_)
          break;
        else
          cache_traverse_[set_cache_idx] = raycast_num_;
      }
    }
  }

  /* ---------- update occupancy in batch ---------- */
  while (!cache_voxel_.empty())
  {
    Eigen::Vector3i idx = cache_voxel_.front();
    int idx_ctns = idx(0) * grid_size_(1) * grid_size_(2) + idx(1) * grid_size_(2) + idx(2);
    cache_voxel_.pop();
    known_buffer_[idx_ctns] += 1;

    double log_odds_update =
        cache_hit_[idx_ctns] >= cache_all_[idx_ctns] - cache_hit_[idx_ctns] ? prob_hit_log_ : prob_miss_log_;
    cache_hit_[idx_ctns] = cache_all_[idx_ctns] = 0;

    if ((log_odds_update >= 0 && occupancy_buffer_[idx_ctns] >= clamp_max_log_) ||
        (log_odds_update <= 0 && occupancy_buffer_[idx_ctns] <= clamp_min_log_))
      continue;

    // Eigen::Vector3i min_id, max_id;
    // posToIndex(local_range_min_, min_id);
    // posToIndex(local_range_max_, max_id);
    // bool in_local = idx(0) >= min_id(0) && idx(0) <= max_id(0) && idx(1) >= min_id(1) && idx(1) <= max_id(1) &&
    //                 idx(2) >= min_id(2) && idx(2) <= max_id(2);
    // if (!in_local)
    // {
    //   occupancy_buffer_[idx_ctns] = clamp_min_log_;
    // }

    occupancy_buffer_[idx_ctns] =
        std::min(std::max(occupancy_buffer_[idx_ctns] + log_odds_update, clamp_min_log_), clamp_max_log_);
    pred_occupancy_buffer_[idx_ctns] = occupancy_buffer_[idx_ctns];
  }
}

void OccMap::raycastPclProcess(const Eigen::Vector3d& t_wc, const sensor_msgs::PointCloud2 pointcloud_msg)
{
  ros::Time t_s = ros::Time::now();
  // t_wc: position of lidar in world frame
  //pointcloud_msg: lidar point cloud of latest frame

  // pcl::PointCloud<pcl::PointXYZI> pointcloud_pcl;
  pcl::PointCloud<pcl::PointXYZ> pointcloud_pcl;

  // pointcloud_pcl is modified below:
  pcl::fromROSMsg(pointcloud_msg, pointcloud_pcl);
  proj_points_cnt_ = pointcloud_pcl.points.size();
  if (proj_points_cnt_ == 0)
    return;

  // raycast_num_ = (raycast_num_ + 1) % 100000;
  raycast_num_ += 1;

  // ROS_INFO_STREAM("proj_points_ size: " << proj_points_cnt_);

  /* ---------- iterate projected points ---------- */
  int set_cache_idx;
  int added_floor_count = 0;
  int cleared_count = 0;

  Eigen::Vector3d half = Eigen::Vector3d(0.5, 0.5, 0.5);
  Eigen::Vector3d ray_pt, tmp;
  Eigen::Vector3i id;

  for (int i = 0; i < proj_points_cnt_; ++i)
  {
    /* ---------- occupancy of ray end ---------- */
    Eigen::Vector3d pt_w(pointcloud_pcl.points[i].x, pointcloud_pcl.points[i].y, pointcloud_pcl.points[i].z);
    // posToIndex(pt_w, id);
    // indexToPos(id, pt_w);
    if (!isInMap(pt_w))
      continue;
		
    // real points
    bool need_anti_raycast = false;
    // if (pointcloud_pcl.points[i].intensity <= 1)
    if (true)
    {
      double length = (pt_w - t_wc).norm();
      // ROS_INFO_STREAM("len: " << length);
      if (length < min_ray_length_)
        continue;
      else if (length > max_ray_length_)
      {
        pt_w = (pt_w - t_wc) * max_ray_length_ / length  + t_wc;
        set_cache_idx = setCacheOccupancy(pt_w, 0);
      }
      else{
        set_cache_idx = setCacheOccupancy(pt_w, 1);
        need_anti_raycast = true;
      }

      /* ---------- raycast will ignore close end ray ---------- */
      if (set_cache_idx != INVALID_IDX)
      {
        // if (cache_rayend_[set_cache_idx] == raycast_num_ && !raycast_fisttime_)
        if (cache_rayend_[set_cache_idx] == raycast_num_)
        {
          continue;
        }
        else
          cache_rayend_[set_cache_idx] = raycast_num_;
      }

      //ray casting backwards from point in world frame to camera pos, 
      //the backwards way skips the overlap grids in each ray end by recording cache_traverse_.
      bool need_ray = raycaster_.setInput(pt_w / resolution_, t_wc / resolution_); //(ray start, ray end)
      if (!need_ray)
        continue;

      // set one grid behind occ as occ
      if (need_anti_raycast)
      {
        raycaster_.anti_step(ray_pt);
        tmp = (ray_pt + half) * resolution_;
        set_cache_idx = setCacheOccupancy(tmp, 1);
      }

      if (!raycaster_.step(ray_pt)) // skip the ray start point since it's the projected point.
        continue;

      // printf("start raycasting!");
      int counter = 0;
      while (raycaster_.step(ray_pt))
      {
        tmp = (ray_pt + half) * resolution_;
        set_cache_idx = setCacheOccupancy(tmp, 0);
        if (set_cache_idx != INVALID_IDX)
        {
          //skip overlap grids in each ray
          if (cache_traverse_[set_cache_idx] == raycast_num_)
            break;
          else
            cache_traverse_[set_cache_idx] = raycast_num_;
        }
      }
    }

    // else if(pointcloud_pcl.points[i].intensity >= 50){
    //   added_floor_count += 1;
    //   double length = (pt_w - t_wc).norm();
    // // // 		ROS_INFO_STREAM("len: " << length);
    //   if (length < min_ray_length_)
    //     continue;
    //   else if (length > max_ray_length_)
    //   {
    //     pt_w = (pt_w - t_wc) / length * max_ray_length_ + t_wc;
    //     set_cache_idx = setCacheOccupancy(pt_w, 0);
    //   }
    //   else{
    //     set_cache_idx = setCacheOccupancy(pt_w, 1);
    //   }
    //   set_cache_idx = setCacheOccupancy(pt_w, 1);
    //   /* ---------- raycast will ignore close end ray ---------- */
    //   if (set_cache_idx != INVALID_IDX)
    //   {
    //     if (cache_rayend_[set_cache_idx] == raycast_num_)
    //     {
    //       continue;
    //     }
    //     else
    //       cache_rayend_[set_cache_idx] = raycast_num_;
    //   }    

    //   //ray casting backwards from point in world frame to camera pos, 
    //   //the backwards way skips the overlap grids in each ray end by recording cache_traverse_.
    //   // step 1: collision check
    //   bool need_ray = raycaster_.setInput(pt_w / resolution_, t_wc / resolution_); //(ray start, ray end)
    //   if (!need_ray)
    //     continue;


    //   if (!raycaster_.step(ray_pt)) // skip the ray start point since it's the projected point.
    //     continue;
    //   bool is_free = true;
    //   while (raycaster_.step(ray_pt))
    //   {
    //     tmp = (ray_pt + half) * resolution_;
    //     if (!isInMap(tmp))
    //       continue;
    //     // stop raycasting if real occ is found
    //     posToIndex(tmp, id);

    //     int idx = id(0) * grid_size_(1) * grid_size_(2) + id(1) * grid_size_(2) + id(2);
    //     if (occupancy_buffer_[idx] > min_occupancy_log_)
    //     {
    //       // ROS_INFO_STREAM("tmp: " << tmp(0) << ", " << tmp(1) << ", " << tmp(2));
    //       // ROS_INFO_STREAM("t_wc: " << t_wc(0) << ", " << t_wc(1) << ", " << t_wc(2));
    //       // ROS_INFO_STREAM("pt_w: " << pt_w(0) << ", " << pt_w(1) << ", " << pt_w(2));
    //       is_free = false;
    //       break;
    //     }
    //   }

    //   //step 2: raycast
    //   if (is_free)
    //   {
    //     need_ray = raycaster_.setInput(pt_w / resolution_, t_wc / resolution_); //(ray start, ray end)
    //     // if (!need_ray){
    //     //   ROS_WARN("DO NOT NEED RAY");
    //     //   continue;
    //     // }

    //     if (!raycaster_.step(ray_pt)) // skip the ray start point since it's the projected point.
    //       continue;
    //     while (raycaster_.step(ray_pt))
    //     {
    //       tmp = (ray_pt + half) * resolution_;
    //       if (!isInMap(tmp))
    //         continue;
    //       cleared_count += 1;
    //       set_cache_idx = setCacheOccupancy(tmp, 0);
    //       if (set_cache_idx != INVALID_IDX)
    //       {
    //         //skip overlap grids in each ray
    //         if (cache_traverse_[set_cache_idx] == raycast_num_)
    //           break;
    //         else
    //           cache_traverse_[set_cache_idx] = raycast_num_;
    //       }
    //     }
    //   }
    // }
  }

  // ROS_INFO_STREAM("added_floor_count: " << added_floor_count << ",  cleared_count: " << cleared_count);

  /* ---------- update occupancy in batch ---------- */
  // ROS_INFO_STREAM("cache_voxel_: " << cache_voxel_.size());
  while (!cache_voxel_.empty())
  {
    id = cache_voxel_.front();
    int idx_ctns = id(0) * grid_size_(1) * grid_size_(2) + id(1) * grid_size_(2) + id(2);
    cache_voxel_.pop();
    known_buffer_[idx_ctns] += 1;

    double log_odds_update =
        cache_hit_[idx_ctns] >= cache_all_[idx_ctns] - cache_hit_[idx_ctns] ? prob_hit_log_ : prob_miss_log_;
    cache_hit_[idx_ctns] = cache_all_[idx_ctns] = 0;

    if ((log_odds_update >= 0 && occupancy_buffer_[idx_ctns] >= clamp_max_log_) ||
        (log_odds_update <= 0 && occupancy_buffer_[idx_ctns] <= clamp_min_log_))
      continue;

    occupancy_buffer_[idx_ctns] =
        std::min(std::max(occupancy_buffer_[idx_ctns] + log_odds_update, clamp_min_log_), clamp_max_log_);
    pred_occupancy_buffer_[idx_ctns] = occupancy_buffer_[idx_ctns];
  }
}

int OccMap::setCacheOccupancy(Eigen::Vector3d pos, int occ)
{
  if (occ != 1 && occ != 0)
  {
    return INVALID_IDX;
  }

  if (!isInMap(pos))
  {
    return INVALID_IDX;
    // --------------- find the nearest point in map range cube --------------------
    // Eigen::Vector3d diff = pos - camera_pos_in_map_;
  }

  Eigen::Vector3i id;
  posToIndex(pos, id);

  int idx_ctns = id(0) * grid_size_(1) * grid_size_(2) + id(1) * grid_size_(2) + id(2);

  cache_all_[idx_ctns] += 1;

  if (cache_all_[idx_ctns] == 1)
  {
    cache_voxel_.push(id);
  }

  if (occ == 1)
    cache_hit_[idx_ctns] += 1;

  return idx_ctns;
}


void OccMap::indepOdomCallback(const nav_msgs::OdometryConstPtr& odom)
{
	latest_odom_time_ = odom->header.stamp;
	curr_posi_[0] = odom->pose.pose.position.x;
  curr_posi_[1] = odom->pose.pose.position.y;
  curr_posi_[2] = odom->pose.pose.position.z;
  curr_twist_[0] = odom->twist.twist.linear.x;
  curr_twist_[1] = odom->twist.twist.linear.y;
  curr_twist_[2] = odom->twist.twist.linear.z;
  curr_q_.w() = odom->pose.pose.orientation.w;
  curr_q_.x() = odom->pose.pose.orientation.x;
  curr_q_.y() = odom->pose.pose.orientation.y;
  curr_q_.z() = odom->pose.pose.orientation.z;
	have_odom_ = true;

  { //TF map^ T ego
    static tf2_ros::TransformBroadcaster br_map_ego;
    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = odom->header.stamp;
    transformStamped.header.frame_id = "map";
    transformStamped.child_frame_id = "ego";
    transformStamped.transform.translation.x = odom->pose.pose.position.x;
    transformStamped.transform.translation.y = odom->pose.pose.position.y;
    transformStamped.transform.translation.z = odom->pose.pose.position.z;
    transformStamped.transform.rotation.x = odom->pose.pose.orientation.x;
    transformStamped.transform.rotation.y = odom->pose.pose.orientation.y;
    transformStamped.transform.rotation.z = odom->pose.pose.orientation.z;
    transformStamped.transform.rotation.w = odom->pose.pose.orientation.w;
    br_map_ego.sendTransform(transformStamped);
  }
  have_odom_ = true;

}

void OccMap::globalCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  if(!use_global_map_ || has_global_cloud_)
    return;

	pcl::PointCloud<pcl::PointXYZ> global_cloud;
  pcl::fromROSMsg(*msg, global_cloud);
  global_map_valid_ = true;

  if (global_cloud.points.size() == 0)
    return;

  pcl::PointXYZ pt;
  Eigen::Vector3d p3d;
  for (size_t i = 0; i < global_cloud.points.size(); ++i)
  {
    pt = global_cloud.points[i];
    p3d(0) = pt.x; p3d(1) = pt.y; p3d(2) = pt.z;
    this->setOccupancy(p3d);
  }
  has_global_cloud_ = true;
}

void OccMap::init(ros::NodeHandle& nh)
{
  node_ = nh;
  /* ---------- param ---------- */
  node_.param("occ_map/origin_x", origin_(0), -20.0);
  node_.param("occ_map/origin_y", origin_(1), -20.0);
  node_.param("occ_map/origin_z", origin_(2), 0.0);
  node_.param("occ_map/map_size_x", map_size_(0), 40.0);
  node_.param("occ_map/map_size_y", map_size_(1), 40.0);
  node_.param("occ_map/map_size_z", map_size_(2), 5.0);

  node_.param("occ_map/local_grid_size_x", local_grid_size_(0), 80);
  node_.param("occ_map/local_grid_size_y", local_grid_size_(1), 80);
  node_.param("occ_map/local_grid_size_z", local_grid_size_(2), 40);
  node_.param("occ_map/known_threshold", known_threshold_, 1);
  node_.param("occ_map/unknown_as_free", unknown_as_free_, true);
  node_.param("occ_map/use_pred_for_collision", use_pred_for_collision_, true);

  node_.param("occ_map/local_radius_x", sensor_range_(0), -1.0);
  node_.param("occ_map/local_radius_y", sensor_range_(1), -1.0);
  node_.param("occ_map/local_radius_z", sensor_range_(2), -1.0);

  node_.param("occ_map/resolution", resolution_, 0.2);
  node_.param("occ_map/use_global_map", use_global_map_, false);

	node_.param("occ_map/depth_scale", depth_scale_, -1.0);
  node_.param("occ_map/use_shift_filter", use_shift_filter_, true);
  node_.param("occ_map/depth_filter_tolerance", depth_filter_tolerance_, -1.0);
  node_.param("occ_map/depth_filter_maxdist", depth_filter_maxdist_, -1.0);
  node_.param("occ_map/depth_filter_mindist", depth_filter_mindist_, -1.0);
  node_.param("occ_map/depth_filter_margin", depth_filter_margin_, -1);
  node_.param("occ_map/skip_pixel", skip_pixel_, -1);
  node_.param("occ_map/show_raw_depth", show_raw_depth_, false);
  node_.param("occ_map/show_filter_proj_depth", show_filter_proj_depth_, false);
  node_.param("occ_map/vis_unknown_", vis_unknown_, false);

  
  node_.param("occ_map/min_ray_length", min_ray_length_, -0.1);
  node_.param("occ_map/max_ray_length", max_ray_length_, -0.1);

  node_.param("occ_map/prob_hit_log", prob_hit_log_, 0.70);
  node_.param("occ_map/prob_miss_log", prob_miss_log_, 0.35);
  node_.param("occ_map/clamp_min_log", clamp_min_log_, 0.12);
  node_.param("occ_map/clamp_max_log", clamp_max_log_, 0.97);
  node_.param("occ_map/min_occupancy_log", min_occupancy_log_, 0.80);
  
  node_.param("occ_map/use_pred", use_pred_, false);
  node_.param("occ_map/pred_occ_thresord", pred_occ_thresord_, 0.50);

  node_.param("occ_map/fx", fx_, -1.0);
  node_.param("occ_map/fy", fy_, -1.0);
  node_.param("occ_map/cx", cx_, -1.0);
  node_.param("occ_map/cy", cy_, -1.0);
  node_.param("occ_map/rows", rows_, 480);
  node_.param("occ_map/cols", cols_, 320);

  std::string enginePath;
  std::string modelPath;
  node_.param<std::string>("occ_map/modelPath", modelPath, "/home/wlz/catkin_ws/src/krrt-planner/opnet/models/simple_80_40.trt");
  node_.param<std::string>("occ_map/enginePath", enginePath, "/home/wlz/catkin_ws/src/krrt-planner/opnet/models/simple_80_40.trt");

  cout << "use_shift_filter_: " << use_shift_filter_ << endl;
  cout << "map size: " << map_size_.transpose() << endl;
  cout << "resolution: " << resolution_ << endl;

  cout << "hit: " << prob_hit_log_ << endl;
  cout << "miss: " << prob_miss_log_ << endl;
  cout << "min: " << clamp_min_log_ << endl;
  cout << "max: " << clamp_max_log_ << endl;
  cout << "thresh: " << min_occupancy_log_ << endl;
  cout << "skip: " << skip_pixel_ << endl;
	cout << "sensor_range: " << sensor_range_.transpose() << endl;



  if (use_pred_)
  {
    cout << "START BUILDING TRT ENGINE"<< endl;

    my_model_.initParam(initializeSampleParams(modelPath));

    if (!my_model_.build())
    {
      cout << "BUILDING TRT ENGINE FAIL"<< endl;
      return;
    }
    else{
      cout << "BUILDING TRT ENGINE SUCCEED"<< endl;
      if (enginePath != modelPath)
      {
        cout << "SAVING TRT ENGINE"<< endl;
        my_model_.saveEngine(enginePath);
        cout << "SAVED TRT ENGINE"<< endl;
      }
    }
  }


  /* ---------- setting ---------- */
  have_odom_ = false;
  global_map_valid_ = true;
  local_map_valid_ = false;
  has_global_cloud_ = false;
  has_first_depth_ = false;
  raycast_fisttime_ = true;

  resolution_inv_ = 1 / resolution_;
  for (int i = 0; i < 3; ++i)
    grid_size_(i) = ceil(map_size_(i) / resolution_);
  
  curr_view_cloud_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  history_view_cloud_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  history_pred_cloud_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  T_ic0_ << 0.0, 0.0, 1.0, 0.0,
           -1.0, 0.0, 0.0, 0.0,
            0.0,-1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0;
								
  cout << "origin_: " << origin_.transpose() << endl;
  min_range_ = origin_;
  max_range_ = origin_ + map_size_;
  cout << "min_range_: " << min_range_.transpose() << endl;
  cout << "max_range_: " << max_range_.transpose() << endl;

  //init proj_points_ buffer
  proj_points_.resize(rows_ * cols_ / skip_pixel_ / skip_pixel_);
  
  K_depth_.setZero();
  K_depth_(0, 0) = fx_; //fx
  K_depth_(1, 1) = fy_; //fy
  K_depth_(0, 2) = cx_; //cx
  K_depth_(1, 2) = cy_; //cy
  K_depth_(2, 2) = 1.0;
  cout << "intrinsic: " << K_depth_ << endl;
								
  // initialize size of buffer
  int buffer_size = grid_size_(0) * grid_size_(1) * grid_size_(2);
  cout << "buffer size: " << buffer_size <<grid_size_(0) <<grid_size_(1) << grid_size_(2) << endl;
  occupancy_buffer_.resize(buffer_size);
  pred_occupancy_buffer_.resize(buffer_size);
  known_buffer_.resize(buffer_size);

  // int local_buffer_size = local_grid_size_(0) * local_grid_size_(1) * local_grid_size_(2);
  // cout << "local buffe size: " << local_buffer_size << endl;
  // local_occupancy_buffer_.resize(local_buffer_size);
  // local_idx_buffer_.resize(local_buffer_size);

  cache_all_.resize(buffer_size);
  cache_hit_.resize(buffer_size);

  cache_rayend_.resize(buffer_size);
  cache_traverse_.resize(buffer_size);
  raycast_num_ = 0;
  
  proj_points_cnt_ = 0;
  collisions_ = 0;

  // fill(occupancy_buffer_.begin(), occupancy_buffer_.end(), clamp_min_log_);
  fill(occupancy_buffer_.begin(), occupancy_buffer_.end(), -1.0);

  fill(pred_occupancy_buffer_.begin(), pred_occupancy_buffer_.end(), clamp_min_log_);

  fill(known_buffer_.begin(), known_buffer_.end(), 0);

  fill(cache_all_.begin(), cache_all_.end(), 0);
  fill(cache_hit_.begin(), cache_hit_.end(), 0);

  fill(cache_rayend_.begin(), cache_rayend_.end(), -1);
  fill(cache_traverse_.begin(), cache_traverse_.end(), -1);

  //set x-y boundary occ
  // for (double cx = min_range_[0]+resolution_/2; cx <= max_range_[0]-resolution_/2; cx += resolution_)
  //   for (double cz = min_range_[2]+resolution_/2; cz <= max_range_[2]-resolution_/2; cz += resolution_)
  //   {
  //     this->setOccupancy(Eigen::Vector3d(cx, min_range_[1]+resolution_/2, cz));
  //     this->setOccupancy(Eigen::Vector3d(cx, max_range_[1]-resolution_/2, cz));
  //   }
  // for (double cy = min_range_[1]+resolution_/2; cy <= max_range_[1]-resolution_/2; cy += resolution_)
  //   for (double cz = min_range_[2]+resolution_/2; cz <= max_range_[2]-resolution_/2; cz += resolution_)
  //   {
  //     this->setOccupancy(Eigen::Vector3d(min_range_[0]+resolution_/2, cy, cz));
  //     this->setOccupancy(Eigen::Vector3d(max_range_[0]-resolution_/2, cy, cz));
  //   }

    /* ---------- sub and pub ---------- */
	if (!use_global_map_)
	{
    // depth_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(node_, "/depth_topic", 1, ros::TransportHints().tcpNoDelay()));
    // odom_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(node_, "/odom_topic", 1, ros::TransportHints().tcpNoDelay()));
    // pcl_sub_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(node_, "/pcl_topic", 1, ros::TransportHints().tcpNoDelay()));
    indep_odom_sub_ = node_.subscribe<nav_msgs::Odometry>("/odom_topic", 10, &OccMap::indepOdomCallback, this, ros::TransportHints().tcpNoDelay());
    cloud_odom_sub_ = node_.subscribe<ocp_msgs::OdomPcl>("/odom_pcl_topic", 10, &OccMap::pclOdomCallback, this, ros::TransportHints().tcpNoDelay());
    // sync_image_odom_.reset(new message_filters::Synchronizer<SyncPolicyImageOdom>(SyncPolicyImageOdom(100), *depth_sub_, *odom_sub_));
    // sync_image_odom_->registerCallback(boost::bind(&OccMap::depthOdomCallback, this, _1, _2, T_ic0_, last_T_wc0_, last_depth0_image_, "camera_front"));
    // sync_pcl_odom_.reset(new message_filters::Synchronizer<SyncPolicyPclOdom>(SyncPolicyPclOdom(100), *pcl_sub_, *odom_sub_));
    // sync_pcl_odom_->registerCallback(boost::bind(&OccMap::pclOdomCallback, this, _1, _2, T_ic0_));
    // global_occ_vis_timer_ = node_.createTimer(ros::Duration(5), &OccMap::globalOccVisCallback, this);
    // local_occ_vis_timer_ = node_.createTimer(ros::Duration(0.2), &OccMap::localOccVisCallback, this);
    local_occ_vis_timer_ = node_.createTimer(ros::Duration(0.05), &OccMap::publishLocalOccCallback, this);
    // collision_check_timer_ = node_.createTimer(ros::Duration(0.1), &OccMap::publishCollisionNum, this);
  }
	else
	{
    indep_odom_sub_ = node_.subscribe<nav_msgs::Odometry>("/odom_topic", 10, &OccMap::indepOdomCallback, this, ros::TransportHints().tcpNoDelay());
		global_occ_vis_timer_ = node_.createTimer(ros::Duration(5), &OccMap::globalOccVisCallback, this);
	}

  curr_view_cloud_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/occ_map/local_view_cloud", 1);
  hist_view_cloud_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/occ_map/history_view_cloud", 1);
  hist_pred_cloud_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/occ_map/history_pred_cloud", 1);

  local_occ_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/occ_map/origin_local_cloud", 1);
  added_occ_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/occ_map/pred_local_cloud", 1);
  collision_pub_ = node_.advertise<std_msgs::Int8>("/occ_map/collisions", 1);

	global_cloud_sub_ = node_.subscribe<sensor_msgs::PointCloud2>("/global_cloud", 1, &OccMap::globalCloudCallback, this);

	origin_pcl_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/occ_map/raw_pcl", 1);
  projected_pc_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/occ_map/filtered_pcl", 1);
  pred_client_ = node_.serviceClient<ocp_msgs::PredictPCL>("/occ_map/pred");
}

}  // namespace fast_planner
