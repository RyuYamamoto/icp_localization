#include <icp_localization/icp_localization.h>

#include <eigen_conversions/eigen_msg.h>
#include <tf2_eigen/tf2_eigen.h>

ICPLocalization::ICPLocalization()
{
  pnh_.param<double>("min_range", min_range_, 0.5);
  pnh_.param<double>("max_range", max_range_, 60.0);
  pnh_.param<double>("downsample_leaf_size", downsample_leaf_size_, 3.0);
  pnh_.param<double>("transformation_epsilon", transformation_epsilon_, 0.01);
  pnh_.param<double>("max_correspondence_distance", max_correspondence_distance_, 1.0);
  pnh_.param<double>("euclidean_fitness_epsilon", euclidean_fitness_epsilon_, 0.1);
  pnh_.param<double>(
    "ransac_outlier_rejection_threshold", ransac_outlier_rejection_threshold_, 1.0);
  pnh_.param<int>("max_iteration", max_iteration_, 20);
  pnh_.param<std::string>("map_frame_id", map_frame_id_, "map");
  pnh_.param<std::string>("base_frame_id", base_frame_id_, "base_link");

  const std::string registration_type = pnh_.param<std::string>("registration_type", "FAST_GICP");
  if (registration_type == "FAST_GICP") {
    boost::shared_ptr<fast_gicp::FastGICP<PointType, PointType>> fast_gicp(
      new fast_gicp::FastGICP<PointType, PointType>);
    const int num_thread = pnh_.param<int>("gicp_num_thread", 0);
    if (0 < num_thread) fast_gicp->setNumThreads(num_thread);
    registration_ = fast_gicp;
  } else {
    boost::shared_ptr<pcl::GeneralizedIterativeClosestPoint<PointType, PointType>> gicp(
      new pcl::GeneralizedIterativeClosestPoint<PointType, PointType>);
    registration_ = gicp;
  }
  registration_->setMaximumIterations(max_iteration_);
  registration_->setTransformationEpsilon(transformation_epsilon_);
  registration_->setMaxCorrespondenceDistance(max_correspondence_distance_);
  registration_->setEuclideanFitnessEpsilon(euclidean_fitness_epsilon_);
  registration_->setRANSACOutlierRejectionThreshold(ransac_outlier_rejection_threshold_);

  map_subscriber_ = pnh_.subscribe("points_map", 1, &ICPLocalization::mapCallback, this);
  points_subscriber_ = pnh_.subscribe("points_raw", 1, &ICPLocalization::pointsCallback, this);
  initialpose_subscriber_ =
    pnh_.subscribe("/initialpose", 1, &ICPLocalization::initialPoseCallback, this);

  icp_align_cloud_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("aligned_cloud", 1);
  icp_pose_publisher_ = pnh_.advertise<geometry_msgs::PoseStamped>("icp_pose", 1);
}

void ICPLocalization::downsample(
  const pcl::PointCloud<PointType>::Ptr & input_cloud_ptr,
  pcl::PointCloud<PointType>::Ptr & output_cloud_ptr)
{
  pcl::VoxelGrid<PointType> voxel_grid;
  voxel_grid.setLeafSize(downsample_leaf_size_, downsample_leaf_size_, downsample_leaf_size_);
  voxel_grid.setInputCloud(input_cloud_ptr);
  voxel_grid.filter(*output_cloud_ptr);
}

void ICPLocalization::crop(
  const pcl::PointCloud<PointType>::Ptr & input_cloud_ptr,
  pcl::PointCloud<PointType>::Ptr output_cloud_ptr, const double min_range, const double max_range)
{
  for (const auto & p : input_cloud_ptr->points) {
    const double dist = std::sqrt(p.x * p.x + p.y * p.y);
    if (min_range < dist && dist < max_range) { output_cloud_ptr->points.emplace_back(p); }
  }
}

void ICPLocalization::mapCallback(const sensor_msgs::PointCloud2 & map)
{
  ROS_INFO("map callback");

  pcl::PointCloud<PointType>::Ptr map_cloud(new pcl::PointCloud<PointType>);
  pcl::fromROSMsg(map, *map_cloud);

  registration_->setInputTarget(map_cloud);
}

void ICPLocalization::pointsCallback(const sensor_msgs::PointCloud2 & points)
{
  if (registration_->getInputTarget() == nullptr) {
    ROS_ERROR("map not received!");
    return;
  }

  if (!localization_ready_) {
    ROS_ERROR("initial pose not received!");
    return;
  }

  const ros::Time current_scan_time = points.header.stamp;

  pcl::PointCloud<PointType>::Ptr input_cloud_ptr(new pcl::PointCloud<PointType>);
  pcl::fromROSMsg(points, *input_cloud_ptr);

  // downsampling input point cloud
  pcl::PointCloud<PointType>::Ptr filtered_cloud(new pcl::PointCloud<PointType>);
  downsample(input_cloud_ptr, filtered_cloud);

  // crop point cloud
  pcl::PointCloud<PointType>::Ptr crop_cloud(new pcl::PointCloud<PointType>);
  crop(filtered_cloud, crop_cloud, min_range_, max_range_);

  // transform base_link to sensor_link
  pcl::PointCloud<PointType>::Ptr transform_cloud_ptr(new pcl::PointCloud<PointType>);
  const std::string sensor_frame_id = points.header.frame_id;
  geometry_msgs::TransformStamped sensor_frame_transform;
  try {
    sensor_frame_transform = tf_buffer_.lookupTransform(
      base_frame_id_, sensor_frame_id, current_scan_time, ros::Duration(1.0));
  } catch (tf2::TransformException & ex) {
    ROS_ERROR("%s", ex.what());
    sensor_frame_transform.header.stamp = current_scan_time;
    sensor_frame_transform.header.frame_id = base_frame_id_;
    sensor_frame_transform.child_frame_id = sensor_frame_id;
    sensor_frame_transform.transform.translation.x = 0.0;
    sensor_frame_transform.transform.translation.y = 0.0;
    sensor_frame_transform.transform.translation.z = 0.0;
    sensor_frame_transform.transform.rotation.w = 1.0;
    sensor_frame_transform.transform.rotation.x = 0.0;
    sensor_frame_transform.transform.rotation.y = 0.0;
    sensor_frame_transform.transform.rotation.z = 0.0;
  }
  const Eigen::Affine3d base_to_sensor_frame_affine = tf2::transformToEigen(sensor_frame_transform);
  const Eigen::Matrix4f base_to_sensor_frame_matrix =
    base_to_sensor_frame_affine.matrix().cast<float>();
  pcl::transformPointCloud(*crop_cloud, *transform_cloud_ptr, base_to_sensor_frame_matrix);
  registration_->setInputSource(transform_cloud_ptr);

  // calculation initial pose for icp
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
  Eigen::Affine3d initial_pose_affine;
  tf2::fromMsg(initial_pose_, initial_pose_affine);
  init_guess = initial_pose_affine.matrix().cast<float>();

  pcl::PointCloud<PointType>::Ptr output_cloud(new pcl::PointCloud<PointType>);
  registration_->align(*output_cloud, init_guess);

  const bool convergenced = registration_->hasConverged();

  const Eigen::Matrix4f result_icp_pose = registration_->getFinalTransformation();

  Eigen::Affine3d result_icp_pose_affine;
  result_icp_pose_affine.matrix() = result_icp_pose.cast<double>();
  const geometry_msgs::Pose icp_pose = tf2::toMsg(result_icp_pose_affine);
  initial_pose_ = icp_pose;

  geometry_msgs::PoseStamped icp_pose_msg;
  icp_pose_msg.header.frame_id = map_frame_id_;
  icp_pose_msg.header.stamp = current_scan_time;
  icp_pose_msg.pose = icp_pose;

  if (convergenced) {
    icp_pose_publisher_.publish(icp_pose_msg);
    publishTF(map_frame_id_, base_frame_id_, icp_pose_msg);
  }

  sensor_msgs::PointCloud2 aligned_cloud_msg;
  pcl::toROSMsg(*output_cloud, aligned_cloud_msg);
  aligned_cloud_msg.header = points.header;
  icp_align_cloud_publisher_.publish(aligned_cloud_msg);
}

void ICPLocalization::initialPoseCallback(
  const geometry_msgs::PoseWithCovarianceStamped & initialpose)
{
  ROS_INFO("initial pose callback.");
  if (initialpose.header.frame_id == map_frame_id_) {
    initial_pose_ = initialpose.pose.pose;
    if (!localization_ready_) localization_ready_ = true;
  } else {
    // TODO transform
    ROS_ERROR(
      "frame_id is not same. initialpose.header.frame_id is %s",
      initialpose.header.frame_id.c_str());
  }
}

void ICPLocalization::publishTF(
  const std::string frame_id, const std::string child_frame_id,
  const geometry_msgs::PoseStamped pose)
{
  geometry_msgs::TransformStamped transform_stamped;

  transform_stamped.header.frame_id = frame_id;
  transform_stamped.header.stamp = pose.header.stamp;
  transform_stamped.child_frame_id = child_frame_id;
  transform_stamped.transform.translation.x = pose.pose.position.x;
  transform_stamped.transform.translation.y = pose.pose.position.y;
  transform_stamped.transform.translation.z = pose.pose.position.z;
  transform_stamped.transform.rotation.w = pose.pose.orientation.w;
  transform_stamped.transform.rotation.x = pose.pose.orientation.x;
  transform_stamped.transform.rotation.y = pose.pose.orientation.y;
  transform_stamped.transform.rotation.z = pose.pose.orientation.z;

  broadcaster_.sendTransform(transform_stamped);
}
