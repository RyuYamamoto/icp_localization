#ifndef _NDT_LOCALIZATION_
#define _NDT_LOCALIZATION_

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32.h>

#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>

class ICPLocalization
{
  using PointType = pcl::PointXYZ;

public:
  ICPLocalization();
  ~ICPLocalization() = default;

private:
  void mapCallback(const sensor_msgs::PointCloud2 & map);
  void pointsCallback(const sensor_msgs::PointCloud2 & points);
  void initialPoseCallback(const geometry_msgs::PoseWithCovarianceStamped & initialpose);

  void downsample(
    const pcl::PointCloud<PointType>::Ptr & input_cloud_ptr,
    pcl::PointCloud<PointType>::Ptr & output_cloud_ptr);
  void crop(
    const pcl::PointCloud<PointType>::Ptr & input_cloud_ptr,
    pcl::PointCloud<PointType>::Ptr output_cloud_ptr, const double min_range,
    const double max_range);

  void publishTF(
    const std::string frame_id, const std::string child_frame_id,
    const geometry_msgs::PoseStamped pose);

private:
  ros::NodeHandle nh_{};
  ros::NodeHandle pnh_{"~"};

  ros::Subscriber map_subscriber_;
  ros::Subscriber points_subscriber_;
  ros::Subscriber initialpose_subscriber_;
  ros::Publisher icp_align_cloud_publisher_;
  ros::Publisher icp_pose_publisher_;

  // icp
  boost::shared_ptr<pcl::GeneralizedIterativeClosestPoint<PointType, PointType>> icp_;

  geometry_msgs::Pose initial_pose_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformBroadcaster broadcaster_;

  // config for icp omp
  double max_correspondence_distance_;
  double euclidean_fitness_epsilon_;
  double ransac_outlier_rejection_threshold_;
  double transformation_epsilon_;
  int max_iteration_;
  std::string map_frame_id_;
  std::string base_frame_id_;

  double downsample_leaf_size_;

  double min_range_;
  double max_range_;

  bool localization_ready_{false};
};

#endif
