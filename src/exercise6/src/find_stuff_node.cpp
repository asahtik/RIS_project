#include <iostream>
#include <ros/ros.h>

#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>

#include <Eigen/Core>

#define LEAF_SIZE 0.01f
#define PLANE_ITERATIONS 1000
#define PLANE_THRESHOLD 0.01f
#define REMAINING_POINTS_PERCENTAGE 0.3f
#define NORMAL_K_NEIGHBOURS 50
#define CYLINDER_WEIGHT 0.1f
#define CYLINDER_ITERATIONS 10000
#define CYLINDER_THRESHOLD 0.05f
#define CYLINDER_MIN_RADIUS 0.06f
#define CYLINDER_MAX_RADIUS 0.2f

tf2_ros::Buffer tfBuffer;
tf2_ros::TransformListener* tfListener;

ros::Publisher pubPlan;
ros::Publisher pubCyl;

void cloud_cb(const pcl::PCLPointCloud2ConstPtr& cloud_blob) {

  ros::Time time_rec;
  time_rec = ros::Time::now();

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  // Read in the cloud data
  pcl::fromPCLPointCloud2(*cloud_blob, *cloud);
  // Build a passthrough filter to remove spurious NaNs
  pcl::PassThrough<pcl::PointXYZ> pass;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pass (new pcl::PointCloud<pcl::PointXYZ>);
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(0, 1.5);
  pass.filter(*cloud_pass);

  // First remove floor and walls
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud_pass);
  sor.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
  sor.filter(*cloud_filtered);

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients(true);
  // Mandatory
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(PLANE_ITERATIONS);
  seg.setDistanceThreshold(PLANE_THRESHOLD);
  seg.setInputCloud(cloud_filtered);
    
  int i = 0, nr_points = (int)cloud_filtered->points.size();
  pcl::IndicesPtr remaining(new std::vector<int>);
  remaining->resize(nr_points);
  for(size_t i = 0; i < remaining->size(); ++i) { (*remaining)[i] = static_cast<int>(i); }

  // While x% of the original cloud is still there
  while(remaining->size() > REMAINING_POINTS_PERCENTAGE * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setIndices(remaining);
    seg.segment(*inliers, *coefficients);
    if(inliers->indices.size() == 0) break;

    // Extract the inliers
    std::vector<int>::iterator it = remaining->begin();
    for(size_t i = 0; i < inliers->indices.size(); ++i)
    {
      int curr = inliers->indices[i];
      // Remove it from further consideration.
      while(it != remaining->end() && *it < curr) { ++it; }
      if(it == remaining->end()) break;
      if(*it == curr) it = remaining->erase(it);
    }
    i++;
  }
  std::cout << "Found " << i << " planes." << std::endl;

  // Extract indices
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outliers(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(cloud_filtered);
  extract.setIndices(remaining);
  extract.setNegative(false);
  extract.filter(*cloud_outliers);

  pcl::PCLPointCloud2 outcloud_plane;
  pcl::toPCLPointCloud2 (*cloud_outliers, outcloud_plane);
  pubPlan.publish (outcloud_plane);


  // Then find cylinders
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);

  // Estimate point normals
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setSearchMethod(tree);
  ne.setInputCloud(cloud_outliers);
  ne.setKSearch(NORMAL_K_NEIGHBOURS);
  ne.compute(*cloud_normals);

  pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg2;
  // Create the segmentation object for cylinder segmentation and set all the parameters
  seg2.setOptimizeCoefficients(true);
  seg2.setModelType(pcl::SACMODEL_CYLINDER);
  seg2.setMethodType(pcl::SAC_RANSAC);
  seg2.setNormalDistanceWeight(CYLINDER_WEIGHT);
  seg2.setMaxIterations(CYLINDER_ITERATIONS);
  seg2.setDistanceThreshold(CYLINDER_THRESHOLD);
  seg2.setRadiusLimits(CYLINDER_MIN_RADIUS, CYLINDER_MAX_RADIUS);
  seg2.setInputCloud(cloud_outliers);
  seg2.setInputNormals(cloud_normals);

  // Obtain the cylinder inliers and coefficients
  seg.segment(*inliers_cylinder, *coefficients_cylinder);

  // Write the cylinder inliers to disk
  extract.setInputCloud(cloud_outliers);
  extract.setIndices(inliers_cylinder);
  extract.setNegative(false);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cylinder(new pcl::PointCloud<pcl::PointXYZ>());
  extract.filter(*cloud_cylinder);

  if (cloud_cylinder->points.empty ()) 
    std::cerr << "Can't find the cylindrical component." << std::endl;
  else {  
    Eigen::Vector4f centroid;   
    pcl::compute3DCentroid (*cloud_cylinder, centroid);
    std::cout << "centroid of the cylindrical component: " << centroid[0] << " " <<  centroid[1] << " " <<   centroid[2] << " " <<   centroid[3] << std::endl;

	  //Create a point in the "camera_rgb_optical_frame"
    geometry_msgs::PointStamped point_camera;
    geometry_msgs::PointStamped point_map;
	  visualization_msgs::Marker marker;
    geometry_msgs::TransformStamped tss;
          
    point_camera.header.frame_id = "camera_depth_optical_frame";
    point_camera.header.stamp = ros::Time::now();

	  point_map.header.frame_id = "map";
    point_map.header.stamp = ros::Time::now();

    point_camera.point.x = centroid[0];
    point_camera.point.y = centroid[1];
    point_camera.point.z = centroid[2];

	  try {
      tss = tfBuffer.lookupTransform("map", "camera_depth_optical_frame", time_rec);
      //tf2_buffer.transform(point_camera, point_map, "map", ros::Duration(2));
    } catch (tf2::TransformException &ex) {
      ROS_WARN("Transform warning: %s\n", ex.what());
    }

    tf2::doTransform(point_camera, point_map, tss);

    // std::cerr << "point_camera: " << point_camera.point.x << " " <<  point_camera.point.y << " " <<  point_camera.point.z << std::endl;
    // std::cerr << "point_map: " << point_map.point.x << " " <<  point_map.point.y << " " <<  point_map.point.z << std::endl;

    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();

    marker.ns = "cylinder";
    marker.id = 0;

    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = point_map.point.x;
    marker.pose.position.y = point_map.point.y;
    marker.pose.position.z = point_map.point.z;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;

    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;

    marker.lifetime = ros::Duration();

    // pubm.publish (marker);

    pcl::PCLPointCloud2 outcloud_cylinder;
    pcl::toPCLPointCloud2 (*cloud_cylinder, outcloud_cylinder);
    pubCyl.publish (outcloud_cylinder);
  }
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "find_stuff");
  ros::NodeHandle nh;

  tfListener = new tf2_ros::TransformListener(tfBuffer);

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("/camera/depth/points", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pubPlan = nh.advertise<pcl::PCLPointCloud2> ("planes", 1);
  pubCyl = nh.advertise<pcl::PCLPointCloud2> ("cylinders", 1);


  // Spin
  ros::spin ();
}

/* #include <iostream>
#include <ros/ros.h>

#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>

#include <Eigen/Core>

#define LEAF_SIZE 0.01f
#define PLANE_ITERATIONS 1000
#define PLANE_THRESHOLD 0.01f
#define REMAINING_POINTS_PERCENTAGE 0.3f
#define PLANE_MIN_SIZE_PERCENTAGE 0.2f
#define NORMAL_K_NEIGHBOURS 50
#define CYLINDER_WEIGHT 0.1f
#define CYLINDER_ITERATIONS 10000
#define CYLINDER_THRESHOLD 0.05f
#define CYLINDER_MIN_RADIUS 0.06f
#define CYLINDER_MAX_RADIUS 0.2f

typedef pcl::PointXYZ PointT;

tf2_ros::Buffer tfBuffer;
tf2_ros::TransformListener* tfListener;

ros::Publisher pubPlan;
ros::Publisher pubCyl;

void cloud_cb(const pcl::PCLPointCloud2ConstPtr& cloud_blob) {

  ros::Time time_rec;
  time_rec = ros::Time::now();

  // All the objects needed
  pcl::PassThrough<PointT> pass;
  pcl::NormalEstimation<PointT, pcl::Normal> ne;
  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg; 
  pcl::ExtractIndices<PointT> extract;
  pcl::ExtractIndices<pcl::Normal> extract_normals;
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

  // Datasets
  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<PointT>::Ptr cloud_filtered2(new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);
  pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients), coefficients_cylinder(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices), inliers_cylinder(new pcl::PointIndices);

  // Read in the cloud data
  pcl::fromPCLPointCloud2(*cloud_blob, *cloud);
  // std::cerr << "PointCloud has: " << cloud->size() << " data points." << std::endl;

  // Build a passthrough filter to remove spurious NaNs
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(0, 1.5);
  pass.filter(*cloud_filtered);
  // std::cerr << "PointCloud after filtering has: " << cloud_filtered->size() << " data points." << std::endl;

  // Estimate point normals
  ne.setSearchMethod(tree);
  ne.setInputCloud(cloud_filtered);
  ne.setKSearch(NORMAL_K_NEIGHBOURS);
  ne.compute(*cloud_normals);

  // Create the segmentation object for the planar model and set all the parameters
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
  seg.setNormalDistanceWeight(0.1);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(PLANE_ITERATIONS);
  seg.setDistanceThreshold(PLANE_THRESHOLD);
  seg.setInputCloud(cloud_filtered);
  seg.setInputNormals(cloud_normals);

  // Obtain the plane inliers and coefficients
  seg.segment (*inliers_plane, *coefficients_plane);
  // std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

  // Extract the planar inliers from the input cloud
  extract.setInputCloud(cloud_filtered);
  extract.setIndices(inliers_plane);
  extract.setNegative(false);

  // Write the planar inliers to disk
  pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());
  extract.filter(*cloud_plane);
  // std::cerr << "PointCloud representing the planar component: " << cloud_plane->size() << " data points." << std::endl;

  // Remove the planar inliers, extract the rest
  extract.setNegative(true);
  extract.filter(*cloud_filtered2);
  extract_normals.setNegative(true);
  extract_normals.setInputCloud(cloud_normals);
  extract_normals.setIndices(inliers_plane);
  extract_normals.filter(*cloud_normals2);

  // Create the segmentation object for cylinder segmentation and set all the parameters
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_CYLINDER);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight(CYLINDER_WEIGHT);
  seg.setMaxIterations(CYLINDER_ITERATIONS);
  seg.setDistanceThreshold(CYLINDER_THRESHOLD);
  seg.setRadiusLimits(CYLINDER_MIN_RADIUS, CYLINDER_MAX_RADIUS);
  seg.setInputCloud(cloud_filtered2);
  seg.setInputNormals(cloud_normals2);

  // Obtain the cylinder inliers and coefficients
  seg.segment(*inliers_cylinder, *coefficients_cylinder);
  // std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

  // Write the cylinder inliers to disk
  extract.setInputCloud(cloud_filtered2);
  extract.setIndices(inliers_cylinder);
  extract.setNegative(false);
  pcl::PointCloud<PointT>::Ptr cloud_cylinder(new pcl::PointCloud<PointT>());
  extract.filter(*cloud_cylinder);
  if(cloud_cylinder->points.empty()) 
    std::cerr << "Can't find the cylindrical component." << std::endl;
  else
  {
	  // std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder->size() << " data points." << std::endl;

    Eigen::Vector4f centroid;   
    pcl::compute3DCentroid (*cloud_cylinder, centroid);
    std::cout << "centroid of the cylindrical component: " << centroid[0] << " " <<  centroid[1] << " " <<   centroid[2] << " " <<   centroid[3] << std::endl;

	  //Create a point in the "camera_rgb_optical_frame"
    geometry_msgs::PointStamped point_camera;
    geometry_msgs::PointStamped point_map;
	  visualization_msgs::Marker marker;
    geometry_msgs::TransformStamped tss;
          
    point_camera.header.frame_id = "camera_depth_optical_frame";
    point_camera.header.stamp = ros::Time::now();

	  point_map.header.frame_id = "map";
    point_map.header.stamp = ros::Time::now();

    point_camera.point.x = centroid[0];
    point_camera.point.y = centroid[1];
    point_camera.point.z = centroid[2];

	  try {
      tss = tfBuffer.lookupTransform("map", "camera_depth_optical_frame", time_rec);
      //tf2_buffer.transform(point_camera, point_map, "map", ros::Duration(2));
    } catch (tf2::TransformException &ex) {
      ROS_WARN("Transform warning: %s\n", ex.what());
    }

    tf2::doTransform(point_camera, point_map, tss);

    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();

    marker.ns = "cylinder";
    marker.id = 0;

    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = point_map.point.x;
    marker.pose.position.y = point_map.point.y;
    marker.pose.position.z = point_map.point.z;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;

    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;

    marker.lifetime = ros::Duration();
    
    
  }
}

int main(int argc, char** argv)
{
  // Initialize ROS
  ros::init(argc, argv, "find_rings");
  ros::NodeHandle nh;

  tfListener = new tf2_ros::TransformListener(tfBuffer);

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe("/camera/depth/points", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pubPlan = nh.advertise<pcl::PCLPointCloud2>("planes", 1);
  pubCyl = nh.advertise<pcl::PCLPointCloud2>("cylinders", 1);


  // Spin
  ros::spin();
} */