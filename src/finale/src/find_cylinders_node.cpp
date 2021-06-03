#include <iostream>
#include <sstream>
#include <ros/ros.h>

#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>
#include <pcl/filters/crop_box.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>

#include <Eigen/Core>

#include "finale/CylCluster.h"
#include "finale/ClusterInCamera.h"
#include "finale/CylClusteringToHub.h"
#include "include/clustering2d_lib.cpp"

#define MIN_Z 0.1
#define MAX_Z 2.0
#define LEAF_SIZE 0.01f
#define PLANE_ITERATIONS 1000
#define PLANE_THRESHOLD 0.01f
#define REMAINING_POINTS_PERCENTAGE 0.15f
#define REMOVED_POINTS_PERCENTAGE 0.1f
#define NORMAL_K_NEIGHBOURS 50
#define CYLINDER_WEIGHT 0.1f
#define CYLINDER_ITERATIONS 10000
#define CYLINDER_THRESHOLD 0.025f // OG 0.05
#define CYLINDER_MIN_RADIUS 0.06f
#define CYLINDER_MAX_RADIUS 0.15f
#define CYLINDER_MAX_AXIS_ANGLE 0.02
#define CYLINDER_POINTS_RATIO 0.08
#define MIN_Y -0.1
#define RATE 2.0
#define MIN_DETECTIONS 2
#define NO_COLORS 16 // do not change
#define MIN_SATURATION 0.01
#define MIN_VALUE 0.01

// Hist, no cylinder points, approach
typedef std::tuple<std::vector<double>, int, geometry_msgs::Pose> cyldata;

tf2_ros::Buffer tfBuffer;
tf2_ros::TransformListener* tfListener;

ros::Publisher pubPlan;
ros::Publisher pubCyl;
ros::Publisher marker_pub;
ros::Publisher cylinder_msg_pub;
ros::Publisher debug_point_pub;

std::list<clustering2d::cluster_t<cyldata>> cylinder_c;

cyldata joinf(const clustering2d::cluster_t<cyldata> &a, const clustering2d::cluster_t<cyldata> &b) {
  std::vector<double> ta = std::get<0>(a.data);
  std::vector<double> tb = std::get<0>(b.data);
  for(int i = 0; i < ta.size(); i++) ta[i] = (ta[i] * (double)a.detections + tb[i] * (double)b.detections) / ((double)a.detections + (double)b.detections);
  return cyldata(ta, (std::get<1>(a.data) > std::get<1>(b.data) ? std::get<1>(a.data) : std::get<1>(b.data)), (std::get<1>(a.data) > std::get<1>(b.data) ? std::get<2>(a.data) : std::get<2>(b.data)));
}

std::string toString(double* arr, int size) {
  std::stringstream ss;
  for(int i = 0; i < size; i++)
    ss << arr[i] << " ";
  return ss.str();
}
std::string toString(std::list<clustering2d::cluster_t<cyldata>> &v) {
  std::stringstream ret;
  for(clustering2d::cluster_t<cyldata> p : v) {
    ret << p.toString() << ", ";
  }
  return ret.str();
}
void sendDebug(geometry_msgs::Pose &p, const char *frame, const char *ns) {
  visualization_msgs::Marker m;
  m.ns = ns;
  m.header.frame_id = frame;
  m.header.stamp = ros::Time::now();
  m.lifetime = ros::Duration(5);
  m.id = 0;
  m.pose = p;
  m.scale.x = 0.3; m.scale.y = 0.05; m.scale.z = 0.05;
  m.type = m.ARROW;
  m.color.a = 1.0; m.color.g = 1.0;
  m.action = m.ADD;
  debug_point_pub.publish(m);
  ROS_INFO("Debug point data for ns %s: position %f %f %f orientation %f %f %f %f", ns, p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
}

int toHubMsg(ros::Time stamp, std::list<clustering2d::cluster_t<cyldata>>& fs, std::tuple<int, geometry_msgs::Pose> &cs, finale::CylClusteringToHub& out, int mind) {
  out.stamp = stamp;
  finale::ClusterInCamera t;
  t.id = std::get<0>(cs);
  t.pose = std::get<1>(cs);
  out.inCamera.push_back(t);

  int no = 0;
  for(clustering2d::cluster_t<cyldata> f : fs) {
    finale::CylCluster t;
    t.id = f.id;
    t.x = f.x;
    t.y = f.y;
    t.cos = f.cos;
    t.sin = f.sin;
    t.status = f.status;
    t.detections = f.detections;
    t.data = std::get<0>(f.data);
    t.approach = std::get<2>(f.data);
    if(t.detections >= mind) {
      out.cyls.push_back(t);
      no++;
    }
  }
  return no;
}

Eigen::Vector4f get_cylinder_centroid(const sensor_msgs::PointCloud2ConstPtr&);

// Subscriber callback
void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &depth_blob) {

  pcl::PCLPointCloud2 cloud_blob;
  pcl_conversions::toPCL(*depth_blob, cloud_blob);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  // Read in the cloud data
  pcl::fromPCLPointCloud2(cloud_blob, *cloud);
  // Get min and max point
  pcl::PointXYZRGB min_pt, max_pt;
  pcl::getMinMax3D(*cloud, min_pt, max_pt);
  // Build a passthrough filter to remove spurious NaNs and irrelevant y values
  pcl::CropBox<pcl::PointXYZRGB> pass;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pass (new pcl::PointCloud<pcl::PointXYZRGB>);
  pass.setInputCloud(cloud);
  pass.setMin(Eigen::Vector4f(min_pt.x, MIN_Y, MIN_Z, 1.0));
  pass.setMax(Eigen::Vector4f(max_pt.x, max_pt.y, MAX_Z, 1.0));
  pass.filter(*cloud_pass);

  /* Remove floor and walls */
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZRGB> sor;
  sor.setInputCloud(cloud_pass);
  sor.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
  sor.filter(*cloud_filtered);

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
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
    int n_inliers = inliers->indices.size();
    // If removed object would be small don't remove
    if(n_inliers < REMOVED_POINTS_PERCENTAGE * nr_points) break;
    // Extract the inliers
    std::vector<int>::iterator it = remaining->begin();
    for(size_t i = 0; i < inliers->indices.size(); ++i) {
      int curr = inliers->indices[i];
      // Remove it from further consideration.
      while(it != remaining->end() && *it < curr) { ++it; }
      if(it == remaining->end()) break;
      if(*it == curr) it = remaining->erase(it);
    }
    i++;
  }
  // std::cout << "Found " << i << " planes." << std::endl;

  if(remaining->size() == 0) return;

  // Extract indices
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_outliers(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  extract.setInputCloud(cloud_filtered);
  extract.setIndices(remaining);
  extract.setNegative(false);
  extract.filter(*cloud_outliers);

  pcl::PCLPointCloud2 outcloud_plane;
  pcl::toPCLPointCloud2 (*cloud_outliers, outcloud_plane);
  pubPlan.publish (outcloud_plane);


  /* Find cylinders */
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);

  // Estimate point normals
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  ne.setSearchMethod(tree);
  ne.setInputCloud(cloud_outliers);
  ne.setKSearch(NORMAL_K_NEIGHBOURS);
  ne.compute(*cloud_normals);

  pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal> seg2;
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
  seg2.segment(*inliers_cylinder, *coefficients_cylinder);
  // ROS_INFO("No points %d, no cyliner inliers %d, ratio %f", (int)nr_points, (int)inliers_cylinder->indices.size(), (float)inliers_cylinder->indices.size() / (float)nr_points);
  int nci_points = (int)inliers_cylinder->indices.size();
  // Write the cylinder inliers to disk
  extract.setInputCloud(cloud_outliers);
  extract.setIndices(inliers_cylinder);
  extract.setNegative(false);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cylinder(new pcl::PointCloud<pcl::PointXYZRGB>());
  extract.filter(*cloud_cylinder);

  if (cloud_cylinder->points.empty()) {
    // std::cerr << "Can't find the cylindrical component." << std::endl;
  } else {
    // coef center x y z, axis x y z, radius
    float axis_x = coefficients_cylinder->values[3], axis_y = coefficients_cylinder->values[4], axis_z = coefficients_cylinder->values[5];
    float axis_v_size = sqrt(pow(axis_x, 2) + pow(axis_y, 2) + pow(axis_z, 2));
    axis_x /= axis_v_size; axis_y /= axis_v_size; axis_z /= axis_v_size;
    // ROS_INFO("Axis x: %f, y: %f, z: %f", axis_x, axis_y, axis_z);
    // Check rotation and size of pointcloud
    if(abs(axis_x) > CYLINDER_MAX_AXIS_ANGLE || (float)nci_points / (float)nr_points < CYLINDER_POINTS_RATIO) {
      // std::cout << "Cyl coef" << *coefficients_cylinder << std::endl;
      // ROS_INFO("Detected cylinder is not valid");
      return;
    }

    // Calculate color histogram
    std::vector<double> clr_hist(NO_COLORS);
    int no_clrs = 0;
    float avg_hue = 0.0;
    for(int i = 0; i < NO_COLORS; i++) clr_hist[i] = 0.0;
    for(pcl::PointCloud<pcl::PointXYZRGB>::iterator it = cloud_cylinder->begin(); it != cloud_cylinder->end(); ++it) {
      pcl::PointXYZHSV pc;
      pcl::PointXYZRGBtoXYZHSV(*it, pc);
      if(pc.s >= MIN_SATURATION && pc.v >= MIN_VALUE) {
        // std::cerr << (int)floor((pc.h / 360.0) * (float)(NO_COLORS)) << std::endl;
        if(no_clrs == 0) avg_hue = pc.h;
        else avg_hue += pc.h;
        no_clrs++;
        clr_hist[(int)floor((pc.h / 360.0) * (float)(NO_COLORS))]++;
      } //else std::cerr << "Sat, Val: " << pc.s << ", " << pc.v << std::endl;
    }
    if(no_clrs == 0) return;
    for(int i = 0; i < NO_COLORS; i++) clr_hist[i] /= (double)no_clrs;
    avg_hue /= no_clrs;
    // std::cout << toString(clr_hist, NO_COLORS) << std::endl;

    Eigen::Vector4f centroid;   
    pcl::compute3DCentroid (*cloud_cylinder, centroid);
    // std::cout << "centroid of the cylindrical component: " << centroid[0] << " " <<  centroid[1] << " " <<   centroid[2] << " " << std::endl;

	  //Create a point in the "camera_rgb_optical_frame"
    geometry_msgs::PointStamped point_camera;
    geometry_msgs::PoseStamped pose_null;
    geometry_msgs::PointStamped point_map;
    geometry_msgs::PoseStamped pose_map;
    geometry_msgs::TransformStamped tss, tss2;
          
    point_camera.header.frame_id = "camera_depth_optical_frame";
    point_camera.header.stamp = depth_blob->header.stamp;
    pose_null.header.stamp = point_camera.header.stamp; pose_null.header.frame_id = "base_link"; pose_null.pose.orientation.w = 1.0;
    pose_map.header.stamp = depth_blob->header.stamp; pose_map.header.frame_id = "map";

	  point_map.header.frame_id = "map";
    point_map.header.stamp = depth_blob->header.stamp;

    point_camera.point.x = centroid[0];
    point_camera.point.y = centroid[1];
    point_camera.point.z = centroid[2];

	  try {
      tss = tfBuffer.lookupTransform("map", "camera_depth_optical_frame", depth_blob->header.stamp);
      tss2 = tfBuffer.lookupTransform("map", "base_link", depth_blob->header.stamp);
      tf2::doTransform(point_camera, point_map, tss);
      tf2::doTransform(pose_null, pose_map, tss2);
      pose_map.pose.position.z = 0.0;
      // sendDebug(pose_map.pose, "map", "cylinder");
      // std::cout << pose_null.pose << std::endl;
      // std::cout << pose_map.pose << std::endl;
      // tfBuffer.transform<geometry_msgs::PointStamped>(point_camera, point_map, "map", ros::Duration(0.1));
      // std::cerr << "point_camera: " << point_camera.point.x << " " <<  point_camera.point.y << " " <<  point_camera.point.z << std::endl;
      // std::cerr << "point_map: " << point_map.point.x << " " <<  point_map.point.y << " " <<  point_map.point.z << std::endl;
      geometry_msgs::Pose p;
      p.position.x = point_map.point.x;
      p.position.y = point_map.point.y;
      p.position.z = point_map.point.z;
      p.orientation.w = 1.0;

      clustering2d::cluster_t<cyldata> *cluster = clustering2d::cluster_t<cyldata>::getCluster(p, 0, cyldata(clr_hist, no_clrs, pose_map.pose), joinf);
      if(cluster != NULL) {
        int cl = cluster->id;
        cylinder_c.push_front(*cluster);
        std::vector<int> joins;
        int no_markers = clustering2d::cluster(cylinder_c, &joins);
        int ncl = clustering2d::clustered_id(joins, cl);
        // ROS_INFO("No markers %d", no_markers);
        finale::CylClusteringToHub carr;
        geometry_msgs::Pose cp; cp.position = point_camera.point; cp.orientation.w = 1.0; std::tuple<int, geometry_msgs::Pose> tdata(ncl, cp);
        int no = toHubMsg(depth_blob->header.stamp, cylinder_c, tdata, carr, MIN_DETECTIONS);
        cylinder_msg_pub.publish(carr);

        pcl::PCLPointCloud2 outcloud_cylinder;
        pcl::toPCLPointCloud2 (*cloud_cylinder, outcloud_cylinder);
        sensor_msgs::PointCloud2 outcloud_cylinder_msg;
        pcl_conversions::fromPCL(outcloud_cylinder, outcloud_cylinder_msg);
        outcloud_cylinder_msg.header.stamp = ros::Time::now();
        pubCyl.publish(outcloud_cylinder);
      }
      // pubm.publish (marker);
    } catch (tf2::TransformException &ex) {
      // ROS_WARN("Transform warning: %s\n", ex.what());
    }
  }
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "find_cylinders2");
  ros::NodeHandle nh;

  tfListener = new tf2_ros::TransformListener(tfBuffer);

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("/camera/depth/points", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pubPlan = nh.advertise<pcl::PCLPointCloud2>("planes", 1);
  pubCyl = nh.advertise<sensor_msgs::PointCloud2>("finale/cylinder_cloud", 1);
  debug_point_pub = nh.advertise<visualization_msgs::Marker>("debug/cylinders", 10);
  // marker_pub = nh.advertise<visualization_msgs::MarkerArray>("finale/markers", 1000);
  cylinder_msg_pub = nh.advertise<finale::CylClusteringToHub>("finale/cylinders", 1000);

  // Spin
  ros::Rate rate(1);
  while(ros::ok()) {
    ros::spinOnce();
    rate.sleep();
  }
  std::cout << "----------------------------------------------" << std::endl;
}