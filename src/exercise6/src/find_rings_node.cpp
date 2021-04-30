#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/rgb_colors.h>
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/Marker.h>
// #include <visualization_msgs/MarkerArray.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/point_cloud.h>

#include <math.h>
#include <sstream>
#include <iterator>

#define CONTOUR_SIZE 20
#define ELIPSE_MAX_CENTER_DISTANCE 5.0
#define MIN_Z 0.1
#define MAX_Z 2.0
#define DETECTION_BORDER 2

ros::Publisher marker_pub;
ros::Publisher ring_cloud_pub;

typedef pcl::PointXYZ PointT;

class tf2_buf {
  public:
  tf2_ros::Buffer buffer;
  tf2_ros::TransformListener *tfListener;
  void initialise() {
    tfListener = new tf2_ros::TransformListener(buffer);
  }
} tfBuf;


int markers = 0;

std::string toString(std::vector<cv::Point> &v) {
  std::stringstream ret;
  for(cv::Point p : v) {
    ret << p.x << " " << p.y << ", ";
  }
  return ret.str();
}

void find_rings(const sensor_msgs::ImageConstPtr &rgb_img, const sensor_msgs::PointCloud2ConstPtr &depth_blob) {
  ros::Time start(0);
  
  cv_bridge::CvImageConstPtr cv_rgb = cv_bridge::toCvShare(rgb_img, sensor_msgs::image_encodings::BGR8);
  int width = cv_rgb->image.cols, height = cv_rgb->image.rows;
  
  cv::Mat cv_gray;
  cv::cvtColor(cv_rgb->image, cv_gray, cv::COLOR_BGR2GRAY);

  cv::equalizeHist(cv_gray, cv_gray);

  cv::Mat cv_bw;
  cv::threshold(cv_gray, cv_bw, 0, 255, cv::ThresholdTypes::THRESH_BINARY | cv::ThresholdTypes::THRESH_OTSU);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(cv_bw, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  std::list<cv::RotatedRect> elps;
  for(std::vector<cv::Point> c : contours) {
    toString(c);
    if(c.size() >= CONTOUR_SIZE) {
      cv::RotatedRect e = cv::fitEllipse(c);
      elps.push_back(e);
    }
  }

  std::list<std::tuple<cv::RotatedRect*, cv::RotatedRect*>> candidates;
  for(std::list<cv::RotatedRect>::iterator n = elps.begin(); n != elps.end(); ++n) {
    for(std::list<cv::RotatedRect>::iterator m = std::next(n, 1); m != elps.end(); ++m) {
      double dist = sqrt(pow(n->center.x - m->center.x, 2) + pow(n->center.y - m->center.y, 2));
      if(dist < ELIPSE_MAX_CENTER_DISTANCE) 
        candidates.push_back(std::tuple<cv::RotatedRect*, cv::RotatedRect*>(&*n, &*m));
    }
  }

  for(std::list<std::tuple<cv::RotatedRect*, cv::RotatedRect*>>::iterator it = candidates.begin(); it != candidates.end(); ++it) {
    cv::RotatedRect *e1 = std::get<0>(*it);
    cv::RotatedRect *e2 = std::get<1>(*it);
    
    cv::Point2f e1_corners[4];
    cv::Point2f e2_corners[4];
    e1->points(e1_corners);
    e2->points(e2_corners);

    float e1_a = sqrt(pow(e1_corners[1].x - e1_corners[2].x, 2) + pow(e1_corners[1].y - e1_corners[2].y, 2)) / 2.0;
    float e1_b = sqrt(pow(e1_corners[0].x - e1_corners[1].x, 2) + pow(e1_corners[0].y - e1_corners[1].y, 2)) / 2.0;

    float e2_a = sqrt(pow(e2_corners[1].x - e2_corners[2].x, 2) + pow(e2_corners[1].y - e2_corners[2].y, 2)) / 2.0;
    float e2_b = sqrt(pow(e2_corners[0].x - e2_corners[1].x, 2) + pow(e2_corners[0].y - e2_corners[1].y, 2)) / 2.0;

    float e1_size = (e1_a + e1_b) / 2.0;
    float e2_size = (e2_a + e2_b) / 2.0;

    cv::RotatedRect *outer = (e1_size > e2_size) ? e1 : e2;
    cv::RotatedRect *inner = (e1_size > e2_size) ? e2 : e1;
    float a = (e1_a > e2_a) ? e1_a : e2_a;
    float b = (e1_b > e2_b) ? e1_b : e2_b;

    cv::Point center(inner->center.x, inner->center.y);

    float x1 = center.x - a, x2 = center.x + a;
    if(x1 <= 0) x1 = 0;
    if(x2 > width) x2 = width;

    float y1 = center.y + b, y2 = center.y - b;
    if(y2 <= 0) y2 = 0;
    if(y1 > height) y1 = height;

    // ROS_INFO("Center %d %d, a %f, b %f", center.x, center.y, a, b);
    ROS_INFO("height %f y1 %f, y2 %f", height / 2.0, y1, y2);

    int k_f = 554;
    float x1_d = x1 - width / 2.0 - DETECTION_BORDER, y1_d = y1 - height / 2.0 + DETECTION_BORDER, x2_d = x2 - width / 2.0 + DETECTION_BORDER, y2_d = y2 - height / 2.0 - DETECTION_BORDER;
    float h_angle_x1 = atan2(x1_d, k_f), h_angle_x2 = atan2(x2_d, k_f);
    float v_angle_y1 = atan2(y1_d, k_f), v_angle_y2 = atan2(y2_d, k_f);

    ROS_INFO("h angle x1 %d, h angle x2 %d, v angle y1 %d, v angle y2 %d", (int)(h_angle_x1 * (180 / M_PI)), (int)(h_angle_x2 * (180 / M_PI)), (int)(v_angle_y1 * (180 / M_PI)), (int)(v_angle_y2 * (180 / M_PI)));

    // Get depth
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*depth_blob, pcl_pc2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    // Read in the cloud data
    pcl::fromPCLPointCloud2(pcl_pc2, *cloud);
    // Create passthrough filter
    pcl::PointCloud<pcl::PointXYZ> cloud_pass;
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(MIN_Z, MAX_Z);
    pass.filter(cloud_pass);

    // Remove invalid points
    /* for(pcl::PointCloud<pcl::PointXYZ>::iterator it = cloud_pass.begin(); it != cloud_pass.end(); ++it) {
      float minx = MAX_Z * sin(h_angle_x1), maxx = MAX_Z * sin(h_angle_x2);
      float miny = MAX_Z * sin(v_angle_y1), maxy = MAX_Z * sin(v_angle_y2);
      if(it->x >= minx && it->x <= maxx && it->y >= miny && it->y <= maxy) ROS_INFO("Point %f %f %f limit x %f %f limit y %f %f", it->x, it->y, it->z, minx, maxx, miny, maxy);
      else ROS_INFO("-Point %f %f %f limit x %f %f limit y %f %f", it->x, it->y, it->z, minx, maxx, miny, maxy);
    } */
    cloud_pass.erase(std::remove_if(cloud_pass.begin(), cloud_pass.end(), [=](const pcl::PointXYZ &p) {
      float minx = p.z * sin(h_angle_x1), maxx = p.z * sin(h_angle_x2);
      float miny = p.z * sin(v_angle_y2), maxy = p.z * sin(v_angle_y1);
      return !(p.x >= minx && p.x <= maxx && p.y >= miny && p.y <= maxy);
    }), cloud_pass.end());

    ROS_INFO("Pointcloud size after filter %d", (int)cloud_pass.points.size());
    if((int)cloud_pass.points.size() == 0) continue;

    // Publish pointcloud
    pcl::PCLPointCloud2 ring_cloud;
    pcl::toPCLPointCloud2(cloud_pass, ring_cloud);
    ring_cloud_pub.publish(ring_cloud);

    // Get centroid
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(cloud_pass, centroid);

    ROS_INFO("Ring centroid %f %f %f", centroid[0], centroid[1], centroid[2]);
    float x = centroid[0], y = centroid[1], z = centroid[2];
    // TODO check valid centroid ???

    /* // Get depth
    cv_bridge::CvImageConstPtr depth_image_msg = cv_bridge::toCvShare(depth_img, sensor_msgs::image_encodings::TYPE_16UC1);
    cv::Mat depth_image = depth_image_msg->image;

    // TODO fix
    cv::Mat ring_img = depth_image(cv::Range(y1, y2), cv::Range(x1, x2));
    long sum_dist = 0, sum_mean = 0; int count = 0;
    ring_img.forEach<uchar>([&](uchar px, const int *position) {
      sum_mean += px;
      if(px > 0) {  
        sum_dist += px;
        count++;
      }
    });
    // float dist = sum_dist / count;
    float dist = (float)sum_dist / (float)count;

    // Get pose
    float elipse_x = width / 2.0 - center.x;
    float elipse_y = height / 2.0 - center.y;

    float angle_to_target = atan2(elipse_x, k_f);
    float vertical_angle = atan2(elipse_y, k_f);

    float x = dist * cos(angle_to_target);
    float y = dist * sin(angle_to_target);
    float z = dist * sin(vertical_angle);

    ROS_INFO("Point in RGB frame: dist %f, x %f, y %f, z %f", dist, x, y, z); */

    geometry_msgs::PointStamped optical;
    optical.header.frame_id = "camera_depth_optical_frame";
    optical.header.stamp = start;
    optical.point.x = x;
    optical.point.y = y;
    optical.point.z = z;
    geometry_msgs::PointStamped map;
    try {
      tfBuf.buffer.transform<geometry_msgs::PointStamped>(optical, map, "map", ros::Duration(0.1));
      geometry_msgs::Pose pose;
      pose.position.x = map.point.x;
      pose.position.y = map.point.y;
      pose.position.z = map.point.z;

      visualization_msgs::Marker m;
      m.id = ++markers;
      m.header.frame_id = "map";
      m.header.stamp = ros::Time::now();
      m.lifetime = ros::Duration(10);
      m.pose = pose;
      m.type = visualization_msgs::Marker::SPHERE;
      m.action = visualization_msgs::Marker::ADD;
      m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.1;
      m.color.a = 1.0; m.color.r = 1.0;

      marker_pub.publish(m);
    } catch(tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
    }
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "find_rings");
  ros::NodeHandle nh;

  tfBuf.initialise();

  marker_pub = nh.advertise<visualization_msgs::Marker>("ring_markers", 1000);
  ring_cloud_pub = nh.advertise<pcl::PCLPointCloud2>("ring_cloud", 1);

  message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/rgb/image_raw", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> depth_sub (nh, "/camera/depth/points", 1);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> sync_policy;

  message_filters::Synchronizer<sync_policy> sync(sync_policy(10), rgb_sub, depth_sub);
  sync.registerCallback(boost::bind(&find_rings, _1, _2));
  
  ros::Rate rate(2);
  while(ros::ok()) {
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}