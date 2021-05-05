#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/rgb_colors.h>
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/MarkerArray.h>
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
#include <pcl/point_types_conversion.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/point_cloud.h>

#include <math.h>
#include <sstream>
#include <iterator>
#include <algorithm>

#include "exercise6/RecogniseColour.h"
#include "include/clustering_2d_lib.hpp"
#include "include/get_color.h"

#define IMG_H_P 0.5
#define KERNEL_SIZE 3
#define BLUR_SIZE 3
#define LOW_THRESHOLD 10
#define HIGH_THRESHOLD 100
#define CONTOUR_SIZE 20
#define ELIPSE_MAX_CENTER_DISTANCE 3.0
#define ELIPSE_MIN_SIZE_DIFF 2
#define MIN_Z 0.1
#define MAX_Z 2.0
#define DETECTION_BORDER 2
#define LEAF_SIZE 0.01
// #define ANGLE_Y_CORRECTION 0.0
// #define ANGLE_X_CORRECTION 0.0
#define MAX_RATIO 0.5
#define MIN_DETECTIONS 5
#define RATE 2.0
#define NO_COLORS 16 // do not change
#define MIN_SATURATION 0.0
#define MIN_VALUE 0.0
#define BINARY_THRESHOLD 50
#define GoF_POINTS 5000
#define GoF_FIT 0.02

ros::Publisher marker_pub;
ros::Publisher ring_cloud_pub;
ros::Publisher ring_msg_pub;
ros::ServiceClient colour_client;

class tf2_buf {
  public:
  tf2_ros::Buffer buffer;
  tf2_ros::TransformListener *tfListener;
  void initialise() {
    tfListener = new tf2_ros::TransformListener(buffer);
  }
} tfBuf;

std::list<clustering2d::cluster_t> ring_c;

std::string toString(std::vector<cv::Point> &v) {
  std::stringstream ret;
  for(cv::Point p : v) {
    ret << p.x << " " << p.y << ", ";
  }
  return ret.str();
}
std::string toString(std::list<clustering2d::cluster_t> &v) {
  std::stringstream ret;
  for(clustering2d::cluster_t p : v) {
    ret << p.toString() << ", ";
  }
  return ret.str();
}

std::string toString(double* arr, int size) {
  std::stringstream ss;
  for(int i = 0; i < size; i++)
    ss << arr[i] << " ";
  return ss.str();
}

void rectify_rectangle(cv::Point2f *b) {
  cv::Point2f temp[4];
  float x[4] = {b[0].x, b[1].x, b[2].x, b[3].x};
  float y[4] = {b[0].y, b[1].y, b[2].y, b[3].y};
  std::sort(x, x + 4);
  std::sort(y, y + 4);
  temp[0].x = x[0]; temp[0].y = y[3];
  temp[1].x = x[0]; temp[1].y = y[0];
  temp[2].x = x[3]; temp[2].y = y[0];
  temp[3].x = x[3]; temp[3].y = y[3];
  for(int i = 0; i < 4; i++) {
    b[i].x = temp[i].x; b[i].y = temp[i].y;
  }
}

// Adapted from https://answers.opencv.org/question/20521/how-do-i-get-the-goodness-of-fit-for-the-result-of-fitellipse/
double goodness_of_fit(cv::RotatedRect &rRect, std::vector<cv::Point> &Coords) {
  double angle = rRect.angle / 180 * M_PI;
  cv::Point2f Center = rRect.center;
  cv::Size2f Sz = rRect.size;
  int csz = Coords.size();

  double g_GOF = 0; //Goodness Of Fit, the smaller the better
  double posx, posy;
  for(int i = 0; i < GoF_POINTS && i < csz; i++) {
      posx = (Coords[i].x - Center.x) * cos(-angle) - (Coords[i].y- Center.y) * sin(-angle);
      posy = (Coords[i].x - Center.x) * sin(-angle) + (Coords[i].y- Center.y) * cos(-angle);
      g_GOF += abs(posx/Sz.width*posx/Sz.width + posy/Sz.height*posy/Sz.height - 0.25);
  }
  return (GoF_POINTS < csz) ? g_GOF / GoF_POINTS : g_GOF / csz;
}

/**
 * @brief Publishes MarkerArray via marker_pub composed of cluster poses
 * @param cs std::list of clusters
*/
void send_marr(std::list<clustering2d::cluster_t> &cs) {
  visualization_msgs::MarkerArray marr;
  // ROS_INFO("Cluster list %s", toString(cs).c_str());
  for(clustering2d::cluster_t c : cs) {
    if(c.detections < MIN_DETECTIONS) continue;
    geometry_msgs::Pose p;
    c.toPose(p);
    p.position.z = 1.0;
    visualization_msgs::Marker m, m_text;
    m.ns = "sphere"; m_text.ns = "text";
    m.id = c.id; m_text.id = c.id;
    m.header.frame_id = "map"; m_text.header.frame_id = "map";
    m.header.stamp = ros::Time::now(); m_text.header.stamp = ros::Time::now();
    m.pose = p; m_text.pose = p; m_text.pose.position.z = 1.5;
    m.type = visualization_msgs::Marker::SPHERE; m_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    m.action = visualization_msgs::Marker::ADD; m_text.action = visualization_msgs::Marker::ADD;
    m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.1; m_text.scale.x = 0.3; m_text.scale.y = 0.1; m_text.scale.z = 0.1;
    m.color.a = 1.0; m.color.r = 1.0; m_text.color.a = 1.0;
    m_text.text = colorFromHSV::enumToString(c.status);
    marr.markers.push_back(m);
  }
  marker_pub.publish(marr);
}

// Subscriber callback
void find_rings(const sensor_msgs::ImageConstPtr &rgb_img, const sensor_msgs::PointCloud2ConstPtr &depth_blob) {
  // Transform image to CV format and get dimensions
  cv_bridge::CvImageConstPtr cv_rgb = cv_bridge::toCvShare(rgb_img, sensor_msgs::image_encodings::BGR8);
  int width = cv_rgb->image.cols, height = cv_rgb->image.rows;
  
  // Convert color to grayscale
  cv::Mat cv_gray;
  // cv::cvtColor(cv_rgb->image, cv_gray, cv::COLOR_BGR2GRAY);

  // cv::equalizeHist(cv_gray, cv_gray);
  int n_h = (int)round((float)height * IMG_H_P);
  cv::Rect ROI(0, 0, width, n_h);
  // cv_gray = cv_gray(ROI);
  cv_gray = cv_rgb->image(ROI);
  // Use Otsu's thresholding to get foreground objects
  cv::Mat cv_bw;
  // cv::threshold(cv_gray, cv_bw, 0, 255, cv::ThresholdTypes::THRESH_BINARY | cv::ThresholdTypes::THRESH_OTSU);
  // cv::threshold(cv_gray, cv_bw, BINARY_THRESHOLD, 255, cv::ThresholdTypes::THRESH_BINARY);
  cv::blur(cv_gray, cv_gray, cv::Size(BLUR_SIZE, BLUR_SIZE));
  cv::Canny(cv_gray, cv_bw, LOW_THRESHOLD, HIGH_THRESHOLD, KERNEL_SIZE);
  // cv::imshow("Edges", cv_bw);
  // int agsa = cv::waitKey();

  // Find contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(cv_bw, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  // Fit ellipses to contours
  std::list<cv::RotatedRect> elps;
  for(std::vector<cv::Point> c : contours) {
    toString(c);
    if(c.size() >= CONTOUR_SIZE) {
      cv::RotatedRect e = cv::fitEllipse(c);
      double GoF = goodness_of_fit(e, c);
      if(GoF < GoF_FIT) elps.push_back(e);
    }
  }

  // Find concentric ellipses - rings
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

    // Calculate ellipse parameters
    float e1_a = sqrt(pow(e1_corners[1].x - e1_corners[2].x, 2) + pow(e1_corners[1].y - e1_corners[2].y, 2)) / 2.0;
    float e1_b = sqrt(pow(e1_corners[0].x - e1_corners[1].x, 2) + pow(e1_corners[0].y - e1_corners[1].y, 2)) / 2.0;

    float e2_a = sqrt(pow(e2_corners[1].x - e2_corners[2].x, 2) + pow(e2_corners[1].y - e2_corners[2].y, 2)) / 2.0;
    float e2_b = sqrt(pow(e2_corners[0].x - e2_corners[1].x, 2) + pow(e2_corners[0].y - e2_corners[1].y, 2)) / 2.0;

    float e1_size = (e1_a + e1_b) / 2.0;
    float e2_size = (e2_a + e2_b) / 2.0;

    // ROS_WARN("Size difference %f", abs(e1_size - e2_size));
    // ROS_INFO("Center %f, %f", e1->center.x, e1->center.y);
    if(abs(e1_size - e2_size) < ELIPSE_MAX_CENTER_DISTANCE) continue;

    // Find out which ellipse is on the inside and which on the outside of the ring
    cv::RotatedRect *outer = (e1_size > e2_size) ? e1 : e2;
    cv::RotatedRect *inner = (e1_size > e2_size) ? e2 : e1;
    cv::Point2f *box = (e1_size > e2_size) ? e1_corners : e2_corners;
    float a = (e1_a > e2_a) ? e1_a : e2_a;
    float b = (e1_b > e2_b) ? e1_b : e2_b;

    cv::Point2f center(inner->center.x, inner->center.y);
    // Get ring bounding box
    // float x1 = center.x - a, x2 = center.x + a + 1;
    // ROS_WARN("0 %f %f, 1 %f %f, 2 %f %f, 3 %f %f", box[0].x, box[0].y, box[1].x, box[1].y, box[2].x, box[2].y, box[3].x, box[3].y);
    rectify_rectangle(box);

    float x1 = box[0].x, x2 = box[2].x, y1 = box[0].y, y2 = box[2].y;
    if(x1 < 0) x1 = 0;
    if(x2 > width) x2 = width;
    if(y2 < 0) y2 = 0;
    if(y1 > height) y1 = height;

    // ROS_WARN("0 %f %f, 1 %f %f, 2 %f %f, 3 %f %f", e1_corners[0].x, e1_corners[0].y, e1_corners[1].x, e1_corners[1].y, e1_corners[2].x, e1_corners[2].y, e1_corners[3].x, e1_corners[3].y);
    // ROS_WARN("0 %f %f, 1 %f %f, 2 %f %f, 3 %f %f", e2_corners[0].x, e2_corners[0].y, e2_corners[1].x, e2_corners[1].y, e2_corners[2].x, e2_corners[2].y, e2_corners[3].x, e2_corners[3].y);

    // ROS_WARN("0 %f %f, 1 %f %f, 2 %f %f, 3 %f %f", box[0].x, box[0].y, box[1].x, box[1].y, box[2].x, box[2].y, box[3].x, box[3].y);
    // ROS_WARN("x1 %f, y2 %f, x2-x1 %f, y1-y2 %f", x1, y2, x2-x1, y1-y2);

    // Get angles from sensor to bounding box corners
    int k_f = 554;
    // float x1_d = x1 - width / 2.0 - DETECTION_BORDER, y1_d = y1 - height / 2.0 + DETECTION_BORDER, x2_d = x2 - width / 2.0 + DETECTION_BORDER, y2_d = y2 - height / 2.0 - DETECTION_BORDER;
    float x1_d = x1 - width / 2.0, y1_d = y1 - height / 2.0, x2_d = x2 - width / 2.0, y2_d = y2 - height / 2.0;
    float h_angle_x1 = atan2(x1_d, k_f), h_angle_x2 = atan2(x2_d, k_f);
    float v_angle_y1 = atan2(y1_d, k_f), v_angle_y2 = atan2(y2_d, k_f);

    // ROS_INFO("h angle x1 %d, h angle x2 %d, v angle y1 %d, v angle y2 %d", (int)(h_angle_x1 * (180 / M_PI)), (int)(h_angle_x2 * (180 / M_PI)), (int)(v_angle_y1 * (180 / M_PI)), (int)(v_angle_y2 * (180 / M_PI)));

    // Get depth
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*depth_blob, pcl_pc2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // Read in the cloud data
    pcl::fromPCLPointCloud2(pcl_pc2, *cloud);
    // Create passthrough filter
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pass(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(MIN_Z, MAX_Z);
    pass.filter(*cloud_pass);

    // Downscale
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_down(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> scd;
    scd.setInputCloud(cloud_pass);
    scd.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
    scd.filter(*cloud_down);

    // Erase invalid points (points that do not project to bounding box)
    cloud_down->erase(std::remove_if(cloud_down->begin(), cloud_down->end(), [=](const pcl::PointXYZRGB &p) {
      float minx = p.z * sin(h_angle_x1), maxx = p.z * sin(h_angle_x2);
      float miny = p.z * sin(v_angle_y2), maxy = p.z * sin(v_angle_y1);
      return !(p.x >= minx && p.x <= maxx && p.y >= miny && p.y <= maxy);
    }), cloud_down->end());

    // ROS_INFO("Pointcloud size after filter %d", (int)cloud_down->points.size());
    if((int)cloud_down->points.size() == 0) continue;

    // Publish pointcloud
    pcl::PCLPointCloud2 ring_cloud;
    pcl::toPCLPointCloud2(*cloud_down, ring_cloud);
    ring_cloud_pub.publish(ring_cloud);

    // Possible improvement: Remove points behind ring
    // Get centroid
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud_down, centroid);

    // ROS_INFO("Ring centroid %f %f %f", centroid[0], centroid[1], centroid[2]);
    float x = centroid[0], y = centroid[1], z = centroid[2];

    // Get area of object in cm2 (project to same plane and downscale) and build a color histogram
    double clr_hist[NO_COLORS + 2]; // add avg value
    int no_clrs = 0;
    float avg_hue = 0.0; float avg_value = 0.0; float avg_sat = 0.0;
    for(int i = 0; i < NO_COLORS; i++) clr_hist[i] = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ring(new pcl::PointCloud<pcl::PointXYZ>);
    for(pcl::PointCloud<pcl::PointXYZRGB>::iterator it = cloud_down->begin(); it != cloud_down->end(); ++it) {
      // Project
      pcl::PointXYZ p; p.x = it->x; p.y = it->y; p.z = z;
      ring->push_back(p);

      // Color histogram
      pcl::PointXYZHSV pc;
      pcl::PointXYZRGBtoXYZHSV(*it, pc);
      if(pc.s >= MIN_SATURATION && pc.v >= MIN_VALUE) {
        if(no_clrs == 0) {avg_hue = pc.h; avg_value = pc.v; avg_sat = pc.s;}
        else {avg_hue += pc.h; avg_value += pc.v; avg_sat += pc.s;}
        no_clrs++;
        clr_hist[(int)floor((pc.h / 360.0) * (float)(NO_COLORS))]++;
      }
    }
    // ROS_INFO("No clrs %d", no_clrs);
    if(no_clrs == 0) continue;
    for(int i = 0; i < NO_COLORS; i++) clr_hist[i] /= (double)no_clrs;
    avg_hue /= (float)no_clrs; avg_value /= (float)no_clrs; avg_sat /= (float)no_clrs;
    clr_hist[NO_COLORS] = avg_value; clr_hist[NO_COLORS + 1] = avg_sat;
    pcl::VoxelGrid<pcl::PointXYZ> ringscd;
    ringscd.setInputCloud(ring);
    ringscd.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
    ringscd.filter(*ring);
    // ROS_INFO("Pointcloud size after filter 2 %d", (int)ring->points.size());

    // Get area of bounding box at distance of ring
    float minx = z * sin(h_angle_x1), maxx = z * sin(h_angle_x2);
    float miny = z * sin(v_angle_y2), maxy = z * sin(v_angle_y1);
    // ROS_INFO("Ring frame x %f %f, y %f %f", minx, maxx, miny, maxy);
    // float ratio = (float)cloud_down->points.size() / (abs(maxx - minx) * abs(maxy - miny) * 10000);
    float ratio = (float)ring->points.size() / (abs(maxx - minx) * abs(maxy - miny) * (1 / (LEAF_SIZE * LEAF_SIZE)));
    ROS_WARN("Frame size %f cm2, ring_size %d, ratio %f", abs(maxx - minx) * abs(maxy - miny) * (1 / (LEAF_SIZE * LEAF_SIZE)), (int)ring->points.size(), ratio);
    // If true it's not a 3d ring
    if(ratio > MAX_RATIO) {
      continue;
    }

    /* cv::Rect ROI2(x1, y2, x2 - x1, (y1 > n_h ? n_h : y1) - y2);
    cv::Mat temp = cv_bw(ROI2);
    cv::imshow("Edges", temp);
    int asg = cv::waitKey(); */

    // std::cout << toString(clr_hist, NO_COLORS + 2) << std::endl;
    geometry_msgs::PointStamped optical;
    optical.header.frame_id = "camera_depth_optical_frame";
    optical.header.stamp = depth_blob->header.stamp;
    optical.point.x = x;
    optical.point.y = y;
    optical.point.z = z;
    geometry_msgs::PointStamped map;
    map.header.frame_id = "map";
    map.header.stamp = depth_blob->header.stamp;
    try {
      // geometry_msgs::TransformStamped transform = tfBuf.buffer.lookupTransform("camera_depth_optical_frame", "map", depth_blob->header.stamp);
      // tf2::doTransform(optical, map, transform);
      tfBuf.buffer.transform<geometry_msgs::PointStamped>(optical, map, "map", ros::Duration(0.1));
      geometry_msgs::Pose pose;
      pose.position.x = map.point.x;
      pose.position.y = map.point.y;
      pose.position.z = map.point.z;
      pose.orientation.w = 1.0;

      clustering2d::cluster_t *cluster = clustering2d::cluster_t::getCluster(pose, colorFromHSV::get_from_hue(avg_hue));
      if(cluster != NULL) {
        ring_c.push_front(*cluster);
        delete(cluster);
      }

      /* exercise6::RecogniseColour srv;
      srv.request.type = exercise6::RecogniseColourRequest::RING;
      srv.request.hist = std::vector<double>(std::begin(clr_hist), std::end(clr_hist));
      if(colour_client.call(srv)) {
        clustering2d::cluster_t *cluster = clustering2d::cluster_t::getCluster(pose, srv.response.colour);
        if(cluster != NULL) {
          ring_c.push_front(*cluster);
          delete(cluster);
        }
      } else ROS_ERROR("Error in colour service"); */

      // marker_pub.publish(m);
    } catch(tf2::TransformException &ex) {
      // ROS_WARN("%s",ex.what());
    }
  }
  int no_markers = clustering2d::cluster(ring_c);
  // ROS_INFO("No markers %d", no_markers);
  if(no_markers > 0) {
    clustering::Cluster2DArray carr;
    carr.header.stamp = ros::Time::now();
    clustering2d::to_cluster_array_msg(ring_c, carr, MIN_DETECTIONS);
    ring_msg_pub.publish(carr);
    send_marr(ring_c);
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "find_rings");
  ros::NodeHandle nh;

  tfBuf.initialise();

  colour_client = nh.serviceClient<exercise6::RecogniseColour>("exercise6/recognise_colour");

  marker_pub = nh.advertise<visualization_msgs::MarkerArray>("exercise6/ring_markers", 1000);
  ring_cloud_pub = nh.advertise<pcl::PCLPointCloud2>("exercise6/ring_cloud", 1);
  ring_msg_pub = nh.advertise<clustering::Cluster2DArray>("exercise6/ring_clusters", 1000);

  message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/rgb/image_raw", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> depth_sub (nh, "/camera/depth/points", 1);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> sync_policy;

  message_filters::Synchronizer<sync_policy> sync(sync_policy(10), rgb_sub, depth_sub);
  sync.registerCallback(boost::bind(&find_rings, _1, _2));
  ros::Rate rate(RATE);
  while(ros::ok()) {
    ros::spinOnce();
    rate.sleep();
  }
  std::cout << "----------------------------------------------" << std::endl;
  return 0;
}