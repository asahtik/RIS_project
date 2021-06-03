#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/rgb_colors.h>
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/Marker.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "include/clustering2d_lib.cpp"
#include "finale/RingCluster.h"
#include "finale/RingClusteringToHub.h"

#include <math.h>

#define RATE 2.0
#define MIN_DETECTIONS 2
#define IMG_H_P 0.5
#define CONTOUR_SIZE 20
#define ELIPSE_MAX_CENTER_DISTANCE 3.0
#define ELIPSE_MIN_SIZE_DIFF 2
#define MIN_Z 0.1
#define MAX_Z 2.0
#define GoF_POINTS 5000
#define GoF_FIT 0.03
#define MIN_SATURATION 0.0
#define MIN_VALUE 0.0
#define NO_COLORS 16 // do not change

// hist, dist, approach
typedef std::tuple<std::vector<double>, double, geometry_msgs::Pose> ringdata;
typedef cv::Point3_<uint8_t> Pixel;

tf2_ros::Buffer tfBuffer;
tf2_ros::TransformListener* tfListener;

ros::Publisher ring_msg_pub;
ros::Publisher debug_point_pub;
// ros::ServiceClient colour_client;

ringdata joinf(const clustering2d::cluster_t<ringdata> &a, const clustering2d::cluster_t<ringdata> &b) {
  std::vector<double> ta = std::get<0>(a.data);
  std::vector<double> tb = std::get<0>(b.data);
  for(int i = 0; i < ta.size(); i++) ta[i] = (ta[i] * (double)a.detections + tb[i] * (double)b.detections) / ((double)a.detections + (double)b.detections);
  return ringdata(ta, (std::get<1>(a.data) > std::get<1>(b.data) ? std::get<1>(a.data) : std::get<1>(b.data)), (std::get<1>(a.data) > std::get<1>(b.data) ? std::get<2>(a.data) : std::get<2>(b.data)));
}

int toHubMsg(ros::Time stamp, std::list<clustering2d::cluster_t<ringdata>>& fs, std::vector<std::tuple<int, geometry_msgs::Pose>> &cs, finale::RingClusteringToHub& out, int mind) {
  out.stamp = stamp;
  for(std::tuple<int, geometry_msgs::Pose> c : cs) {
    finale::ClusterInCamera t;
    t.id = std::get<0>(c);
    t.pose = std::get<1>(c);
    out.inCamera.push_back(t);
  }
  int no = 0;
  for(clustering2d::cluster_t<ringdata> f : fs) {
    finale::RingCluster t;
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
      out.rings.push_back(t);
      no++;
    }
  }
  return no;
}

std::string vectorToString(std::vector<double> &hs) {
  std::stringstream ss;
  for(std::vector<double>::iterator it = hs.begin(); it != hs.end(); ++it) {
    ss << *it;
    if(it + 1 != hs.end()) ss << ",";
  }
  return ss.str();
}

void sendDebug(geometry_msgs::Point &p, const char *frame, const char *ns) {
  visualization_msgs::Marker m;
  m.ns = ns;
  m.header.frame_id = frame;
  m.header.stamp = ros::Time::now();
  m.id = 0;
  m.pose.position = p;
  m.pose.orientation.w = 1.0;
  m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.1;
  m.type = m.SPHERE;
  m.color.a = 1.0; m.color.r = 1.0;
  m.action = m.ADD;
  debug_point_pub.publish(m);
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

float max(float a, float b) {
  return (a > b) ? a : b;
}
float min(float a, float b) {
  return (a < b) ? a : b;
}
// Adapted from https://gist.github.com/fairlight1337/4935ae72bcbcc1ba5c72
void RGBtoHSV(const float& fR, const float& fG, const float fB, float& fH, float& fS, float& fV) {
  float fCMax = max(max(fR, fG), fB);
  float fCMin = min(min(fR, fG), fB);
  float fDelta = fCMax - fCMin;
  if(fDelta > 0) {
    if(fCMax == fR) {
      fH = 60 * (fmod(((fG - fB) / fDelta), 6));
    } else if(fCMax == fG) {
      fH = 60 * (((fB - fR) / fDelta) + 2);
    } else if(fCMax == fB) {
      fH = 60 * (((fR - fG) / fDelta) + 4);
    }
    if(fCMax > 0) {
      fS = fDelta / fCMax;
    } else {
      fS = 0;
    }
    fV = fCMax;
  } else {
    fH = 0;
    fS = 0;
    fV = fCMax;
  }
  if(fH < 0) {
    fH = 360 + fH;
  }
}

std::list<clustering2d::cluster_t<ringdata>> ring_c;

void find_rings(const sensor_msgs::Image::ConstPtr &rgb_msg, const sensor_msgs::Image::ConstPtr &depth_msg) {
  cv_bridge::CvImageConstPtr cv_depth = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
  cv_bridge::CvImageConstPtr cv_rgb = cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGR8);
  int width = depth_msg->width, height = depth_msg->height;

  int n_h = (int)round((float)height * IMG_H_P);
  cv::Rect ROI(0, 0, width, n_h);
  cv::Mat cv_depth2 = cv_depth->image(ROI);
  cv::Mat cv_rgb2 = cv_rgb->image(ROI);

  double min_depth_v, max_depth_v; 
  cv::minMaxLoc(cv_depth2, &min_depth_v, &max_depth_v);
  // ROS_INFO("Min, max in depth image: %f, %f", min_depth_v, max_depth_v);
  cv::Mat cv_bw;
  cv_depth2.convertTo(cv_bw, CV_8UC1, 255.0 / (max_depth_v - min_depth_v), -min_depth_v);
  cv::threshold(cv_bw, cv_bw, 0, 255, cv::ThresholdTypes::THRESH_BINARY | cv::ThresholdTypes::THRESH_OTSU);
  // cv::imshow("depth2", cv_depth2);
  // int agsa = cv::waitKey();
  // cv::imshow("bw", cv_bw);
  // agsa = cv::waitKey();

  // Find contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(cv_bw, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
  // Fit ellipses to contours
  std::list<cv::RotatedRect> elps;
  for(std::vector<cv::Point> c : contours) {
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

  std::list<std::tuple<int, geometry_msgs::Pose>> inCamera_temp;
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

    if(abs(e1_size - e2_size) < ELIPSE_MAX_CENTER_DISTANCE) continue;

    // Find out which ellipse is on the inside and which on the outside of the ring
    cv::RotatedRect *outer = (e1_size > e2_size) ? e1 : e2;
    cv::RotatedRect *inner = (e1_size > e2_size) ? e2 : e1;
    cv::Point2f *box = (e1_size > e2_size) ? e1_corners : e2_corners;
    float a = (e1_a > e2_a) ? e1_a : e2_a;
    float b = (e1_b > e2_b) ? e1_b : e2_b;

    cv::Point2f center(inner->center.x, inner->center.y);
    // Get ring bounding box
    cv::Rect box2 = outer->boundingRect();

    float x1 = box2.x, x2 = box2.x + box2.width, y1 = box2.y, y2 = box2.y + box2.height;
    // float x1 = box[0].x, x2 = box[2].x, y1 = box[0].y, y2 = box[2].y;
    if(x1 < 0) x1 = 0;
    if(x2 > width) x2 = width;
    if(y1 < 0) y1 = 0;
    if(y2 > n_h) y2 = n_h;
    
    // Get angles from sensor to bounding box corners
    int k_f = 554;
    float x1_d = width / 2.0 - x1, y1_d = height / 2.0 - y1, x2_d = width / 2.0 - x2, y2_d = height / 2.0 - y2;
    float h_angle_x1 = atan2(x1_d, k_f), h_angle_x2 = atan2(x2_d, k_f);
    float v_angle_y1 = atan2(y1_d, k_f), v_angle_y2 = atan2(y2_d, k_f);

    float angle_to_target = atan2(width/2.0 - (x1+x2)/2.0, k_f), h_angle_to_target = atan2(height/2.0 - (y1+y2)/2.0, k_f);
    // ROS_INFO("Angles %f %f", angle_to_target, h_angle_to_target);
    
    cv::Rect ROI2(x1, y1, x2 - x1, y2 - y1);
    cv::Mat depth_bw = cv_bw(ROI2), depth_bw2;
    cv::Mat bw_temp[3] = {depth_bw, depth_bw, depth_bw}; cv::merge(bw_temp, 3, depth_bw2);
    cv::Mat rgb_img = cv_rgb2(ROI2); cv::bitwise_and(rgb_img, depth_bw2, rgb_img);
    depth_bw.convertTo(depth_bw, CV_32FC1, 1.0 / 255.0);
    cv::Mat depth_img = cv_depth2(ROI2).mul(depth_bw), depth_temp; // cv::Mat depth_img = cv_depth2(ROI2);

    // depth_img.convertTo(depth_temp, CV_8UC1, 255.0 / (max_depth_v - min_depth_v), -min_depth_v);
    // cv::imshow("Depth", depth_temp);
    // int asgs = cv::waitKey();
    cv::imshow("RGB", rgb_img);
    int asg = cv::waitKey(1);

    // Build a color histogram
    std::vector<double> clr_hist(NO_COLORS + 2);
    int no_clrs = 0;
    float avg_hue = 0.0; float avg_value = 0.0; float avg_sat = 0.0;
    for(int i = 0; i < NO_COLORS; i++) clr_hist[i] = 0;
    rgb_img.forEach<Pixel>([&](Pixel &p, const int *position) {
      unsigned int b = p.x, g = p.y, r = p.z;
      float h, s, v;
      RGBtoHSV((float)r, (float)g, (float)b, h, s, v);
      if(!(b < 1 && g < 1 && r < 1)) {
        if(no_clrs == 0) {avg_hue = h; avg_value = v; avg_sat = s;}
        else {avg_hue += h; avg_value += v; avg_sat += s;}
        no_clrs++;
        clr_hist[(int)floor((h / 360.0) * (float)(NO_COLORS))]++;
      }
    });
    double min_depth_v2, max_depth_v2; 
    cv::minMaxLoc(depth_img, &min_depth_v2, &max_depth_v2);
    // ROS_INFO("Min, max in depth image: %f, %f", min_depth_v2, max_depth_v2);
    // Get distance to ring
    float sum = 0.0; int no = 0;
    depth_img.forEach<float>([&](float &point, const int *position) {
      if(point > 0.1) {
        sum += point;
        no++;
      }
    });
    float distance_to_target = sum / (float)no;
    // ROS_INFO("Ring distance %f", distance_to_target);

    // Normalise colour histogram
    if(no_clrs == 0) continue;
    for(int i = 0; i < NO_COLORS; i++) clr_hist[i] /= (double)no_clrs;
    avg_hue /= (float)no_clrs; avg_value /= (float)no_clrs; avg_sat /= (float)no_clrs;
    clr_hist[NO_COLORS] = avg_value; clr_hist[NO_COLORS + 1] = avg_sat;

    // std::cout << toString(clr_hist, NO_COLORS + 2) << std::endl;
    geometry_msgs::PointStamped optical;
    optical.header.frame_id = "camera_depth_optical_frame";
    optical.header.stamp = depth_msg->header.stamp;
    optical.point.x = -distance_to_target * sin(angle_to_target);
    optical.point.y = -distance_to_target * sin(h_angle_to_target);
    optical.point.z = distance_to_target * cos(angle_to_target);
    // sendDebug(optical.point, "camera_depth_optical_frame", "camera");
    geometry_msgs::TransformStamped transform, tss2;
    geometry_msgs::PoseStamped curr_pose_null, pose; curr_pose_null.header.stamp = optical.header.stamp; curr_pose_null.header.frame_id = "base_link"; curr_pose_null.pose.orientation.w = 1.0;
    geometry_msgs::PointStamped map;
    map.header.frame_id = "map";
    map.header.stamp = depth_msg->header.stamp;
    pose.header = map.header;
    // ROS_WARN("Point in camera %f %f %f", optical.point.x, optical.point.y, optical.point.z);
    try {
      transform = tfBuffer.lookupTransform("map", "camera_depth_optical_frame", depth_msg->header.stamp);
      tss2 = tfBuffer.lookupTransform("map", "base_link", depth_msg->header.stamp);
      tf2::doTransform(optical, map, transform);
      tf2::doTransform(curr_pose_null, pose, tss2);
      pose.pose.position.z = 0.0;
      // ROS_WARN("Point in map %f %f %f", map.point.x, map.point.y, map.point.z);
      geometry_msgs::Pose ring_pose; ring_pose.position = map.point; ring_pose.orientation.w = 1.0;
      geometry_msgs::Pose inCamera_pose; inCamera_pose.position = optical.point; inCamera_pose.orientation.w = cos(angle_to_target / 2.0); inCamera_pose.orientation.z = sin(angle_to_target / 2.0); // Should it be /2 ?
      // sendDebug(map.point, "map", "map");
      clustering2d::cluster_t<ringdata> *cluster = clustering2d::cluster_t<ringdata>::getCluster(ring_pose, 0, ringdata(clr_hist, distance_to_target, pose.pose), joinf);
      if(cluster != NULL) {
        ring_c.push_front(*cluster);
        inCamera_temp.push_back(std::tuple<int, geometry_msgs::Pose>(cluster->id, inCamera_pose));
        delete(cluster);
      }
      // marker_pub.publish(m);
    } catch(tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
    }
  }
  std::vector<int> joins;
  int no_markers = clustering2d::cluster(ring_c, &joins);
  // ROS_INFO("No markers %d", no_markers);
  if(no_markers > 0) {
    std::vector<std::tuple<int, geometry_msgs::Pose>> inCamera;
    for(std::tuple<int, geometry_msgs::Pose> i : inCamera_temp) {
      int ncl = clustering2d::clustered_id(joins, std::get<0>(i));
      inCamera.push_back(std::tuple<int, geometry_msgs::Pose>(ncl, std::get<1>(i)));
    }
    finale::RingClusteringToHub rcl;
    int no = toHubMsg(depth_msg->header.stamp, ring_c, inCamera, rcl, MIN_DETECTIONS);
    ring_msg_pub.publish(rcl);
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "find_rings2");
  ros::NodeHandle nh;

  tfListener = new tf2_ros::TransformListener(tfBuffer);

  // colour_client = nh.serviceClient<exercise6::RecogniseColour>("exercise6/recognise_colour");

  ring_msg_pub = nh.advertise<finale::RingClusteringToHub>("finale/rings", 100);
  //
  debug_point_pub = nh.advertise<visualization_msgs::Marker>("debug/ring", 100);
  //

  message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "camera/rgb/image_raw", 1);
  message_filters::Subscriber<sensor_msgs::Image> depth_sub (nh, "camera/depth/image_raw", 1);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_policy;

  message_filters::Synchronizer<sync_policy> sync(sync_policy(10), rgb_sub, depth_sub);
  sync.registerCallback(boost::bind(&find_rings, _1, _2));
  ros::Rate rate(RATE);
  while(ros::ok()) {
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}