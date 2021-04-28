#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/rgb_colors.h>
#include <geometry_msgs/Pose.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <math.h>
#include <sstream>
#include <iterator>

#define CONTOUR_SIZE 20
#define ELIPSE_MAX_CENTER_DISTANCE 5.0

ros::Publisher marker_pub;

std::string toString(std::vector<cv::Point> &v) {
  std::stringstream ret;
  for(cv::Point p : v) {
    ret << p.x << " " << p.y << ", ";
  }
  return ret.str();
}

void find_rings(const sensor_msgs::ImageConstPtr &rgb_img, const sensor_msgs::ImageConstPtr &depth_img) {
  cv_bridge::CvImageConstPtr cv_rgb = cv_bridge::toCvShare(rgb_img, sensor_msgs::image_encodings::BGR8);
  cv::Mat cv_gray;
  cv::cvtColor(cv_rgb->image, cv_gray, cv::COLOR_BGR2GRAY);
  ROS_INFO("rgb pic dims %d", cv_rgb->image.dims);

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

    float e1_size = (sqrt(pow(e1_corners[1].x - e1_corners[2].x, 2) + pow(e1_corners[1].y - e1_corners[2].y, 2)) +  sqrt(pow(e1_corners[0].x - e1_corners[1].x, 2) + pow(e1_corners[0].y - e1_corners[1].y, 2))) / 2.0;
    float e2_size = (sqrt(pow(e2_corners[1].x - e2_corners[2].x, 2) + pow(e2_corners[1].y - e2_corners[2].y, 2)) +  sqrt(pow(e2_corners[0].x - e2_corners[1].x, 2) + pow(e2_corners[0].y - e2_corners[1].y, 2))) / 2.0;

    cv::RotatedRect *outer = (e1_size > e2_size) ? e1 : e2;
    cv::RotatedRect *inner = (e1_size > e2_size) ? e2 : e1;

    cv::Point center;
    center.x = inner->center.x;
    center.y = inner->center.y;
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "find_rings");
  ros::NodeHandle nh;

  message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/rgb/image_raw", 1);
  message_filters::Subscriber<sensor_msgs::Image> depth_sub (nh, "/camera/depth/image_raw", 1);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_policy;

  message_filters::Synchronizer<sync_policy> sync(sync_policy(10), rgb_sub, depth_sub);
  sync.registerCallback(boost::bind(&find_rings, _1, _2));
  
  ros::spin();
  return 0;
}