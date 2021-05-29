#include "ros/ros.h"

#include <nav_msgs/GetMap.h>
#include <geometry_msgs/Twist.h>
#include <move_base_msgs/MoveBaseActionResult.h>
#include <actionlib_msgs/GoalID.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <actionlib_msgs/GoalStatusArray.h>
#include <actionlib_msgs/GoalStatus.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <Eigen/Core>

#include <random>
#include <time.h>
#include <sstream>
#include <math.h>

#include "include/object_data.cpp"
#include "finale/FaceCluster.h"
#include "finale/ClusterInCamera.h"
#include "finale/FaceClusteringToHub.h"
#include "finale/ToggleQRNumService.h"
#include "finale/QRNumDataToHub.h"

#define ROS_RATE 10
#define LIFO false
#define NO_RADIUS_CHECKS 50
#define SAFETY_RADIUS 0.3
#define MAX_TIME_DIFF 1.0
#define MAX_APPROACH_ANGLE 5.0 * M_PI / 180.0
#define MAX_APPROACH_DISTANCE 0.7
#define MIN_APPROACH_DISTANCE 0.5
#define MAX_SEARCH_ROTATIONS 10

typedef std::tuple<int, int> facedata;
typedef std::vector<int> cyldata;
typedef std::vector<int> ringdata;

enum {STOPPED, WANDERING, GOTO_FACE, APPROACH_FACE, INTERACT_FACE, GOTO_CYLINDER, APPROACH_CYLINDER, INTERACT_CYLINDER, GOTO_RING, APPROACH_RING, INTERACT_RING, GOTO_DELIVERY, APPROACH_DELIVERY, DELIVER} state;

int STATE = STOPPED;
int GOAL_ID = 0;

ros::NodeHandle *n;

float map_resolution = 0;
geometry_msgs::TransformStamped map_transform;

ros::Publisher goal_pub;
ros::Publisher goal_cancel_pub;
ros::Publisher velocity_pub;
ros::Publisher qrnum_pub;
ros::Publisher debug_point_pub;

bool got_map = false, got_pose = false;
int map_width = 0, map_height = 0;
std::vector<int8_t> map;

// Faces
std::list<data_t<facedata>> faces;
std::list<data_t<facedata>*> face_list;
// Cylinders
std::list<data_t<cyldata>> cylinders;
std::list<data_t<cyldata>*> cyl_list;
// Rings
std::list<data_t<ringdata>> rings;
std::list<data_t<ringdata>*> ring_list;
// Deliveries
std::list<data_t<facedata>*> delivery_list;

std::tuple<ros::Time, geometry_msgs::Pose> recent_pose;
std::tuple<ros::Time, int, geometry_msgs::Pose> recent_face;
std::tuple<ros::Time, int, geometry_msgs::Pose> recent_cyl;
std::tuple<ros::Time, int, geometry_msgs::Pose> recent_ring;
std::tuple<ros::Time, actionlib_msgs::GoalStatus> goal_status;
std::tuple<ros::Time, std::string> recent_qr;
std::tuple<ros::Time, int, int> recent_num;

template<typename T>
void add_to_list(T &el, std::list<T*> &list) {
  list.push_back(&el);
}
template<typename T>
void add_to_end(T &el, std::list<T*> &list, bool lifo = false) {
  if(lifo) list.push_front(&el);
  else list.push_back(&el);
}
void sendDebug(geometry_msgs::Pose &p, const char *frame) {
  visualization_msgs::Marker m;
  m.header.frame_id = frame;
  m.header.stamp = ros::Time::now();
  m.id = 0;
  m.pose = p;
  m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.1;
  m.type = m.SPHERE;
  m.color.a = 1.0; m.color.r = 1.0;
  m.action = m.ADD;
  debug_point_pub.publish(m);
}
template<typename T>
T* pop_from_list(std::list<T*> &list, bool lifo = false) {
  if(lifo) {
    T* ret = &*list.back();
    list.pop_back();
    return ret;
  } else {
    T* ret = &*list.front();
    list.pop_front();
    return ret;
  }
}

int getState() {
  int fsize = face_list.size();
  int csize = cyl_list.size();
  int rsize = ring_list.size();
  int dsize = delivery_list.size();
  if(fsize != 0) return GOTO_FACE;
  else if(csize != 0) return GOTO_CYLINDER;
  else if(rsize != 0) return GOTO_RING;
  else if(dsize != 0) return GOTO_DELIVERY;
  else return WANDERING;
}

geometry_msgs::Twist getTwist(float lx, float az) {
  geometry_msgs::Twist ret;
  ret.linear.x = lx;
  ret.angular.z = az;
  return ret;
}

std::string poseToString(const geometry_msgs::Pose &p, bool angle = true) {
  std::stringstream ss;
  ss << "x: " << p.position.x << ", y: " << p.position.y;
  if(angle) ss << ", angle: " << atan2(p.orientation.z, p.orientation.w) * M_PI / 180;
  else ss << ", z: " << p.orientation.z << ", w: " << p.orientation.w;
  return ss.str();
}

double get_ang_vel(double angle, double max_speed, int rate) {
  double n_rot = ceil((abs(angle) * (double)rate) / max_speed);
  return (angle * (double)rate) / n_rot;
}

double get_lin_vel(double dist, double max_speed, int rate) {
  double n_mv = ceil((abs(dist) * (double)rate) / max_speed);
  return (dist * (double)rate) / n_mv;
}

float random_float(float from, float to, int resolution = 0) {
    int mult = 1;
    if(resolution > 0) mult = 10 * resolution;
    float range = (to - from) * mult;
    int r = rand() % (int)range;
    return ((float)r / (float)mult) + from;
}

float get_angle(double a_cos, double a_sin) {
  float arccos = (acos(a_cos) * 2.0);
  if(a_sin >= 0) return arccos;
  else return -arccos;
}

bool check_valid_position(int x, int y) {
    // ROS_INFO("map at x:%d,y:%d %d", x, y, map[y * map_width + x]);
    if(map[y * map_width + x] != 0) return false;

    float angle_inc = (2 * M_PI / NO_RADIUS_CHECKS);
    for(int i = 0; i < NO_RADIUS_CHECKS; i++) {
        float angle = i * angle_inc;
        int map_r_x = x + ceil((cos(angle) * SAFETY_RADIUS) / map_resolution);
        int map_r_y = y + ceil((sin(angle) * SAFETY_RADIUS) / map_resolution);
        if(map[map_r_y * map_width + map_r_x] != 0) return false;
    }
    return true;
}

data_t<facedata> *curr_face;
data_t<cyldata> *curr_cyl;
data_t<ringdata> *curr_ring;

void handleStopped() {
  STATE = getState();
}
void handleWandering() {
  // Goto next goal TODO
  STATE = getState();
}
void handleGotoFace() {
  curr_face = pop_from_list(face_list, LIFO);
  goal_cancel_pub.publish(actionlib_msgs::GoalID());
  geometry_msgs::PoseStamped goal;
  goal.header.frame_id = "map"; goal.header.stamp = ros::Time().now(); goal.header.seq = GOAL_ID++;
  goal.pose = curr_face->approach;
  ROS_INFO("GOTO FACE POSE %s", poseToString(goal.pose).c_str());
  goal_pub.publish(goal);
  finale::ToggleQRNumService tqr; tqr.what = tqr.BOTH; tqr.to = true;
  qrnum_pub.publish(tqr);
  STATE = APPROACH_FACE;
}
int no_rotations = 0;
void handleApproachFace() {
  goal_cancel_pub.publish(actionlib_msgs::GoalID());
  std::get<1>(goal_status).status = std::get<1>(goal_status).ACTIVE;
  if(std::get<1>(recent_face) == curr_face->id && (std::get<0>(recent_face) - ros::Time::now()).toSec() <= MAX_TIME_DIFF) {
    no_rotations = 0;
    // Valid face - move to
    geometry_msgs::Pose *recent = &std::get<2>(recent_face);
    sendDebug(*recent, "camera_rgb_frame");
    double angle_to_target = atan2(recent->position.y, recent->position.x);
    double distance_to_target = sqrt(pow(recent->position.x, 2) + pow(recent->position.y, 2));
    if(abs(angle_to_target) > MAX_APPROACH_ANGLE) {
      // Rotate
      ROS_INFO("ROTATE");
      double ang_vel = get_ang_vel(angle_to_target, 0.5, ROS_RATE);
      velocity_pub.publish(getTwist(0, ang_vel));
      double dang = -ang_vel * (1.0 / (double)ROS_RATE);
      recent->position.x = cos(angle_to_target + dang) * distance_to_target;
      recent->position.y = sin(angle_to_target + dang) * distance_to_target;
    } else if(abs(distance_to_target) > MAX_APPROACH_DISTANCE) {
      // Move forward
      ROS_INFO("FORWARD %f %f", distance_to_target, MAX_APPROACH_DISTANCE);
      double lin_vel = get_lin_vel(distance_to_target, 0.2, ROS_RATE);
      velocity_pub.publish(getTwist(lin_vel, 0));
      double dlin = lin_vel * (1.0 / (double)ROS_RATE);
      recent->position.x -= dlin;
    } else if(abs(distance_to_target) < MIN_APPROACH_DISTANCE) {
      // Move backward
      ROS_INFO("BACKWARD %f %f", distance_to_target, MIN_APPROACH_DISTANCE);
      double lin_vel = -0.1;
      velocity_pub.publish(getTwist(lin_vel, 0));
      double dlin = lin_vel * (1.0 / (double)ROS_RATE);
      recent->position.x -= dlin;
    } else {
      // Stop
      ROS_INFO("STOP");
      velocity_pub.publish(getTwist(0, 0));
      STATE = INTERACT_FACE;
    }
    // ROS_INFO("%s", poseToString(*recent).c_str());
    ROS_INFO("Approach %f m, %f deg", distance_to_target, angle_to_target * M_PI / 180.0);
  } else {
    // Invalid face - rotate w/ pose and cluster loc
    actionlib_msgs::GoalStatus st = std::get<1>(goal_status);
    if(st.status == st.SUCCEEDED) {
      if(no_rotations == 0) {
        geometry_msgs::Pose p = std::get<1>(recent_pose);
        double fvec[2] = {-p.position.x + curr_face->x, -p.position.y + curr_face->y};
        double pvec[2] = {p.orientation.w, p.orientation.z};
        double cosang = (fvec[0] * pvec[0] + fvec[1] * pvec[1]) / (sqrt(pow(fvec[0], 2) + pow(fvec[1], 2)) * sqrt(pow(pvec[0], 2) + pow(pvec[1], 2)));
        velocity_pub.publish(getTwist(0, 0.2 * (abs(cosang) / cosang)));
        no_rotations++;
      } else if(no_rotations > MAX_SEARCH_ROTATIONS) {
        // Try again later
        no_rotations = 0;
        add_to_end(*curr_face, face_list, LIFO);
        STATE = getState();
        if(face_list.size() == 1) STATE = WANDERING;
      } else no_rotations++;
    } 
  }
}
void handleInteractFace() {

  curr_face == NULL;
  finale::ToggleQRNumService tqr; tqr.what = tqr.BOTH; tqr.to = false;
  qrnum_pub.publish(tqr);
  STATE = getState();
}
void handleGotoCylinder() {

  finale::ToggleQRNumService tqr; tqr.what = tqr.QR; tqr.to = true;
  qrnum_pub.publish(tqr);
  STATE = APPROACH_CYLINDER;
}
void handleApproachCylinder() {

  STATE = INTERACT_CYLINDER;
}
void handleInteractCylinder() {

  finale::ToggleQRNumService tqr; tqr.what = tqr.QR; tqr.to = false;
  qrnum_pub.publish(tqr);
  STATE = getState();
}
void handleGotoRing() {
  
  STATE = APPROACH_RING;
}
void handleApproachRing() {

  STATE = INTERACT_RING;
}
void handleInteractRing() {

  STATE = getState();
}
void handleGotoDelivery() {
  
  STATE = APPROACH_DELIVERY;
}
void handleApproachDelivery() {

  STATE = DELIVER;
}
void handleDeliver() {

  STATE = getState();
}

void mapCallback(const nav_msgs::OccupancyGridConstPtr& msg_map) {
    if(got_map) return;

    got_map = true;

    map = msg_map->data;

    map_resolution = msg_map->info.resolution;
    map_width = msg_map->info.width;
    map_height = msg_map->info.height;

    map_transform.transform.translation.x = msg_map->info.origin.position.x;
    map_transform.transform.translation.y = msg_map->info.origin.position.y;
    map_transform.transform.translation.z = msg_map->info.origin.position.z;

    map_transform.transform.rotation = msg_map->info.origin.orientation;
}

void poseCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg) {
  got_pose = true;
  recent_pose = std::tuple<ros::Time, geometry_msgs::Pose>(pose_msg->header.stamp, pose_msg->pose.pose);
}

void faceCallback(const finale::FaceClusteringToHub::ConstPtr &face_msg) {
  int no_new = face_msg->inCamera.size();
  if(no_new != 0) {
    for(int i = 0; i < face_msg->faces.size(); i++) {
      for(int j = 0; j < no_new; j++) if(face_msg->faces[i].id == face_msg->inCamera[j].id) {
        // Freshly detected cluster
        data_t<facedata> *cl = findById(face_msg->faces[i].id, faces);
        finale::FaceCluster fcl = face_msg->faces[i];

        if(cl == NULL) {
          // Add cluster
          faces.push_back(data_t<facedata>(fcl.id, fcl.x, fcl.y, fcl.cos, fcl.sin, fcl.detections, fcl.status, facedata(-1, -1), fcl.approach));
          add_to_list<data_t<facedata>>(faces.back(), face_list);
        } else {
          // Update data
          cl->update(fcl);
        }
      }
    }
    for(int i = 0; i < no_new; i++) if(curr_face != NULL && curr_face->id == face_msg->inCamera[i].id) {
      // ROS_INFO("INCAMERA %s", poseToString(face_msg->inCamera[i].pose).c_str());
      recent_face = std::tuple<ros::Time, int, geometry_msgs::Pose>(face_msg->stamp, face_msg->inCamera[i].id, face_msg->inCamera[i].pose);
    }
  }
}

void cylinderCallback() {

}

void ringCallback() {

}

void qrCallback(const finale::QRNumDataToHub::ConstPtr &msg) {
  ROS_INFO("QR data %s", msg->data.c_str());
  recent_qr = std::tuple<ros::Time, std::string>(msg->stamp, msg->data);
}

void numCallback(const finale::QRNumDataToHub::ConstPtr &msg) {
  ROS_INFO("Num data %s", msg->data.c_str());
  recent_num = std::tuple<ros::Time, int, int>(msg->stamp, msg->data.at(0), msg->data.at(1));
}

void goalCallback(const actionlib_msgs::GoalStatusArray::ConstPtr &msg) {
  // TODO fix states not working correctly
  if(msg->status_list.size() > 0) goal_status = std::tuple<ros::Time, actionlib_msgs::GoalStatus>(msg->header.stamp, msg->status_list[0]);
}

int main(int argc, char** argv) {

    srand(time(NULL));

    ros::init(argc, argv, "move_it2");
    ros::NodeHandle nh;
    n = &nh;

    std::get<1>(recent_face) = -1; std::get<1>(recent_cyl) = -1; std::get<1>(recent_ring) = -1; 

    ros::Subscriber map_sub = nh.subscribe("map", 10, mapCallback);
    ros::Subscriber est_pose_sub = nh.subscribe("amcl_pose", 100, poseCallback);
    ros::Subscriber goal_sub = nh.subscribe("move_base/status", 100, goalCallback);
    ros::Subscriber face_detection_sub = nh.subscribe("finale/faces", 10, faceCallback);
    // ros::Subscriber cylinder_detection_sub = nh.subscribe("finale/cylinders", 10, cylinderCallback);
    // ros::Subscriber ring_detection_sub = nh.subscribe("finale/rings", 10, ringCallback);
    ros::Subscriber qr_detection_sub = nh.subscribe("finale/qr_data", 100, qrCallback);
    ros::Subscriber num_detection_sub = nh.subscribe("finale/number_data", 100, numCallback);

    goal_pub = nh.advertise<geometry_msgs::PoseStamped>("move_base_simple/goal", 10);
    velocity_pub = nh.advertise<geometry_msgs::Twist>("mobile_base/commands/velocity", 10);
    goal_cancel_pub = nh.advertise<actionlib_msgs::GoalID>("move_base/cancel", 10);
    qrnum_pub = nh.advertise<finale::ToggleQRNumService>("finale/toggle_num", 100);
    //
    debug_point_pub = nh.advertise<visualization_msgs::Marker>("check_point", 100);
    //

    ROS_INFO("Waiting for goal subscriber");
    while(goal_pub.getNumSubscribers() < 1 && ros::ok()) {}
    
    ros::Rate rate(ROS_RATE);
    ROS_INFO("Waiting for map and pose");
    while(ros::ok()) {
      ROS_INFO("%d", STATE);
      if(got_map && got_pose) {
        switch(STATE) {
          case STOPPED: handleStopped();
          break;
          case WANDERING: handleWandering();
          break;
          case GOTO_FACE: handleGotoFace();
          break;
          case APPROACH_FACE: handleApproachFace();
          break;
          case INTERACT_FACE: handleInteractFace();
          break;
          case GOTO_CYLINDER: handleGotoCylinder();
          break;
          case APPROACH_CYLINDER: handleApproachCylinder();
          break;
          case INTERACT_CYLINDER: handleInteractCylinder();
          break;
          case GOTO_RING: handleGotoRing();
          break;
          case APPROACH_RING: handleApproachRing();
          break;
          case INTERACT_RING: handleInteractRing();
          break;
          case GOTO_DELIVERY: handleGotoDelivery();
          break;
          case APPROACH_DELIVERY: handleApproachDelivery();
          break;
          case DELIVER: handleDeliver();
          break;
        }
      }
      rate.sleep();
      ros::spinOnce();
    }
    return 0;

}