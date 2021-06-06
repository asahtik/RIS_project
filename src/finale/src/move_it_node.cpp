#include "ros/ros.h"

#include <nav_msgs/GetMap.h>
#include <geometry_msgs/Twist.h>
#include <move_base_msgs/MoveBaseActionResult.h>
#include <actionlib_msgs/GoalID.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/ColorRGBA.h>
#include <actionlib_msgs/GoalStatusArray.h>
#include <actionlib_msgs/GoalStatus.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "sound_play/sound_play.h"
#include <sensor_msgs/LaserScan.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <Eigen/Core>

#include <random>
#include <time.h>
#include <sstream>
#include <math.h>

#include "include/object_data.cpp"
#include "finale/ClusterInCamera.h"
#include "finale/FaceCluster.h"
#include "finale/FaceClusteringToHub.h"
#include "finale/CylCluster.h"
#include "finale/CylClusteringToHub.h"
#include "finale/RingCluster.h"
#include "finale/RingClusteringToHub.h"
#include "finale/ToggleQRNumService.h"
#include "finale/QRNumDataToHub.h"
#include "finale/RecogniseSpeech.h"
#include "finale/BarvaCilindrov.h"
#include "finale/BarvaRingov.h"
#include "finale/KateraCepiva.h"
#include "finale/WanderControl.h"

#define ROS_RATE 10
#define LIFO false
#define NO_RADIUS_CHECKS 100
#define SAFETY_RADIUS 0.3
#define MAX_TIME_DIFF 2.0
#define APPROACH_TIME 15.0
#define MAX_APPROACH_ANGLE 5.0 * M_PI / 180.0
#define MAX_APPROACH_DISTANCE 0.7
#define MIN_APPROACH_DISTANCE 0.5
#define PERSONAL_SPACE 1.0
#define LASER_ANGLE 10.0 * M_PI / 180.0
#define APPROACH_OBSTACLE_DIFF 0.1
#define OBSTACLE_DANGER_ZONE 1.0
#define CLEAR_FORWARD_ITERATIONS 10

// Age, Exercise
typedef std::tuple<int, int> facedata;
// Colour, person
typedef std::tuple<unsigned int, data_t<facedata>*> cyldata;
typedef std::tuple<unsigned int, data_t<facedata>*> ringdata;

enum {STOPPED, WANDERING, GOTO_FACE, APPROACH_FACE, INTERACT_FACE, GOTO_CYLINDER, APPROACH_CYLINDER, INTERACT_CYLINDER, GOTO_RING, APPROACH_RING, INTERACT_RING, GOTO_DELIVERY, APPROACH_DELIVERY, DELIVER} state;
enum {RED = finale::RecogniseSpeech::Response::RED, GREEN = finale::RecogniseSpeech::Response::GREEN, BLUE = finale::RecogniseSpeech::Response::BLUE, YELLOW = finale::RecogniseSpeech::Response::YELLOW, BLACK} colours;

volatile int STATE = STOPPED;
volatile int GOAL_ID = 0;

ros::NodeHandle *n;

float map_resolution = 0;
geometry_msgs::TransformStamped map_transform;

ros::Publisher goal_pub;
ros::Publisher goal_cancel_pub;
ros::Publisher velocity_pub;
ros::Publisher qrnum_pub;
ros::Publisher debug_point_pub;
ros::Publisher marker_pub;
ros::Publisher wander_pub;
ros::ServiceClient conv_client;
ros::ServiceClient ring_clr_client;
ros::ServiceClient cyl_clr_client;
ros::ServiceClient vacc_client;
sound_play::SoundClient *sc;

bool got_map = false, got_pose = false;
int map_width = 0, map_height = 0;
std::vector<int8_t> map;

// Faces
std::list<data_t<facedata>> faces;
std::list<data_t<facedata>*> face_list;
// Cylinders
std::list<data_t<cyldata>> cylinders;
std::list<data_t<cyldata>*> cyl_list;
std::list<cyldata> cyl_requests;
// Rings
std::list<data_t<ringdata>> rings;
std::list<data_t<ringdata>*> ring_list;
std::list<ringdata> ring_requests;
// Deliveries
std::list<data_t<facedata>*> delivery_list;

std::tuple<ros::Time, geometry_msgs::Pose> recent_pose;
std::tuple<ros::Time, int, geometry_msgs::Pose> recent_face;
std::tuple<ros::Time, int, geometry_msgs::Pose> recent_cyl;
std::tuple<ros::Time, int, geometry_msgs::Pose> recent_ring;
std::tuple<float, float, float> recent_ranges;
std::tuple<ros::Time, actionlib_msgs::GoalStatus> goal_status;
std::tuple<ros::Time, std::string> recent_qr;
std::tuple<ros::Time, volatile int, volatile int> recent_num;

template<typename T>
void add_to_list(T *el, std::list<T*> &list) {
  list.push_back(el);
}
template<typename T>
void add_to_end(T &el, std::list<T*> &list, bool lifo = false) {
  if(lifo) list.push_front(&el);
  else list.push_back(&el);
}
void sendDebug(geometry_msgs::Pose &p, const char *frame, const char *ns) {
  visualization_msgs::Marker m;
  m.ns = ns;
  m.header.frame_id = frame;
  m.header.stamp = ros::Time::now();
  m.lifetime = ros::Duration(5);
  m.id = 0;
  m.pose = p;
  m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.1;
  m.type = m.SPHERE;
  m.color.a = 1.0; m.color.r = 1.0;
  m.action = m.ADD;
  debug_point_pub.publish(m);
  ROS_INFO("Debug point data for ns %s: position %f %f %f orientation %f %f %f %f", ns, p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
}
void sendToggleWander(bool wander) {
  finale::WanderControl w; w.wander = wander;
  wander_pub.publish(w);
}
void sendMarkers();
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

bool check_valid_position(int x, int y) {
    // ROS_INFO("map at x:%d,y:%d %d", x, y, map[y * map_width + x]);
    if(x >= map_width) return false;
    else if(y >= map_height) return false;
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
template<typename T>
geometry_msgs::Pose getApproach(data_t<T> *cl, geometry_msgs::Pose apprch, float dist = 0.5) {
  tf2::Stamped<tf2::Transform> temp_inv;
  tf2::fromMsg(map_transform, temp_inv);
  geometry_msgs::TransformStamped inv_transform;
  inv_transform.transform = tf2::toMsg(temp_inv.inverse());

  float angle_inc = (2 * M_PI / NO_RADIUS_CHECKS);
  for(int i = 0; i < NO_RADIUS_CHECKS; i++) {
      float angle = i * angle_inc;
      float x = (cl->x + cos(angle) * (dist));
      float y = (cl->y + sin(angle) * (dist));

      geometry_msgs::Point pt;
      geometry_msgs::Point transformed_pt;
      pt.x = x;
      pt.y = y;
      pt.z = 0.0;
      tf2::doTransform(pt, transformed_pt, inv_transform);

      int map_r_x = round(transformed_pt.x / map_resolution);
      int map_r_y = round(transformed_pt.y / map_resolution);
      if(check_valid_position(map_r_x, map_r_y)) {
        geometry_msgs::Pose p;
        p.position.x = x;
        p.position.y = y;
        float ang = atan2(cl->y - y, cl->x - x);
        p.orientation.z = sin(ang); p.orientation.w = cos(ang);
        return p;
      }
    }
  geometry_msgs::Pose p;
  p.position.x = cl->x - apprch.orientation.w * dist;
  p.position.y = cl->y - apprch.orientation.z * dist;
  p.orientation = apprch.orientation;
  return p;
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
std::string vectorToString(std::vector<double> &hs) {
  std::stringstream ss;
  for(std::vector<double>::iterator it = hs.begin(); it != hs.end(); ++it) {
    ss << *it;
    if(it + 1 != hs.end()) ss << ",";
  }
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
bool check_approach(double dist, double &ang_vel, double max_speed, int rate) {
  // true clear, false obstacle
  float left = std::get<0>(recent_ranges), center = std::get<1>(recent_ranges), right = std::get<2>(recent_ranges);
  bool chLeft = left < dist - APPROACH_OBSTACLE_DIFF && left < OBSTACLE_DANGER_ZONE, chCenter = center < dist - APPROACH_OBSTACLE_DIFF && center < OBSTACLE_DANGER_ZONE, chRight = right < dist - APPROACH_OBSTACLE_DIFF && right < OBSTACLE_DANGER_ZONE;
  ROS_INFO("Checking approach %d %d %d, distances: %f %f %f", chLeft, chCenter, chRight, left, center, right);
  if(chLeft && chRight && chCenter) {
    ang_vel = get_ang_vel(-2.0 * LASER_ANGLE, max_speed, rate);
    ROS_WARN("Path not clear, turning right");
    return false;
  } else if(chLeft && chCenter || chLeft) {
    ang_vel = get_ang_vel(-2.0 * LASER_ANGLE, max_speed, rate);
    ROS_WARN("Path not clear, turning right");
    return false;
  } else if(chCenter && chRight || chRight) {
    ang_vel = get_ang_vel(2.0 * LASER_ANGLE, max_speed, rate);
    ROS_WARN("Path not clear, turning left");
    return false;
  } else return true;
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

unsigned int getClr(bool ring, std::vector<double> &hist) {
  if(ring) {
    finale::BarvaRingov srv; srv.request.tabela = vectorToString(hist);
    if(ring_clr_client.call(srv)) {
      const char *clr_in = srv.response.barva.c_str();
      unsigned int clr_out = -1;
      if(strcmp(clr_in, "red") == 0) clr_out = RED;
      else if(strcmp(clr_in, "green") == 0) clr_out = GREEN;
      else if(strcmp(clr_in, "blue") == 0) clr_out = BLUE;
      else if(strcmp(clr_in, "black") == 0) clr_out = BLACK;
      return clr_out;
    } else {
      ROS_WARN("Error calling ring clr service");
      return -1;
    }
  } else {
    finale::BarvaCilindrov srv; srv.request.tabela = vectorToString(hist);
    if(cyl_clr_client.call(srv)) {
      const char *clr_in = srv.response.barva.c_str();
      unsigned int clr_out = -1;
      if(strcmp(clr_in, "red") == 0) clr_out = RED;
      else if(strcmp(clr_in, "green") == 0) clr_out = GREEN;
      else if(strcmp(clr_in, "blue") == 0) clr_out = BLUE;
      else if(strcmp(clr_in, "yellow") == 0) clr_out = YELLOW;
      return clr_out;
    } else {
      ROS_WARN("Error calling cylinder clr service");
      return -1;
    }
  }
}
std_msgs::ColorRGBA clrToColour(unsigned int clr) {
  std_msgs::ColorRGBA c; c.a = 1.0;
  switch(clr) {
    case RED:
    c.r = 1.0;
    break;
    case GREEN:
    c.g = 1.0;
    break;
    case BLUE:
    c.b = 1.0;
    break;
    case YELLOW:
    c.r = 1.0; c.g = 1.0;
    break;
    case BLACK:
    break;
    default:
    c.r = 0.8; c.g = 0.8; c.b = 0.8;
    break;
  }
  return c;
}
unsigned int getVacc(std::string &url, int age, int exercise) {
  finale::KateraCepiva srv;
  srv.request.url = url; srv.request.prvaStevilka = age; srv.request.drugaStevilka = exercise;
  if(vacc_client.call(srv)) {
    const char *c = srv.response.cepivo.c_str();
    unsigned int clr = -1;
    if(strcmp(c, "Greenzer") == 0) clr = GREEN;
    else if(strcmp(c, "Rederna") == 0) clr = RED;
    else if(strcmp(c, "BlacknikV") == 0) clr = BLACK;
    else if(strcmp(c, "StellaBluera") == 0) clr = BLUE;
    return clr;
  } else {
    ROS_WARN("Error calling vaccine service");
    return -1;
  }
}
template<typename T>
geometry_msgs::Pose getObjectPose(data_t<T> *cl, geometry_msgs::Pose &apprch) {
  geometry_msgs::Pose p;
  p.position.x = cl->x; p.position.y = cl->y;
  p.orientation = apprch.orientation;
  return p;
}
void warn(data_t<facedata> *person) {
  if(person->status > 0) return;
  double dist = person->find_nearest(faces);
  if(dist < PERSONAL_SPACE && dist >= 0.0) {
    ROS_WARN("Person too close (%f m)", dist);
    std::string s = "Please follow the social distancing rules";
    sc->say(s);
    person->status = 1;
  }
}

data_t<facedata> *curr_face;
data_t<cyldata> *curr_cyl;
data_t<ringdata> *curr_ring;

ros::Time approach_start(0);
bool approach_started = false, going_to_goal = false;
int clear_iter = 0;

void handleStopped() {
  STATE = getState();
}
void handleWandering() {
  STATE = getState();
  sendToggleWander(STATE == WANDERING);
}
void handleGotoFace() {
  curr_face = pop_from_list(face_list, LIFO);
  goal_cancel_pub.publish(actionlib_msgs::GoalID());
  geometry_msgs::PoseStamped goal;
  goal.header.frame_id = "map"; goal.header.stamp = ros::Time().now(); goal.header.seq = GOAL_ID++;
  goal.pose = curr_face->approach;
  sendDebug(goal.pose, "map", "goto");
  ROS_INFO("GOTO FACE POSE %s", poseToString(goal.pose).c_str());
  goal_pub.publish(goal);
  STATE = APPROACH_FACE;
}
void handleApproachFace() {
  finale::ToggleQRNumService tqr; tqr.what = tqr.BOTH; tqr.to = true;
  qrnum_pub.publish(tqr);
  if(!approach_started) {
    approach_started = true;
    approach_start = ros::Time::now();
  }
  if(std::get<1>(recent_face) == curr_face->id && (ros::Time::now() - std::get<0>(recent_face)).toSec() <= MAX_TIME_DIFF) {
    goal_cancel_pub.publish(actionlib_msgs::GoalID());
    std::get<0>(goal_status) = ros::Time::now();
    // Valid face - move to
    geometry_msgs::Pose *recent = &std::get<2>(recent_face);
    sendDebug(*recent, "camera_rgb_frame", "face");
    double angle_to_target = atan2(recent->position.y, recent->position.x);
    double distance_to_target = sqrt(pow(recent->position.x, 2) + pow(recent->position.y, 2));

    double vel; bool clear = check_approach(distance_to_target, vel, 0.5, ROS_RATE);

    if(abs(angle_to_target) > MAX_APPROACH_ANGLE && clear_iter < 1 || !clear) {
      if(!clear) clear_iter = CLEAR_FORWARD_ITERATIONS;
      else clear_iter = 0;
      // Rotate
      ROS_INFO("ROTATE");
      double ang_vel;
      if(clear) ang_vel = get_ang_vel(angle_to_target, 0.5, ROS_RATE);
      else ang_vel = vel;
      velocity_pub.publish(getTwist(0, ang_vel));
      double dang = -ang_vel * (1.0 / (double)ROS_RATE);
      ROS_INFO("ang: %f, dang: %f", angle_to_target, dang);
      recent->position.x = cos(angle_to_target + dang) * distance_to_target;
      recent->position.y = sin(angle_to_target + dang) * distance_to_target;
    } else if(abs(distance_to_target) > 0.5 || clear_iter > 0) {
      clear_iter--;
      // Move forward
      ROS_INFO("FORWARD %f %f", distance_to_target, MAX_APPROACH_DISTANCE);
      double lin_vel = get_lin_vel(distance_to_target, 0.2, ROS_RATE);
      velocity_pub.publish(getTwist(lin_vel, 0));
      double dlin = lin_vel * (1.0 / (double)ROS_RATE);
      recent->position.x -= dlin;
    } else if(abs(distance_to_target) < 0.3) {
      // Move backward
      ROS_INFO("BACKWARD %f %f", distance_to_target, MIN_APPROACH_DISTANCE);
      double lin_vel = -0.1;
      velocity_pub.publish(getTwist(lin_vel, 0));
      double dlin = lin_vel * (1.0 / (double)ROS_RATE);
      recent->position.x -= dlin;
    } else {
      clear_iter = 0;
      approach_started = false;
      going_to_goal = false;
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
    if((st.status == st.SUCCEEDED) && (ros::Time::now() - std::get<0>(goal_status)).toSec() <= MAX_TIME_DIFF && (ros::Time::now() - approach_start).toSec() <= APPROACH_TIME) {
      velocity_pub.publish(getTwist(0, 0.4));
    } else if((ros::Time::now() - approach_start).toSec() > APPROACH_TIME) {
      ROS_INFO("Trying to find face %d", going_to_goal);
      geometry_msgs::PoseStamped goal;
      goal.header.frame_id = "map"; goal.header.stamp = ros::Time().now(); goal.header.seq = GOAL_ID++;
      goal.pose = getApproach(curr_face, curr_face->approach);
      if(!going_to_goal) {
        ROS_INFO("Going to face");
        going_to_goal = true;
        goal_pub.publish(goal);
      }
    } else if(going_to_goal && sqrt(pow(std::get<1>(recent_pose).position.x - curr_face->x, 2) + pow(std::get<1>(recent_pose).position.y - curr_face->y, 2)) <= 0.5) {
      goal_cancel_pub.publish(actionlib_msgs::GoalID());
      approach_started = false;
      going_to_goal = false;
      STATE = INTERACT_FACE;
    }
  }
}
void handleInteractFace() {
  goal_cancel_pub.publish(actionlib_msgs::GoalID());
  finale::ToggleQRNumService tqr; tqr.what = tqr.BOTH; tqr.to = false;
  qrnum_pub.publish(tqr);
  finale::RecogniseSpeech srv; srv.request.question = srv.request.Q1;
  if((ros::Time::now() - std::get<0>(recent_num)).toSec() <= MAX_TIME_DIFF) {
    int x1 = std::get<1>(recent_num), x2 = std::get<2>(recent_num);
    ROS_INFO("Recent num %d %d", x1, x2);
    if(x1 * 10 + x2 > 200) std::get<0>(curr_face->data) = -1;
    else std::get<0>(curr_face->data) = x1 * 10 + x2;
  }
  srv.request.askAge = std::get<0>(curr_face->data) < 0;
  ROS_INFO("Ask age? %d %d %d", srv.request.askAge, std::get<0>(curr_face->data), curr_face->id);
  if(conv_client.call(srv)) {
    if(srv.request.askAge) std::get<0>(curr_face->data) = srv.response.age;
    std::get<1>(curr_face->data) = srv.response.exercise;
    if(srv.response.confirm == srv.response.NO) {
      data_t<cyldata> *cl = findByColour(srv.response.colour, cylinders);
      if(cl != NULL) {
        // Already found cylinder
        ROS_INFO("Requested cylinder already found");
        if(srv.request.askAge) std::get<0>(curr_face->data) = srv.response.age;
        if(curr_face == NULL) ROS_INFO("Curr face is null");
        std::get<1>(cl->data) = curr_face;
        if(std::get<1>(cl->data) == NULL) ROS_INFO("Assign error");
        cyl_list.push_back(cl);
      } else {
        // Cylinder not yet found
        ROS_INFO("Requested cylinder not yet found");
        cyl_requests.push_back(cyldata(srv.response.colour, curr_face));
      }
    }
    curr_face == NULL;
    STATE = getState();
  } else {
    ROS_WARN("Service call unsuccessful");
  }
  warn(curr_face);
}
void handleGotoCylinder() {
  curr_cyl = pop_from_list(cyl_list, LIFO);
  goal_cancel_pub.publish(actionlib_msgs::GoalID());
  geometry_msgs::PoseStamped goal;
  goal.header.frame_id = "map"; goal.header.stamp = ros::Time().now(); goal.header.seq = GOAL_ID++;
  goal.pose = curr_cyl->approach;
  sendDebug(goal.pose, "map", "goto");
  ROS_INFO("GOTO CYLINDER POSE %s", poseToString(goal.pose).c_str());
  goal_pub.publish(goal);
  STATE = APPROACH_CYLINDER;
}
void handleApproachCylinder() {
  finale::ToggleQRNumService tqr; tqr.what = tqr.QR; tqr.to = true;
  qrnum_pub.publish(tqr);
  if(!approach_started) {
    approach_started = true;
    approach_start = ros::Time::now();
  }
  if(std::get<1>(recent_cyl) == curr_cyl->id && (ros::Time::now() - std::get<0>(recent_cyl)).toSec() <= MAX_TIME_DIFF) {
    goal_cancel_pub.publish(actionlib_msgs::GoalID());
    std::get<0>(goal_status) = ros::Time::now();
    // Valid cylinder - move to
    geometry_msgs::Pose *recent = &std::get<2>(recent_cyl);
    sendDebug(*recent, "camera_depth_optical_frame", "cylinder");
    double angle_to_target = atan2(-recent->position.x, recent->position.z); // Should it be *2 ?
    double distance_to_target = sqrt(pow(recent->position.z, 2) + pow(recent->position.x, 2));

    double vel; bool clear = check_approach(distance_to_target, vel, 0.5, ROS_RATE);

    if(abs(angle_to_target) > MAX_APPROACH_ANGLE && clear_iter < 1 || !clear) {
      if(!clear) clear_iter = CLEAR_FORWARD_ITERATIONS;
      else clear_iter = 0;
      // Rotate
      ROS_INFO("ROTATE");
      double ang_vel;
      if(clear) ang_vel = get_ang_vel(angle_to_target, 0.5, ROS_RATE);
      else ang_vel = vel;
      velocity_pub.publish(getTwist(0, ang_vel));
      double dang = -ang_vel * (1.0 / (double)ROS_RATE);
      recent->position.z = cos(angle_to_target + dang) * distance_to_target;
      recent->position.x = -sin(angle_to_target + dang) * distance_to_target;
    } else if(abs(distance_to_target) > MAX_APPROACH_DISTANCE || clear_iter > 0) {
      clear_iter--;
      // Move forward
      ROS_INFO("FORWARD %f %f", distance_to_target, MAX_APPROACH_DISTANCE);
      double lin_vel = get_lin_vel(distance_to_target, 0.2, ROS_RATE);
      velocity_pub.publish(getTwist(lin_vel, 0));
      double dlin = lin_vel * (1.0 / (double)ROS_RATE);
      recent->position.z -= dlin;
    } else if(abs(distance_to_target) < MIN_APPROACH_DISTANCE) {
      // Move backward
      ROS_INFO("BACKWARD %f %f", distance_to_target, MIN_APPROACH_DISTANCE);
      double lin_vel = -0.1;
      velocity_pub.publish(getTwist(lin_vel, 0));
      double dlin = lin_vel * (1.0 / (double)ROS_RATE);
      recent->position.z -= dlin;
    } else {
      clear_iter = 0;
      approach_started = false;
      going_to_goal = false;
      // Stop
      ROS_INFO("STOP");
      velocity_pub.publish(getTwist(0, 0));
      STATE = INTERACT_CYLINDER;
    }
    // ROS_INFO("%s", poseToString(*recent).c_str());
    ROS_INFO("Approach %f m, %f deg", distance_to_target, angle_to_target * M_PI / 180.0);
  } else {
    // Invalid cylinder - rotate w/ pose and cluster loc
    actionlib_msgs::GoalStatus st = std::get<1>(goal_status);
    if((st.status == st.SUCCEEDED) && (ros::Time::now() - std::get<0>(goal_status)).toSec() <= MAX_TIME_DIFF && (ros::Time::now() - approach_start).toSec() <= APPROACH_TIME) {
      velocity_pub.publish(getTwist(0, 0.4));
    } else if((ros::Time::now() - approach_start).toSec() > APPROACH_TIME) {
      ROS_INFO("Trying to find cylinder %d", going_to_goal);
      geometry_msgs::PoseStamped goal;
      goal.header.frame_id = "map"; goal.header.stamp = ros::Time().now(); goal.header.seq = GOAL_ID++;
      goal.pose = getApproach(curr_cyl, curr_cyl->approach);
      if(!going_to_goal) {
        going_to_goal = true;
        goal_pub.publish(goal);
      }
    } else if(going_to_goal && sqrt(pow(std::get<1>(recent_pose).position.x - curr_cyl->x, 2) + pow(std::get<1>(recent_pose).position.y - curr_cyl->y, 2)) <= 0.5) {
      goal_cancel_pub.publish(actionlib_msgs::GoalID());
      approach_started = false;
      going_to_goal = false;
      STATE = INTERACT_CYLINDER;
    }
  }
}
void handleInteractCylinder() {
  goal_cancel_pub.publish(actionlib_msgs::GoalID());
  if((ros::Time::now() - std::get<0>(recent_qr)).toSec() < MAX_TIME_DIFF) {
    ROS_INFO("QR info %s", std::get<1>(recent_qr).c_str());
    data_t<facedata> *face = std::get<1>(curr_cyl->data);
    int age = std::get<0>(face->data), ex = std::get<1>(face->data);
    unsigned int clr = getVacc(std::get<1>(recent_qr), age, ex);
    finale::ToggleQRNumService tqr; tqr.what = tqr.QR; tqr.to = false;
    qrnum_pub.publish(tqr);
    ROS_INFO("Classified colour %d", clr);

    if(clr < 0) {
      add_to_end(*curr_cyl, cyl_list, LIFO);
      STATE = WANDERING;
      return;
    }

    data_t<ringdata> *cl = findByColour(clr, rings);
    if(cl != NULL) {
      // Already found ring
      ROS_INFO("Requested ring already found");
      std::get<1>(cl->data) = face;
      ring_list.push_back(cl);
    } else {
      // Ring not yet found
      ROS_INFO("Requested ring not yet found");
      ring_requests.push_back(ringdata(clr, face));
    }
    curr_cyl == NULL;
    ROS_INFO("Done interacting with cilinder");
    STATE = getState();
  } else {
    velocity_pub.publish(getTwist(-0.1, 0.0));
  }
}
void handleGotoRing() {
  curr_ring = pop_from_list(ring_list, LIFO);
  goal_cancel_pub.publish(actionlib_msgs::GoalID());
  geometry_msgs::PoseStamped goal;
  goal.header.frame_id = "map"; goal.header.stamp = ros::Time().now(); goal.header.seq = GOAL_ID++;
  goal.pose = curr_ring->approach;
  sendDebug(goal.pose, "map", "goto");
  ROS_INFO("GOTO RING POSE %s", poseToString(goal.pose).c_str());
  goal_pub.publish(goal);
  STATE = APPROACH_RING;
}
void handleApproachRing() {
  if(!approach_started) {
    approach_started = true;
    approach_start = ros::Time::now();
  }
  if(std::get<1>(recent_ring) == curr_ring->id && (ros::Time::now() - std::get<0>(recent_ring)).toSec() <= MAX_TIME_DIFF) {
    goal_cancel_pub.publish(actionlib_msgs::GoalID());
    std::get<0>(goal_status) = ros::Time::now();
    // Valid ring - move to
    geometry_msgs::Pose *recent = &std::get<2>(recent_ring);
    sendDebug(*recent, "camera_depth_optical_frame", "ring");
    double angle_to_target = atan2(-recent->position.x, recent->position.z); // Should it be *2 ?
    double distance_to_target = sqrt(pow(recent->position.z, 2) + pow(recent->position.x, 2));

    double vel; bool clear = check_approach(distance_to_target, vel, 0.5, ROS_RATE);

    if(abs(angle_to_target) > MAX_APPROACH_ANGLE && clear_iter < 1 || !clear) {
      if(!clear) clear_iter = CLEAR_FORWARD_ITERATIONS;
      else clear_iter = 0;
      // Rotate
      ROS_INFO("ROTATE");
      double ang_vel;
      if(clear) ang_vel = get_ang_vel(angle_to_target, 0.5, ROS_RATE);
      else ang_vel = vel;
      velocity_pub.publish(getTwist(0, ang_vel));
      double dang = -ang_vel * (1.0 / (double)ROS_RATE);
      recent->position.z = cos(angle_to_target + dang) * distance_to_target;
      recent->position.x = -sin(angle_to_target + dang) * distance_to_target;
      std::get<0>(recent_ring) = ros::Time::now();
    } else if(abs(distance_to_target) > 0.5 || clear_iter > 0) {
      clear_iter--;
      // Move forward
      ROS_INFO("FORWARD %f %f", distance_to_target, 0.1);
      double lin_vel = get_lin_vel(distance_to_target, 0.2, ROS_RATE);
      velocity_pub.publish(getTwist(lin_vel, 0));
      double dlin = lin_vel * (1.0 / (double)ROS_RATE);
      recent->position.z -= dlin;
      std::get<0>(recent_ring) = ros::Time::now();
    } else {
      clear_iter = 0;
      approach_started = false;
      going_to_goal = false;
      // Stop
      ROS_INFO("STOP");
      velocity_pub.publish(getTwist(0, 0));
      STATE = INTERACT_RING;
    }
    // ROS_INFO("%s", poseToString(*recent).c_str());
    ROS_INFO("Approach %f m, %f deg", distance_to_target, angle_to_target * M_PI / 180.0);
  } else {
    ROS_INFO("Trying to find ring");
    // Invalid ring - rotate w/ pose and cluster loc
    actionlib_msgs::GoalStatus st = std::get<1>(goal_status);
    if((st.status == st.SUCCEEDED) && (ros::Time::now() - std::get<0>(goal_status)).toSec() <= MAX_TIME_DIFF && (ros::Time::now() - approach_start).toSec() <= APPROACH_TIME) {
      velocity_pub.publish(getTwist(0, 0.4));
    } else if((ros::Time::now() - approach_start).toSec() > APPROACH_TIME) {
      ROS_INFO("Trying to find ring %d", going_to_goal);
      geometry_msgs::PoseStamped goal;
      goal.header.frame_id = "map"; goal.header.stamp = ros::Time().now(); goal.header.seq = GOAL_ID++;
      goal.pose = getApproach(curr_ring, curr_ring->approach);
      if(!going_to_goal) {
        going_to_goal = true;
        goal_pub.publish(goal);
      }
    } else if(going_to_goal && sqrt(pow(std::get<1>(recent_pose).position.x - curr_ring->x, 2) + pow(std::get<1>(recent_pose).position.y - curr_ring->y, 2)) <= 0.5) {
      goal_cancel_pub.publish(actionlib_msgs::GoalID());
      approach_started = false;
      going_to_goal = false;
      STATE = INTERACT_RING;
    }
  }
}
void handleInteractRing() {
  goal_cancel_pub.publish(actionlib_msgs::GoalID());
  delivery_list.push_back(std::get<1>(curr_ring->data));
  curr_ring = NULL;
  STATE = getState();
}
void handleGotoDelivery() {
  curr_face = pop_from_list(delivery_list, LIFO);
  goal_cancel_pub.publish(actionlib_msgs::GoalID());
  geometry_msgs::PoseStamped goal;
  goal.header.frame_id = "map"; goal.header.stamp = ros::Time().now(); goal.header.seq = GOAL_ID++;
  goal.pose = curr_face->approach;
  ROS_INFO("GOTO DELIVERY POSE %s", poseToString(goal.pose).c_str());
  goal_pub.publish(goal);
  STATE = APPROACH_DELIVERY;
}
void handleApproachDelivery() {
  if(!approach_started) {
    approach_started = true;
    approach_start = ros::Time::now();
  }
  if(std::get<1>(recent_face) == curr_face->id && (ros::Time::now() - std::get<0>(recent_face)).toSec() <= MAX_TIME_DIFF) {
    goal_cancel_pub.publish(actionlib_msgs::GoalID());
    std::get<0>(goal_status) = ros::Time::now();
    // Valid face - move to
    geometry_msgs::Pose *recent = &std::get<2>(recent_face);
    sendDebug(*recent, "camera_rgb_frame", "face");
    double angle_to_target = atan2(recent->position.y, recent->position.x);
    double distance_to_target = sqrt(pow(recent->position.x, 2) + pow(recent->position.y, 2));

    double vel; bool clear = check_approach(distance_to_target, vel, 0.5, ROS_RATE);

    if(abs(angle_to_target) > MAX_APPROACH_ANGLE && clear_iter < 1 || !clear) {
      if(!clear) clear_iter = CLEAR_FORWARD_ITERATIONS;
      else clear_iter = 0;
      // Rotate
      ROS_INFO("ROTATE");
      double ang_vel;
      if(clear) ang_vel = get_ang_vel(angle_to_target, 0.5, ROS_RATE);
      else ang_vel = vel;
      velocity_pub.publish(getTwist(0, ang_vel));
      double dang = -ang_vel * (1.0 / (double)ROS_RATE);
      recent->position.x = cos(angle_to_target + dang) * distance_to_target;
      recent->position.y = sin(angle_to_target + dang) * distance_to_target;
    } else if(abs(distance_to_target) > MAX_APPROACH_DISTANCE || clear_iter > 0) {
      clear_iter--;
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
      clear_iter = 0;
      approach_started = false;
      going_to_goal = false;
      // Stop
      ROS_INFO("STOP");
      velocity_pub.publish(getTwist(0, 0));
      STATE = DELIVER;
    }
    // ROS_INFO("%s", poseToString(*recent).c_str());
    ROS_INFO("Approach %f m, %f deg", distance_to_target, angle_to_target * M_PI / 180.0);
  } else {
    ROS_INFO("Trying to find face");
    // Invalid face - rotate w/ pose and cluster loc
    actionlib_msgs::GoalStatus st = std::get<1>(goal_status);
    if((st.status == st.SUCCEEDED) && (ros::Time::now() - std::get<0>(goal_status)).toSec() <= MAX_TIME_DIFF && (ros::Time::now() - approach_start).toSec() <= APPROACH_TIME) {
      velocity_pub.publish(getTwist(0, 0.4));
    } else if((ros::Time::now() - approach_start).toSec() > APPROACH_TIME) {
      geometry_msgs::PoseStamped goal;
      goal.header.frame_id = "map"; goal.header.stamp = ros::Time().now(); goal.header.seq = GOAL_ID++;
      goal.pose = getApproach(curr_face, curr_face->approach);
      if(!going_to_goal) {
        going_to_goal = true;
        goal_pub.publish(goal);
      }
    } else if(going_to_goal && sqrt(pow(std::get<1>(recent_pose).position.x - curr_face->x, 2) + pow(std::get<1>(recent_pose).position.y - curr_face->y, 2)) <= 0.5) {
      goal_cancel_pub.publish(actionlib_msgs::GoalID());
      approach_started = false;
      going_to_goal = false;
      STATE = DELIVER;
    }
  }
}
void handleDeliver() {
  goal_cancel_pub.publish(actionlib_msgs::GoalID());
  finale::RecogniseSpeech srv; srv.request.question = srv.request.Q2;
  srv.request.askAge = false;
  if(conv_client.call(srv)) {
    if(srv.response.confirm == srv.response.YES) {

    }
    curr_face == NULL;
    STATE = getState();
  } else {
    ROS_WARN("Service call unsuccessful");
  }
  warn(curr_face);
  STATE = getState();
}

void mapCallback(const nav_msgs::OccupancyGridConstPtr& msg_map) {
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
          faces.push_back(data_t<facedata>(fcl.id, fcl.x, fcl.y, fcl.cos, fcl.sin, fcl.detections, 0, facedata(-1, -1), fcl.approach));
          data_t<facedata> *fce = &faces.back();
          std::get<0>(fce->data) = -1;
          add_to_list<data_t<facedata>>(fce, face_list);
        } else {
          // Update data
          cl->update(fcl);
        }
      }
      break;
    }
    for(int i = 0; i < no_new; i++) if(curr_face != NULL && curr_face->id == face_msg->inCamera[i].id) {
      // ROS_INFO("INCAMERA %s", poseToString(face_msg->inCamera[i].pose).c_str());
      recent_face = std::tuple<ros::Time, int, geometry_msgs::Pose>(face_msg->stamp, face_msg->inCamera[i].id, face_msg->inCamera[i].pose);
    }
  }
}

void cylinderCallback(const finale::CylClusteringToHub::ConstPtr &msg) {
  int no_new = msg->inCamera.size();
  if(no_new != 0) {
    for(int i = 0; i < msg->cyls.size(); i++) {
      for(int j = 0; j < no_new; j++) if(msg->cyls[i].id == msg->inCamera[j].id) {
        // Freshly detected cluster
        data_t<cyldata> *cl = findById(msg->cyls[i].id, cylinders);
        finale::CylCluster ccl = msg->cyls[i];

        if(cl == NULL) {
          unsigned int clr = getClr(false, ccl.data);
          if(clr < 0) break;
          // Add cluster
          cylinders.push_back(data_t<cyldata>(ccl.id, ccl.x, ccl.y, ccl.cos, ccl.sin, ccl.detections, ccl.status, cyldata(clr, NULL), ccl.approach));
          // Check requests
          int size = cyl_requests.size(), i = 0;
          for(std::list<cyldata>::iterator it = cyl_requests.begin(); it != cyl_requests.end() && i < size; ++it) {
            if(std::get<0>(*it) == clr) {
              data_t<cyldata> *cyl = &cylinders.back();
              std::get<1>(cyl->data) = std::get<1>(*it);
              add_to_list<data_t<cyldata>>(&cylinders.back(), cyl_list);
              cyl_requests.erase(it);
              break;
            }
            i++;
          }
        } else {
          // Update data
          unsigned int clr = getClr(false, ccl.data);
          if(clr >= 0) std::get<0>(cl->data) = clr;
          cl->update(ccl);
        }
        break;
      }
    }
    for(int i = 0; i < no_new; i++) if(curr_cyl != NULL && curr_cyl->id == msg->inCamera[i].id) {
      // ROS_INFO("INCAMERA %s", poseToString(face_msg->inCamera[i].pose).c_str());
      recent_cyl = std::tuple<ros::Time, int, geometry_msgs::Pose>(msg->stamp, msg->inCamera[i].id, msg->inCamera[i].pose);
    }
  }
}

void ringCallback(const finale::RingClusteringToHub::ConstPtr &msg) {
  int no_new = msg->inCamera.size();
  if(no_new != 0) {
    for(int i = 0; i < msg->rings.size(); i++) {
      for(int j = 0; j < no_new; j++) if(msg->rings[i].id == msg->inCamera[j].id) {
        // Freshly detected cluster
        data_t<ringdata> *cl = findById(msg->rings[i].id, rings);
        finale::RingCluster ccl = msg->rings[i];

        if(cl == NULL) {
          unsigned int clr = getClr(true, ccl.data);
          if(clr < 0) break;
          // Add cluster
          rings.push_back(data_t<ringdata>(ccl.id, ccl.x, ccl.y, ccl.cos, ccl.sin, ccl.detections, ccl.status, ringdata(clr, NULL), ccl.approach));
          // Check requests
          int size = ring_requests.size(), i = 0;
          for(std::list<ringdata>::iterator it = ring_requests.begin(); it != ring_requests.end() && i < size; ++it) {
            if(std::get<0>(*it) == clr) {
              data_t<ringdata> *rng = &rings.back();
              std::get<1>(rng->data) = std::get<1>(*it);
              add_to_list<data_t<ringdata>>(&rings.back(), ring_list);
              ring_requests.erase(it);
              break;
            }
            i++;
          }
        } else {
          // Update data
          unsigned int clr = getClr(true, ccl.data);
          if(clr >= 0) std::get<0>(cl->data) = clr;
          cl->update(ccl);
        }
      }
    }
    for(int i = 0; i < no_new; i++) if(curr_ring != NULL && curr_ring->id == msg->inCamera[i].id) {
      // ROS_INFO("INCAMERA %s", poseToString(face_msg->inCamera[i].pose).c_str());
      recent_ring = std::tuple<ros::Time, int, geometry_msgs::Pose>(msg->stamp, msg->inCamera[i].id, msg->inCamera[i].pose);
    }
  }
}

void qrCallback(const finale::QRNumDataToHub::ConstPtr &msg) {
  ROS_INFO("QR data %s", msg->data.c_str());
  recent_qr = std::tuple<ros::Time, std::string>(msg->stamp, msg->data.substr(2, msg->data.size() - 3));
}

void numCallback(const finale::QRNumDataToHub::ConstPtr &msg) {
  ROS_INFO("Num data %s", msg->data.c_str());
  recent_num = std::tuple<ros::Time, volatile int, volatile int>(msg->stamp, msg->data.at(0) - '0', msg->data.at(1) - '0');
}

void goalCallback(const move_base_msgs::MoveBaseActionResult::ConstPtr &msg) {
  // TODO fix states not working correctly
  goal_status = std::tuple<ros::Time, actionlib_msgs::GoalStatus>(msg->header.stamp, msg->status);
}

void laserCallback(const sensor_msgs::LaserScan::ConstPtr &msg) {
  float min = msg->angle_min, max = msg->angle_max;
  int size = msg->ranges.size();
  int size2 = floor((float)size / 2.0);
  float radPerData = (max - min) / (float)size;
  int rec = round(LASER_ANGLE / radPerData);
  float right = msg->ranges[size2 - rec < 0 ? 0 : size2 - rec], center = msg->ranges[size2], left = msg->ranges[size2 + rec >= size ? size - 1 : size2 + rec];
  if(left < msg->range_min) left = 10.0; else if(left > msg->range_max) left = 10.0;
  if(center < msg->range_min) center = 10.0; else if(center > msg->range_max) center = 10.0;
  if(right < msg->range_min) right = 10.0; else if(right > msg->range_max) right = 10.0;

  recent_ranges = std::tuple<float, float, float>(left, center, right);
}

int main(int argc, char** argv) {

    srand(time(NULL));

    ros::init(argc, argv, "move_it2");
    ros::NodeHandle nh;
    n = &nh;

    sound_play::SoundClient s;
    sc = &s;

    std::get<1>(recent_face) = -1; std::get<1>(recent_cyl) = -1; std::get<1>(recent_ring) = -1; std::get<1>(recent_num) = -1; std::get<2>(recent_num) = -1;
    std::get<0>(recent_face) = ros::Time(0); std::get<0>(recent_cyl) = ros::Time(0); std::get<0>(recent_ring) = ros::Time(0); std::get<0>(recent_num) = ros::Time(0); std::get<0>(recent_pose) = ros::Time(0); std::get<0>(recent_qr) = ros::Time(0);

    ros::Subscriber map_sub = nh.subscribe("map", 10, mapCallback);
    ros::Subscriber est_pose_sub = nh.subscribe("amcl_pose", 100, poseCallback);
    ros::Subscriber goal_sub = nh.subscribe("move_base/result", 100, goalCallback);
    ros::Subscriber face_detection_sub = nh.subscribe("finale/faces", 10, faceCallback);
    ros::Subscriber cylinder_detection_sub = nh.subscribe("finale/cylinders", 10, cylinderCallback);
    ros::Subscriber ring_detection_sub = nh.subscribe("finale/rings", 10, ringCallback);
    ros::Subscriber qr_detection_sub = nh.subscribe("finale/qr_data", 100, qrCallback);
    ros::Subscriber num_detection_sub = nh.subscribe("finale/number_data", 100, numCallback);
    ros::Subscriber laser_scan_sub = nh.subscribe("scan", 100, laserCallback);
    
    conv_client = nh.serviceClient<finale::RecogniseSpeech>("finale/speech_service");
    ring_clr_client = nh.serviceClient<finale::BarvaRingov>("finale/prepoznaj_ring");
    cyl_clr_client = nh.serviceClient<finale::BarvaCilindrov>("finale/prepoznaj_cilinder");
    vacc_client = nh.serviceClient<finale::KateraCepiva>("finale/prepoznaj_cepivo");

    goal_pub = nh.advertise<geometry_msgs::PoseStamped>("move_base_simple/goal", 10);
    velocity_pub = nh.advertise<geometry_msgs::Twist>("mobile_base/commands/velocity", 10);
    goal_cancel_pub = nh.advertise<actionlib_msgs::GoalID>("move_base/cancel", 10);
    qrnum_pub = nh.advertise<finale::ToggleQRNumService>("finale/toggle_num", 100);
    marker_pub = nh.advertise<visualization_msgs::MarkerArray>("finale/markers", 100);
    wander_pub = nh.advertise<finale::WanderControl>("finale/wandering", 100);
    //
    debug_point_pub = nh.advertise<visualization_msgs::Marker>("debug/move_it", 100);
    //

    ROS_INFO("Waiting for goal subscriber");
    while(goal_pub.getNumSubscribers() < 1 && ros::ok()) {}
    
    ros::Rate rate(ROS_RATE);
    ROS_INFO("Waiting for map and pose");
    while(ros::ok()) {
      if((faces.size() + rings.size() + cylinders.size()) > 0) sendMarkers();
      ROS_INFO("STATE: %d", STATE);
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

void sendMarkers() {
  visualization_msgs::MarkerArray marr;
  for(std::list<data_t<facedata>>::iterator it = faces.begin(); it != faces.end(); ++it) {
    visualization_msgs::Marker m, m_text;
    geometry_msgs::Pose p; p.position.x = it->x; p.position.y = it->y; p.position.z = 0.2; p.orientation.w = 1.0;
    m.ns = "faces";
    m.id = it->id;
    m.header.frame_id = "map";
    m.header.stamp = ros::Time::now();
    m.pose = p;
    m.type = visualization_msgs::Marker::CUBE;
    m.action = visualization_msgs::Marker::ADD;
    m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.1;
    m.color.a = 1.0;
    marr.markers.push_back(m);
    m_text = m;
    m_text.ns = "face_text";
    m_text.pose.position.z = 1.0;
    m_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    m_text.color.r = 0.0; m_text.color.g = 0.0; m_text.color.b = 0.0;
    m_text.text = std::to_string(std::get<0>(it->data)) + " " + std::to_string(std::get<1>(it->data));
    marr.markers.push_back(m_text);
  }
  for(std::list<data_t<ringdata>>::iterator it = rings.begin(); it != rings.end(); ++it) {
    visualization_msgs::Marker m;
    geometry_msgs::Pose p; p.position.x = it->x; p.position.y = it->y; p.position.z = 0.2; p.orientation.w = 1.0;
    m.ns = "rings";
    m.id = it->id;
    m.header.frame_id = "map";
    m.header.stamp = ros::Time::now();
    m.pose = p;
    m.type = visualization_msgs::Marker::SPHERE;
    m.action = visualization_msgs::Marker::ADD;
    m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.1;
    m.color = clrToColour(std::get<0>(it->data));
    marr.markers.push_back(m);
  }
  for(std::list<data_t<cyldata>>::iterator it = cylinders.begin(); it != cylinders.end(); ++it) {
    visualization_msgs::Marker m;
    geometry_msgs::Pose p; p.position.x = it->x; p.position.y = it->y; p.position.z = 0.2; p.orientation.w = 1.0;
    m.ns = "cylinders";
    m.id = it->id;
    m.header.frame_id = "map";
    m.header.stamp = ros::Time::now();
    m.pose = p;
    m.type = visualization_msgs::Marker::CYLINDER;
    m.action = visualization_msgs::Marker::ADD;
    m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.3;
    m.color = clrToColour(std::get<0>(it->data));
    marr.markers.push_back(m);
  }
  marker_pub.publish(marr);
}