#include <ros/ros.h>
#include "std_msgs/ColorRGBA.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"
#include "geometry_msgs/Vector3.h"
#include "move_base_msgs/MoveBaseActionResult.h"
#include "sound_play/sound_play.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>

const unsigned int N_GOALS = 25;

enum {TO_GOAL = 0, TO_FACE = 1, APPROACH = 2};

ros::NodeHandle *n;
sound_play::SoundClient *sc;
ros::Publisher marker_pub;
ros::Publisher goal_pub;
ros::Timer timer;

std::tuple<double, double, double, double> goals[N_GOALS] = { // pos x, y, orient z, w
  std::make_tuple(0.003934413194656372, 0.0822521448135376, -0.014871905840120438, 0.9998894070929457),
  std::make_tuple(-1.0047004222869873, 0.002727031707763672, 0.9991567485232693, 0.04105839597948799),
  std::make_tuple(-0.13222908973693848, -0.598783016204834, -0.748326562731764, -0.748326562731764),
  std::make_tuple(-0.051183462142944336, -1.3333165645599365, 0.9979036312988675, 0.06471740600900136),
  std::make_tuple(-0.26892685890197754, -1.6013619899749756, -0.7878396729025193, 0.615880385952379),
  std::make_tuple(-0.051183462142944336, -1.3333165645599365, 0.9979036312988675, 0.06471740600900136),
  std::make_tuple(0.42502129077911377, -2.046574354171753, 0.43818298476035084, 0.8988857946738896),
  std::make_tuple(1.3439583778381348, -1.954862117767334, -0.5804239569677728, 0.8143144541133192),
  std::make_tuple(2.9228553771972656, -2.0107920169830322, -0.534465281364818, 0.8451904300307866),
  std::make_tuple(2.640591859817505, -2.0018045902252197, -0.24842960576553522, 0.9686499527585705),
  std::make_tuple(2.8628547191619873, -1.8034391403198242, 0.4197893378911042, 0.9076215686027675),
  std::make_tuple(2.5169644355773926, -1.516801118850708, 0.9818941542090391, 0.18943038280623203),
  std::make_tuple(2.1053600311279297, -0.41121774911880493, 0.08512003568798963, 0.9963707038670273),
  std::make_tuple(1.1619327068328857, -0.41463005542755127, -0.7686498065681026, 0.6396698170640995),
  std::make_tuple(1.4824655055999756, 0.4113864600658417, 0.9474588230617326, 0.31987775571689336),
  std::make_tuple(1.4192171096801758, 0.3678373396396637, 0.993892570341084, 0.1103519760620229),
  std::make_tuple(1.3571312427520752, 0.9290643334388733, -0.13812570492010018, 0.9904147058885613),
  std::make_tuple(1.5366005897521973, 1.633037805557251, 0.4305347521808318, 0.90257400093543),
  std::make_tuple(2.165714979171753, 1.7576401233673096, -0.7173462157173455, 0.6967168770713135),
  std::make_tuple(2.8150665760040283, 0.8676133155822754, 0.07755756913994735, 0.9969878752869076),
  std::make_tuple(1.1273107528686523, 1.628039836883545, 0.7857051274901583, 0.6186012064615409),
  std::make_tuple(0.5483073592185974, 2.1121902465820312, -0.7660692722923166, 0.6427580182693333),
  std::make_tuple(-0.38890355825424194, 1.2998301982879639, 0.20957937392619566, 0.9777916373260225),
  std::make_tuple(-0.43698635697364807, 1.6141595840454102, 0.1791615812136009, 0.983819662243565),
  std::make_tuple(-0.8588401079177856, 1.618564486503601, 0.99325125304743619, 0.11598253454593041)
};

double min_dist, max_angle, approach_start = 0.5;
int min_det, max_unvisited = 1;

int goal_i = 0;
std::atomic<int> task_ind(TO_GOAL);
std::tuple<double, double> prev_goal = std::make_tuple(0.0, 0.0);

// gets angle from quaternions
double get_angle(double a_cos, double a_sin) {
  double arccos = (acos(a_cos) * 2.0);
  if(a_sin >= 0) return arccos;
  else return -arccos;
}

int cluster_id = 0;
class cluster {

  public:
  int id;
  double n_detected;
  double sum_x;
  double sum_y;
  double sum_sin_ang;
  double sum_cos_ang;
  bool visited = false;

  cluster(int i, const geometry_msgs::PoseArray::ConstPtr& markerarr) {
    sum_x = markerarr->poses[i].position.x;
    sum_y = markerarr->poses[i].position.y;
    sum_sin_ang = sin(get_angle(markerarr->poses[i].orientation.w, markerarr->poses[i].orientation.z));
    sum_cos_ang = cos(get_angle(markerarr->poses[i].orientation.w, markerarr->poses[i].orientation.z));
    n_detected = 1;
    id = cluster_id++;
  }

  double get_x(){
    return sum_x / n_detected;
  }

  double get_y() {
    return sum_y / n_detected;
  }

  double get_orientation() {
    return atan2(sum_sin_ang / n_detected, sum_cos_ang / n_detected);
  }

  cluster* join(const cluster &b) {
    this->n_detected += b.n_detected;
    this->sum_x += b.sum_x;
    this->sum_y += b.sum_y;
    this->sum_sin_ang += b.sum_sin_ang;
    this->sum_cos_ang += b.sum_cos_ang;
    this->visited = this->visited || b.visited;
    this->id = this->id < b.id ? this->id : b.id;
    return this;
  }

  std::string toString() {
    std::stringstream ss;
    ss << n_detected << ": x = " << sum_x/n_detected << ", y = " << sum_y/n_detected;
    return ss.str();
  }

};

// stores detected faces and is used for clustering
std::list<cluster> faces;

// distance between two clusters
double cluster_dist(cluster &a, cluster &b) {
  return sqrt(pow(a.get_x() - b.get_x(), 2) + pow(a.get_y() - b.get_y(), 2));
}

double angle_diff(double a, double b) {
  if(a <=0 && b <= 0 || a >= 0 && b >= 0) return abs(a - b);
  else {
    double d1 = abs(a) + abs(b);
    double d2 = M_PI - d1;
    return d1 < d2 ? d1 : d2;
  }
}

// minimal distance between clusters
std::tuple<std::list<cluster>::iterator, std::list<cluster>::iterator, double> min_cluster_dist(std::list<cluster> &clusters) {
  std::list<cluster>::iterator i_1, i_2; 
  int n = clusters.size();
  double min = -1.0;
  for(std::list<cluster>::iterator c_i = faces.begin(); c_i != faces.end(); ++c_i) 
    for(std::list<cluster>::iterator c_j = faces.begin(); c_j != faces.end(); ++c_j) {
      if(&*c_i != &*c_j && angle_diff(c_i->get_orientation(), c_j->get_orientation()) < max_angle) { // two nearby clusters facing in different directions are different
        double dist = cluster_dist(*c_i, *c_j);
        if(dist < min || min < 0) {
          min = dist;
          i_1 = c_i; i_2 = c_j;
        }
      }
      if(&*c_i != &*c_j && cluster_dist(*c_i, *c_j) < min_dist && angle_diff(c_i->get_orientation(), c_j->get_orientation()) >= max_angle) ROS_INFO("ERROR Diff ang %f, %f, diff %f", c_i->get_orientation(), c_j->get_orientation(), angle_diff(c_i->get_orientation(), c_j->get_orientation()));
    }

  return std::make_tuple(i_1, i_2, min);
}

// filters and transforms cluster list to MarkerArray, could optimise min_detections criteria (e.g. by using limit n-times smallest value)
int to_markers(std::list<cluster> &clusters, int min_detections, visualization_msgs::MarkerArray &ret) {
  int no_faces = 0;
  for(std::list<cluster>::iterator c = clusters.begin(); c != clusters.end(); ++c) {
    if(c->n_detected - min_detections > -0.01) {
      visualization_msgs::Marker mark; geometry_msgs::Pose p;
      mark.header.stamp = ros::Time(0);
      mark.header.frame_id = "map";
      mark.id = c->id;
      mark.type = visualization_msgs::Marker::ARROW;
      mark.action = visualization_msgs::Marker::ADD;
      mark.frame_locked = false;
      geometry_msgs::Vector3 scale; scale.x = 0.2; scale.y = 0.05; scale.z = 0.05;
      mark.scale = scale;
      // mark.lifetime = ros::Duration(1.0);

      p.position.x = c->get_x();
      p.position.y = c->get_y();
      p.position.z = 0.3; // fixed value because irrelevant
      p.orientation.x = 0;
      p.orientation.y = 0;
      p.orientation.z = sin(c->get_orientation() / 2.0);
      p.orientation.w = cos(c->get_orientation() / 2.0);

      if(isnan(p.position.x) || isnan(p.position.y) || isnan(p.orientation.z) || isnan(p.orientation.w)) {
        std::list<cluster>::iterator temp = c;
        c--;
        clusters.erase(temp);
        continue;
      }

      mark.pose = p;
      mark.id = no_faces;

      std_msgs::ColorRGBA clr; clr.a = 1; clr.r = 0;
      if(c->visited) {
        clr.b = 1; clr.g = 0;
      } else {
        clr.b = 0; clr.g = 1;
      }
      mark.color = clr;

      ret.markers.push_back(mark);
      no_faces++;
    }
  }
  return no_faces;
}

std::tuple<int, cluster*> nearest() {
  double min_d = -1;
  cluster* ret = NULL;
  int unvisited = 0;
  for(std::list<cluster>::iterator i = faces.begin(); i != faces.end(); ++i) {
    double dist = sqrt(pow(i->get_x() - std::get<0>(prev_goal), 2) + pow(i->get_y() - std::get<1>(prev_goal), 2));
    if(!(i->visited) && i->n_detected - min_det > -0.01 && (dist < min_d || min_d < 0)) {
      min_d = dist;
      ret = &*i;
      unvisited++;
    }
  }
  double dist = sqrt(pow(std::get<0>(goals[goal_i+1]) - std::get<0>(prev_goal), 2) + pow(std::get<1>(goals[goal_i+1]) - std::get<1>(prev_goal), 2));
  if((min_dist < 0 || min_d < dist) && unvisited <= max_unvisited) {
    ret = NULL;
  }
  
  if(ret == NULL) return std::make_tuple(++goal_i, ret);
  else return std::make_tuple(-1, ret);
}

// returns aproach start point and orientation TODO
std::tuple<double, double, double> get_approach_point(cluster &c) {
  double p_x = c.get_x();
  double p_y = c.get_y();
  double p_angle = c.get_orientation();
  double dir;
  if(p_angle < 0) dir = M_PI + p_angle;
  else dir = -M_PI + p_angle;
  // ROS_INFO("angle %f", p_angle * (180 / M_PI));
  double a_x_start = p_x - approach_start * cos(dir);
  double a_y_start = p_y - approach_start * sin(dir);
  // ROS_INFO("dir %f", dir * (180 / M_PI));

  //
  visualization_msgs::Marker mark; geometry_msgs::Pose p;
  mark.header.stamp = ros::Time(0);
  mark.header.frame_id = "map";
  mark.id = c.id;
  mark.type = visualization_msgs::Marker::CUBE;
  mark.action = visualization_msgs::Marker::ADD;
  mark.frame_locked = false;
  geometry_msgs::Vector3 scale; scale.x = 0.1; scale.y = 0.1; scale.z = 0.1;
  mark.scale = scale;
  //mark.lifetime = ros::Duration(10.0);
  mark.id = 12094052;
  p.position.x = a_x_start;
  p.position.y = a_y_start;
  p.position.z = 0.3; // fixed value because irrelevant
  mark.pose = p;
  std_msgs::ColorRGBA clr; clr.a = 1; clr.r = 1;
  mark.color = clr;
  visualization_msgs::MarkerArray marr;
  marr.markers.push_back(mark);
  ROS_INFO("Publishing approach marker");
  marker_pub.publish(marr);
  //

  return std::make_tuple(a_x_start, a_y_start, dir);
}

// find nearest goal and go
void goNext() {
  geometry_msgs::PoseStamped goal;
  goal.header.frame_id = "map";
  goal.header.stamp = ros::Time(0);
  
  std::tuple<int, cluster*> n = nearest();
  if(std::get<0>(n) >= 0) { // go to goal
    ROS_INFO("To Goal %d", goal_i);
    goal_i %= N_GOALS;
    goal.pose.position.x = std::get<0>(goals[goal_i]);
    goal.pose.position.y = std::get<1>(goals[goal_i]);
    goal.pose.orientation.z = std::get<2>(goals[goal_i]);
    goal.pose.orientation.w = std::get<3>(goals[goal_i]);
  } else { // go to face
    ROS_INFO("To Face");
    std::tuple<double, double, double> a_p = get_approach_point(*(std::get<1>(n)));
    std::get<1>(n)->visited = true;
    goal.pose.position.x = std::get<0>(a_p);
    goal.pose.position.y = std::get<1>(a_p);
    goal.pose.orientation.z = sin(std::get<2>(a_p)/2.0);
    goal.pose.orientation.w = cos(std::get<2>(a_p)/2.0);

    task_ind = TO_FACE;
  }
  goal_pub.publish(goal);
  prev_goal = std::make_tuple(goal.pose.position.x, goal.pose.position.y);
}

void cluster_markers(const geometry_msgs::PoseArray::ConstPtr& markerarr) {
  // join closest clusters until smallest distance between clusters > min_dist
  // ROS_INFO("Got markers");
  int n = markerarr->poses.size();
  if(n > 0) {
    for(int i = 0; i < n; i++) {
      cluster c(i, markerarr);
      faces.push_back(c);
    }

    while(true) {
      std::tuple<std::list<cluster>::iterator, std::list<cluster>::iterator, double> d = min_cluster_dist(faces);
      if(std::get<2>(d) > min_dist || faces.size() < 2) {
        break;
      }
      else {
        std::list<cluster>::iterator a = std::get<0>(d);
        std::list<cluster>::iterator b = std::get<1>(d);
        if(std::get<2>(d) < 0) break;
        a->join(*b);
        faces.erase(b);
      }
    }
  }

  // filter by # detections and transform to marker array
  visualization_msgs::MarkerArray marr;
  int no = to_markers(faces, min_det, marr);
  // std::cout << "No markers " << no << " ";
  marker_pub.publish(marr);
}

// ends approach procedure and moves on
void proceed(const ros::TimerEvent& time) {
  ROS_INFO("Proceed");
  goNext();
}

// begins approach procedure
void approach() {
  ROS_INFO("Approach");
  task_ind = APPROACH;
  std::string s = "Stop right there";
  sc->say(s);
  timer = n->createTimer(ros::Duration(5), proceed, true);
}

// decides the next task
void next_goal(move_base_msgs::MoveBaseActionResult msg) {
  int ti = task_ind;
  if(msg.status.status < 3) ROS_INFO("Goal pending");
  else if(msg.status.status == 3) {
    if(ti == TO_FACE) approach();
    else goNext();
  } else {
    task_ind = TO_GOAL;
    ROS_INFO("Could not reach goal");
    goNext();
  }
}

int main(int argc, char **argv) {

	ros::init(argc, argv, "face_clustering");
  ros::NodeHandle nh;
  sound_play::SoundClient s;
  n = &nh;
  sc = &s;

  ROS_INFO("Initialized clustering node");

  marker_pub = nh.advertise<visualization_msgs::MarkerArray>("face_markers/faces", 1000);
  goal_pub = nh.advertise<geometry_msgs::PoseStamped>("move_base_simple/goal", 100);

  while(goal_pub.getNumSubscribers() < 1 && ros::ok()) {}

  if(!nh.getParam("/face_clustering/min_dist", min_dist)) min_dist = 0.5;
  if(!nh.getParam("/face_clustering/max_angle", max_angle)) max_angle = 0.7854;
  if(!nh.getParam("/face_clustering/min_det", min_det)) min_det = 2;

	ros::Subscriber sub = nh.subscribe("face_markers/processing", 1000, cluster_markers);
  ros::Subscriber goal_sub = nh.subscribe<move_base_msgs::MoveBaseActionResult>("/move_base/result", 100, next_goal);

  goNext();

	ros::spin();
	
}
