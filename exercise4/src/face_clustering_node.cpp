#include <ros/ros.h>
#include "std_msgs/ColorRGBA.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseArray.h"
#include "geometry_msgs/Vector3.h"

#include <iostream>
#include <vector>
#include <math.h>

ros::Publisher marker_pub;

double min_dist, max_angle;
int min_det;

// gets angle from quaternions
double get_angle(double a_cos, double a_sin) {
  double arccos = acos(a_cos);

  if(a_sin >= 0) return arccos;
  else return -arccos;
}

class cluster {

  public:
  int n_detected;
  double sum_x;
  double sum_y;
  double sum_ang;
  bool visited = false;

  cluster(int i, const geometry_msgs::PoseArray::ConstPtr& markerarr) {
    sum_x = markerarr->poses[i].position.x;
    sum_y = markerarr->poses[i].position.y;
    sum_ang = get_angle(markerarr->poses[i].orientation.w, markerarr->poses[i].orientation.z);
    n_detected = 1;
  }

  double get_x(){
    return sum_x / (double)n_detected;
  }

  double get_y() {
    return sum_y / (double)n_detected;
  }

  double get_orientation() {
    return sum_ang / (double)n_detected;
  }

  cluster* join(const cluster &b) {
    this->n_detected += b.n_detected;
    this->sum_x += b.sum_x;
    this->sum_y += b.sum_y;
    this->sum_ang += b.sum_ang;
    this->visited = this->visited || b.visited;
    return this;
  }

};

// stores detected faces and is used for clustering
std::list<cluster> faces;

// distance between two clusters
double cluster_dist(cluster &a, cluster &b) {
  return sqrt(pow(a.get_x() - b.get_x(), 2) + pow(a.get_y() - b.get_y(), 2));
}

// minimal distance between clusters
std::tuple<std::list<cluster>::iterator, std::list<cluster>::iterator, double> min_cluster_dist(std::list<cluster> &clusters) {
  std::list<cluster>::iterator i_1, i_2; 
  int n = clusters.size();
  double min = -1.0;

  for(std::list<cluster>::iterator c_i = faces.begin(); c_i != faces.end(); ++c_i) 
    for(std::list<cluster>::iterator c_j = faces.begin(); c_j != faces.end(); ++c_j)
      if(&*c_i != &*c_j && abs(c_i->get_orientation() - c_j->get_orientation()) < max_angle) { // two nearby clusters facing in different directions are different
        double dist = cluster_dist(*c_i, *c_j);
        if(dist < min || min < 0) {
          min = dist;
          i_1 = c_i; i_2 = c_j;
        }
      }
    
  return std::make_tuple(i_1, i_2, min);
}

// filters and transforms cluster list to MarkerArray, could optimise min_detections criteria (e.g. by using limit n-times smallest value)
int to_markers(std::list<cluster> &clusters, int min_detections, visualization_msgs::MarkerArray &ret) {
  int no_faces = 0;
  for(cluster c : clusters) {
    if(c.n_detected > min_detections) {
      visualization_msgs::Marker mark; geometry_msgs::Pose p;
      mark.header.stamp = ros::Time(0);
      mark.header.frame_id = "map";
      mark.type = visualization_msgs::Marker::CUBE;
      mark.action = visualization_msgs::Marker::ADD;
      mark.frame_locked = false;
      geometry_msgs::Vector3 scale; scale.x = 0.1; scale.y = 0.1; scale.z = 0.1;
      mark.scale = scale;

      p.position.x = c.get_x();
      p.position.y = c.get_y();
      p.position.z = 0.3; // fixed value because irrelevant

      mark.pose = p;
      mark.id = no_faces;

      std_msgs::ColorRGBA clr; clr.a = 1; clr.r = 0;
      if(c.visited) {
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

void cluster_markers(const geometry_msgs::PoseArray::ConstPtr& markerarr) {
  // join closest clusters until smallest distance between clusters > min_dist
  ROS_INFO("Got markers");
  int n = markerarr->poses.size();
  if(n > 0) {
    for(int i = 0; i < n; i++) {
      cluster c(i, markerarr);
      faces.push_back(c);
    }

    while(true) {
      std::tuple<std::list<cluster>::iterator, std::list<cluster>::iterator, double> d = min_cluster_dist(faces);
      if(std::get<2>(d) > min_dist || faces.size() < 2) break;
      else {
        std::list<cluster>::iterator a = std::get<0>(d);
        std::list<cluster>::iterator b = std::get<1>(d);
        a->join(*b);
        faces.erase(b);
      }
    }
  }

  // filter by # detections and transform to marker array
  visualization_msgs::MarkerArray marr;
  int no = to_markers(faces, min_det, marr);
  marker_pub.publish(marr);

  // TODO go to face
}

int main(int argc, char **argv) {

	ros::init(argc, argv, "face_clustering");
	ros::NodeHandle nh;

  ROS_INFO("Initialized clustering node");

  marker_pub = nh.advertise<visualization_msgs::MarkerArray>("face_markers/faces", 1000);

  if(!nh.getParam("/face_clustering/min_dist", min_dist)) min_dist = 0.5;
  if(!nh.getParam("/face_clustering/max_angle", max_angle)) max_angle = 0.7854;
  if(!nh.getParam("/face_clustering/min_det", min_det)) min_det = 2;

	ros::Subscriber sub=nh.subscribe("face_markers/processing", 1000, cluster_markers);

	ros::spin();
	
}
