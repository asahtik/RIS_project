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

#include "include/clustering_2d_lib.hpp"

#define SAFETY_DISTANCE 2.0

ros::NodeHandle *n;
sound_play::SoundClient *sc;
ros::Publisher marker_pub;

double min_dist, max_angle;
int min_det, max_unvisited = 1;

// filters and transforms cluster list to MarkerArray, could optimise min_detections criteria (e.g. by using limit n-times smallest value)
int to_markers(std::list<clustering2d::cluster_t> &clusters, int min_detections, visualization_msgs::MarkerArray &ret) {
  int no_faces = 0;
  for(std::list<clustering2d::cluster_t>::iterator c = clusters.begin(); c != clusters.end(); ++c) {
    if(c->detections - min_detections > -0.01) {
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
      c->toPose(p);
      if(isnan(p.position.x) || isnan(p.position.y) || isnan(p.orientation.z) || isnan(p.orientation.w)) {
        std::list<clustering2d::cluster_t>::iterator temp = c;
        c--;
        clusters.erase(temp);
        continue;
      }
      mark.pose = p;
      mark.id = no_faces;
      std_msgs::ColorRGBA clr; clr.a = 1; clr.r = 0;
      if(c->status) {
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

std::list<clustering2d::cluster_t> faces;

void cluster_markers(const geometry_msgs::PoseArray::ConstPtr& posearr) {
  std::list<int> new_clusters;
  for(geometry_msgs::Pose p : posearr->poses) {
    clustering2d::cluster_t *cluster = clustering2d::cluster_t::getCluster(p);
    if(cluster != NULL) {
      new_clusters.push_back(cluster->id);
      faces.push_front(*cluster);
    }
  }
  std::vector<int> joins;
  int no_markers = clustering2d::cluster(faces, &joins);

  std::list<int> new2_clusters;
  std::string s = "Stay two meters apart";
  for(int cl : new_clusters) {
    int ncl = clustering2d::clustered_id(joins, cl);
    bool isNew = true;
    for(int i : new2_clusters) if(i == ncl) {isNew = false; break;}
    if(isNew) {
      new2_clusters.push_back(ncl);
      clustering2d::cluster_t *nearestcl = clustering2d::find_by_id(faces, ncl);
      double dist = nearestcl->get_closest(faces);
      if(dist >= 0.0) {
        if(dist < SAFETY_DISTANCE) {
          if(nearestcl-> status < 1) {
            ROS_INFO("Saying: %s", s.c_str());
            sc->stopSaying(s);
            sc->say(s);
          }
          nearestcl->status = 1;
        }
      }
    } else {}
  }
  // filter by # detections and transform to marker array
  visualization_msgs::MarkerArray marr;
  int no = to_markers(faces, min_det, marr);
  // std::cout << "No markers " << no << " ";
  marker_pub.publish(marr);
}

int main(int argc, char **argv) {

	ros::init(argc, argv, "face_clustering");
  ros::NodeHandle nh;
  sound_play::SoundClient s;
  n = &nh;
  sc = &s;

  ROS_INFO("Initialized clustering node");

  marker_pub = nh.advertise<visualization_msgs::MarkerArray>("face_markers/faces", 1000);

  if(!nh.getParam("/face_clustering/min_dist", min_dist)) min_dist = 0.5;
  if(!nh.getParam("/face_clustering/max_angle", max_angle)) max_angle = 0.7854;
  if(!nh.getParam("/face_clustering/min_det", min_det)) min_det = 2;

	ros::Subscriber sub = nh.subscribe("finale/face_clustering", 1000, cluster_markers);

	ros::spin();
	
}
