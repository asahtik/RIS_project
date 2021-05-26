#include <ros/ros.h>
#include "std_msgs/ColorRGBA.h"
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

#include "finale/FaceCluster.h"
#include "finale/ClusterInCamera.h"
#include "finale/FaceDetectorToClustering.h"
#include "finale/FaceClusteringToHub.h"
#include "include/clustering2d_lib.cpp"

#define SAFETY_DISTANCE 1.0

// approach angle, approach Pose
typedef std::tuple<float, geometry_msgs::Pose> approachp;

ros::NodeHandle *n;
sound_play::SoundClient *sc;
ros::Publisher marker_pub;

double min_dist, max_angle;
int min_det, max_unvisited = 1;

approachp joinf(const clustering2d::cluster_t<approachp> &a, const clustering2d::cluster_t<approachp> &b) {
  if(abs(std::get<0>(a.data)) < abs(std::get<0>(b.data))) return a.data;
  else return b.data;
}

int toHubMsg(ros::Time stamp, std::list<clustering2d::cluster_t<approachp>>& fs, std::vector<std::tuple<int, geometry_msgs::Pose>>& cs, finale::FaceClusteringToHub& out, int mind) {
  out.stamp = stamp;
  for(std::tuple<int, geometry_msgs::Pose> c : cs) {
    finale::ClusterInCamera t;
    t.id = std::get<0>(c);
    t.pose = std::get<1>(c);
    out.inCamera.push_back(t);
  }
  int no = 0;
  for(clustering2d::cluster_t<approachp> f : fs) {
    finale::FaceCluster t;
    t.id = f.id;
    t.x = f.x;
    t.y = f.y;
    t.cos = f.cos;
    t.sin = f.sin;
    t.status = f.status;
    t.detections = f.detections;
    t.approach = std::get<1>(f.data);
    if(t.detections >= mind) {
      out.faces.push_back(t);
      no++;
    }
  }
  return no;
}

geometry_msgs::Pose get_approach_point(geometry_msgs::Pose &p, geometry_msgs::Pose &c) {
  double c_x = c.position.x, c_y = c.position.y;
  double d_x = c_x * c.orientation.w, d_y = c_y * c.orientation.z;

  double c_cos = c.orientation.w, c_sin = c.orientation.z;
  double p_cos = -p.orientation.w, p_sin = -p.orientation.z;

  double vcos = p_cos * c_cos - p_sin * c_sin, vsin = p_sin * c_cos + p_cos * c_sin;
  vcos = vcos / (vcos + vsin);
  vsin = vsin / (vcos + vsin);

  geometry_msgs::Pose ret;
  ret.orientation.w = vcos;
  ret.orientation.z = vsin;
  ret.position.x = p.position.x + d_x;
  ret.position.y = p.position.y + d_y;

  return ret;
}

std::list<clustering2d::cluster_t<approachp>> faces;

void cluster_markers(const finale::FaceDetectorToClustering::ConstPtr &posearr) {
  std::vector<int> new_clusters;
  std::vector<geometry_msgs::Pose> fcs = posearr->faces.poses;
  std::vector<geometry_msgs::Pose> inCamera = posearr->inCamera.poses;
  std::vector<float> angls = posearr->angles;
  
  assert(fcs.size() == inCamera.size() && inCamera.size() == angls.size());

  int size = fcs.size();
  new_clusters.resize(size);

  for(int i = 0; i < size; i++) {
    geometry_msgs::Pose aprch;
    clustering2d::cluster_t<approachp> *cluster = clustering2d::cluster_t<approachp>::getCluster(fcs[i], 0, approachp(angls[i], aprch), &joinf);
    if(cluster != NULL) {
      new_clusters[i] = cluster->id;
      faces.push_front(*cluster);
    }
  }
  std::vector<int> joins;
  int no_markers = clustering2d::cluster(faces, &joins);

  std::list<int> new2_clusters;
  // id, pose in camera
  std::vector<std::tuple<int, geometry_msgs::Pose>> cam_pose(size);
  std::string s = "Stay two meters apart";
  for(int i = 0; i < size; i++) {
    int cl = new_clusters[i];
    int ncl = clustering2d::clustered_id(joins, cl);
    cam_pose[i] = std::tuple<int, geometry_msgs::Pose>(ncl, inCamera[i]);
    bool isNew = true;
    for(int i : new2_clusters) if(i == ncl) {isNew = false; break;}
    if(isNew) {
      new2_clusters.push_back(ncl);
      clustering2d::cluster_t<approachp> *nearestcl = clustering2d::find_by_id(faces, ncl);
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
  // filter by # detections and transform to msg
  finale::FaceClusteringToHub fcl2hub;
  int no = toHubMsg(posearr->faces.header.stamp, faces, cam_pose, fcl2hub, min_det);
  // std::cout << "No markers " << no << " ";
  marker_pub.publish(fcl2hub);
}

int main(int argc, char **argv) {

	ros::init(argc, argv, "face_clustering");
  ros::NodeHandle nh;
  sound_play::SoundClient s;
  n = &nh;
  sc = &s;

  ROS_INFO("Initialized clustering node");

  marker_pub = nh.advertise<finale::FaceClusteringToHub>("finale/faces", 1000);

  if(!nh.getParam("/face_clustering/min_dist", min_dist)) min_dist = 0.5;
  if(!nh.getParam("/face_clustering/max_angle", max_angle)) max_angle = 0.7854;
  if(!nh.getParam("/face_clustering/min_det", min_det)) min_det = 2;

	ros::Subscriber sub = nh.subscribe("finale/face_clustering", 1000, cluster_markers);

	ros::spin();
	
}
