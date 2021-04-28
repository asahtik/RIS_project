#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <clustering/Cluster2D.h>
#include <clustering/Clustering2DService.h>

#include <math.h>

double angle_diff(double a, double b) {
  if(a <=0 && b <= 0 || a >= 0 && b >= 0) return abs(a - b);
  else {
    double d1 = abs(a) + abs(b);
    double d2 = M_PI - d1;
    return d1 < d2 ? d1 : d2;
  }
}

struct cluster_t {
  public:
  double x, y, cos, sin;
  int detections, id;

  cluster_t(int id) {
    this->id = id;
  }
  cluster_t(clustering::Cluster2D &c) {
    this->x = c.x;
    this->y = c.y;
    this->detections = c.detections;
    this->id = c.id;
    this->sin = c.sin;
    this->cos = c.cos;
  }
  cluster_t(int id, geometry_msgs::Pose &pose) {
    this->x = pose.position.x;
    this->y = pose.position.y;
    this->detections = 1;
    this->id = id;
    this->sin = pose.orientation.z;
    this->cos = pose.orientation.w;
  }

  /**
   * \brief gets orientation in rad
  */ 
  double get_orientation() {
    return atan2(sin / detections, cos / detections);
  }

  /**
   * \brief joins two clusters
   * \param b second join source
  */ 
  cluster_t* join(const cluster_t &b) {
    int prev_detections = this->detections;
    this->detections += b.detections;

    this->id = this->id < b.id ? this->id : b.id;
    this->x = (this->x * prev_detections + b.x) / this->detections;
    this->y = (this->y * prev_detections + b.y) / this->detections;
    this->sin = (this->sin * prev_detections + b.sin) / this->detections;
    this->cos = (this->cos * prev_detections + b.cos) / this->detections;

    return this;
  }

  /**
   * \brief transforms cluster class to geometry_msgs::Pose type
   * \param p object in which to store data
  */ 
  void toPose(geometry_msgs::Pose &p) {
    p.position.x = this->x;
    p.position.y = this->y;
    p.orientation.z = this->sin;
    p.orientation.w = this->cos;
  }

  /**
   * \brief transforms cluster class to clustering::Cluster2D type
   * \param c object in which to store data
  */ 
  void toCluster2D(clustering::Cluster2D &c) {
    c.x = this->x;
    c.y = this->y;
    c.cos = this->cos;
    c.sin = this->sin;
    c.detections = this->detections;
    c.id = this->id;
  }
};

/**
 * \brief get distance between centers of clusters
 * \param a first cluster
 * \param b second cluster
*/ 
double cluster_dist(cluster_t &a, cluster_t &b) {
  return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

/**
 * \brief joins nearby poses
*/
bool cluster(clustering::Clustering2DService::Request &req, clustering::Clustering2DService::Response &res) {
  std::list<cluster_t> ret; 
  
  double max_dist = req.max_dist > 0.001 ? req.max_dist : 0.5;
  double max_angle = req.max_angle > 0.001 ? req.max_angle : 0.7854;
  
  int max_id = -1;
  // load data to single vector
  for(clustering::Cluster2D i : req.clusters) {
    if(i.id > max_id) max_id = i.id;
    ret.push_back(i);
  }
  for(geometry_msgs::Pose i : req.poses) {
    cluster_t c(++max_id, i);
    ret.push_back(c);
  }

  while(ros::ok()) {
    int n = ret.size();
    double min = -1.0;
    std::list<cluster_t>::iterator c_i, c_j;
    for(c_i = ret.begin(); c_i != ret.end(); ++c_i) 
      for(c_j = ret.begin(); c_j != ret.end(); ++c_j) {
        if(&*c_i != &*c_j && angle_diff(c_i->get_orientation(), c_j->get_orientation()) < max_angle) { // two nearby clusters facing in different directions are different
          double dist = cluster_dist(*c_i, *c_j);
          if(dist < min || min < 0) {
            min = dist;
          }
        }
        if(&*c_i != &*c_j && cluster_dist(*c_i, *c_j) < max_dist && angle_diff(c_i->get_orientation(), c_j->get_orientation()) >= max_angle) ROS_INFO("ERROR Diff ang %f, %f, diff %f", c_i->get_orientation(), c_j->get_orientation(), angle_diff(c_i->get_orientation(), c_j->get_orientation()));
      }

    if(min > max_dist || ret.size() < 2) {
      break;
    }
    else {
      if(min < 0) break;
      c_i->join(*c_j);
      ret.erase(c_j);
    }
  }

  std::vector<geometry_msgs::Pose> p_a;
  std::vector<clustering::Cluster2D> c_a;
  int no = 0;
  for(cluster_t i : ret) {
    geometry_msgs::Pose p;
    clustering::Cluster2D c;
    i.toPose(p);
    i.toCluster2D(c);
    p_a.push_back(p);
    c_a.push_back(c);
    no++;
  }

  res.clustered_poses = p_a;
  res.data = c_a;
  res.no = no;

  return true;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "clustering_2d");
  ros::NodeHandle nh;

  ros::ServiceServer service = nh.advertiseService("clustering/clustering_2d", cluster);
  
  ros::spin();
  return 0;
}