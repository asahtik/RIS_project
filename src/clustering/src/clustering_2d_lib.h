#include <geometry_msgs/Pose.h>
#include <ros/ros.h>
#include <math.h>
#include <iostream>

namespace clustering2d {
  double max_dist = 0.5, max_angle = 0.7854;

  /**
   * \brief sets value (distance) to be used to determine whether two clusters are to be joined
   * \param d max distance, default 0.5
  */ 
  void set_max_dist(double d) {
    max_dist = d;
  }

  /**
   * \brief sets value (angle difference) to be used to determine whether two clusters are to be joined
   * \param d max angle diff, default 0.7854
  */ 
  void set_max_diff(double d) {
    max_angle = d;
  }

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
    int status = 0;

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
      this->status = this->id < b.id ? this->status : b.status;
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
  };

  /**
   * \brief get distance between centers of clusters
   * \param a first cluster
   * \param b second cluster
   * \return std::tuple (vector<Pose>, vector<cluster_t>, int no_clusters)
  */ 
  double cluster_dist(cluster_t &a, cluster_t &b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
  }

  /**
   * \brief joins nearby poses
   * \param clusters of type std::list<cluster_t>. Clusters from previous iterations. Gets rewritten with clusters.
   * \param poses of type std::list<geometry_msgs::Pose>. Poses to be clustered.
   * \return number of clusters
  */
  int cluster(std::list<cluster_t> &clusters, std::list<geometry_msgs::Pose> &poses) {
    int max_id = -1;
    // load data to single vector
    for(cluster_t i : clusters) {
      if(i.id > max_id) max_id = i.id;
    }
    for(geometry_msgs::Pose i : poses) {
      if(std::isinf(i.position.x) || std::isinf(i.position.y) || std::isinf(i.position.z) ||
          std::isinf(i.orientation.x) || std::isinf(i.orientation.y) || std::isinf(i.orientation.z) || std::isinf(i.orientation.w)) continue;
      cluster_t c(++max_id, i);
      clusters.push_back(c);
    }

    int n = clusters.size();
    while(ros::ok()) {
      double min = -1.0;
      std::list<cluster_t>::iterator min_i, min_j;
      for(std::list<cluster_t>::iterator c_i = clusters.begin(); c_i != clusters.end(); ++c_i) 
        for(std::list<cluster_t>::iterator c_j = clusters.begin(); c_j != clusters.end(); ++c_j) {
          if(&*c_i != &*c_j && angle_diff(c_i->get_orientation(), c_j->get_orientation()) < max_angle) { // two nearby clusters facing in different directions are different
            double dist = cluster_dist(*c_i, *c_j);
            if(dist < min || min < 0) {
              min_i = c_i; min_j = c_j;
              min = dist;
            }
          }
          if(&*c_i != &*c_j && cluster_dist(*c_i, *c_j) < max_dist && angle_diff(c_i->get_orientation(), c_j->get_orientation()) >= max_angle) ROS_INFO("ERROR Diff ang %f, %f, diff %f", c_i->get_orientation(), c_j->get_orientation(), angle_diff(c_i->get_orientation(), c_j->get_orientation()));
        }

      if(min > max_dist || n < 2) {
        break;
      } else {
        if(min < 0) break;
        min_i->join(*min_j);
        clusters.erase(min_j);
        if(--n < 2) break;
      }
    }

    return n;
  }
}