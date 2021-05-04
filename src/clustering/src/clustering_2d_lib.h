#include <geometry_msgs/Pose.h>
#include <ros/ros.h>
#include <math.h>
#include <iostream>

#include "clustering/Cluster2D.h"
#include "clustering/Cluster2DArray.h"

namespace clustering2d {
  double max_dist = 0.5, max_angle = 0.7854;
  bool ignore_angle = false;

  int max_id = 0;
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
    int status;

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

      // ROS_WARN("this: %s, b: %s", this->toString().c_str(), b.toString().c_str());

      this->id = this->id < b.id ? this->id : b.id;
      this->status = this->id < b.id ? this->status : b.status;
      this->x = (this->x * prev_detections + b.x * b.detections) / this->detections;
      this->y = (this->y * prev_detections + b.y * b.detections) / this->detections;
      this->sin = (this->sin * prev_detections + b.sin * b.detections) / this->detections;
      this->cos = (this->cos * prev_detections + b.cos * b.detections) / this->detections;

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
     * \param p object in which to store data
    */ 
    void toCluster2D(clustering::Cluster2D &c) {
      c.x = this->x;
      c.y = this->y;
      c.cos = this->cos;
      c.sin = this->sin;
      c.detections = this->detections;
      c.id = this->id;
      c.status = this->status;
    }

    std::string toString() {
      std::stringstream ret;
      ret << this->id << " " << this->x << " " << this->y << " " << this->cos << " " << this->sin << " " << this->detections; 
      return ret.str();
    }

    /**
     * @brief Get a cluster_t object
     * 
     * @param pose Pose of the cluster_t object
     * @param c pointer to new cluster_t object
     * @param status value in which to store some data
     * @return true if Pose is valid
     * @return false if Pose is invalid
    */
    static cluster_t *getCluster(geometry_msgs::Pose &pose, int status = 0) {
      if(std::isinf(pose.position.x) || std::isinf(pose.position.y) || std::isinf(pose.position.z) ||
          std::isinf(pose.orientation.x) || std::isinf(pose.orientation.y) || std::isinf(pose.orientation.z) || std::isinf(pose.orientation.w)) return NULL;
      return new cluster_t(max_id++, pose, status);
    }

    private:
    cluster_t(int id, geometry_msgs::Pose &pose, int status = 0) {
      this->x = pose.position.x;
      this->y = pose.position.y;
      this->detections = 1;
      this->id = id;
      this->sin = pose.orientation.z;
      this->cos = pose.orientation.w;
      this->status = status;
    }
  };

  /**
   * @brief Transforms list of cluster_t objects to clustering::Cluster2DArray message
   * 
   * @param l list of cluster_t objects
   * @param arr object in which to store clustering::Cluster2DArray
   */
  void to_cluster_array_msg(std::list<cluster_t> l, clustering::Cluster2DArray &arr) {
    int no = 0;
    for(cluster_t c : l) {
      clustering::Cluster2D c2;
      c.toCluster2D(c2);
      arr.clusters.push_back(c2);
      no++;
    }
    arr.no = no;
  }

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
  int cluster(std::list<cluster_t> &clusters) {
    int n = clusters.size();
    while(ros::ok() && n >= 2) {
      double min = -1.0;
      std::list<cluster_t>::iterator min_i, min_j;
      for(std::list<cluster_t>::iterator c_i = clusters.begin(); c_i != clusters.end(); ++c_i) 
        for(std::list<cluster_t>::iterator c_j = clusters.begin(); c_j != clusters.end(); ++c_j) {
          if(&*c_i != &*c_j && (ignore_angle || angle_diff(c_i->get_orientation(), c_j->get_orientation()) < max_angle)) { // two nearby clusters facing in different directions are different
            double dist = cluster_dist(*c_i, *c_j);
            if(dist < min || min < 0) {
              min_i = c_i; min_j = c_j;
              min = dist;
            }
          }
          if(&*c_i != &*c_j && cluster_dist(*c_i, *c_j) < max_dist && angle_diff(c_i->get_orientation(), c_j->get_orientation()) >= max_angle) ROS_INFO("ERROR Diff ang %f, %f, diff %f", c_i->get_orientation(), c_j->get_orientation(), angle_diff(c_i->get_orientation(), c_j->get_orientation()));
        }
      if(min > max_dist) {
        ROS_INFO("Not working %f", min);
        break;
      } else {
        if(min < 0) break;
        min_i->join(*min_j);
        clusters.erase(min_j);
        n--;
      }
    }

    return n;
  }
}