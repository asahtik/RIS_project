#include <geometry_msgs/Pose.h>
#include <ros/ros.h>
#include <math.h>
#include <iostream>

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

  template<typename T>
  struct cluster_t {
    public:
    double x, y, cos, sin;
    int detections, id;
    int status;
    T data;
    T (*joinf)(const cluster_t<T>&, const cluster_t<T>&);

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
    cluster_t<T>* join(const cluster_t<T> &b) {
      int prev_detections = this->detections;
      this->data = joinf(*this, b);
      
      this->detections += b.detections;

      // ROS_WARN("this: %s, b: %s", this->toString().c_str(), b.toString().c_str());
      this->id = this->id <= b.id ? this->id : b.id;
      this->status = this->id <= b.id ? b.status : this->status;
      this->x = (this->x * prev_detections + b.x * b.detections) / this->detections;
      this->y = (this->y * prev_detections + b.y * b.detections) / this->detections;
      this->sin = (this->sin * prev_detections + b.sin * b.detections) / this->detections;
      this->cos = (this->cos * prev_detections + b.cos * b.detections) / this->detections;
      return this;
    }

    /**
     * @brief Get the closest cluster
     * 
     * @param cs std::list<cluster_t> containing all clusters
     * @return pointer to the closest cluster
     */
    double get_closest(std::list<cluster_t<T>> &cs, cluster_t<T> *closest = NULL, bool save_closest = false) {
      if(cs.size() == 1) return -1.0;
      double min_d = -1.0;
      cluster_t<T> *min = NULL;
      for(typename std::list<cluster_t<T>>::iterator it = cs.begin(); it != cs.end(); ++it) {
        if(this->id != it->id) {
          double dist = sqrt(pow(this->x - it->x, 2) + pow(this->y - it->y, 2));
          if(min_d < 0.0 || dist < min_d) {
            min_d = dist;
            min = &*it;
          }
        }
      }
      if(save_closest) closest = min;
      return min_d;
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
    static cluster_t<T> *getCluster(geometry_msgs::Pose &pose, int status, T data, T (*joinf)(const cluster_t<T>&, const cluster_t<T>&)) {
      if(std::isinf(pose.position.x) || std::isnan(pose.position.x) || std::isinf(pose.position.y) || std::isnan(pose.position.y) || 
          std::isinf(pose.position.z) || std::isnan(pose.position.z) || std::isinf(pose.orientation.x) || std::isnan(pose.orientation.x) || 
          std::isinf(pose.orientation.y) || std::isnan(pose.orientation.y) || std::isinf(pose.orientation.z) || std::isnan(pose.orientation.z) || 
          std::isinf(pose.orientation.w) || std::isnan(pose.orientation.w)) return NULL;
      return new cluster_t<T>(max_id++, pose, status, data, joinf);
    }

    private:
    cluster_t(int id, geometry_msgs::Pose &pose, int status, T data, T (*joinf)(const cluster_t<T>&, const cluster_t<T>&)) {
      this->x = pose.position.x;
      this->y = pose.position.y;
      this->detections = 1;
      this->id = id;
      this->sin = pose.orientation.z;
      this->cos = pose.orientation.w;
      this->status = status;
      this->data = data;
      this->joinf = joinf;
    }
  };

  /**
   * @brief Finds cluster by id.
   * 
   * @param cs list of type std::list<cluster_t> holding all clusters.
   * @param id id to look for.
   * @return cluster_t*. NULL if not found.
   */
  template<typename T>
  cluster_t<T> *find_by_id(std::list<cluster_t<T>> &cs, int id) {
    for(typename std::list<cluster_t<T>>::iterator i = cs.begin(); i != cs.end(); ++i)
      if(i->id == id) return &*i;
    
    return NULL;
  }

  /**
   * @brief Finds cluster id to which cluster was joined.
   * 
   * @param joins of type std::vector<std::list<int>> where joins are stored.
   * @param id original id of cluster.
   * @return int cluster id. If not found equals original id.
   */
  int clustered_id(std::vector<int> &joins, int id) {
    if(id == 0) return 0;
    for(int i = 0; i < id; i++) {
      if(joins[i] == id) return clustered_id(joins, i);
    }
    return id;
  }

  /**
   * \brief get distance between centers of clusters
   * \param a first cluster
   * \param b second cluster
   * \return std::tuple (vector<Pose>, vector<cluster_t>, int no_clusters)
   */ 
  template<typename T>
  double cluster_dist(cluster_t<T> &a, cluster_t<T> &b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
  }

  /**
   * \brief joins nearby poses
   * \param clusters of type std::list<cluster_t>. Clusters from previous iterations. Gets rewritten with clusters.
   * \param poses of type std::list<geometry_msgs::Pose>. Poses to be clustered.
   * \param joins of type std::vector<int>*. Optional. Stores vector of joined clusters for each output cluster id.
   * \return number of clusters
   */
  template<typename T>
  int cluster(std::list<cluster_t<T>> &clusters, std::vector<int> *joins = NULL) {
    if(joins != NULL) joins->clear();
    int n = clusters.size();
    joins->resize(n, -1);
    while(ros::ok() && n >= 2) {
      double min = -1.0;
      typename std::list<cluster_t<T>>::iterator min_i, min_j;
      for(typename std::list<cluster_t<T>>::iterator c_i = clusters.begin(); c_i != clusters.end(); ++c_i) 
        for(typename std::list<cluster_t<T>>::iterator c_j = clusters.begin(); c_j != clusters.end(); ++c_j) {
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
        break;
      } else {
        if(min < 0) break;
        if(joins != NULL) (*joins)[(min_i->id < min_j->id) ? min_i->id : min_j->id] = ((min_i->id < min_j->id) ? min_j->id : min_i->id);
        min_i->join(*min_j);
        clusters.erase(min_j);
        n--;
      }
    }
    int max_id_temp = 0;
    for(cluster_t<T> c : clusters) if(c.id >= max_id_temp) max_id_temp = c.id + 1;
    max_id = max_id_temp;
    return n;
  }
}