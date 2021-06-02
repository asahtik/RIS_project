#include "geometry_msgs/Pose.h"
#include "finale/FaceCluster.h"
#include "finale/CylCluster.h"
#include "finale/RingCluster.h"

#include <math.h>
#include <iostream>

typedef std::tuple<int, int> facedata;

template<typename T>
struct data_t {
  public:
  double x, y, cos, sin;
  int detections, id;
  int status;
  T data;
  geometry_msgs::Pose approach;

  data_t(int id, double x, double y, double cos, double sin, int detections, int status, T data, geometry_msgs::Pose approach) {
    this->x = x;
    this->y = y;
    this->cos = cos;
    this->sin = sin;
    this->detections = detections;
    this->id = id;
    this->status = status;
    this->data = data;
    this->approach = approach;
  }

  double find_nearest(std::list<data_t<T>> &ls) {
    if(ls.size() == 1) return -1.0;
    double min_d = -1.0;
    data_t<T> *min = NULL;
    for(typename std::list<data_t<T>>::iterator it = ls.begin(); it != ls.end(); ++it) {
      if(this->id != it->id) {
        double dist = sqrt(pow(this->x - it->x, 2) + pow(this->y - it->y, 2));
        if(min_d < 0.0 || dist < min_d) {
          min_d = dist;
          min = &*it;
        }
      }
    }
    // if(save_closest) closest = min;
    return min_d;
  }

  void update(finale::FaceCluster &fcl) {
    this->x = fcl.x;
    this->y = fcl.y;
    this->cos = fcl.cos;
    this->sin = fcl.sin;
    this->detections = fcl.detections;
    this->approach = fcl.approach;
  }
  void update(finale::CylCluster &ccl) {
    this->x = ccl.x;
    this->y = ccl.y;
    this->cos = ccl.cos;
    this->sin = ccl.sin;
    this->detections = ccl.detections;
    this->approach = ccl.approach;
  }
  void update(finale::RingCluster &ccl) {
    this->x = ccl.x;
    this->y = ccl.y;
    this->cos = ccl.cos;
    this->sin = ccl.sin;
    this->detections = ccl.detections;
    this->approach = ccl.approach;
  }
};

template<typename T>
data_t<T> *findById(int id, std::list<data_t<T>> ls) {
  for(typename std::list<data_t<T>>::iterator it = ls.begin(); it != ls.end(); ++it)
    if(it->id == id) return &*it;
  return NULL;
}
template<typename T>
data_t<T> *findByColour(unsigned int clr, std::list<data_t<T>> &ls) {
  if(!std::is_same<T, std::tuple<unsigned int, data_t<facedata>*>>::value) return NULL;
  data_t<T> *cl = NULL;
  for(typename std::list<data_t<T>>::iterator it = ls.begin(); it != ls.end(); ++it) {
    if(std::get<0>(it->data) == clr) return &*it;
  }
  return NULL;
}