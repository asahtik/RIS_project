#include "geometry_msgs/Pose.h"
#include "finale/FaceCluster.h"

#include <math.h>
#include <iostream>

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
    this-> approach = approach;
  }

  void update(finale::FaceCluster &fcl) {
    this->x = fcl.x;
    this->y = fcl.y;
    this->cos = fcl.cos;
    this->sin = fcl.sin;
    this->detections = fcl.detections;
    this->approach = fcl.approach;
  }
};

template<typename T>
data_t<T> *findById(int id, std::list<data_t<T>> ls) {
    for(typename std::list<data_t<T>>::iterator it = ls.begin(); it != ls.end(); ++it)
      if(it->id == id) return &*it;
    return NULL;
  }