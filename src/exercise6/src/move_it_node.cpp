#include "ros/ros.h"

#include <nav_msgs/GetMap.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/PointCloud2.h>
#include <move_base_msgs/MoveBaseActionResult.h>
#include <actionlib_msgs/GoalID.h>
#include <visualization_msgs/Marker.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <Eigen/Core>

#include <random>
#include <time.h>
#include <math.h>

#include "include/clustering_2d_lib.hpp"
#include "clustering/Cluster2DArray.h"

#define MIN_DISTANCE 0.5
#define MAX_DISTANCE 3
#define MAX_DIST_TRIES 10
#define DISTANCE_DECIMALS 3
#define ANGLE_DECIMALS 3
#define SAFETY_RADIUS 0.3
#define NO_RADIUS_CHECKS 16
#define MAX_DIRECTION_ANGLE_START (M_PI / 2.0)
#define MAX_DIRECTION_ANGLE_INC 0.1
#define MAX_GOAL_ANGLE (M_PI / 1.5)
#define MAX_APPROACH_ANGLE 0.2
#define APPROACH_DISTANCE 0.1
#define TIME_CORRECTION 0.5

enum {APPROACHING, INTERACTING, WANDERING} state;

ros::NodeHandle *n;

float map_resolution = 0;
geometry_msgs::TransformStamped map_transform;

ros::Publisher goal_pub;
ros::Publisher goal_cancel_pub;
ros::Publisher velocity_pub;
ros::Publisher debug_point_pub;

bool got_map = false, got_pose = false;
int map_width = 0, map_height = 0;
std::vector<int8_t> map;
geometry_msgs::Pose est_pose;

float random_float(float from, float to, int resolution = 0) {
    int mult = 1;
    if(resolution > 0) mult = 10 * resolution;
    float range = (to - from) * mult;
    int r = rand() % (int)range;
    return ((float)r / (float)mult) + from;
}

float get_angle(double a_cos, double a_sin) {
  float arccos = (acos(a_cos) * 2.0);
  if(a_sin >= 0) return arccos;
  else return -arccos;
}

bool check_valid_position(int x, int y) {
    // ROS_INFO("map at x:%d,y:%d %d", x, y, map[y * map_width + x]);

    if(map[y * map_width + x] != 0) return false;

    float angle_inc = (2 * M_PI / NO_RADIUS_CHECKS);
    for(int i = 0; i < NO_RADIUS_CHECKS; i++) {
        float angle = i * angle_inc;
        int map_r_x = x + ceil((cos(angle) * SAFETY_RADIUS) / map_resolution);
        int map_r_y = y + ceil((sin(angle) * SAFETY_RADIUS) / map_resolution);
        if(map[map_r_y * map_width + map_r_x] != 0) return false;
    }
    return true;
}

void mapCallback(const nav_msgs::OccupancyGridConstPtr& msg_map) {
    if(got_map) return;

    got_map = true;

    map = msg_map->data;

    map_resolution = msg_map->info.resolution;
    map_width = msg_map->info.width;
    map_height = msg_map->info.height;

    map_transform.transform.translation.x = msg_map->info.origin.position.x;
    map_transform.transform.translation.y = msg_map->info.origin.position.y;
    map_transform.transform.translation.z = msg_map->info.origin.position.z;

    map_transform.transform.rotation = msg_map->info.origin.orientation;
    // ROS_INFO("map resolution %f", map_resolution);
    // ROS_INFO("map orientation %f, %f, %f, %f", msg_map->info.origin.orientation.x, msg_map->info.origin.orientation.y, msg_map->info.origin.orientation.z, msg_map->info.origin.orientation.w);
}

void poseCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg) {
    got_pose = true;
    est_pose = pose_msg->pose.pose;
}

int goal_id = 0;

void goNext() {
    float robot_angle = get_angle(est_pose.orientation.w, est_pose.orientation.z);
    float t_x = 0, t_y = 0, map_angle = 0, max_angle = MAX_DIRECTION_ANGLE_START;
    bool appropriate = false;
    while(!appropriate && ros::ok()) {
        float dist = 0;
        // Choose random direction (improvement: Higher likelyhood to go forward)
        float dir = random_float(-max_angle, max_angle, ANGLE_DECIMALS);
        max_angle += MAX_DIRECTION_ANGLE_INC;
        map_angle = robot_angle + dir;

        int tries = 0;
        while(!appropriate && tries < MAX_DIST_TRIES && ros::ok()) {
            
            dist = random_float(MIN_DISTANCE, MAX_DISTANCE, DISTANCE_DECIMALS);

            float d_x = dist * cos(map_angle), d_y = dist * sin(map_angle);

            geometry_msgs::Point pt;
            geometry_msgs::Point transformed_pt;

            pt.x = est_pose.position.x + d_x;
            pt.y = est_pose.position.y + d_y;
            pt.z = 0.0;

            // Get point in c. s. of map array
            tf2::Stamped<tf2::Transform> temp_inv;
            tf2::fromMsg(map_transform, temp_inv);
            geometry_msgs::TransformStamped inv_transform;
            inv_transform.transform = tf2::toMsg(temp_inv.inverse());
            tf2::doTransform(pt, transformed_pt, inv_transform);

            ROS_INFO("Potential target angle %f, potential target distance %f", map_angle, dist);
            visualization_msgs::Marker m;
            m.header.stamp = ros::Time(0);
            m.header.frame_id = "map";
            m.type = visualization_msgs::Marker::SPHERE;
            m.pose.position.x = pt.x;
            m.pose.position.y = pt.y;
            m.pose.position.z = SAFETY_RADIUS;
            m.scale.x = SAFETY_RADIUS * 2;
            m.scale.y = SAFETY_RADIUS * 2;
            m.scale.z = SAFETY_RADIUS * 2;
            m.lifetime.fromSec(1);
            m.color.a = 1.0; m.color.r = 1.0;
            debug_point_pub.publish(m);

            appropriate = check_valid_position((int)round(transformed_pt.x / map_resolution), (int)round(transformed_pt.y / map_resolution));

            t_x = pt.x; t_y = pt.y;

            tries++;
            // Get distance to barrier in said direction
            // d = interpolate_distance(round(transformed_pose.position.x / map_resolution), round(transformed_pose.position.y / map_resolution), transformed_map_angle);
            // if(d > MIN_DISTANCE) appropriate = true;
        }
        // ROS_INFO("Angle %f", map_angle * (180 / M_PI));
        // ROS_INFO("transformed map angle %f", transformed_map_angle * (180 / M_PI));
    }

    // float dist = random_float(MIN_DISTANCE, d * map_resolution, DISTANCE_DECIMALS);

    float d_goal_angle = random_float(-MAX_GOAL_ANGLE, MAX_GOAL_ANGLE, ANGLE_DECIMALS);

    geometry_msgs::PoseStamped goal;
    goal.header.frame_id = "map";
    goal.header.stamp = ros::Time(0);
    goal.pose.position.x = t_x;
    goal.pose.position.y = t_y;
    goal.pose.orientation.z = sin((map_angle + d_goal_angle) / 2.0);
    goal.pose.orientation.w = cos((map_angle + d_goal_angle) / 2.0);

    goal_id++;
    goal_pub.publish(goal);
}

void next_goal(move_base_msgs::MoveBaseActionResult msg) {
    if(msg.status.status < 3) ROS_INFO("Goal pending");
    else if(msg.status.status == 3) {
        goNext();
    } else {
        ROS_INFO("Goal cannot be reached");
        goNext();
    }
}


void ring_cloud_callback(const sensor_msgs::PointCloud2ConstPtr &depth_blob) {

}
void ring_cluster_callback(const clustering::Cluster2DArray::ConstPtr &clusters) {

}

std::list<int> visited_ids;

ros::Timer timer;
int what = WANDERING;
void approach(float angle, float dist,  int id, float lin_speed = 0.2, float ang_speed = 0.5) {
  ROS_INFO("Approach");
  geometry_msgs::Twist go;
  if(angle <= MAX_APPROACH_ANGLE && false) {
    if(what != APPROACHING) {
      what = APPROACHING;
      go.linear.x = lin_speed;
      float time = dist / lin_speed * TIME_CORRECTION;
      timer.stop();
      velocity_pub.publish(go);
      timer = n->createTimer(ros::Duration(time), [&](const ros::TimerEvent& time) {
        visited_ids.push_back(id);
        geometry_msgs::Twist stop;
        stop.angular.z = 0;
        velocity_pub.publish(stop);
        what = WANDERING;
        // goNext();
      }, true);
    }
  } else {
    if(angle < 0.0) ang_speed = -ang_speed;
    go.angular.z = ang_speed;
    float time = angle / ang_speed;
    timer.stop();
    velocity_pub.publish(go);
    timer = n->createTimer(ros::Duration(time), [&](const ros::TimerEvent& time) {
      geometry_msgs::Twist stop;
      stop.angular.z = 0;
      velocity_pub.publish(stop);
    }, true);
  }
}

bool is_visited(int id) {
  for(int i : visited_ids) if(i == id) return true;
  return false;
}

void cylinder_callback(const sensor_msgs::PointCloud2ConstPtr &depth_blob, const clustering::Cluster2DArray::ConstPtr &clusters) {
  ROS_INFO("Got cylinder");
  for(int i = 0; i < clusters->no; i++) {
    if(!is_visited(clusters->clusters[i].id)) {
      // New cylinder -> approach
      // ROS_INFO("Found unvisited cylinder");
      actionlib_msgs::GoalID gid; gid.id = goal_id; gid.stamp = ros::Time::now();
      goal_cancel_pub.publish(gid);

      pcl::PCLPointCloud2 cloud_blob;
      pcl_conversions::toPCL(*depth_blob, cloud_blob);

      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
      // Read in the cloud data
      pcl::fromPCLPointCloud2(cloud_blob, *cloud);
      // Get min and max point
      float minz = 100;
      for(pcl::PointCloud<pcl::PointXYZ>::iterator it = cloud->begin(); it != cloud->end(); ++it)
        if(it->z < minz) minz = it->z;

      Eigen::Vector4f centroid;   
      pcl::compute3DCentroid(*cloud, centroid);

      float x = centroid[0], y = centroid[1];

      float angle_to_target = -atan2(x, minz), distance_to_target = minz / cos(angle_to_target);
      ROS_INFO("Angle to target %f, distance %f", angle_to_target, distance_to_target);
      // approach(angle_to_target, minz / cos(angle_to_target), clusters->clusters[i].id);
    }
  }
}

int main(int argc, char** argv) {

    srand(time(NULL));

    ros::init(argc, argv, "auto_goals");
    ros::NodeHandle nh;
    n = &nh;

    ros::Subscriber map_sub = nh.subscribe("map", 10, mapCallback);
    ros::Subscriber est_pose_sub = nh.subscribe("amcl_pose", 100, poseCallback);
    // ros::Subscriber goal_sub = nh.subscribe("move_base/result", 100, next_goal);
    ros::Subscriber ring_cloud_sub = nh.subscribe("exercise6/ring_cloud", 1, ring_cloud_callback);
    ros::Subscriber ring_cluster_sub = nh.subscribe("exercise6/ring_clusters", 10, ring_cluster_callback);
    // ros::Subscriber cylinder_cloud_sub = nh.subscribe("exercise6/cylinder_cloud", 1, cylinder_cloud_callback);
    // ros::Subscriber cylinder_cluster_sub = nh.subscribe("exercise6/cylinder_clusters", 10, cylinder_cluster_callback);

    message_filters::Subscriber<sensor_msgs::PointCloud2> cylinder_cloud_sub(nh, "exercise6/cylinder_cloud", 10);
    message_filters::Subscriber<clustering::Cluster2DArray> cylinder_cluster_sub(nh, "exercise6/cylinder_clusters", 10);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, clustering::Cluster2DArray> sync_policy;
    message_filters::Synchronizer<sync_policy> sync(sync_policy(10), cylinder_cloud_sub, cylinder_cluster_sub);
    sync.registerCallback(boost::bind(cylinder_callback, _1, _2));


    goal_pub = nh.advertise<geometry_msgs::PoseStamped>("move_base_simple/goal", 10);
    velocity_pub = nh.advertise<geometry_msgs::Twist>("mobile_base/commands/velocity", 10);
    goal_cancel_pub = nh.advertise<actionlib_msgs::GoalID>("move_base/cancel", 10);
    //
    debug_point_pub = nh.advertise<visualization_msgs::Marker>("check_point", 1000);
    //

    ROS_INFO("Waiting for goal subscriber");
    while(goal_pub.getNumSubscribers() < 1 && ros::ok()) {}
    
    ros::Rate rate(10);
    ROS_INFO("Waiting for map and pose");
    bool started = false;
    while(ros::ok()) {
      /* if(!started && got_map && got_pose) {
          ROS_INFO("map %d, pose %d", got_map, got_pose);
          started = true;
          goNext();
      } */
      rate.sleep();
      ros::spinOnce();
    }
    return 0;

}