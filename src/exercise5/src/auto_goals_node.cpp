#include "ros/ros.h"

#include <nav_msgs/GetMap.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <move_base_msgs/MoveBaseActionResult.h>
#include <visualization_msgs/Marker.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <random>
#include <time.h>
#include <math.h>

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

float map_resolution = 0;
geometry_msgs::TransformStamped map_transform;

ros::Publisher goal_pub;
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
    got_map = true;

    map = msg_map->data;

    map_resolution = msg_map->info.resolution;
    map_width = msg_map->info.width;
    map_height = msg_map->info.height;

    map_transform.transform.translation.x = msg_map->info.origin.position.x;
    map_transform.transform.translation.y = msg_map->info.origin.position.y;
    map_transform.transform.translation.z = msg_map->info.origin.position.z;

    map_transform.transform.rotation = msg_map->info.origin.orientation;
    ROS_INFO("map resolution %f", map_resolution);
    ROS_INFO("map orientation %f, %f, %f, %f", msg_map->info.origin.orientation.x, msg_map->info.origin.orientation.y, msg_map->info.origin.orientation.z, msg_map->info.origin.orientation.w);
}

void poseCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg) {
    got_pose = true;
    est_pose = pose_msg->pose.pose;
}

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
int main(int argc, char** argv) {

    srand(time(NULL));

    ros::init(argc, argv, "auto_goals");
    ros::NodeHandle nh;

    ros::Subscriber map_sub = nh.subscribe("map", 10, mapCallback);
    ros::Subscriber est_pose_sub = nh.subscribe("amcl_pose", 100, poseCallback);
    ros::Subscriber goal_sub = nh.subscribe<move_base_msgs::MoveBaseActionResult>("move_base/result", 100, next_goal);
    goal_pub = nh.advertise<geometry_msgs::PoseStamped>("move_base_simple/goal", 10);

    //
    debug_point_pub = nh.advertise<visualization_msgs::Marker>("check_point", 1000);
    //

    ROS_INFO("Waiting for goal subscriber");
    while(goal_pub.getNumSubscribers() < 1 && ros::ok()) {}
    
    ROS_INFO("Waiting for map and pose");
    bool started = false;
    while(ros::ok()) {
        if(!started && got_map && got_pose) {
            ROS_INFO("map %d, pose %d", got_map, got_pose);
            started = true;
            goNext();
        }
        ros::spinOnce();
    }
    return 0;

}