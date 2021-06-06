#!/usr/bin/python3

# import ros stuff
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from time import sleep
from finale.msg import WanderControl
import math

drive = False

def callback_laser(dt):

    global sredina
    global levo
    global desno

    sredina = dt.ranges[320]
    levo = dt.ranges[639]
    desno = dt.ranges[0]
    
    if math.isnan(levo):
      levo = 0
    if math.isnan(sredina):
      sredina = 0
    if math.isnan(desno):
      desno = 0

def callback_toggle(tg):
  global drive 
  drive = tg.wander
      
def vozi():
    rospy.init_node('moving', anonymous=True)
    laser_sub = rospy.Subscriber("/scan", LaserScan, callback_laser)
    toggle_sub = rospy.Subscriber("finale/wandering", WanderControl, callback_toggle)
    pub = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size=10)
    move = Twist()
    rate = rospy.Rate(5)
    stevc = 0
    
    while not rospy.is_shutdown():
      if drive:
        print("Driving")
        if stevc % 100 == 0:
          stevc = 0
          i = 0
          while i < 80:
            move.linear.x = 0.0
            move.angular.z = 0.4
            pub.publish(move)
            print('Looking around.')
            i += 1
            rate.sleep()
        stevc += 1
        print('Check surroundings in T - %s.' % (100 - stevc))
      
        if levo > thr and desno > thr and sredina > thr:
          move.linear.x = 0.2
          move.angular.z = 0.0
          print('Free path, roll out!')
          pub.publish(move)
        else:
          if levo > desno:
            while levo < thr or sredina < thr or desno < 1:
              move.linear.x = 0.0
              move.angular.z = 0.3
              pub.publish(move)
              print('Obstacle! Turning left.')
              rate.sleep()
          else:
            while levo < 1 or sredina < thr or desno < thr:
              move.linear.x = 0.0
              move.angular.z = -0.3
              pub.publish(move)
              print('Obstacle! Turning right.')
              rate.sleep()
      else:
        print("Not driving")
        stevc = 1
        
    	
        
      rate.sleep()
      
      
    
    
if __name__ == '__main__':
    sredina = 0
    levo = 0
    desno = 0
    thr = 0.7
    vozi()
