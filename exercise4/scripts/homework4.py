#!/usr/bin/python3

import sys
import rospy
import dlib
import cv2
import math
import numpy as np
import tf2_geometry_msgs
import tf2_ros
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

# max diff between depth and rgb
TIME_DIFF_LIMIT = 0.1 * 10**9
# size of map
MAP_SIZE = 8
# abs map origin (in acc cs) y^ x>
MAP_ORIGIN = (3, 4)
# resolution of accumulator matrix
ACC_RESOLUTION = 0.25
# distance to which we evaluate markers as same location
MIN_SEP_DIST = 0.5
## index translating to 0,0 in map coordinates
# ACC_ORIGIN = (math.floor(MAP_ORIGIN[0] / ACC_RESOLUTION), math.floor(MAP_ORIGIN[1] / ACC_RESOLUTION))
# number of neighbours affected when face is detected
PSIGMA = 1
NO_REDET_PENALTY = 0.1

gauss = np.zeros((2*PSIGMA + 1, 2*PSIGMA + 1))
for i in range(0, PSIGMA*2 + 1):
    for j in range(0, PSIGMA*2 + 1):
        y = i - PSIGMA
        x = j - PSIGMA
        gauss[i][j] = float(1/(2*math.pi*(PSIGMA/3)**2) * math.exp(-((x**2 + y**2)/(2*(PSIGMA/3)**2))))

class face_localizer:
    def __init__(self):
        rospy.init_node('face_localizer', anonymous=True)

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # The function for performin HOG face detection
        self.face_detector = dlib.get_frontal_face_detector()

        # A help variable for holding the dimensions of the image
        self.dims = (0, 0, 0)

        # Marker array object used for showing markers in Rviz
        self.marker_array = MarkerArray()
        self.marker_num = 1

        # Subscribe to the image and/or depth topic
        # self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        # self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher('face_markers', MarkerArray, queue_size=1000)

        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        
        # holding accumulator array (accumulator, no detected, orientation)
        size = math.floor(MAP_SIZE / ACC_RESOLUTION) + 1
        self.face_acc = [np.zeros((size, size), dtype=float), np.zeros((size, size), dtype=int), np.zeros((size, size), dtype=float)]
        self.prev_detect = []

    def add_to_acc(x, y, angle, prev):
        for i in range(y-PSIGMA, y+PSIGMA+1):
            for j in range(x-PSIGMA, x+PSIGMA+1):
                 if i>=0 and j>=0 and i<len(self.face_acc) and j<len(self.face_acc[0]):
                     g_i = i - y + PSIGMA
                     g_j = j - x + PSIGMA
                     self.face_acc[0][i][j] += gauss[g_i][g_j]
        if len(prev) > 0 and NO_REDET_PENALTY > 0:
            for point in prev:
                if point[0] < x-PSIGMA and point[0] > x+PSIGMA and point[1] < y-PSIGMA and point[1] > x+PSIGMA:
                    self.face_acc[0][point[1]][point[0]] -= NO_REDET_PENALTY


        self.face_acc[1][y][x] += 1
        self.face_acc[2][y][x] = angle

    def get_pose(self,coords,dist,stamp):
        # Calculate the position of the detected face

        k_f = 554 # kinect focal length in pixels

        x1, x2, y1, y2 = coords

        face_x = self.dims[1] / 2 - (x1+x2)/2.
        face_y = self.dims[0] / 2 - (y1+y2)/2.

        angle_to_target = np.arctan2(face_x,k_f)

        # Get the angles in the base_link relative coordinate system
        x, y = dist*np.cos(angle_to_target), dist*np.sin(angle_to_target)

        ### Define a stamped message for transformation - directly in "base_link"
        #point_s = PointStamped()
        #point_s.point.x = x
        #point_s.point.y = y
        #point_s.point.z = 0.3
        #point_s.header.frame_id = "base_link"
        #point_s.header.stamp = rospy.Time(0)

        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped()
        point_s.point.x = -y
        point_s.point.y = 0
        point_s.point.z = x
        point_s.header.frame_id = "camera_rgb_optical_frame"
        point_s.header.stamp = stamp

        # Get the point in the "map" coordinate system
        try:
            #point_world = self.tf_buf.transform(point_s, "map")
            point_world = self.tf_buf.transform(point_s, "map", timeout=rospy.Duration(0.4))

            # Create a Pose object with the same position
            pose = Pose()
            pose.position.x = point_world.point.x
            pose.position.y = point_world.point.y
            pose.position.z = point_world.point.z
        except Exception as e:
            print(e)
            pose = None

        return pose
    

    def find_faces(self):
        print('I got a new image!')

        # Get the next rgb and depth images that are posted from the camera
        while 1==1:  
          try:
              rgb_image_message = rospy.wait_for_message("/camera/rgb/image_raw", Image)
          except Exception as e:
              print(e)
              return 0

          try:
              depth_image_message = rospy.wait_for_message("/camera/depth/image_raw", Image)
          except Exception as e:
              print(e)
              return 0
          
          # Limit time differences between images
          if abs(rgb_image_message.header.stamp.nsecs - depth_image_message.header.stamp.nsecs) < TIME_DIFF_LIMIT:
              break

        # Convert the images into a OpenCV (numpy) format

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_message, "bgr8")
        except CvBridgeError as e:
            print(e)

        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_message, "32FC1")
        except CvBridgeError as e:
            print(e)

        # Set the dimensions of the image
        self.dims = rgb_image.shape

        # Tranform image to gayscale
        #gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Do histogram equlization
        #img = cv2.equalizeHist(gray)

        # Detect the faces in the image
        face_rectangles = self.face_detector(rgb_image, 0)

        # For each detected face, extract the depth from the depth image
        detections = []
        for face_rectangle in face_rectangles:
            print('Faces were detected')

            # The coordinates of the rectanle
            x1 = face_rectangle.left()
            x2 = face_rectangle.right()
            y1 = face_rectangle.top()
            y2 = face_rectangle.bottom()

            # Extract region containing face
            face_region = rgb_image[y1:y2,x1:x2]

            # Visualize the extracted face
            # cv2.imshow("Depth window", face_region)
            # cv2.waitKey(1)

            # Find the distance to the detected face
            face_distance = float(np.nanmean(depth_image[y1:y2,x1:x2]))

            print('Distance to face', face_distance)

            # Get the time that the depth image was recieved
            depth_time = depth_image_message.header.stamp

            # Find the location of the detected face
            pose = self.get_pose((x1,x2,y1,y2), face_distance, depth_time)

            if pose is not None:
                print("pose")

                # Create a marker used for visualization
                self.marker_num += 1
                marker = Marker()
                marker.header.stamp = rospy.Time(0)
                marker.header.frame_id = 'map'
                marker.pose = pose
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.frame_locked = False
                marker.lifetime = rospy.Duration.from_sec(10)
                marker.id = self.marker_num
                marker.scale = Vector3(0.1, 0.1, 0.1)
                marker.color = ColorRGBA(0, 1, 0, 1)
                self.marker_array.markers.append(marker)

                self.markers_pub.publish(self.marker_array)

                # Add to acc matrix TODO
                acc_x = math.floor((pose.point.x + MAP_ORIGIN[0]) / ACC_RESOLUTION)
                acc_y = math.floor((pose.point.y + MAP_ORIGIN[1]) / ACC_RESOLUTION)
                print("x={pose.point.x:f}, y={pose.point.y:f}, acc_x={acc_x:f}, acc_y={acc_y:f}")
                self.add_to_acc(acc_x, acc_y, 0, self.prev_detect)
                detections.append((acc_x, acc_y))
        self.prev_detect = detections

        plt.imshow(plt.pcolor(self.face_acc[0]), cmap="gray")
        plt.show()

    def depth_callback(self,data):

        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)

        # Do the necessairy conversion so we can visuzalize it in OpenCV
        
        image_1 = depth_image / np.nanmax(depth_image)
        image_1 = image_1*255
        
        image_viz = np.array(image_1, dtype=np.uint8)

        #cv2.imshow("Depth window", image_viz)
        #cv2.waitKey(1)

        #plt.imshow(depth_image)
        #plt.show()

def main():

        face_finder = face_localizer()

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            face_finder.find_faces()
            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
