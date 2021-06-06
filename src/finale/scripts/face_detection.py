#!/usr/bin/python3

import sys
import rospy
import dlib
import cv2
import math
import numpy as np
import tf2_geometry_msgs
import tf2_ros
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray
from cv_bridge import CvBridge, CvBridgeError
from finale.msg import FaceDetectorToClustering

MIN_CONF = 0.3
MAX_DETECTION_ANGLE = 1.05


class face_localizer:
    def __init__(self):
        rospy.init_node('face_localizer', anonymous=True)

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # The function for performin HOG face detection
        #self.face_detector = dlib.get_frontal_face_detector()
        self.face_net = cv2.dnn.readNetFromCaffe(rospy.get_param('/face_localizer/proto_loc', '') + 'deploy.prototxt.txt', rospy.get_param('~proto_loc', '') + 'res10_300x300_ssd_iter_140000.caffemodel')
        # A help variable for holding the dimensions of the image
        self.dims = (0, 0, 0)
        # # Marker array object used for showing markers in Rviz
        # self.marker_array = MarkerArray()
        # self.marker_num = 1

        # Subscribe to the image and/or depth topic
        # self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        # self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher('finale/face_clustering', FaceDetectorToClustering, queue_size=1000)
        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)


    def get_pose(self,coords,dist,stamp):
        # (dist to center, dist to left edge, dist to right edge, avg dist to top edge, avg dist to bottom edge)
        # Calculate the position of the detected face

        k_f = 554 # kinect focal length in pixels

        x1, x2, y1, y2 = coords

        # Get the angles in the base_link relative coordinate system
        face_x = self.dims[1] / 2 - (x1+x2)/2.
        angle_to_target = np.arctan2(face_x, k_f)
        x, y = dist*np.cos(angle_to_target), dist*np.sin(angle_to_target)

        if(angle_to_target < 0):
            angle_to_target = math.pi + angle_to_target
        else:
            angle_to_target = -math.pi + angle_to_target

        pose_s = PoseStamped()
        pose_s.pose.position.x = x
        pose_s.pose.position.y = y
        pose_s.pose.position.z = 0
        pose_s.pose.orientation.x = 0.0
        pose_s.pose.orientation.y = 0.0
        pose_s.pose.orientation.z = np.sin(angle_to_target / 2.0)
        pose_s.pose.orientation.w = np.cos(angle_to_target / 2.0)
        pose_s.header.frame_id = "camera_rgb_frame"
        pose_s.header.stamp = stamp

        pose_null = PoseStamped()
        pose_null.header = pose_s.header
        pose_null.pose.orientation.w = 1.0

        # Get the point in the "map" coordinate system
        try:
            # point_world = self.tf_buf.transform(point_s, "map")
            transform = self.tf_buf.lookup_transform("map", pose_s.header.frame_id, rospy.Time(0), rospy.Duration(0.5))
            # TODO revert to original angle calc. 
            apprch_pose = tf2_geometry_msgs.do_transform_pose(pose_null, transform)
            pose = tf2_geometry_msgs.do_transform_pose(pose_s, transform)
            # print(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w)
        except Exception as e:
            print(e)
            pose = None
            apprch_pose = None

        return (pose, pose_s, angle_to_target, apprch_pose)
    

    def find_faces(self):
        # print('I got a new image!')
        # Get the next rgb and depth images that are posted from the camera
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
        
        # # Limit time differences between images
        # if abs(rgb_image_message.header.stamp.nsecs - depth_image_message.header.stamp.nsecs) < TIME_DIFF_LIMIT:
        #     break

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
        h = self.dims[0]
        w = self.dims[1]

        # Tranform image to gayscale
        #gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Do histogram equlization
        #img = cv2.equalizeHist(gray)

        # Detect the faces in the image
        #face_rectangles = self.face_detector(rgb_image, 0)
        blob = cv2.dnn.blobFromImage(cv2.resize(rgb_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        face_detections = self.face_net.forward()

        pose_array = []
        camera_pose_array = []
        angle_array = []
        apprch_array = []

        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]
            if confidence>MIN_CONF:
                box = face_detections[0,0,i,3:7] * np.array([w,h,w,h])
                box = box.astype('int')
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                # Extract region containing face
                face_region = rgb_image[y1:y2, x1:x2]

                # Visualize the extracted face
                #cv2.imshow("ImWindow", face_region)
                #cv2.waitKey(1)

                # Find the distance to the detected face (dist to center, dist to left edge, dist to right edge, avg dist to top edge, avg dist to bottom edge)
                face_distance = float(np.nanmean(depth_image[y1:y2,x1:x2]))

                print('Distance to face', face_distance)

                # Get the time that the depth image was recieved
                depth_time = depth_image_message.header.stamp

                # Find the location of the detected face
                (pose, pose_s, angle, apprch_pose) = self.get_pose((x1,x2,y1,y2), face_distance, depth_time)

                if pose is not None:
                    pose_array.append(pose.pose)
                    camera_pose_array.append(pose_s.pose)
                    angle_array.append(angle)
                    apprch_array.append(apprch_pose.pose)

        data = FaceDetectorToClustering()
        data.angles = angle_array
        data.inCamera.header.stamp = depth_image_message.header.stamp
        data.inCamera.poses = camera_pose_array
        data.faces.header.stamp = depth_image_message.header.stamp
        data.faces.poses = pose_array
        data.approaches.header.stamp = depth_image_message.header.stamp
        data.approaches.poses = apprch_array
        self.markers_pub.publish(data)

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
        rospy.loginfo("Initialized DNN node")
        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown():
            face_finder.find_faces()
            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
