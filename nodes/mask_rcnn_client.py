#!/usr/bin/env python

import threading
import roslib
import rospy
from sensor_msgs.msg import Image
from mask_rcnn_ros.srv import MaskRcnn, MaskRcnnRequest


class MaskRcnnClient:
    def __init__(self):
        # Image subscriber
        self.rgb_subscriber = rospy.Subscriber(rospy.get_param("~image_topic"), Image, self._rgb_callback)
        self.rgb = None
        self.ready = False
        
        # Service to call
        service_name = rospy.get_param('~service_name', 'mask_rcnn_service')
        rospy.loginfo('Waiting for the service: %s', service_name)
        rospy.wait_for_service(service_name)
        self.maskrcnn_service = rospy.ServiceProxy(service_name, MaskRcnn)
        
    def _rgb_callback(self, msg):
        rospy.logdebug('Getting image')
        self.rgb = msg
        self.ready = True
        
    def call_mask_rcnn_service(self):
        # Wait for an image
        self.count = 10
        rate = rospy.Rate(5)  # 5hz
        while not self.ready:
            rospy.loginfo('Waiting for data')
            rate.sleep()
            self.count -= 1
            if self.count < 0:
                break
        # If did not get an image
        if not self.ready:
            rospy.logerr('Did not get any data')
            return None

        # Got an image, call the service
        try:
            req = MaskRcnnRequest()
            req.image = self.rgb
            res = self.maskrcnn_service(req)
            if res.success:
                rospy.loginfo('Detection result\n%s', len(res.result.boxes))
                return res.result
            else:
                rospy.logwarning('Failed to get a detections')
        except rospy.ServiceException as e:
            rospy.logerr('Service call failed: %s', e)

        return None
          

if __name__ == "__main__":
    rospy.init_node('mask_rcnn_client')
    rospy.loginfo('Starting mask_rcnn_client')
    mask_rcnn_client = MaskRcnnClient()
    mask_rcnn_client.call_mask_rcnn_service()
    
