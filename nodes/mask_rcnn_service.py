#!/usr/bin/env python

import os
import threading
from Queue import Queue
import numpy as np
from timeit import default_timer as timer

import cv2
from cv_bridge import CvBridge
import rospy
import rospkg
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest
from std_msgs.msg import UInt8MultiArray

from mask_rcnn_ros import coco
from mask_rcnn_ros import utils
from mask_rcnn_ros import model as modellib
from mask_rcnn_ros import visualize
from mask_rcnn_ros.msg import Result
from mask_rcnn_ros.srv import MaskRcnn, MaskRcnnResponse


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: CLASS_NAMES.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskRCNNNode(object):
    def __init__(self):
        # Get parameters
        self._visualization = rospy.get_param('~visualize', True)
        self._class_names = rospy.get_param('~class_names', CLASS_NAMES)
        self._class_colors = visualize.random_colors(len(CLASS_NAMES))
        self._publish_rate = rospy.get_param('~publish_rate', 100)
        vis_topic = rospy.get_param('~visualization_topic', '/mask_rcnn_service/visualization')
        self._vis_pub = rospy.Publisher(vis_topic, Image, queue_size=1)

        # Create model object in inference mode
        config = InferenceConfig()
        config.display()
        self._model = modellib.MaskRCNN(mode="inference", model_dir="",
                                        config=config)
        # Load weights
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('mask_rcnn_ros')
        coco_model_path = os.path.join(pkg_path, 'models/mask_rcnn_coco.h5')
        model_path = rospy.get_param('~model_path', coco_model_path)
        rospy.loginfo('model path : %s', model_path)
        # Download COCO trained weights from Releases if needed
        if model_path == coco_model_path and not os.path.exists(coco_model_path):
            model_dir = os.path.join(pkg_path, 'models')
            os.mkdir(model_dir)
            utils.download_trained_weights(coco_model_path)
        self._model.load_weights(model_path, by_name=True)
        
        # Create cv bridge object
        self._cv_bridge = CvBridge()

        # Creating ROS service
        self._service_name = rospy.get_param('~service_name', 'mask_rcnn_service')
        rospy.logdebug('service name : %s', self._service_name)
        s = rospy.Service(self._service_name, MaskRcnn, self._service_callback)
        rospy.loginfo('Ready to get requests...')

    def _service_callback(self, req):
        # Service called, check the requested image is valid
        if req.image is not None:
            rospy.loginfo('Received request')
            
            # Convert to numpy array
            np_image = self._cv_bridge.imgmsg_to_cv2(req.image, 'bgr8')

            # Run detection
            results = self._model.detect([np_image], verbose=0)
            result = results[0]
            result_msg = self._build_result_msg(req.image, result)
            
            # Visualize results
            if self._visualization:
                print('Visualizing')
                vis_image = self._visualize(result, np_image)
                cv_result = np.zeros(shape=vis_image.shape, dtype=np.uint8)
                cv2.convertScaleAbs(vis_image, cv_result)
                image_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                self._vis_pub.publish(image_msg)
            
            # Return Service Result
            rospy.loginfo('Returning result')
            res = MaskRcnnResponse()
            res.success = True
            res.result = result_msg
            return res
        else:
            rospy.logerror('Service request called with empty image')
            res = MaskRcnnResponse()
            res.success = False
            return res

    def _build_result_msg(self, msg, result):
        result_msg = Result()
        result_msg.header = msg.header
        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
            box = RegionOfInterest()
            box.x_offset = np.asscalar(x1)
            box.y_offset = np.asscalar(y1)
            box.height = np.asscalar(y2 - y1)
            box.width = np.asscalar(x2 - x1)
            result_msg.boxes.append(box)

            class_id = result['class_ids'][i]
            result_msg.class_ids.append(class_id)

            class_name = self._class_names[class_id]
            result_msg.class_names.append(class_name)

            score = result['scores'][i]
            result_msg.scores.append(score)

            mask = Image()
            mask.header = msg.header
            mask.height = result['masks'].shape[0]
            mask.width = result['masks'].shape[1]
            mask.encoding = "mono8"
            mask.is_bigendian = False
            mask.step = mask.width
            mask.data = (result['masks'][:, :, i] * 255).tobytes()
            result_msg.masks.append(mask)
        return result_msg

    def _visualize(self, result, image):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        axes = fig.gca()
        visualize.display_instances(image, result['rois'], result['masks'],
                                    result['class_ids'], CLASS_NAMES,
                                    result['scores'], ax=axes,
                                    class_colors=self._class_colors)
        fig.tight_layout()
        canvas.draw()
        result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        _, _, w, h = fig.bbox.bounds
        result = result.reshape((int(h), int(w), 3))
        return result
        

if __name__ == '__main__':
    rospy.init_node('mask_rcnn')
    node = MaskRCNNNode()
    rospy.spin()
    
