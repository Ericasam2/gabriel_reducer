#
# Cloudlet Infrastructure for Mobile Computing
#   - Task Assistance
#
#   Author: Zhuo Chen <zhuoc@cs.cmu.edu>
#           Roger Iyengar <iyengar@cmu.edu>
#
#   Copyright (C) 2011-2019 Carnegie Mellon University
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import numpy as np
import logging
from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2
import instruction_pb2
import instructions
import sys
import os
import cv2
import time

# sys.path.append("/home/samgao1999/gabriel-jiansuqi-yolo/yolo3_tx2")
# import cmd_yolo as dn
sys.path.append('/home/samgao1999/gabriel-pytorch-yolo-test/PyTorch-YOLOv3')
from pytorchyolo.models import load_model
from pytorchyolo.detect import detect_image

#import _init_paths  # this is necessary
'''
darknet_cfg = b"/home/samgao1999/gabriel-sandwich-yolo/darknet/cfg/yolov3.cfg"
darknet_weights = b"/home/samgao1999/gabriel-sandwich-yolo/darknet/backup/yolov3_last.weights"
darknet_meta = b"/home/samgao1999/gabriel-sandwich-yolo/darknet/cfg/jiansuqi.data"
'''

yolo_cfg_path = "PyTorch-YOLOv3/config/yolov3.cfg"
yolo_weights_path = "PyTorch-YOLOv3/weights/yolov3.weights"
yolo_meta_path = "PyTorch-YOLOv3/data/coco.names"

# darknet_meta = "/home/samgao1999/gabriel-jiansuqi-yolo/yolo3_tx2/jiansuqi_weight/voc.data"
# darknet_cfg = "/home/samgao1999/gabriel-jiansuqi-yolo/yolo3_tx2/jiansuqi_weight/yolov3.cfg"
# darknet_weights = "/home/samgao1999/gabriel-jiansuqi-yolo/yolo3_tx2/jiansuqi_weight/backup/yolov3_last519.weights"

IMAGE_MAX_WH = 640  # Max image width and height

CONF_THRESH = 0.5
NMS_THRESH = 0.3

CLASS_IDX_LIMIT = instructions.BOTTOM_AND_BIG_GEARING + 1  # BOTTOM_AND_BIG_GEARING has largest index


# if not os.path.isfile(CAFFEMODEL):
    # raise IOError(('{:s} not found.').format(CAFFEMODEL))


# faster_rcnn_config.TEST.HAS_RPN = True  # Use RPN for proposals

logger = logging.getLogger(__name__)


class SandwichEngine(cognitive_engine.Engine):
    def __init__(self, cpu_only = False):
        if cpu_only:
            raise IOError("Cpu mode is not supported in this version!")

        launch_start = time.time()

        self.net = load_model(yolo_cfg_path, yolo_weights_path)
        
        # Warmup on a dummy image
        img = 128 * np.ones((300, 500, 3), dtype=np.uint8)
        # cv2.imwrite("/sys_test/sys_test_img.img", img)
        # img = "/sys_test/sys_test_img.img"
        for i in range(1):
            _ = detect_image(self.net, img)  
        logger.info("Darknet net has been initilized")
        # launch_time = open("/home/samgao1999/gabriel-jiansuqi-yolo/launch_time_test.txt", "a", encoding="utf-8")
        # text = "{}\n".format(time.time()-launch_start)
        # launch_time.write(text)


    def _detect_object(self, img):
        detection_res = detect_image(self.net, img)
        det_for_class = {}
        # null mode
        
        '''# running mode
        if (detection_res == None):
            det_for_class = {}
        else:
            det_for_class = {}
            det_for_class[detection_res[-1]] = detection_res[:-1] ''' 
            # det_for_class : {class : [x1,y1,x2,y2]}

        return det_for_class


    def handle(self, from_client):
        if from_client.payload_type != gabriel_pb2.PayloadType.IMAGE:
            return cognitive_engine.wrong_input_format_error(
                from_client.frame_id)
        
        engine_fields = cognitive_engine.unpack_engine_fields(
            instruction_pb2.EngineFields, from_client)
        
        img_start = time.time()
        img_array = np.asarray(bytearray(from_client.payload), dtype=np.int8)
        img = cv2.imdecode(img_array, -1)
        
        
        # img resize
        if max(img.shape) > IMAGE_MAX_WH:
            resize_ratio = float(IMAGE_MAX_WH) / max(img.shape[0], img.shape[1])

            img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio,  # resize the image
                             interpolation=cv2.INTER_AREA)
            det_for_class = self._detect_object(img)
            for class_idx in det_for_class:
                det_for_class[class_idx][:4] /= resize_ratio
        else:
            det_for_class = self._detect_object(img)
        '''#####################img_time
        proc_time = open("/home/samgao1999/gabriel-jiansuqi-yolo/proc_test.txt", "a", encoding="utf-8")
        text = "img\t{}\n".format(time.time()-img_start)
        proc_time.write(text)
        #####################'''
        logger.info("object detection result: %s", det_for_class)
        wrapper_start = time.time()  # start track img wrapper time
        result_wrapper = instructions.get_instruction(
            engine_fields, det_for_class)
        result_wrapper.frame_id = from_client.frame_id
        result_wrapper.status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        '''######################wrapper_time
        proc_time = open("/home/samgao1999/gabriel-jiansuqi-yolo/proc_test.txt", "a", encoding="utf-8")
        text = "wrapper\t{}\n".format(time.time()-wrapper_start)
        proc_time.write(text)
        ######################'''
        return result_wrapper


if __name__ == "__main__":
    SandwichEngine()
