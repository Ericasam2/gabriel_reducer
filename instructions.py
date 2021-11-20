# Cloudlet Infrastructure for Mobile Computing
#   - Task Assistance
#
#   Author: Zhuo Chen <zhuoc@cs.cmu.edu>
#           Roger Iyengar <iyengar@cmu.edu>
#
#   Copyright (C) 2011-2013 Carnegie Mellon University
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

import os
import math
import instruction_pb2
from gabriel_protocol import gabriel_pb2
from collections import namedtuple
# import cv2

ENGINE_NAME = "instruction"

# Class indexes come from the following code:
# LABELS = ["lid","total","bottom_and_two_gearing","big_gearing","small_gearing",
#           "bottom","bottom_and_two_gearing","done"]
# with open(os.path.join('model', 'labels.txt')) as f:
#     idx = 1
#     for line in f:
#         line = line.strip()
#         print(line.upper(), '=', idx)
#         idx += 1


LID = 1
TOTAL = 2
BOTTOM_AND_TWO_GEARING = 3
BIG_GEARING = 4
SMALL_GEARING = 5
BOTTOM = 6
BOTTOM_AND_BIG_GEARING = 7
DONE = 8  # This is not a class the DNN will output

label_dict = {b"lid":1, b"total":2, b"button_and_twogearing":3, b"big_gearing":4,
              b"small_gearing":5, b"bottom":6, b"bottom_and_biggearing":7}

Hologram = namedtuple('Hologram', ['dist', 'x', 'y'])

BOTTOM_HOLO = Hologram(dist=6500, x=0.5, y=0.36)
BIG_GEARING_HOLO = Hologram(dist=6800, x=0.5, y=0.32)
SMALL_GEARING_HOLO = Hologram(dist=7100, x=0.5, y=0.3)
LID_HOLO = Hologram(dist=7500, x=0.5, y=0.26)
# BREAD_TOP_HOLO = Hologram(dist=7800, x=0.5, y=0.22)


INSTRUCTIONS = {
    BOTTOM: 'Now place the bottom on the table.',
    BIG_GEARING: 'Now put the big gear onto the bottom.',
    BOTTOM_AND_BIG_GEARING: 'Now you get the bottom with the big gear, prepare the small gear',
    SMALL_GEARING: 'Now put the small gear onto the bottom.',
    BOTTOM_AND_TWO_GEARING: 'Now you get the bottom with two gears, prepare the lid',
    LID: 'Now place the lid onto the gears, covering the whole model.',
    TOTAL: 'Put the lid to cover the model and the decelerator will be done!',
    DONE: 'Congratulations! You have made a decelerator!',
}


IMAGE_FILENAMES = {
    BOTTOM: 'bottom.jpg',
    BIG_GEARING: 'big_gearing.jpg',
    SMALL_GEARING: 'small_gearing.jpg',
    LID: 'whole.jpg',
    BOTTOM_AND_BIG_GEARING: 'big_gearing.jpg',
    BOTTOM_AND_TWO_GEARING: 'small_gearing.jpg',
    TOTAL: 'whole.jpg',
    DONE: 'whole.jpg',
}

'''
create dict to accommodate the relation  
'class_idx:image_path' 
'''
IMAGES = {
    class_idx: open(os.path.join('/home/samgao1999/gabriel-jiansuqi-yolo/images_feedback', filename), 'rb').read()
    for class_idx, filename in IMAGE_FILENAMES.items()
}



def _result_without_update(engine_fields):
    result_wrapper = gabriel_pb2.ResultWrapper()
    result_wrapper.engine_fields.Pack(engine_fields)
    return result_wrapper


def _result_with_update(engine_fields, class_idx):  # class_idx is used to look up the img_file
    """
    update the result
    initialize the class: ResultWrapper.Result()
    the Result() is used to generate a gabriel-style result to pass
    update the result's ---- name,image_path,type
    append the results to form a list
    """
    #####################################
    # print('============')
    # print('class_idx:{}'.format(class_idx))
    # cv2.imshow(' ',IMAGES[class_idx])
    # cv2.waitKey(0)
    # print('============')
    #####################################
    engine_fields.update_count += 1

    result_wrapper = _result_without_update(engine_fields)

    result = gabriel_pb2.ResultWrapper.Result()
    result.payload_type = gabriel_pb2.PayloadType.IMAGE
    result.engine_name = ENGINE_NAME
    result.payload = IMAGES[class_idx]
    result_wrapper.results.append(result)

    result = gabriel_pb2.ResultWrapper.Result()
    result.payload_type = gabriel_pb2.PayloadType.TEXT
    result.engine_name = ENGINE_NAME
    result.payload = INSTRUCTIONS[class_idx].encode(encoding="utf-8")
    result_wrapper.results.append(result)
    return result_wrapper


def _start_result(engine_fields):
    engine_fields.decelerator.state = instruction_pb2.Decelerator.State.NOTHING
    return _result_with_update(engine_fields, BOTTOM)


def _nothing_result(det_for_class, engine_fields):
    if BOTTOM not in det_for_class:
        return _result_without_update(engine_fields)

    engine_fields.decelerator.state = instruction_pb2.Decelerator.State.BOTTOM
    hologram_updater = _HologramUpdater(engine_fields)
    hologram_updater.update_holo_location(det_for_class[BOTTOM], BIG_GEARING_HOLO)

    return _result_with_update(engine_fields, BOTTOM_AND_BIG_GEARING)


def _bottom_result(det_for_class, engine_fields):
    if BOTTOM_AND_BIG_GEARING not in det_for_class:
        if BOTTOM in det_for_class:

            # We have to increase this so the client will process the hologram
            # update
            engine_fields.update_count += 1

            hologram_updater = _HologramUpdater(engine_fields)
            hologram_updater.update_holo_location(
                det_for_class[BOTTOM], BIG_GEARING_HOLO)
        return _result_without_update(engine_fields)

    engine_fields.decelerator.state = instruction_pb2.Decelerator.State.BOTTOM_AND_BIG_GEARING
    hologram_updater = _HologramUpdater(engine_fields)
    hologram_updater.update_holo_location(det_for_class[BOTTOM_AND_BIG_GEARING], SMALL_GEARING_HOLO)

    return _result_with_update(engine_fields, BOTTOM_AND_TWO_GEARING)


def _small_gearing_helper(det_for_class, engine_fields):
    engine_fields.decelerator.state = instruction_pb2.Decelerator.State.BOTTOM_AND_TWO_GEARING
    hologram_updater = _HologramUpdater(engine_fields)
    hologram_updater.update_holo_location(det_for_class[BOTTOM_AND_TWO_GEARING], BOTTOM_HOLO)
    return _result_with_update(engine_fields, LID)


def _big_gearing_result(det_for_class, engine_fields):
    if BOTTOM_AND_TWO_GEARING in det_for_class:
        return _small_gearing_helper(det_for_class, engine_fields)
    elif (BOTTOM_AND_BIG_GEARING not in det_for_class) and (BOTTOM not in det_for_class):
        return _nothing_result(det_for_class, engine_fields)

    if BOTTOM_AND_BIG_GEARING in det_for_class:
        engine_fields.update_count += 1
        hologram_updater = _HologramUpdater(engine_fields)
        hologram_updater.update_holo_location(det_for_class[BOTTOM_AND_BIG_GEARING], SMALL_GEARING_HOLO)
    return _result_without_update(engine_fields)


def _lid_helper(det_for_class, engine_fields):
    engine_fields.decelerator.state = instruction_pb2.Decelerator.State.DONE
    hologram_updater = _HologramUpdater(engine_fields)
    hologram_updater.update_holo_location(det_for_class[DONE], LID_HOLO)
    return _result_with_update(engine_fields, DONE)
  
  
def _small_gearing_result(det_for_class, engine_fields):
    if LID in det_for_class:
        return _lid_helper(det_for_class, engine_fields)
    elif (BOTTOM_AND_BIG_GEARING not in det_for_class) and (BOTTOM_AND_TWO_GEARING not in det_for_class):
        return _big_gearing_result(det_for_class, engine_fields)
    
    if BOTTOM_AND_TWO_GEARING in det_for_class:
        engine_fields.update_count += 1
        hologram_updater = _HologramUpdater(engine_fields)
        hologram_updater.update_holo_location(det_for_class[BOTTOM_AND_TWO_GEARING], LID_HOLO)
    return _result_without_update(engine_fields)


def _lid_result(det_for_class, engine_fields):
    if TOTAL in det_for_class:
        engine_fields.decelerator.state = instruction_pb2.Decelerator.State.DONE
        return _result_with_update(engine_fields, DONE)
    elif (BOTTOM_AND_TWO_GEARING not in det_for_class) and (LID not in det_for_class):
        return _small_gearing_result(det_for_class, engine_fields)
    
    if TOTAL in det_for_class:
        engine_fields.update_count += 1
        hologram_updater = _HologramUpdater(engine_fields)
        hologram_updater.update_holo_location(det_for_class[LID], LID_HOLO)  ## !
    return _result_without_update(engine_fields)


class _HologramUpdater:
    def __init__(self, engine_fields):
        self._engine_fields = engine_fields

    def update_holo_location(self, det, holo):
        """
        det : the location of an item
        the location is depicted by vertex of the diagonal

        """
        '''
        print('============')
        print('det:{}'.format(det))
        print('holo:{}'.format(holo))
        print('============')
        '''
        
        x1, y1, x2, y2 = det[:4]
        x = x1 * (1 - holo.x) + x2 * holo.x
        y = y1 * (1 - holo.y) + y2 * holo.y
        area = abs((y2 - y1) * (x2 - x1))
        if (area == 0): area = area + 0.1
        
        depth = math.sqrt(holo.dist / float(area))
        self._engine_fields.decelerator.holo_x = x
        self._engine_fields.decelerator.holo_y = y
        self._engine_fields.decelerator.holo_depth = depth


def get_instruction(engine_fields, det_for_class):
    state = engine_fields.decelerator.state


    if state == instruction_pb2.Decelerator.State.START:
        return _start_result(engine_fields)

    if len(det_for_class) < 1:
        return _result_without_update(engine_fields)

    if state == instruction_pb2.Decelerator.State.NOTHING:
        return _nothing_result(det_for_class, engine_fields)
    elif state == instruction_pb2.Decelerator.State.BOTTOM:
        return _bottom_result(det_for_class, engine_fields)
    elif state == instruction_pb2.Decelerator.State.BOTTOM_AND_BIG_GEARING:
        return _big_gearing_result(det_for_class, engine_fields)
    elif state == instruction_pb2.Decelerator.State.BOTTOM_AND_TWO_GEARING:
        return _small_gearing_result(det_for_class, engine_fields)
    elif state == instruction_pb2.Decelerator.State.LID:
        return _lid_result(det_for_class, engine_fields)
    elif state == instruction_pb2.Decelerator.State.TOTAL:
        return _result_without_update(engine_fields)

    raise Exception("Invalid state")