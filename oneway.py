"""
Copyright (C) 2020 Intel Corporation

SPDX-License-Identifier: BSD-3-Clause
"""

from openvino.inference_engine import IENetwork, IECore
from libs.person_trackers import PersonTrackers, TrackableObject
from libs.geometric import get_polygon, get_point, get_line
from libs.draw import Draw
from collections import OrderedDict
import math
import cv2
import json
import os
from libs.validate import validate


class OneWay(object):
    def __init__(self):
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        cfg = json.load(open(config_file_path))
        validate(cfg)
        self.running = True
        self.videosource = cfg.get("video")
        self.model_modelfile = cfg.get("pedestrian_model_weights")
        self.model_configfile = cfg.get("pedestrian_model_description")
        self.model_modelfile_reid = cfg.get("reidentification_model_weights")
        self.model_configfile_reid = cfg.get("reidentification_model_description")
        self.coords = cfg.get("coords")
        self.area = cfg.get("area")

        # OPENVINO VARS
        self.ov_input_blob = None
        self.out_blob = None
        self.net = None
        self.ov_n = None
        self.ov_c = None
        self.ov_h = None
        self.ov_w = None
        self.ov_input_blob_reid = None
        self.out_blob_reid = None
        self.net_reid = None
        self.ov_n_reid = None
        self.ov_c_reid = None
        self.ov_h_reid = None
        self.ov_w_reid = None
        self.confidence_threshold = .85

        self.line = None
        self.polygon = None
        self.direction = ""
        self.opposite = ""
        self.samples_quantity = 10
        self.threshold_displacement = 20

        self.trackers = PersonTrackers(OrderedDict())

    def load_openvino(self):
        try:
            ie = IECore()
            net = ie.read_network(model=self.model_configfile, weights=self.model_modelfile)
            self.ov_input_blob = next(iter(net.inputs))
            self.out_blob = next(iter(net.outputs))
            self.net = ie.load_network(network=net, num_requests=2, device_name="CPU")
            # Read and pre-process input image
            self.ov_n, self.ov_c, self.ov_h, self.ov_w = net.inputs[self.ov_input_blob].shape
            del net
        except Exception as e:
            raise Exception(f"Load Openvino error:{e}")
        self.load_openvino_reid()

    def load_openvino_reid(self):
        try:
            ie = IECore()
            net = ie.read_network(model=self.model_configfile_reid, weights=self.model_modelfile_reid)
            self.ov_input_blob_reid = next(iter(net.inputs))
            self.out_blob_reid = next(iter(net.outputs))
            self.net_reid = ie.load_network(network=net, num_requests=2, device_name="CPU")
            # Read and pre-process input image
            self.ov_n_reid, self.ov_c_reid, self.ov_h_reid, self.ov_w_reid = net.inputs[self.ov_input_blob_reid].shape
            del net
        except Exception as e:
            raise Exception(f"Load Openvino reidentification error:{e}")

    def set_direction(self):
        dify = (self.coords[1][1] - self.coords[0][1]) ** 2
        difx = (self.coords[1][0] - self.coords[0][0]) ** 2
        if dify < difx:
            self.direction = "R" if self.coords[1][0] - self.coords[0][0] > 0 else "L"
            self.opposite = "R" if self.direction == "L" else "R"
        else:
            self.direction = "D" if self.coords[1][1] - self.coords[0][1] > 0 else "U"
            self.opposite = "U" if self.direction == "D" else "D"
        return self.direction

    def get_direction(self, direction):
        if self.direction in ["U", "D"]:
            if direction > 0:
                return "D"
            elif direction < 0:
                return "U"
        else:
            if direction > 0:
                return "R"
            elif direction < 0:
                return "L"
        return None

    def config_env(self, frame):
        h, w = frame.shape[:2]
        line = ((int(self.coords[0][0] * w / 100), int(self.coords[0][1] * h / 100)),
                (int(self.coords[1][0] * w / 100), int(self.coords[1][1] * h / 100)))
        self.coords = line
        line = get_line(line)
        self.line = line
        self.polygon = line.buffer(self.area)
        self.polygon = get_polygon(list(self.polygon.exterior.coords))

    def get_frame(self):
        h = w = None
        try:
            cap = cv2.VideoCapture(self.videosource)
        except Exception as e:
            raise Exception(f"Video source error: {e}")

        while self.running:
            has_frame, frame = cap.read()
            if has_frame:
                if frame.shape[1] > 2000:
                    frame = cv2.resize(frame, (int(frame.shape[1] * .3), int(frame.shape[0] * .3)))
                elif frame.shape[1] > 1000:
                    frame = cv2.resize(frame, (int(frame.shape[1] * .8), int(frame.shape[0] * .8)))
                if w is None or h is None:
                    h, w = frame.shape[:2]
                    self.config_env(frame)
                yield frame
            else:
                self.running = False
        return None

    def process_frame(self, frame):

        _frame = frame.copy()
        trackers = []

        frame = cv2.resize(frame, (self.ov_w, self.ov_h))
        frame = frame.transpose((2, 0, 1))
        frame = frame.reshape((self.ov_n, self.ov_c, self.ov_h, self.ov_w))
        self.net.start_async(request_id=0, inputs={self.ov_input_blob: frame})

        if self.net.requests[0].wait(-1) == 0:
            res = self.net.requests[0].outputs[self.out_blob]

            frame = _frame
            h, w = frame.shape[:2]
            out = res[0][0]
            for i, detection in enumerate(out):

                confidence = detection[2]
                if confidence > self.confidence_threshold and int(detection[1]) == 1:  # 1 => CLASS Person

                    xmin = int(detection[3] * w)
                    ymin = int(detection[4] * h)
                    xmax = int(detection[5] * w)
                    ymax = int(detection[6] * h)

                    cX = int((xmin + xmax) / 2.0)
                    cY = int((ymin + ymax) / 2.0)
                    point = get_point([cX, cY])
                    if not self.polygon.contains(point):
                        continue

                    trackers.append(
                        TrackableObject((xmin, ymin, xmax, ymax), None, (cX, cY))
                    )

        for tracker in trackers:
            person = frame[tracker.bbox[1]:tracker.bbox[3], tracker.bbox[0]:tracker.bbox[2]]

            try:
                person = cv2.resize(person, (self.ov_w_reid, self.ov_h_reid))
            except cv2.error as e:
                print(f"CV2 RESIZE ERROR: {e}")
                continue

            person = person.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            person = person.reshape((self.ov_n_reid, self.ov_c_reid, self.ov_h_reid, self.ov_w_reid))
            self.net_reid.start_async(request_id=0, inputs={self.ov_input_blob: person})

            if self.net_reid.requests[0].wait(-1) == 0:
                res = self.net_reid.requests[0].outputs[self.out_blob_reid]
                tracker.reid = res

        self.trackers.similarity(trackers)

        if len(self.trackers.trackers) > 0:
            for trackId, v in self.trackers.trackers.items():

                if len(v.centroids) > self.samples_quantity:

                    centroids = v.centroids

                    if self.direction in ["U", "D"]:
                        y = [c[1] for c in centroids]
                        dire = sum(y[-2:]) / 2 - y[0]
                    else:
                        x = [c[0] for c in centroids]
                        dire = sum(x[-2:]) / 2 - x[0]

                    if math.sqrt(dire ** 2) < self.threshold_displacement:
                        continue

                    direction = self.get_direction(dire)

                    if direction == self.opposite and direction is not None:
                        Draw.rectangle(frame, (v.bbox[0], v.bbox[1], v.bbox[2], v.bbox[3]), "red", 2)
                        Draw.arrowed_line(frame,
                                          (v.centroids[-1:][0][0], v.centroids[-1:][0][1],
                                           v.centroids[-1:][0][0] + 60, v.centroids[-1:][0][1]),
                                          "red", tipLength=0.06, thickness=1)
                    else:
                        Draw.rectangle(frame, (v.bbox[0], v.bbox[1], v.bbox[2], v.bbox[3]), "green", 2)
                        Draw.arrowed_line(frame,
                                          (v.centroids[-1:][0][0], v.centroids[-1:][0][1],
                                           v.centroids[-1:][0][0] - 60, v.centroids[-1:][0][1]),
                                          "green", tipLength=0.06, thickness=1)

        Draw.arrowed_line(frame, (self.coords[0][0], self.coords[0][1], self.coords[1][0], self.coords[1][1]), "orange")
        return frame

    def render(self, frame):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            exit()

    def run(self):
        self.load_openvino()
        self.set_direction()
        for frame in self.get_frame():
            frame = self.process_frame(frame)
            self.render(frame)


if __name__ == "__main__":
    try:
        oneway = OneWay()
        oneway.run()
    except Exception as exception:
        print(exception)
