import time
import cv2
import numpy as np
from datetime import date, datetime

from numpy.core.arrayprint import DatetimeFormat

from utils.VideoGet import VideoGet
from utils.VideoShow import VideoShow


class ObjectDetection:
    def __init__(self):

        count = cv2.cuda.getCudaEnabledDeviceCount()
        if(count > 0):
            print(r"""
                ▒█▀▀█ ▒█▀▀█ ▒█░▒█ ▄ 　 ▒█▀▀▀ ▒█▄░▒█ ░█▀▀█ ▒█▀▀█ ▒█░░░ ▒█▀▀▀ ▒█▀▀▄ 
                ▒█░▄▄ ▒█▄▄█ ▒█░▒█ ░ 　 ▒█▀▀▀ ▒█▒█▒█ ▒█▄▄█ ▒█▀▀▄ ▒█░░░ ▒█▀▀▀ ▒█░▒█ 
                ▒█▄▄█ ▒█░░░ ░▀▄▄▀ ▀ 　 ▒█▄▄▄ ▒█░░▀█ ▒█░▒█ ▒█▄▄█ ▒█▄▄█ ▒█▄▄▄ ▒█▄▄▀
                """)
        net = cv2.dnn.readNet("models/yolov4.cfg", "models/yolov4.weights")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.CONFIDENCE_THRESHOLD = 0.2
        self.NMS_THRESHOLD = 0.4

        self.MODEL = cv2.dnn_DetectionModel(net)
        self.MODEL.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

        self.CLASSES = []
        with open("models/coco.names", "r") as f:
            self.CLASSES = [line.strip() for line in f.readlines()]

        # self.OUTPUT_LAYERS = [self.MODEL.getLayerNames()[i[0] - 1] for i in self.MODEL.getUnconnectedOutLayers()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self.COLORS /= (np.sum(self.COLORS**2, axis=1)**0.5/255)[np.newaxis].T

        self.prevDetectionList = []

    def checkNumpySimilarity(self, A, B):
        number_of_equal_elements = np.sum(A == B)
        total_elements = A.size
        percentage = number_of_equal_elements/total_elements

        # print('total number of elements: \t\t{}'.format(total_elements))
        # print('number of identical elements: \t\t{}'.format(
        #     number_of_equal_elements))
        # print('number of different elements: \t\t{}'.format(
        #     total_elements-number_of_equal_elements))
        # print('percentage of identical elements: \t{:.2f}%'.format(
        #     percentage*100))

        return percentage

    def logDetection(self, classes, scores, boxes):
        OUTPUTLOG = open('detection.log', 'a')
        label = None
        for (classid, score, box) in zip(classes, scores, boxes):
            label = ("%s : %f" % (self.CLASSES[classid[0]], score))
            if(len(self.prevDetectionList) > 0):
                for item in self.prevDetectionList:
                    if((item["label"] in label) or (self.checkNumpySimilarity(item["box"], box) >= 0) or (item["bigBox"] > box)):
                        if(int(time.time()) > item["time"]+5):
                            self.prevDetectionList.append({
                                "label": label,
                                "box": box,
                                "time": int(time.time()),
                                "bigBox": box+10
                            })
                            self.prevDetectionList.remove(item)
                            OUTPUTLOG.write(
                                "TIME: " + str(datetime.now())+" DETECTION: " + str(label) + "\n")
                        else:
                            pass
                            # print("OLD DETECTION FOUND:", item)

                    else:
                        # print('NO OLD FOUND')
                        self.prevDetectionList.append({
                            "label": label,
                            "box": box,
                            "time": int(time.time()),
                            "bigBox": box+10
                        })
                        OUTPUTLOG.write(
                            "TIME: " + str(datetime.now())+" DETECTION: " + str(label) + "\n")
            else:
                
                OUTPUTLOG.write(
                    "TIME: " + str(datetime.now())+" DETECTION: " + str(label) + "\n")
                self.prevDetectionList.append({
                    "label": label,
                    "box": box,
                    "time": int(time.time()),
                    "bigBox": box+20
                })
            

            

    def detectObj(self, snap):
        start = time.time()
        classes, scores, boxes = self.MODEL.detect(
            snap, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        end = time.time()

        self.logDetection(classes, scores, boxes)

        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            color = self.COLORS[int(classid) % len(self.COLORS)]
            label = "%s : %f" % (self.CLASSES[classid[0]], score)
            cv2.rectangle(snap, box, color, 2)
            cv2.putText(snap, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        end_drawing = time.time()

        # fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
        fps_label = "FPS: %.2f" % (1 / (end - start))
        cv2.putText(snap, fps_label, (0, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return snap


class VideoStreaming(object):
    def __init__(self, SOURCE):
        super(VideoStreaming, self).__init__()

        # self.VIDEO = cv2.VideoCapture(self.URL)

        self.video_getter = VideoGet(SOURCE).start()

        self.MODEL = ObjectDetection()

        self._preview = True
        self._flipH = False
        self._detect = True
        self._exposure = self.video_getter.stream.get(cv2.CAP_PROP_EXPOSURE)
        self._contrast = self.video_getter.stream.get(cv2.CAP_PROP_CONTRAST)

    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        self._preview = bool(value)

    @property
    def flipH(self):
        return self._flipH

    @flipH.setter
    def flipH(self, value):
        self._flipH = bool(value)

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        self._detect = bool(value)

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, value):
        self._exposure = value
        self.video_getter.stream.set(cv2.CAP_PROP_EXPOSURE, self._exposure)

    @property
    def contrast(self):
        return self._contrast

    @contrast.setter
    def contrast(self, value):
        self._contrast = value
        self.video_getter.stream.set(cv2.CAP_PROP_CONTRAST, self._contrast)

    def show(self):
        
        while(self.video_getter.stream.isOpened()):
            if (cv2.waitKey(1) == ord("q")) or self.video_getter.stopped:
                self.video_getter.stop()
                break
            snap = self.video_getter.frame
            ret = self.video_getter.grabbed
            if self.flipH:
                snap = cv2.flip(snap, 1)

            if ret == True:
                if self._preview:
                    # snap = cv2.resize(snap, (0, 0), fx=0.5, fy=0.5)
                    if self.detect:
                        snap = self.MODEL.detectObj(snap)

                else:
                    snap = np.zeros((
                        int(self.video_getter.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        int(self.video_getter.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ), np.uint8)
                    label = 'camera disabled'
                    H, W = snap.shape
                    font = cv2.FONT_HERSHEY_PLAIN
                    color = (255, 255, 255)
                    cv2.putText(snap, label, (W//2 - 100, H//2),
                                font, 2, color, 2)

                frame = cv2.imencode('.jpg', snap)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.01)

            else:
                break
        print('off')
