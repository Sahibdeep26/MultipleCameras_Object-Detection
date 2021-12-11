"""
This script downloads the weight file
"""
import requests

URL = "https://pjreddie.com/media/files/yolov3.weights"
r = requests.get(URL, allow_redirects=True)
open('yolov3.weights', 'wb').write(r.content)


URL2 = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
r = requests.get(URL2, allow_redirects=True)
open('yolov4.weights', 'wb').write(r.content)