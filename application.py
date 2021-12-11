from cv2 import VideoCapture
from flask import Flask, app, render_template, request, Response, redirect, url_for
from flask_bootstrap import Bootstrap

from object_detection import *

import pafy

# For youtube streams use pafy
url = "https://www.youtube.com/watch?v=J6LiOrQoih4"
best = pafy.new(url).getbest(preftype="mp4")


url = "https://www.youtube.com/watch?v=AdUw5RdyZxI"
play = pafy.new(url).getbest(preftype="mp4")


application = Flask(__name__)
Bootstrap(application)

#if not using youtube streams and using othere streams like rtsp then use
# VIDEO = VideoStreaming(SOURCE = 'YOUR RTSP STREAM LINK')
# EXAMPLE:  VIDEO = VideoStreaming(SOURCE = '')
VIDEO2 = VideoStreaming(SOURCE=best.url)
VIDEO = VideoStreaming(SOURCE=play.url)


@application.route('/')
def home():
    TITLE = 'Object Detection'
    return render_template('index.html', TITLE=TITLE)


@application.route('/video_feed')
def video_feed():
    '''
    Video streaming route.
    '''
    return Response(
        VIDEO.show(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@application.route('/video_feed2')
def video_feed2():
    '''
    Video streaming route.
    '''
    return Response(
        VIDEO2.show(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )



if __name__ == '__main__':
    application.run(debug=True)
