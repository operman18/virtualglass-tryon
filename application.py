"""
Serve webcam images (from a remote socket dictionary server)
using Tornado (to a WebSocket browser client.)

Usage:

   python server.py <host> <port>

"""

# Import standard modules.
import sys
import io as StringIO
from PIL import Image

# Import 3rd-party modules.
from head_pose_estimation import PnpHeadPoseEstimator
from tornado import websocket, web, ioloop, wsgi
import numpy as np
import coils
from stream_pose import StreamProcessor

import os
import json

from imutils import face_utils
import dlib

import  cv2
from time import time

import base64
from io import BytesIO

import pdb

app_directory = os.path.dirname(os.path.abspath(__file__))
LANDMARK_FILE = os.path.join(app_directory,'files/shape_predictor_68_face_landmarks.dat')

class IndexHandler(web.RequestHandler):
    def get(self):
        self.render('public/index.html')

class SocketHandler(websocket.WebSocketHandler):
    def __init__(self, *args, **kwargs):
        super(SocketHandler, self).__init__(*args, **kwargs)
        
        # Client to the socket server.
        # self._map_client = coils.MapSockClient(host, port, encode=False)
        # Monitor the framerate at 1s, 5s, 10s intervals.
        self._fps = coils.RateTicker((1,5,10))
        
        self.now = time()
        self.speed = 0
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(LANDMARK_FILE)
        
        self.rects = []
        self.shape = 0
        self.tvec = np.array([160,120,-300])
        self.rvec = np.array([0,0,0])
        
        self.cam_w = 320
        self.cam_h = 240
        
        self.aT = np.array([2,2,1])
        self.bT = np.array([-320,-250,0])

        self.aR = np.array([-1.0,0.75,-1.])
        self.bR = np.array([0,0,0])

        self.poseEstimator = PnpHeadPoseEstimator(self.cam_w,self.cam_h)
        self.stream_processor = StreamProcessor(np.zeros((self.cam_w,self.cam_h)))
        
        self.computing = False
        self.rvecInc=0
        self.tvecInc=0
        self.found=None
        self.image = ""
        
        self.max_time = 0 

    def on_message(self, data):
        if data == '1':
            self.write_message("1")
            return 

        msg = json.loads(data)
        image_str = msg['image']
        date = msg['timestamp']

        if date > self.max_time:
            image = Image.open(StringIO.BytesIO(base64.b64decode(image_str.encode('ascii'))))
            #image.save('tmp.png','PNG')

            frame = np.array(image)

            # Swap red and blue channel
            red = frame[:,:,0]
            blue = frame[:,:,2]
            frame[:,:,0] = blue
            frame[:,:,2] = red
    
            frame_old = self.stream_processor.draw_shapes(frame.copy())
            # print(base64.b64encode(frame_old))
            pose_old, is_new = self.stream_processor.get_last_pose()

            if pose_old is not None and is_new:
                # print(pose_old)
                self.rvec, self.tvec = post_process(pose_old[0],pose_old[1],self.aT,self.bT,self.aR,self.bR)
                # print(self.tvec)
                self.found = True
                #print(self.rvec)
                self.image = image_str #message
                image_str_new = image_str
            else:
                blur = cv2.blur(np.array(image),(15,15))
                image_str_new = array2string(blur)
                self.found = False

            posSizRot={
                'position':{ 'x': float(self.tvec[0]), 'y': float(self.tvec[1]), 'z': float(self.tvec[2]) }
                ,'rotation':{ 'x': float(self.rvec[0]), 'y': float(self.rvec[1]), 'z': float(self.rvec[2])}
                ,'size':{ 'x':120*700/self.tvec[2] }
                ,'image':"data:image/jpeg;base64,"+image_str_new
                ,'speed':self.speed
                ,'state':self.found
                ,"timestamp":date
            }        
        
            #print "After count update"
            self.write_message(json.dumps(posSizRot))
            self.now = time()
        
            rate1,rate5,rate10 = self._fps.tick()
            self.speed = rate1
        
            # Print object ID and the framerate.
            text = '{} {:.2f}, {:.2f}, {:.2f} fps'.format(id(self), rate1 , rate5 , rate10 )
            print( text)


def post_process(rvecInc,tvecInc,aT,bT,aR,bR):
    rvec = aR*rvecInc+bR
    tvecInc = aT*tvecInc+bT
    tvec = tvecInc
    return rvec,tvec


def polar2vec(p,thtx,thty):
    return p*np.array([np.cos(thtx)*np.sin(thty),-np.sin(thtx),-np.cos(thtx)*np.cos(thty)])

def array2string(img):
    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return new_image_string

# Retrieve command line arguments.


public_root = os.path.join(os.path.dirname(__file__), 'public')

settings = dict(
  debug=True,
  template_path=public_root
)

handlers = [
    (r'/', IndexHandler),
    (r'/ws', SocketHandler),
    (r'/(.*)', web.StaticFileHandler, {'path': public_root})
    ]

tornado_app = web.Application(handlers)

if __name__ == '__main__':
    tornado_app.listen(8081)
    ioloop.IOLoop.instance().start()

