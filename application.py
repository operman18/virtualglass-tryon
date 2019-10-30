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

import os
import json

from imutils import face_utils
import dlib

import  cv2
from time import time

import base64
import pdb

app_directory = os.path.dirname(os.path.abspath(__file__))

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
        self.time = time()
        self.count = 10**20
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(app_directory,'files/shape_predictor_68_face_landmarks.dat'))
        
        self.rects = []
        self.shape = 0
        self.tvec = np.array([160,120,-300])
        self.rvec = np.array([0,0,0])
        
        self.cam_w = 320
        self.cam_h = 240
        
        self.aT=np.array([1,-1,-1])
        self.bT=np.array([0,20,-240])
        self.aR=np.array([1,-1,-0.2])
        self.bR=np.array([-1.15,-1.23,1.15])
        self.poseEstimator=PnpHeadPoseEstimator(self.cam_w,self.cam_h)
        self.speed = 1
        
        self.computing = False
        self.rvecInc=0
        self.tvecInc=0
        self.found=None

    def on_message(self, data):
        if data == '1':
            self.write_message("1")
            return 

        msg = json.loads(data)
        message = msg['image']
        date = msg['date']
        if len(message)>2000 : # and self.speed>count :
            image = Image.open(StringIO.BytesIO(base64.b64decode(message.encode('ascii'))))
            cv_image = np.array(image)
            
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            # pdb.set_trace()
            # detect faces in the grayscale frame
            self.rects = self.detector(gray, 0)
            if self.rects:
                self.found = True
            else:
                self.found = False
                # loop over the face detections
            for rect in self.rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shapeInc = self.predictor(gray, rect)
                shapeInc = face_utils.shape_to_np(shapeInc)
                
                print( "Face found")
                self.shape = self.shape*0.+shapeInc*1.
                rvecInc,tvecInc=self.poseEstimator.return_roll_pitch_yaw(self.shape,self.cam_w,self.cam_h)
                pose = self.poseEstimator.return_roll_pitch_yaw_slow(self.shape,self.cam_w,self.cam_h)
                self.rvecInc = pose.get_rotation_euler_angles()
                self.rvec = self.rvec*0+self.rvecInc*1
                tvecInc = self.aT*(tvecInc+self.bT)
                self.tvec = self.tvec*0+tvecInc*1
        
        tvec = self.tvec + polar2vec(50,-self.rvec[0],-self.rvec[1])
        posSizRot={
            'position':{ 'x': tvec[0], 'y': tvec[1], 'z': tvec[2] }
            ,'rotation':{ 'x': float(self.rvec[0]), 'y': float(self.rvec[1]), 'z': float(self.rvec[2])}
            ,'size':{ 'x': 150 }
            ,'image':"data:image/jpeg;base64,"+message
            ,'speed':self.speed
            ,'state':self.found
            ,'timestamp':date
        }        
        
        #print "After count update"
        self.write_message(json.dumps(posSizRot))
        self.time = time()
        
        rate1,rate5,rate10 = self._fps.tick()
        self.speed = rate1
        
        # Print object ID and the framerate.
        text = '{} {:.2f}, {:.2f}, {:.2f} fps'.format(id(self), rate1 , rate5 , rate10 )
        print( text)


def polar2vec(p,thtx,thty):
    return p*np.array([np.cos(thtx)*np.sin(thty),-np.sin(thtx),-np.cos(thtx)*np.cos(thty)])

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

application = wsgi.WSGIAdapter(tornado_app)

if __name__ == '__main__':
    tornado_app.listen(8080)
    ioloop.IOLoop.instance().start()

