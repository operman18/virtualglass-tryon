#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2018 Operman Levy
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import  cv2
import eos

import os
import pdb

#Enbale if you need printing utilities
DEBUG = False

app_directory = os.path.dirname(os.path.abspath(__file__))

MODEL = os.path.join(app_directory,"files/sfm_shape_3448.bin")
BLENDSHAPES = os.path.join(app_directory,"files/expression_blendshapes_3448.bin")

L_MAPPER = os.path.join(app_directory,"files/ibug_to_sfm.txt")
TOPOLOGY = os.path.join(app_directory,"files/sfm_3448_edge_topology.json")
CONTOUR_L = os.path.join(app_directory,"files/ibug_to_sfm.txt")
CONTOUR_M = os.path.join(app_directory,"files/sfm_model_contours.json")

class PnpHeadPoseEstimator:
    """ Head pose estimation class which uses the OpenCV PnP algorithm.

        It finds Roll, Pitch and Yaw of the head given a figure as input.
        It uses the PnP algorithm and it requires the dlib library
    """

    def __init__(self, cam_w, cam_h):
        """ Init the class

        @param cam_w the camera width. If you are using a 640x480 resolution it is 640
        @param cam_h the camera height. If you are using a 640x480 resolution it is 480
        @dlib_shape_predictor_file_path path to the dlib file for shape prediction (look in: deepgaze/etc/dlib/shape_predictor_68_face_landmarks.dat)
        """
        
        #Defining the camera matrix.
        #To have better result it is necessary to find the focal
        # lenght of the camera. fx/fy are the focal lengths (in pixels) 
        # and cx/cy are the optical centres. These values can be obtained 
        # roughly by approximation, for example in a 640x480 camera:
        # cx = 640/2 = 320
        # cy = 480/2 = 240
        # fx = fy = cx/tan(60/2 * pi / 180) = 554.26
        c_x = cam_w / 2
        c_y = cam_h / 2
        f_x = c_x / np.tan(60/2 * np.pi / 180)
        f_y = f_x
        
        #Estimated camera matrix values.
        self.camera_matrix = np.float32([[f_x, 0.0, c_x],
                                         [0.0, f_y, c_y], 
                                         [0.0, 0.0, 1.0] ])
        #These are the camera matrix values estimated on my webcam with
        # the calibration code (see: src/calibration):
        #camera_matrix = np.float32([[602.10618226,          0.0, 320.27333589],
        #[         0.0, 603.55869786,  229.7537026], 
        #[         0.0,          0.0,          1.0] ])
        
        #Distortion coefficients
        self.camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
        #Distortion coefficients estimated by calibration in my webcam
        #camera_distortion = np.float32([ 0.06232237, -0.41559805,  0.00125389, -0.00402566,  0.04879263])

        if(DEBUG==True): print("[DEEPGAZE] PnpHeadPoseEstimator: estimated camera matrix: \n" + str(self.camera_matrix) + "\n")
        # load face models
        
        self.model = eos.morphablemodel.load_model(MODEL)
        self.blendshapes = eos.morphablemodel.load_blendshapes(BLENDSHAPES)
        self.morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(self.model.get_shape_model(), self.blendshapes,
                                                                        eos.morphablemodel.PcaModel(),
                                                                        self.model.get_texture_coordinates())
        self.landmark_mapper = eos.core.LandmarkMapper(L_MAPPER)
        self.edge_topology = eos.morphablemodel.load_edge_topology(TOPOLOGY)
        self.contour_landmarks = eos.fitting.ContourLandmarks.load(CONTOUR_L)
        self.model_contour = eos.fitting.ModelContour.load(CONTOUR_M)
        self.landmark_ids = list(map(str, range(1, 69))) # generates the numbers 1 to 68, as strings        
        
        
        
        
        
        
    def return_roll_pitch_yaw(self, landmarks, img_w, img_h, img_d=3, radians=True):
         """ Return the the roll pitch and yaw angles associated with the input image.
         
         @param image It is a colour image. It must be >= 64 pixel.
         @param radians When True it returns the angle in radians, otherwise in degrees.
         """
         
         #The dlib shape predictor returns 68 points, we are interested only in a few of those
         TRACKED_POINTS = [0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62]
         
         #Antropometric constant values of the human head. 
         #Check the wikipedia EN page and:
         #"Head-and-Face Anthropometric Survey of U.S. Respirator Users"
         #
         #X-Y-Z with X pointing forward and Y on the left and Z up.
         #The X-Y-Z coordinates used are like the standard
         # coordinates of ROS (robotic operative system)
         #OpenCV uses the reference usually used in computer vision: 
         #X points to the right, Y down, Z to the front
         #
         #The Male mean interpupillary distance is 64.7 mm (https://en.wikipedia.org/wiki/Interpupillary_distance)
         #
         P3D_RIGHT_SIDE = np.float32([-100.0, -77.5, -5.0]) #0
         P3D_GONION_RIGHT = np.float32([-110.0, -77.5, -85.0]) #4
         P3D_MENTON = np.float32([0.0, 0.0, -122.7]) #8
         P3D_GONION_LEFT = np.float32([-110.0, 77.5, -85.0]) #12
         P3D_LEFT_SIDE = np.float32([-100.0, 77.5, -5.0]) #16
         P3D_FRONTAL_BREADTH_RIGHT = np.float32([-20.0, -56.1, 10.0]) #17
         P3D_FRONTAL_BREADTH_LEFT = np.float32([-20.0, 56.1, 10.0]) #26
         P3D_SELLION = np.float32([0.0, 0.0, 0.0]) #27 This is the world origin
         P3D_NOSE = np.float32([21.1, 0.0, -48.0]) #30
         P3D_SUB_NOSE = np.float32([5.0, 0.0, -52.0]) #33
         P3D_RIGHT_EYE = np.float32([-20.0, -32.35,-5.0]) #36 
         P3D_RIGHT_TEAR = np.float32([-10.0, -20.25,-5.0]) #39
         P3D_LEFT_TEAR = np.float32([-10.0, 20.25,-5.0]) #42
         P3D_LEFT_EYE = np.float32([-20.0, 32.35,-5.0]) #45
         #P3D_LIP_RIGHT = np.float32([-20.0, 65.5,-5.0]) #48
         #P3D_LIP_LEFT = np.float32([-20.0, 65.5,-5.0]) #54
         P3D_STOMION = np.float32([10.0, 0.0, -75.0]) #62
         
         #This matrix contains the 3D points of the
         # 11 landmarks we want to find. It has been
         # obtained from antrophometric measurement
         # of the human head.
         landmarks_3D = np.float32([P3D_RIGHT_SIDE,
                                  P3D_GONION_RIGHT,
                                  P3D_MENTON,
                                  P3D_GONION_LEFT,
                                  P3D_LEFT_SIDE,
                                  P3D_FRONTAL_BREADTH_RIGHT,
                                  P3D_FRONTAL_BREADTH_LEFT,
                                  P3D_SELLION,
                                  P3D_NOSE,
                                  P3D_SUB_NOSE,
                                  P3D_RIGHT_EYE,
                                  P3D_RIGHT_TEAR,
                                  P3D_LEFT_TEAR,
                                  P3D_LEFT_EYE,
                                  P3D_STOMION])
         
         #Return the 2D position of our landmarks
         landmarks_2D = landmarks[TRACKED_POINTS]
         
         #Print som red dots on the image       
         #for point in landmarks_2D:
             #cv2.circle(frame,( point[0], point[1] ), 2, (0,0,255), -1)
             
             
         #Applying the PnP solver to find the 3D pose
         #of the head from the 2D position of the
         #landmarks.
         #retval - bool
         #rvec - Output rotation vector that, together with tvec, brings 
         #points from the world coordinate system to the camera coordinate system.
         #tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
         retval, rvec, tvec = cv2.solvePnP(landmarks_3D, 
                                           landmarks_2D, 
                                           self.camera_matrix, 
                                           self.camera_distortion)
         
         return rvec.T[0],tvec.T[0]
     
    def return_roll_pitch_yaw_slow(self,landmarks,cam_w,cam_h):
        landmarks_new = [eos.core.Landmark(str(i+1),[l[0],l[1]]) for i,l in enumerate(landmarks)]
        (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(self.morphablemodel_with_expressions, landmarks_new, self.landmark_mapper, cam_w, cam_h, self.edge_topology, self.contour_landmarks, self.model_contour, num_iterations=1)
        return pose
