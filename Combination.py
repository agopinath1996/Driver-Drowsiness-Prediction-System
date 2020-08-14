#!/usr/bin/env python

##
# Massimiliano Patacchiola, Plymouth University 2016
#
# This is an example of head pose estimation with solvePnP.
# It uses the dlib library and openCV
# The original code has been modified to detect drowsy behavior
# requires deepgaze library
import os
import tensorflow as tf
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import numpy
import cv2
import sys
sys.path[0] = "/usr/local/lib/python2.7/dist-packages"
from deepgaze.haar_cascade import haarCascade
from deepgaze.face_landmark_detection import faceLandmarkDetection
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator





#If True enables the verbose mode
DEBUG = True

#Antropometric constant values of the human head.
#Found on wikipedia and on:
# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
#
#X-Y-Z with X pointing forward and Y on the left.
#The X-Y-Z coordinates used are like the standard
# coordinates of ROS (robotic operative system)
P3D_RIGHT_SIDE = numpy.float32([-100.0, -77.5, -5.0]) #0
P3D_GONION_RIGHT = numpy.float32([-110.0, -77.5, -85.0]) #4
P3D_MENTON = numpy.float32([0.0, 0.0, -122.7]) #8
P3D_GONION_LEFT = numpy.float32([-110.0, 77.5, -85.0]) #12
P3D_LEFT_SIDE = numpy.float32([-100.0, 77.5, -5.0]) #16
P3D_FRONTAL_BREADTH_RIGHT = numpy.float32([-20.0, -56.1, 10.0]) #17
P3D_FRONTAL_BREADTH_LEFT = numpy.float32([-20.0, 56.1, 10.0]) #26
P3D_SELLION = numpy.float32([0.0, 0.0, 0.0]) #27
P3D_NOSE = numpy.float32([21.1, 0.0, -48.0]) #30
P3D_SUB_NOSE = numpy.float32([5.0, 0.0, -52.0]) #33
P3D_RIGHT_EYE = numpy.float32([-20.0, -65.5,-5.0]) #36
P3D_RIGHT_TEAR = numpy.float32([-10.0, -40.5,-5.0]) #39
P3D_LEFT_TEAR = numpy.float32([-10.0, 40.5,-5.0]) #42
P3D_LEFT_EYE = numpy.float32([-20.0, 65.5,-5.0]) #45
#P3D_LIP_RIGHT = numpy.float32([-20.0, 65.5,-5.0]) #48
#P3D_LIP_LEFT = numpy.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = numpy.float32([10.0, 0.0, -75.0]) #62

#The points to track
#These points are the ones used by PnP
# to estimate the 3D pose of the face
TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
ALL_POINTS = list(range(0,68)) #Used for debug only


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def main():
    #Defining the video capture object
    video_capture = cv2.VideoCapture(1)
    thresh = 0.25
    frame_check = 15
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("/home/agopinath1996/git_ws/deepgaze/scripts/shape_predictor_68_face_landmarks.dat")# change to path where landmark points are stored

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]   # get the left eye index
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]  # get the right eye index
    flag=0

    sess = tf.Session()
    my_head_pose_estimator = CnnHeadPoseEstimator(sess)
    my_head_pose_estimator.load_roll_variables(os.path.realpath("/home/agopinath1996/git_ws/deepgaze/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))# change to deepgaze directory path
    my_head_pose_estimator.load_pitch_variables(os.path.realpath("/home/agopinath1996/git_ws/deepgaze/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))
    my_head_pose_estimator.load_yaw_variables(os.path.realpath("/home/agopinath1996/git_ws/deepgaze/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"))


    #Start of Eye Gaze Tracking
    win = dlib.image_window()
    
    predictor_path = "/home/agopinath1996/git_ws/deepgaze/scripts/shape_predictor_68_face_landmarks.dat"
    roi = []
    ref_point = 0 
    index1 = 0
    pt_lefteye_corner_x= 0
    pt_lefteye_corner_y = 0
    pt_pos1 = 0
    predictor = dlib.shape_predictor(predictor_path)
    pt_x2 =0
    pt_y2 = 0
    pt_x1 = 0
    pt_y1 = 0
    pt_actualx = 0
    pt_actualy = 0
    detector = dlib.get_frontal_face_detector()
    flag = 0
    flag1 = 0
    pt_righteye_corner_x = 0
    pt_righteye_corner_y = 0


    if(video_capture.isOpened() == False):
        print("Error: the resource is busy or unvailable")
    else:
        print("The video source has been opened correctly...")

    #Create the main window and move it
    cv2.namedWindow('Video')
    cv2.moveWindow('Video', 20, 20)

    #Obtaining the CAM dimension
    cam_w = int(video_capture.get(3))
    cam_h = int(video_capture.get(4))

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
    f_x = c_x / numpy.tan(60/2 * numpy.pi / 180)
    f_y = f_x

    #Estimated camera matrix values.
    camera_matrix = numpy.float32([[f_x, 0.0, c_x],
                                   [0.0, f_y, c_y],
                                   [0.0, 0.0, 1.0] ])

    print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")

    #These are the camera matrix values estimated on my webcam with
    # the calibration code (see: src/calibration):
    camera_matrix = numpy.float32([[602.10618226,          0.0, 320.27333589],
                                   [         0.0, 603.55869786,  229.7537026],
                                   [         0.0,          0.0,          1.0] ])

    #Distortion coefficients
    #camera_distortion = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    #Distortion coefficients estimated by calibration
    camera_distortion = numpy.float32([ 0.06232237, -0.41559805,  0.00125389, -0.00402566,  0.04879263])


    #This matrix contains the 3D points of the
    # 11 landmarks we want to find. It has been
    # obtained from antrophometric measurement
    # on the human head.
    landmarks_3D = numpy.float32([P3D_RIGHT_SIDE,
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

    #Declaring the two classifiers
    my_cascade = haarCascade("/home/agopinath1996/git_ws/deepgaze/etc/xml/haarcascade_frontalface_alt.xml", "/home/agopinath1996/git_ws/deepgaze/etc/xml/haarcascade_profileface.xml")
    #TODO If missing, example file can be retrieved from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    my_detector = faceLandmarkDetection('/home/agopinath1996/git_ws/deepgaze/scripts/shape_predictor_68_face_landmarks.dat')





    #Error counter definition
    no_face_counter = 0

    #Variables that identify the face
    #position in the main frame.
    face_x1 = 0
    face_y1 = 0
    face_x2 = 0
    face_y2 = 0
    face_w = 0
    face_h = 0

    #Variables that identify the ROI
    #position in the main frame.
    roi_x1 = 0
    roi_y1 = 0
    roi_x2 = cam_w
    roi_y2 = cam_h
    roi_w = cam_w
    roi_h = cam_h
    roi_resize_w = int(cam_w/10)
    roi_resize_h = int(cam_h/10)

    while(True):

        # Capture frame-by-frame
        ret, frame = video_capture.read()
        ret, frame_eye = video_capture.read()

        #print("Estimated [roll, pitch, yaw] ..... [" + str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")
        gray = cv2.cvtColor(frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)

        img = cv2.cvtColor(frame_eye, cv2.COLOR_RGB2BGR) #for eye gaze detection
        drowsyframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(drowsyframe, 0)
        for subject in subjects:
            shape = predict(frame,subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = leftEAR + rightEAR / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0),1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
            if ear<thresh:
                flag+= 1
                print(flag)
                if flag >= frame_check:
                    cv2.putText(frame, "WAKEUPPPP", (10,30), cv2.FONT_HERSHEY_PLAIN, 1.6, (10,10,255), 2)
                    cv2.putText(frame, "WAKEUPPPP", (10, 325), cv2.FONT_HERSHEY_PLAIN, 1.6, (10,10,255),2)
            else:
                flag=0

#Eye Gaze Detetction
        dets = detect(img, 0)
        check = 5
        shapes_eye = []
        for k,d in enumerate(dets):
            #print("dets{}".format(d))
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))

            shape_eye = predict(img, d)

            for index, pt in enumerate(shape_eye.parts()):
                #print('Part {}: {}'.format(index, pt))
                pt_pos = (pt.x, pt.y)
                cv2.circle(img, pt_pos, 1, (0,225, 0), 2)
                if index == 29:
                    pt_x2 = int(pt.x)
                    pt_y2 = int(pt.y)
                if index == 18:
                    pt_x1 = int(pt.x)
                    pt_y1 = int(pt.y)
                if index == 37:
                    pt_righteye_corner_x = pt.x
                    pt_righteye_corner_y = pt.y
                if index == 40:
                    pt_lefteye_corner_x = pt.x
                    pt_lefteye_corner_y = pt.y
                roi =  frame_eye[pt_y1:pt_y2,pt_x1:pt_x2]
                roi_gray = cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)
                _, threshold = cv2.threshold(roi_gray, 30, 255, cv2.THRESH_BINARY_INV)
                try:
                    M = cv2.moments(threshold)
                    #print(M)
                    cX = int(M["m10"]/M["m00"])
                    cY = int(M["m01"]/M["m00"])
                    #print(cX,cY)
                    pt_actualx = pt_x1+cX
                    pt_actualy = pt_y1+cY
                    #print(pt_actualx,pt_actualy)
                    diff_right = pt_actualx-pt_righteye_corner_x
                    diff_left = pt_lefteye_corner_x - pt_actualx
                    print(diff_right,diff_left)
                    #print(cX,cY)
                    if diff_right < 3:
                        cv2.putText(frame,'Look straight!',(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(10,10,255),2)
                    if diff_left <3:
                        cv2.putText(frame,'Look straight!',(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(10,10,255),2)


                except:
                    pass
                cv2.circle(frame,(pt_actualx,pt_actualy), 2,(255,0,255),-1)
                #print(pt_actualx,pt_actualy)


            #print(pt_x1,pt_x2,pt_y1,pt_y2)
            #print(roi.shape_eye)
            #print(img.shape_eye)
            try:
                cv2.imshow("threshold", threshold)
                cv2.waitKey(1)
            except:
                pass

        win.clear_overlay()
        win.set_image(img)
        if len(shapes_eye)!= 0 :
            for i in range(len(shapes_eye)):
                win.add_overlay(shapes_eye[i])





        #Looking for faces with cascade
        #The classifier moves over the ROI
        #starting from a minimum dimension and augmentig
        #slightly based on the scale factor parameter.
        #The scale factor for the frontal face is 1.10 (10%)
        #Scale factor: 1.15=15%,1.25=25% ...ecc
        #Higher scale factors means faster classification
        #but lower accuracy.
        #
        #Return code: 1=Frontal, 2=FrontRotLeft,
        # 3=FrontRotRight, 4=ProfileLeft, 5=ProfileRight.
        my_cascade.findFace(gray, True, True, True, True, 1.10, 1.10, 1.15, 1.15, 40, 40, rotationAngleCCW=30, rotationAngleCW=-30, lastFaceType=my_cascade.face_type)
        #print(returnvalue)
        #Accumulate error values in a counter
        if(my_cascade.face_type == 0):
            no_face_counter += 1

        #If any face is found for a certain
        #number of cycles, then the ROI is reset
        if(no_face_counter == 50):
            no_face_counter = 0
            roi_x1 = 0
            roi_y1 = 0
            roi_x2 = cam_w
            roi_y2 = cam_h
            roi_w = cam_w
            roi_h = cam_h

        #Checking wich kind of face it is returned
        if(my_cascade.face_type > 0):

            #Face found, reset the error counter
            no_face_counter = 0

            #Because the dlib landmark detector wants a precise
            #boundary box of the face, it is necessary to resize
            #the box returned by the OpenCV haar detector.
            #Adjusting the frame for profile left
            if(my_cascade.face_type == 4):
                face_margin_x1 = 20 - 10 #resize_rate + shift_rate
                face_margin_y1 = 20 + 5 #resize_rate + shift_rate
                face_margin_x2 = -20 - 10 #resize_rate + shift_rate
                face_margin_y2 = -20 + 5 #resize_rate + shift_rate
                face_margin_h = -0.7 #resize_factor
                face_margin_w = -0.7 #resize_factor
            #Adjusting the frame for profile right
            elif(my_cascade.face_type == 5):
                face_margin_x1 = 20 + 10
                face_margin_y1 = 20 + 5
                face_margin_x2 = -20 + 10
                face_margin_y2 = -20 + 5
                face_margin_h = -0.7
                face_margin_w = -0.7
            #No adjustments
            else:
                face_margin_x1 = 0
                face_margin_y1 = 0
                face_margin_x2 = 0
                face_margin_y2 = 0
                face_margin_h = 0
                face_margin_w = 0

            #Updating the face position
            face_x1 = my_cascade.face_x + roi_x1 + face_margin_x1
            face_y1 = my_cascade.face_y + roi_y1 + face_margin_y1
            face_x2 = my_cascade.face_x + my_cascade.face_w + roi_x1 + face_margin_x2
            face_y2 = my_cascade.face_y + my_cascade.face_h + roi_y1 + face_margin_y2
            face_w = my_cascade.face_w + int(my_cascade.face_w * face_margin_w)
            face_h = my_cascade.face_h + int(my_cascade.face_h * face_margin_h)

            crop_img = frame[face_y1:face_y2, face_x1:face_x2]
            cv2.imshow("cropped", crop_img)

            roll = my_head_pose_estimator.return_roll(crop_img)
            pitch = my_head_pose_estimator.return_pitch(crop_img)
            yaw = my_head_pose_estimator.return_yaw(crop_img)
            #print("Estimated [roll, pitch, yaw] ..... [" + str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")

            if yaw > 30:
                cv2.putText(frame, "You are facing right!", (10,30), cv2.FONT_HERSHEY_PLAIN, 1.6, (10,10,255), 2)
            if yaw < -30:
                cv2.putText(frame, "You are facing left!", (10,30), cv2.FONT_HERSHEY_PLAIN, 1.6, (10,10,255), 2)








            #Updating the ROI position
            roi_x1 = face_x1 - roi_resize_w
            if (roi_x1 < 0): roi_x1 = 0
            roi_y1 = face_y1 - roi_resize_h
            if(roi_y1 < 0): roi_y1 = 0
            roi_w = face_w + roi_resize_w + roi_resize_w
            if(roi_w > cam_w): roi_w = cam_w
            roi_h = face_h + roi_resize_h + roi_resize_h
            if(roi_h > cam_h): roi_h = cam_h
            roi_x2 = face_x2 + roi_resize_w
            if (roi_x2 > cam_w): roi_x2 = cam_w
            roi_y2 = face_y2 + roi_resize_h
            if(roi_y2 > cam_h): roi_y2 = cam_h

            #Debugging printing utilities
            if(DEBUG == True):
                #print("FACE: ", face_x1, face_y1, face_x2, face_y2, face_w, face_h)
                #print("ROI: ", roi_x1, roi_y1, roi_x2, roi_y2, roi_w, roi_h)

                #Drawing a green rectangle
                # (and text) around the face.
                text_x1 = face_x1
                text_y1 = face_y1 - 3
                if(text_y1 < 0): text_y1 = 0
                cv2.putText(frame, "FACE", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1);
                cv2.rectangle(frame,
                             (face_x1, face_y1),
                             (face_x2, face_y2),
                             (0, 255, 0),
                              2)

            #In case of a frontal/rotated face it
            # is called the landamark detector
            if(my_cascade.face_type > 0):
                landmarks_2D = my_detector.returnLandmarks(frame, face_x1, face_y1, face_x2, face_y2, points_to_return=TRACKED_POINTS)

                if(DEBUG == True):
                    #cv2.drawKeypoints(frame, landmarks_2D)

                    for point in landmarks_2D:
                        cv2.circle(frame,( point[0], point[1] ), 2, (0,0,255), -1)


                #Applying the PnP solver to find the 3D pose
                # of the head from the 2D position of the
                # landmarks.
                #retval - bool
                #rvec - Output rotation vector that, together with tvec, brings
                # points from the model coordinate system to the camera coordinate system.
                #tvec - Output translation vector.
                retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                                  landmarks_2D,
                                                  camera_matrix, camera_distortion)

                #Now we project the 3D points into the image plane
                #Creating a 3-axis to be used as reference in the image.
                axis = numpy.float32([[50,0,0],
                                      [0,50,0],
                                      [0,0,50]])
                imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)

                #Drawing the three axis on the image frame.
                #The opencv colors are defined as BGR colors such as:
                # (a, b, c) >> Blue = a, Green = b and Red = c
                #Our axis/color convention is X=R, Y=G, Z=B
                sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
                cv2.line(frame, sellion_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
                cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE
                cv2.line(frame, sellion_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED

        #Drawing a yellow rectangle
        # (and text) around the ROI.
        if(DEBUG == True):
            text_x1 = roi_x1
            text_y1 = roi_y1 - 3
            if(text_y1 < 0): text_y1 = 0
            cv2.putText(frame, "ROI", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1);
            cv2.rectangle(frame,
                         (roi_x1, roi_y1),
                         (roi_x2, roi_y2),
                         (0, 255, 255),
                         2)

        #Showing the frame and waiting
        # for the exit command
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    #Release the camera
    video_capture.release()
    print("Bye...")



if __name__ == "__main__":
    main()
