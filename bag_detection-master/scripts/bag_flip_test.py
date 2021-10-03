#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:52:37 2021

@author: jamesxiu
"""

import util_test

import numpy as np
# import rospy
import cv2

# from std_msgs.msg import UInt16
# from sensor_msgs.msg import Image, CompressedImage
# from geometry_msgs.msg import Point
# from visualization_msgs.msg import Marker
#from bag_detection.msg import FlipPos

# from cv_bridge import CvBridge, CvBridgeError
from cv2 import RANSAC

# class bagFlipModule:

    # BAG_CAMERA_TOPIC  = rospy.get_param("bag_flip_detection/flip_camera_topic")
    # TEST_CAMERA_TOPIC = rospy.get_param("bag_flip_detection/test_flip_camera_topic")
    # BAG_POS_TOPIC     = rospy.get_param("bag_flip_detection/flip_pos_topic")
    # MODE_TOPIC        = rospy.get_param("bag_flip_detection/mode_topic")

    # #LOWER_COLOR       = rospy.get_param("bag_flip_detection/lower_color_range")
    # #UPPER_COLOR       = rospy.get_param("bag_flip_detection/upper_color_range")
    # #AREA              = rospy.get_param("bag_flip_detection/flip_area")
    # DILATION          = rospy.get_param("bag_flip_detection/dilation", 2)
    # THRES_AREA_PROP   = rospy.get_param("bag_flip_detection/thres_area_prop", 0.05)
    # SHEAR_FACTOR      = rospy.get_param("bag_flip_detection/shear_factor", 50)
    
    # TL                = rospy.get_param("bag_flip_detection/tl")
    # BR                = rospy.get_param("bag_flip_detection/br")
    # VERTICAL          = rospy.get_param("bag_flip_detection/vertical", 1)
    # DIRECTION         = rospy.get_param("bag_flip_detection/direction", 1)
    # AREA_PROP         = rospy.get_param("bag_flip_detection/area_prop", 0.5)

    # WIDTH             = rospy.get_param("bag_flip_detection/width")
    # HEIGHT            = rospy.get_param("bag_flip_detection/height")

    # SIDE              = rospy.get_param("bag_flip_detection/side")

    # MODE              = 0

    # def __init__(self):
    #     # Initialize your publishers and
    #     # subscribers here
    #     self.bag_camera_sub = rospy.Subscriber(self.BAG_CAMERA_TOPIC, Image, callback=self.react_to_camera)
    #     self.mode_sub = rospy.Subscriber(self.MODE_TOPIC, UInt16, callback=self.update_mode)
    #     self.image_pub = rospy.Publisher(self.TEST_CAMERA_TOPIC, Image, queue_size=10)
    #     self.edge_pub = rospy.Publisher("edgetest", Image, queue_size=10)
    #     self.blob_pub = rospy.Publisher("blobtest", Image, queue_size=10)
    #     self.bag_pos_pub = rospy.Publisher(self.BAG_POS_TOPIC, UInt16, queue_size=10)
    #     self.bridge = CvBridge()     


    # def update_bag_message(self, bag_msg, rect):
    #     x, y, w, h = rect

    #     bot_x = (x+(w/2)) - ((self.bot_zone[0][0] + self.bot_zone[1][0])/2)
    #     bot_y = (y+(h/2)) - ((self.bot_zone[0][1] + self.bot_zone[1][1])/2)
    #     top_x = (x+(w/2)) - ((self.top_zone[0][0] + self.top_zone[1][0])/2)
    #     top_y = (y+(h/2)) - ((self.top_zone[0][1] + self.top_zone[1][1])/2)

    #     if ((bot_x**2)+(bot_y**2)**2)**0.5 < ((top_x**2)+(top_y**2)**2)**0.5:
    #         if abs(bot_x) < abs(bag_msg.bot_x) and abs(bot_y) < abs(bag_msg.bot_y):
    #             bag_msg.bot_x = bot_x
    #             bag_msg.bot_y = bot_y
    #     else:
    #         if abs(top_x) < abs(bag_msg.top_x) and abs(top_y) < abs(bag_msg.top_y):
    #             bag_msg.top_x = top_x
    #             bag_msg.top_y = top_y

    #     return bag_msg


    # def react_to_camera(self, data):
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #     except CvBridgeError as e:
    #         print(e)

    #     bag_msg = None

    #     top_color = (0,255,0)
    #     bot_color = (0,255,0)
    #     green = (0,255,0)
    
    #     # if bag_msg:
    #     #     if bag_msg.top:
    #     #         top_color = green
    #     #     if bag_msg.bot:
    #     #         bot_color = green

    #     element = util.get_element(self.DILATION)
    #     edges = util.canny(cv_image)
    #     closed = util.dilate_bag_row(edges, element)
    #     closedS = util.directional_shear(closed, element, self.VERTICAL, self.SHEAR_FACTOR) 
    #     h, w = cv_image.shape[:2]
    #     threshold_area = self.THRES_AREA_PROP*h*w
    #     c_rects = util.get_rectangles(closedS, threshold_area)
    #     #c_rects = util.bag_rect_detection(cv_image, self.VERTICAL)
    #     inbound_area = 0
    #     error = 9999
    #     for rect in c_rects:
    #         top_l = (rect[0], rect[1])
    #         bot_r = (rect[0]+rect[2], rect[1]+rect[3])
    #         cv2.rectangle(cv_image, top_l, bot_r, (255,0,0), 5)
    #         if (self.VERTICAL):
    #             if ((top_l[1] > self.TL[1] and top_l[1] < self.BR[1]) and (bot_r[1] > self.TL[1] and bot_r[1] < self.BR[1])):
    #                 inbound_area += (bot_r[0] - top_l[0]) * (bot_r[1] - top_l[1])
    #             elif (top_l[1] < self.TL[1] and bot_r[1] > self.TL[1]):
    #                 # GO FORWARD
    #                 error = self.TL[1] - top_l[1]
               
    #         else:
    #             if ((top_l[0] > self.TL[0] and top_l[0] < self.BR[0]) and (bot_r[0] > self.TL[0] and bot_r[0] < self.BR[0])):
    #                 inbound_area += (bot_r[0] - top_l[0]) * (bot_r[1] - top_l[1])
    #             elif (top_l[0] < self.TL[0] and bot_r[0] > self.TL[0]):
    #                 # GO FORWARD
    #                 error = self.TL[0] - top_l[0]

    #     if (inbound_area >= self.AREA_PROP*((self.BR[0] - self.TL[0]) * (self.BR[1] - self.TL[1]))):
    #         error = 0

        

    #     self.bag_pos_pub.publish(error)
                 

    #     #FOR VIEWING
    #     cv2.rectangle(cv_image, tuple(self.TL), tuple(self.BR), (0,0,255), 10)
    #     self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding="passthrough"))
    #     self.edge_pub.publish(self.bridge.cv2_to_imgmsg(edges, encoding="passthrough"))
    #     self.blob_pub.publish(self.bridge.cv2_to_imgmsg(closed, encoding="passthrough"))


    # def update_mode(self, data):
    #     print("UPDATE MODE: ", data)
    #     self.MODE = data.data]
    
def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        cv2.imwrite("first_frame.jpg", image)
    vidcap.release()
  

def findFPS(videofile):
    video = cv2.VideoCapture("GP066349.MP4");
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps
      
def getFrame(videofile, frame_number, save = False):
    cap = cv2.VideoCapture(videofile)
    total_frames = cap.get(7)
    cap.set(1, frame_number)
    success, frame = cap.read()
    cap.release()
    if success:
        if save:
            filename = str(frame_number) + '_frame.jpg'
            cv2.imwrite(filename, frame)
        return frame
    else:
        raise ValueError('No success matey')
        
def cvshow(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)    
    
def frame_difference(FILENAME):
    cap = cv2.VideoCapture(FILENAME)
    ret, current_frame = cap.read()
    previous_frame = current_frame
    frames = 0
    
    while(cap.isOpened()):
        if current_frame is not None and previous_frame is not None:
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    
        
            frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)
            frame_diff = cv2.convertScaleAbs(frame_diff, alpha=3, beta=0)
            cv2.imshow('frame diff ',frame_diff)      
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if current_frame is not None:
            previous_frame = current_frame.copy()
        else:
            previous_frame = None
        ret, current_frame = cap.read()
        frames += 1
        if frames > 1000:
            break
    
    cap.release()
    cv2.destroyAllWindows()


def find_flippy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    blank = np.zeros(gray.shape)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if gray[r, c] == 255:
                blank[r, c] = 255
                img[r, c] = 50
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blank = cv2.dilate(blank, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    gray = cv2.erode(gray, kernel, iterations = 1)
    return gray, blank

def react_to_camera(FILENAME, frame):
    img = getFrame(FILENAME, frame)
    original = img.copy()
    cvshow(img)
    cv2.imwrite('img.jpg', img)
    print(img.shape)
    #Could contrast the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    img[thresh == 255] = 50
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    img = cv2.erode(img, kernel, iterations = 1)
    cvshow(img)
    cv2.imwrite('removed.jpg', img)

    vertical = 1
    TL = [120,80]
    BR = [432, 240]
    
    top_color = (0,255,0)
    bot_color = (0,255,0)

    element = util_test.get_element(6)
    edges = util_test.canny(img)
    # cvshow(edges)
    closed = util_test.dilate_bag_row(edges, element)
    # closedS = util_test.directional_shear(closed, element, vertical, 100) 
    # cvshow(closed)
    h, w = img.shape[:2]
    threshold_area = 0.01*h*w
    # c_rects = util_test.get_rectangles_new(closed, threshold_area)
    
    # for rect in c_rects:
    #     x, y, w, h = rect
    #     cv2.rectangle(original,(x,y),(x+w,y+h),(0,255,0),3)
    contours, hierarchy = cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    blank = np.zeros((2160, 3840), dtype = 'uint8')
    cv2.drawContours(blank, contours, -1, (255), 3)
    cv2.imwrite('contours.jpg', blank)
    element = util_test.get_element(100)
    closedcontours = util_test.dilate_bag_row(blank, element)
    combined = np.hstack((closed, blank))
    cvshow(combined)
    cv2.imwrite('combined.jpg', combined)
    cvshow(closedcontours)
    cv2.imwrite('closedcontours.jpg', closedcontours)
    # print(c_rects)
    #c_rects = util.bag_rect_detection(cv_image, self.VERTICAL)
    # inbound_area = 0
    # error = 9999
    # for rect in c_rects:
    #     top_l = (rect[0], rect[1])
    #     bot_r = (rect[0]+rect[2], rect[1]+rect[3])
    #     cv2.rectangle(img, top_l, bot_r, (255,0,0), 5)
    #     if vertical:
    #         if ((top_l[1] > TL[1] and top_l[1] < BR[1]) and (bot_r[1] > TL[1] and bot_r[1] < BR[1])):
    #             inbound_area += (bot_r[0] - top_l[0]) * (bot_r[1] - top_l[1])
    #         elif (top_l[1] < TL[1] and bot_r[1] > TL[1]):
    #             # GO FORWARD
    #             error = TL[1] - top_l[1]
           
    #     else:
    #         if ((top_l[0] > TL[0] and top_l[0] < BR[0]) and (bot_r[0] > TL[0] and bot_r[0] < BR[0])):
    #             inbound_area += (bot_r[0] - top_l[0]) * (bot_r[1] - top_l[1])
    #         elif (top_l[0] < TL[0] and bot_r[0] > TL[0]):
    #             # GO FORWARD
    #             error = TL[0] - top_l[0]

    # if (inbound_area >= 0.5*((BR[0] - TL[0]) * (BR[1] - TL[1]))):
    #     error = 0
    
    
    # squares = util_test.find_squares(closed)
    # cv2.drawContours(original, squares, -1, (0, 255, 0), 3)
    # cvshow(original)
    # #FOR VIEWING
    # print('Error ', error)
    # cv2.rectangle(img, tuple(TL), tuple(BR), (0,0,255), 10)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

def react_video_ziptie(FILENAME, frame, armdown = False):
    img = getFrame(FILENAME, frame)
    cvshow(img)
    # cv2.imwrite('img.jpg', img)
    print(img.shape)
    
    if armdown:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
        img[thresh == 255] = np.array([255, 255, 255])
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        img = cv2.dilate(img, kernel, iterations = 1)
        cvshow(img)
        cv2.imwrite('removed.jpg', img)
    
    red_lower_bound = (0, 0, 80)
    red_upper_bound = (65, 65, 255)
    mask = cv2.inRange(img, red_lower_bound, red_upper_bound)
    print(mask.shape)
    cvshow(mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
    cvshow(opening)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    cvshow(closing)

def crop_video(FILENAME, OUTNAME, armdown = False):
    cap = cv2.VideoCapture(FILENAME)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = 1000
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    w = cv2.VideoWriter(OUTNAME, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (3840,2160))
        
    frames = 0
    # failed_frames = []
    
    while(cap.isOpened() and frames < total_frames):
        success, img = cap.read()
        frames += 1
        # out.write(np.ones((2160,3840,3), np.uint8))
        if success == True:
            if img is None:
                print(frames, 'LMAO')
                continue 
            
            w.write(img)
            
            if frames % 100 == 0:
                print(frames)
        # else:
        #     failed_frames.append(frames)
    
    # print(failed_frames)
    cap.release()
    w.release()
    cv2.destroyAllWindows()

    
def react_video(FILENAME, OUTNAME, armdown = False):
    cap = cv2.VideoCapture(FILENAME)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = 1000
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    w = cv2.VideoWriter(OUTNAME, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (3840,2160))
        
    frames = 0
    # failed_frames = []
    
    while(cap.isOpened() and frames < total_frames):
        success, img = cap.read()
        frames += 1
        # out.write(np.ones((2160,3840,3), np.uint8))
        if success == True:
            if img is None:
                print(frames, 'LMAO')
                continue 
            
            if armdown:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
                img[thresh == 255] = np.array([255, 255, 255])
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
                img = cv2.dilate(img, kernel, iterations = 1)
            
            red_lower_bound = (0, 0, 80)
            red_upper_bound = (65, 65, 255)
            mask = cv2.inRange(img, red_lower_bound, red_upper_bound)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            closing = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
            
            w.write(closing)
            if frames % 100 == 0:
                print(frames)
        # else:
        #     failed_frames.append(frames)
    
    # print(failed_frames)
    cap.release()
    w.release()
    cv2.destroyAllWindows()

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 ) ** (1/2)

def average(*pts):
    center = np.zeros(2)
    print(center)
    for pt in pts:
        center += pt
    center /= len(pts)
    return center

    
def react_video_original(FILENAME, OUTNAME, armdown = False):
    cap = cv2.VideoCapture(FILENAME)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_cap = 1000
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    #IMG Resolution is 3840x2160
    w = cv2.VideoWriter(OUTNAME, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (3840,2160))
        
    frames = 0
    # failed_frames = []
    
    while(cap.isOpened() and frames < frame_cap):
        success, img = cap.read()
        frames += 1
        # out.write(np.ones((2160,3840,3), np.uint8))
        if success == True:
            if img is None:
                print(frames, 'LMAO')
                continue 
            
            if armdown:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
                img[thresh == 255] = np.array([255, 255, 255])
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
                img = cv2.dilate(img, kernel, iterations = 1)
            
            red_lower_bound = (0, 0, 80)
            red_upper_bound = (65, 65, 255)
            mask = cv2.inRange(img, red_lower_bound, red_upper_bound)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            # closing = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
            
            contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            TL, BL, TR, BR = None, None, None, None
            
            for c in contours:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(img, (cX, cY), 50, (255, 0, 0), -1)
                
                if cX < 1920:
                    if cY < 1080:
                        TL = np.array([cX, cY])
                    else:
                        BL = np.array([cX, cY])
                else:
                    if cY < 1080:
                        TR = np.array([cX, cY])
                    else:
                        BR = np.array([cX, cY])
            
            #Box height is typically 2000 pixels
            #width between zipties is typically 1450 pixels
            #We try to refine these later
            box_height = 2000
            box_width = 1450
            center = None
            
            if TL is not None and BL is not None and TR is not None and BR is not None:
                box_height = distance(TL, BL)
                box_width = distance(TL, TR)
                center = average(TL, BL, TR, BR)
            
            elif TL is not None and BL is not None:
                box_height = distance(TL, BL)
                hyp = ((box_height/2) ** 2 + (box_width/2) ** 2) ** (1/2)
                angle = np.arctan2(BL[1]-TL[1], BL[0] - TL[0]) + np.arctan2(box_height, box_width)
                center = TL + hyp * np.array(np.cos(angle), np.sin(angle))
                
            elif TL is not None and TR is not None:
                box_width = distance(TL, TR)
                hyp = ((box_height/2) ** 2 + (box_width/2) ** 2) ** (1/2)
                angle = np.arctan2(TR[1]-TL[1], TR[0] - TL[0]) - np.arctan2(box_height, box_width)
                center = TL + hyp * np.array(np.cos(angle), np.sin(angle))
            
            elif TR is not None and BR is not None:
                box_height = distance(TR, BR)
                hyp = ((box_height/2) ** 2 + (box_width/2) ** 2) ** (1/2)
                angle = np.arctan2(BR[1]-TR[1], BR[0] - TR[0]) - np.arctan2(box_height, box_width)
                center = TR + hyp * np.array(np.cos(angle), np.sin(angle))
            
            elif BL is not None and BR is not None:
                box_width = distance(BL, BR)
                hyp = ((box_height/2) ** 2 + (box_width/2) ** 2) ** (1/2)
                angle = np.arctan2(BR[1]-BL[1], BR[0] - BL[0]) + np.arctan2(box_height, box_width)
                center = BL + hyp * np.array(np.cos(angle), np.sin(angle))
                
            else:
                center = None
                print('Box not detected', frames)
            if center is not None:
                center = (int(center[0]), int(center[1]))
                cv2.circle(img, tuple(center), 50, (255, 0, 0), -1)
            toWrite = 'Center: ' + str(center)
            cv2.putText(img, toWrite, (100, 1500),  cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
            
            w.write(img)
            if frames % 100 == 0:
                print(frames)
        # else:
        #     failed_frames.append(frames)
    
    # print(failed_frames)
    print(box_width)
    cap.release()
    w.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # FILENAME = '2021-08-10 11.41.06.mp4'
    # OUTNAME = 'Original-' + FILENAME
    # # cap = cv2.VideoCapture(FILENAME)
    # # print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # # getFrame(FILENAME, 1, save = True)
    
    # crop_video(FILENAME, OUTNAME)

    pass