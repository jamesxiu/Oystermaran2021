#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some code that finds the box in Flippy's down-looking camera' and some utilities

"""
import numpy as np
import cv2

#DIMENSION OF VIDEO IS width = 3840, height = 2160.

def getFrame(videofile, frame_number, save = False):
    """
    Given a file containing a video, return the frame in the video specified by frame_number.
    The frame is represented as a numpy array (1 or 3 channels). Save the frame by setting save = True
    """
    
    cap = cv2.VideoCapture(videofile)
    total_frames = cap.get(7)
    assert frame_number < total_frames, "Not a valid frame number"
    
    cap.set(1, frame_number)
    success, frame = cap.read()
    cap.release()
    
    if success:
        if save:
            filename = str(frame_number) + '_frame.jpg'
            cv2.imwrite(filename, frame)
        return frame
    else:
        raise ValueError('The frame does not exist')
  
        
def crop_video(FILENAME, OUTNAME, right_bound):
    """
    Crop the video in FILENAME from the beginning frame up to the frame specified by right_bound.
    Save the result in OUTNAME. 
    """
    cap = cv2.VideoCapture(FILENAME)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    w = cv2.VideoWriter(OUTNAME, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (3840,2160))
        
    frames = 0
    # failed_frames = []
    
    while(cap.isOpened() and frames < right_bound):
        success, img = cap.read()
        frames += 1
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


def cvshow(img):
    """
    MacOS specific method for displaying an image, because that is weird in MacOS. Shows an image in a 
    separate window, click any key to close the window
    """
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)    
    

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 ) ** (1/2)

def average(*pts):
    center = np.zeros(2)
    print(center)
    for pt in pts:
        center += pt
    center /= len(pts)
    return center

    
def react_video_original(FILENAME, OUTNAME, armdown = False, crop = False, save_tracking_video = False):
    """
    FILENAME is a video from the bottom facing camera of Flippy. Save a video that tracks the center of the box. 
    armdown = True if the arm is down, crop = True if want to run the program on just the first 1000 frames of the video
    """
    cap = cv2.VideoCapture(FILENAME)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if crop:
        total_frames = min(total_frames, 1000)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if save_tracking_video:
        tracker_name = "Tracker-" + OUTNAME
        tracker = cv2.VideoWriter(OUTNAME, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (3840,2160))
        
    w = cv2.VideoWriter(OUTNAME, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (3840,2160))
        
    frames = 0
    
    while(cap.isOpened() and frames < total_frames):
        success, img = cap.read()
        frames += 1

        if success == True:
            if img is None:
                print(frames, 'LMAO')
                continue 
            
            #Black out the region containing the arm, if the arm is down
            if armdown:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
                img[thresh == 255] = np.array([255, 255, 255])
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
                img = cv2.dilate(img, kernel, iterations = 1)
            
            #Create a mask for where the image has red
            red_lower_bound = (0, 0, 80)
            red_upper_bound = (65, 65, 255)
            mask = cv2.inRange(img, red_lower_bound, red_upper_bound)
            
            #Get rid of smaller blobs of red (could be red wire, noise)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
            
            #Join the blobs of red together that form the zipties
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            
            if save_tracking_video:
                closing = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
                tracker.write(closing)
            
            #Get the edges of the blobs
            contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            TL, BL, TR, BR = None, None, None, None
            
            for c in contours:
                #Find the center of the given blob
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                #Draw the center of the blob onto the output video
                cv2.circle(img, (cX, cY), 50, (255, 0, 0), -1)
                
                #Determine the quadrant of the blob
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
            
            #Logic for where the center of the box is located
            
            #Case where 4 zipties are detected
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
    cap.release()
    w.release()
    if save_tracking_video:
        tracker.release()
    cv2.destroyAllWindows()