import cv2 as cv
# import pybgs as bgs
import lib.algorithm as alg
from enum import Enum
import numpy as np
from scipy.spatial import distance
from lib.utils import Bbox, CentroidTracker, ContourTracker, FRAME_SHAPE

# algorithm = bgs.FrameDifference()
algorithm = alg.FrameDifference()

def empty(int):
    pass

def split_contours(contours, fg):
    split_contours = []
    for cnt in contours:
        # Get bounding box
        x, y, w, h = cv.boundingRect(cnt)
        
        # Create a mask of the contour
        mask = np.zeros_like(fg)
        cv.drawContours(mask, [cnt], 0, 255, -1)
        
        # Calculate width at top and bottom portions
        top_third = mask[y:y+h//3, :]
        bottom_third = mask[y+2*h//3:y+h, :]
        
        # Find width of the object at top and bottom
        top_indices = np.where(top_third > 0)
        bottom_indices = np.where(bottom_third > 0)
        
        if len(top_indices[1]) > 0 and len(bottom_indices[1]) > 0:
            top_width = np.max(top_indices[1]) - np.min(top_indices[1])
            bottom_width = np.max(bottom_indices[1]) - np.min(bottom_indices[1])
            
            # If width difference is significant, split the contour
            if abs(top_width - bottom_width) > 0.3 * max(top_width, bottom_width):
                mid_y = y + h // 2
                
                # Create upper and lower masks
                upper_mask = mask.copy()
                upper_mask[mid_y:, :] = 0
                lower_mask = mask.copy()
                lower_mask[:mid_y, :] = 0
                
                # Find contours in the split masks
                upper_cnts, _ = cv.findContours(upper_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                lower_cnts, _ = cv.findContours(lower_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                
                # Add the split contours if they exist
                if len(upper_cnts) > 0:
                    split_contours.extend(upper_cnts)
                if len(lower_cnts) > 0:
                    split_contours.extend(lower_cnts)
            else:
                split_contours.append(cnt)
        else:
            split_contours.append(cnt)

def perform_processing(cap: cv.VideoCapture) -> dict[str, int]:
    # cv.namedWindow('Camera_Stream_BG', cv.WINDOW_NORMAL)
    cv.namedWindow('Camera_Stream_FG', cv.WINDOW_NORMAL)
    cv.namedWindow('Camera_Stream_FG_2', cv.WINDOW_NORMAL)
    # cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.createTrackbar('trackbar', 'Camera_Stream_FG' , 25, 255, empty)
    tracker_cnt = ContourTracker()
    tracker = CentroidTracker()
    counts = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred_frame = cv.GaussianBlur(grey, (9, 9), 0)
        th_val = cv.getTrackbarPos('trackbar', 'Camera_Stream_FG')

        fg = algorithm.apply(blurred_frame, threshold=th_val, update_background=False, two_background=True)
        bg = algorithm.getBackgroundModel()

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
        fg = cv.morphologyEx(fg, cv.MORPH_OPEN, kernel)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
        fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, kernel)
        contours, _ = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Split large contours that might represent multiple objects
        # contours = split_contours(contours, fg)
        bcontours = [Bbox(cnt) for cnt in contours if cv.contourArea(cnt) > 500]
        tracker.update(bcontours)
        fg = cv.cvtColor(fg, cv.COLOR_GRAY2BGR)
        tracker.draw(fg)

        frame_counts = tracker.get_counts()
        for key, value in frame_counts.items():
            counts[key] = counts.get(key, 0) + value

        fg2 = fg.copy()
        for bbox in bcontours:
            bbox.draw(fg2)

        cv.imshow('Camera_Stream_FG_2', fg2)
        cv.imshow('Camera_Stream_FG', fg)
        # cv.imshow('frame', frame)
        # cv.imshow('Camera_Stream_BG', bg)
        cv.waitKey(1)

    return {
        "liczba_samochodow_osobowych_z_prawej_na_lewa": counts.get('R2L_car', 0),
        "liczba_samochodow_osobowych_z_lewej_na_prawa": counts.get('L2R_car', 0),
        "liczba_samochodow_ciezarowych_autobusow_z_prawej_na_lewa": counts.get('R2L_truck', 0),
        "liczba_samochodow_ciezarowych_autobusow_z_lewej_na_prawa": counts.get('L2R_truck', 0),
        "liczba_tramwajow": counts.get('TRAM', 0),
        "liczba_pieszych": counts.get('PED', 0),
        "liczba_rowerzystow": counts.get('BIKE', 0)
    }