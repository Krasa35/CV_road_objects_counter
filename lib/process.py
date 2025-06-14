import cv2 as cv
# import pybgs as bgs
import lib.algorithm as alg
from enum import Enum
import numpy as np
from scipy.spatial import distance
from lib.utils import Bbox, CentroidTracker, Row

# algorithm = bgs.FrameDifference()
algorithm = alg.FrameDifference()

def empty(int):
    pass

def connect_vertical_cnts(cnts, width_thresh=0.2, y_gap_thresh=80):
    cnts = sorted(cnts, key=lambda c: cv.boundingRect(c)[1])  # sort by y
    merged = []
    used = set()

    for i, c1 in enumerate(cnts):
        if i in used:
            continue
        x1, y1, w1, h1 = cv.boundingRect(c1)
        merged_cnt = c1.copy()
        for j in range(i+1, len(cnts)):
            if j in used:
                continue
            x2, y2, w2, h2 = cv.boundingRect(cnts[j])
            # Check width similarity
            if abs(w1 - w2) / max(w1, w2) < width_thresh:
                # Check horizontal overlap
                if abs(x1 - x2) < max(w1, w2) * 0.5:
                    # Check vertical proximity
                    if 0 < y2 - (y1 + h1) < y_gap_thresh:
                        merged_cnt = np.vstack((merged_cnt, cnts[j]))
                        used.add(j)
        merged.append(merged_cnt)
        used.add(i)
    return merged

def perform_processing(cap: cv.VideoCapture) -> dict[str, int]:
    cv.namedWindow('Camera_Stream_BG', cv.WINDOW_NORMAL)
    cv.namedWindow('Camera_Stream_FG', cv.WINDOW_NORMAL)
    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.createTrackbar('trackbar', 'Camera_Stream_FG' , 25, 255, empty)
    tracker = CentroidTracker()
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
        connected = connect_vertical_cnts(contours)

        bboxes = [Bbox(cnt, frame.shape) for cnt in connected if cv.contourArea(cnt) > 4000]

        print(f"Detected: \n\
               {len([b for b in bboxes if b.row == Row.R2L])} R2L vehicles\n\
               {len([b for b in bboxes if b.row == Row.TRAM])} TRAM vehicles\n\
               {len([b for b in bboxes if b.row == Row.L2R])} L2R vehicles\n\
               {len([b for b in bboxes if b.row == Row.PED])} PED vehicles")
        
        fg = cv.cvtColor(fg, cv.COLOR_GRAY2BGR)
        for bbox in bboxes:
            bbox.draw(fg)
        tracker.update(bboxes)
        tracker.draw(frame)
        # tracker.draw(fg)

        cv.imshow('frame', frame)
        cv.imshow('Camera_Stream_FG', fg)
        cv.imshow('Camera_Stream_BG', bg)
        cv.waitKey(50)
        # if key == ord('a'):
        #     bg = grey
        # if bg is not None:
        #     fg = cv.absdiff(bg, grey)
    return {
        "liczba_samochodow_osobowych_z_prawej_na_lewa": 5,
        "liczba_samochodow_osobowych_z_lewej_na_prawa": 5,
        "liczba_samochodow_ciezarowych_autobusow_z_prawej_na_lewa": 5,
        "liczba_samochodow_ciezarowych_autobusow_z_lewej_na_prawa": 5,
        "liczba_tramwajow": 5,
        "liczba_pieszych": 5,
        "liczba_rowerzystow": 5
    }