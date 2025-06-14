# Background Subtraction using pybgs library
# This script demonstrates the usage of the pybgs library for background subtraction in a video.
# It processes a video file, extracts the moving objects by applying the FrameDifference algorithm,
# and displays the original video, the foreground mask, and the background model in real-time.

# Import necessary libraries
import numpy as np
import cv2
import pybgs as bgs

cv2.namedWindow('Original Video', cv2.WINDOW_NORMAL)
cv2.namedWindow('Foreground Mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('Background Model', cv2.WINDOW_NORMAL)

# Initialize the background subtraction algorithm
algorithm = bgs.FrameDifference()
video_file = "data/00000_trimmed.mp4"

# Create a video capture object to read the video file
capture = cv2.VideoCapture(video_file)

# Wait for the video file to be opened
while not capture.isOpened():
  capture = cv2.VideoCapture(video_file)
  cv2.waitKey(1000)
  print("Waiting for the video to be loaded...")

# Main loop to process the video frames
while True:
  # Read a new frame
  flag, frame = capture.read()
  
  # If a frame was successfully read
  if flag:
    # Display the original video frame
    # Blur the frame before displaying
    blurred_frame = cv2.GaussianBlur(frame, (17, 17), 0)
    # cv2.imshow('Original Video', blurred_frame)
    
    # Apply the background subtraction algorithm
    img_output = algorithm.apply(blurred_frame)
    # Retrieve the current background model
    img_bgmodel = algorithm.getBackgroundModel()
    

    # Apply morphological closing to remove noise and fill gaps in the foreground mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed_mask = cv2.morphologyEx(img_output, cv2.MORPH_CLOSE, kernel)
    # Display the foreground mask and the background model
    cv2.imshow('Foreground Mask', closed_mask)
    cv2.imshow('Background Model', img_bgmodel)
    # Find contours (blobs) in the closed foreground mask
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_with_rects = frame.copy()

    # Dictionary to store previous positions of blobs
    if 'prev_centroids' not in globals():
        prev_centroids = {}
        next_blob_id = 0

    curr_centroids = {}
    velocities = {}
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500:  # Lower threshold to allow smaller blobs
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

            # Predict positions of previous blobs using velocity
            predicted_positions = {}
            for blob_id, (pcx, pcy) in prev_centroids.items():
                vx, vy = velocities.get(blob_id, (0, 0))
                predicted_positions[blob_id] = (pcx + vx, pcy + vy)

            # Try to match with predicted positions (nearest neighbor)
            min_dist = float('inf')
            matched_id = None
            for blob_id, (pred_x, pred_y) in predicted_positions.items():
                dist = np.hypot(cx - pred_x, cy - pred_y)
                if dist < 60 and dist < min_dist:  # Slightly larger threshold for prediction
                    min_dist = dist
                    matched_id = blob_id

            if matched_id is not None:
                # Update velocity based on actual movement
                prev_cx, prev_cy = prev_centroids[matched_id]
                velocities[matched_id] = (cx - prev_cx, cy - prev_cy)
                curr_centroids[matched_id] = (cx, cy)
                # Remember the largest area seen for this blob
                if 'max_area' not in globals():
                    max_area = {}
                prev_max = max_area.get(matched_id, 0)
                max_area[matched_id] = max(prev_max, area)
            else:
                curr_centroids[next_blob_id] = (cx, cy)
                velocities[next_blob_id] = (0, 0)
                if 'max_area' not in globals():
                    max_area = {}
                max_area[next_blob_id] = area
                matched_id = next_blob_id
                next_blob_id += 1

            # Draw rectangle and ID, and optionally show max area
            cv2.rectangle(frame_with_rects, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_with_rects, f"ID:{matched_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            # Optionally display max area
            # cv2.putText(frame_with_rects, f"MaxA:{max_area[matched_id]}", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

    prev_centroids = curr_centroids

    cv2.imshow('Original Video', frame_with_rects)
  else:
    # Wait for a bit and exit the loop if no frame is captured
    cv2.waitKey(1000)
    print("No more frames to read or error in reading the frame.")
    break
  
  # Break the loop if the user presses 'Esc'
  if cv2.waitKey(10) & 0xFF == 27:
    print("Exiting...")
    break

# Clean up: close all OpenCV windows and release the video capture object
cv2.destroyAllWindows()
capture.release()