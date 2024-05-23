import os
from flask import Flask, render_template, jsonify
import cv2
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not
import threading

app = Flask(__name__)

# Directory of the current script
current_dir = os.path.dirname(__file__)

# Function to calculate difference between images
def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# Paths to resources
mask_path = os.path.join(current_dir, 'resources', 'mask_1920_1080.png')
video_path = os.path.join(current_dir, 'resources', 'parking_1920_1080_loop3.mp4')

# Load mask and video
mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)

# Perform connected components analysis on the mask
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

# Initialize spots status and diffs
spots_status = [None for _ in spots]
diffs = [None for _ in spots]

previous_frame = None
ret = True
frame_nmr = 0
step = 30

# Function to update parking status
def update_parking_status():
    global previous_frame, frame_nmr, spots_status, diffs, cap

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_nmr % step == 0 and previous_frame is not None:
            for spot_indx, spot in enumerate(spots):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        if frame_nmr % step == 0:
            if previous_frame is None:
                arr_ = range(len(spots))
            else:
                arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]

            for spot_indx in arr_:
                spot = spots[spot_indx]
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                spot_status = empty_or_not(spot_crop)
                spots_status[spot_indx] = spot_status

        if frame_nmr % step == 0:
            previous_frame = frame.copy()

        for spot_indx, spot in enumerate(spots):
            spot_status = spots_status[spot_indx]
            x1, y1, w, h = spots[spot_indx]

            if spot_status:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            else:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

        cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
        cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_nmr += 1

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to get parking status
@app.route('/parking_status', methods=['GET'])
def get_parking_status():
    return jsonify({
        "available_spots": sum(spots_status),
        "total_spots": len(spots_status)
    })

if __name__ == '__main__':
    # Start the video thread
    video_thread = threading.Thread(target=update_parking_status)
    video_thread.start()

    # Start the Flask application
    app.run(host='0.0.0.0', port=5000, debug=False)

    # Ensure the video thread finishes
    video_thread.join()
