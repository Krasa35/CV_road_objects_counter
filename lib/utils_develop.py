import cv2 as cv
from enum import Enum
from scipy.spatial import distance

class Row_utils():
    R2L_LINE = 350
    TRAM_LINE = 450
    L2R_LINE = 700

    R2L_COLOR = (0, 255, 0)
    TRAM_COLOR = (255, 0, 0)
    L2R_COLOR = (0, 0, 255)
    PED_COLOR = (255, 255, 0)


class Side(Enum):
    RIGHT = 1
    LEFT = 2

class Row(Enum):
    R2L = 1
    TRAM = 2
    L2R = 3
    PED = 4

class Bbox():
    def __init__(self, cnt, frame_shape):
        self.cnt = cnt
        self.x, self.y, self.w, self.h = cv.boundingRect(cnt)
        self.coming_side = self._find_coming_side(frame_shape[1])
        self.row = self._find_row()
        self.square_front = self._is_front_square()
        self.color = self._assign_color()
        M = cv.moments(cnt)
        if M['m00'] != 0:
            self.cx = int(M['m10'] / M['m00'])
            self.cy = int(M['m01'] / M['m00'])
        else:
            print("[WARN] Zero moment detected, using bbox center")
            self.cx = self.x + self.w // 2
            self.cy = self.y + self.h // 2

    def start_corner(self) -> tuple[int, int]:
        return self.x, self.y
    
    def end_corner(self) -> tuple[int, int]:
        return self.x + self.w, self.y + self.h
    
    def _find_coming_side(self, frame_width: int) -> Side:
        if self.x + self.w / 2 < frame_width / 2:
            return Side.LEFT
        else:
            return Side.RIGHT
        
    def _find_row(self) -> Row:
        y_low = self.end_corner()[1]
        if y_low > Row_utils.L2R_LINE:
            return Row.PED
        elif y_low > Row_utils.TRAM_LINE:
            return Row.L2R
        elif y_low > Row_utils.R2L_LINE:
            return Row.TRAM
        else:
            return Row.R2L
        
    def _assign_color(self) -> tuple[int, int, int]:
        if self.row == Row.R2L:
            return Row_utils.R2L_COLOR
        elif self.row == Row.TRAM:
            return Row_utils.TRAM_COLOR
        elif self.row == Row.L2R:
            return Row_utils.L2R_COLOR
        elif self.row == Row.PED:
            return Row_utils.PED_COLOR
        
    def _is_front_square(self) -> bool:
        pass

    def draw(self, frame):
        cv.drawContours(frame, [self.cnt], -1, self.color, 2)
        cv.rectangle(frame, self.start_corner(), self.end_corner(), self.color, 2)
        cv.putText(
            frame,
            f"{self.row.name} {self.coming_side.name}",
            self.start_corner(),
            cv.FONT_HERSHEY_COMPLEX,
            2,
            self.color,
            2
        )
        cv.line(
            frame,
            (0, Row_utils.R2L_LINE),
            (frame.shape[1], Row_utils.R2L_LINE),
            Row_utils.TRAM_COLOR,
            2
        )
        cv.line(
            frame,
            (0, Row_utils.TRAM_LINE),
            (frame.shape[1], Row_utils.TRAM_LINE),
            Row_utils.L2R_COLOR,
            2
        )
        cv.line(
            frame,
            (0, Row_utils.L2R_LINE),
            (frame.shape[1], Row_utils.L2R_LINE),
            Row_utils.PED_COLOR,
            2
        )

class CentroidTracker:
    def __init__(self, max_missed=50, recovery_zone_x=[[800, 1800]], max_distance=500, height_thresh=40, min_frames_detected=20):
        self.objects = {}      # object_id: {'bbox': Bbox, 'missed': int}
        self.next_id = 0
        self.max_missed = max_missed
        self.recovery_zone_x = recovery_zone_x
        self.max_distance = max_distance
        self.height_thresh = height_thresh
        self.min_frames_detected = min_frames_detected

    def update(self, bboxes: list[Bbox]):
        detected_centroids = [(b.cx, b.cy) for b in bboxes]
        used_indices = set()
        new_objects = {}

        # Match existing objects to new detections
        for oid, data in self.objects.items():
            old_centroid = (data['bbox'].cx, data['bbox'].cy)
            if detected_centroids:
                dists = [distance.euclidean(old_centroid, c) for c in detected_centroids]
                min_dist = min(dists)
                idx = dists.index(min_dist)
                if min_dist < self.max_distance and data['bbox'].h - bboxes[idx].h < self.height_thresh and idx not in used_indices:
                    velocity = ((detected_centroids[idx][0] - old_centroid[0]) + data['velocity'] ) // 2
                    print(f"Object {oid} matched with new detection at index {idx}, distance {min_dist}, velocity {velocity}")
                    new_objects[oid] = {'bbox': bboxes[idx], 'missed': 0, 'detected': data['detected'] + 1, 'velocity': velocity}
                    used_indices.add(idx)
                else:
                    data['missed'] += 1
                    data['bbox'].cx += data['velocity']
                    if data['missed'] < self.max_missed:
                        new_objects[oid] = data
            else:
                data['missed'] += 1
                data['bbox'].cx += data['velocity']
                if data['missed'] < self.max_missed and data['detected'] > self.min_frames_detected:
                    new_objects[oid] = data

        # Add new objects or recover lost ones inside recovery zone
        for i, bbox in enumerate(bboxes):
            if i not in used_indices:
                # for rec_zone in self.recovery_zone_x:
                    # if bbox.cx > rec_zone[0] and bbox.cx < rec_zone[1]:
                        for oid, data in self.objects.items():
                            if data['missed'] >= self.max_missed:
                                dist = distance.euclidean((data['bbox'].cx, data['bbox'].cy), (bbox.cx, bbox.cy))
                                print(f"Recovery check: {oid} missed {data['missed']} times, distance {dist}")
                                if dist < self.max_distance:
                                    new_objects[oid] = {'bbox': bbox, 'missed': 0, 'detected': 0, 'velocity': 0}
                                    break
                            else:
                                new_objects[self.next_id] = {'bbox': bbox, 'missed': 0, 'detected': 0, 'velocity': 0}
                                # new_objects[self.next_id] = {'bbox': bbox, 'missed': 0, 'detected': data['detected'] + 1,  'velocity': data['velocity']}
                                self.next_id += 1
                    # else:
                        # new_objects[self.next_id] = {'bbox': bbox, 'missed': 0, 'detected': 0, 'velocity': 0}
                        # self.next_id += 1

        self.objects = new_objects

    def draw(self, frame):
        for object_id, data in self.objects.items():
            bbox = data['bbox']
            # Draw bounding box
            cv.rectangle(frame, bbox.start_corner(), bbox.end_corner(), bbox.color, 2)
            # Draw centroid and ID
            cv.circle(frame, (bbox.cx, bbox.cy), 5, bbox.color, -1)
            cv.putText(frame, f"ID {object_id}", (bbox.cx + 10, bbox.cy), cv.FONT_HERSHEY_SIMPLEX, 0.5, bbox.color, 2)

        # Draw recovery zone
        for rec_zone in self.recovery_zone_x:
            cv.rectangle(frame, (rec_zone[0], 1), (rec_zone[1], frame.shape[1]), (255, 0, 0), 2)
