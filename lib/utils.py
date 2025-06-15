import cv2 as cv
from enum import Enum
from scipy.spatial import distance
from typing import List
import numpy as np

FRAME_SHAPE = (1080, 1920)  

class Row_utils():
    R2L_LINE = 300
    TRAM_LINE = 450
    L2R_LINE = 700

    R2L_COLOR = (0, 255, 0)
    TRAM_COLOR = (255, 0, 0)
    L2R_COLOR = (0, 0, 255)
    PED_COLOR = (255, 255, 0)


class Side(Enum):
    RIGHT = 1
    LEFT = 2

class Type(Enum):
    R2L_car = 1
    R2L_truck = 2
    TRAM = 3
    L2R_car = 4
    L2R_truck = 5
    PED = 6
    BIKE = 7

class Bbox():
    def __init__(self, cnt):
        self.cnt = cnt
        self.x, self.y, self.w, self.h = cv.boundingRect(cnt)
        M = cv.moments(cnt)
        if M['m00'] != 0:
            self.cx = int(M['m10'] / M['m00'])
            self.cy = int(M['m01'] / M['m00'])
        else:
            print("[WARN] Zero moment detected, using bbox center")
            self.cx = self.x + self.w // 2
            self.cy = self.y + self.h // 2
        self.centroid = (self.cx, self.cy)
        self.color = (0, 255, 0)  # Default color for drawing
        self.type = self._find_type()
        self.color = self._assign_color() 
        self.square_front = self._is_front_square()

    def start_corner(self) -> tuple[int, int]:
        return self.x, self.y
    
    def end_corner(self) -> tuple[int, int]:
        return self.x + self.w, self.y + self.h
    
    def _find_type(self) -> Type:
        y_low = self.end_corner()[1]
        if y_low > Row_utils.L2R_LINE:
            return Type.PED
        elif y_low > Row_utils.TRAM_LINE:
            if self.w > self.h * 2:
                return Type.L2R_truck
            else:
                return Type.L2R_car
        elif y_low > Row_utils.R2L_LINE and self.h > 150:
            return Type.TRAM
        else:
            if self.w > self.h * 2:
                return Type.R2L_truck
            else:
                return Type.R2L_car
            
    def _is_front_square(self) -> bool:
        pass

    def _assign_color(self) -> tuple[int, int, int]:
        if self.type == Type.R2L_car or self.type == Type.R2L_truck:
            return Row_utils.R2L_COLOR
        elif self.type == Type.TRAM:
            return Row_utils.TRAM_COLOR
        elif self.type == Type.L2R_car or self.type == Type.L2R_truck:
            return Row_utils.L2R_COLOR
        elif self.type == Type.PED or self.type == Type.BIKE:
            return Row_utils.PED_COLOR

    def draw(self, frame):
        cv.drawContours(frame, [self.cnt], -1, self.color, 2)
        cv.rectangle(frame, self.start_corner(), self.end_corner(), self.color, 2)
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

class TrackerObject(Bbox):
    def __init__(self, bbox, missed: int = 0, detected_count: int = 1, blackout_zones=[[1000, 1200], [1400, 1700]]):
        super().__init__(bbox.cnt)
        self.missed = missed
        self.detected_count = detected_count
        self.coming_side = self._find_coming_side()
        self.velocity = 1 if self.coming_side == Side.LEFT else -1
        self.blackout_zones = blackout_zones

    def update(self, bbox):
        self.velocity = self._find_velocity(bbox)
        self.cnt =  bbox.cnt
        self.x, self.y, self.w, self.h = cv.boundingRect(bbox.cnt)
        self.detected_count += 1
        self.missed = 0
        # M = cv.moments(bbox.cnt)
        # if M['m00'] != 0:
        #     self.cx = int(M['m10'] / M['m00'])
        #     self.cy = int(M['m01'] / M['m00'])
        # else:
        #     print("[WARN] Zero moment detected, using bbox center")
        self.cx = self.x + self.w // 2
        self.cy = self.y + self.h // 2
        self.centroid = (self.cx, self.cy)

    def predict(self):
        self.x += self.velocity
        self.cx += self.velocity
        self.centroid = (self.cx, self.cy)
        self.missed += 1

    def _find_velocity(self, new_bbox : Bbox) -> int:
        if new_bbox.x < 5 or \
           -20 < new_bbox.x - self.blackout_zones[0][1] < 5 or \
           -20 < new_bbox.x - self.blackout_zones[1][1] < 5:
            return new_bbox.end_corner()[0] - self.end_corner()[0]
        elif new_bbox.end_corner()[0] > FRAME_SHAPE[1] - 5 or \
             -20 < self.blackout_zones[0][0] - new_bbox.end_corner()[0] < 5 or \
             -20 < self.blackout_zones[1][0] - new_bbox.end_corner()[0] < 5:
            return new_bbox.x - self.x
        else:
            return new_bbox.cx - self.cx

    def _find_coming_side(self) -> Side:
        if self.x + self.w / 2 < FRAME_SHAPE[1] / 2:
            return Side.LEFT
        else:
            return Side.RIGHT
        


class CentroidTracker:
    def __init__(self, max_missed=50, blackout_zones=[[1000, 1200], [1400, 1700]], max_distance=300, height_thresh=100, min_frames_detected=10):
        self.objects: List[TrackerObject] = []
        self.pending_objects: List[TrackerObject] = []
        self.next_id = 0
        self.max_missed = max_missed
        self.blackout_zones = blackout_zones
        self.max_distance = max_distance
        self.height_thresh = height_thresh
        self.min_frames_detected = min_frames_detected

    def _predict_all(self):
        for t_object in self.objects:
            t_object.predict()
            if t_object.missed > self.max_missed:
                self.objects.remove(t_object)
        for pt_object in self.pending_objects:
            pt_object.predict()
            if pt_object.missed > self.max_missed:
                self.pending_objects.remove(pt_object)

    def _match_confirmed_objects(self, bboxes: list[Bbox], used_indices: set):
        # t_objest - tracked object
        for t_object in self.objects:
            dists = [distance.euclidean(t_object.centroid, bbox.centroid) for bbox in bboxes]
            min_dist = min(dists)
            idx = dists.index(min_dist) if dists else -1
            if min_dist < self.max_distance and abs(t_object.h - bboxes[idx].h) < self.height_thresh and idx not in used_indices:
                t_object.update(bboxes[idx])
                # print(f"Object matched with new detection at index {idx}, distance {min_dist}, velocity {t_object.velocity}")
                used_indices.add(idx)
            else:
                t_object.predict()
                if t_object.missed > self.max_missed:
                    self.objects.remove(t_object)

    def _match_pending_objects(self, bboxes: list[Bbox], used_indices: set):
        # pt_object - pending tracked object
        for pt_object in self.pending_objects:
            dists = [distance.euclidean(pt_object.centroid, bbox.centroid) for bbox in bboxes]
            min_dist = min(dists)
            idx = dists.index(min_dist)
            if min_dist < self.max_distance and idx not in used_indices:
                # Calculate new velocity but preserve its sign
                pt_object.update(bboxes[idx])
                used_indices.add(idx)
            else:
                pt_object.predict()
                if pt_object.missed > self.max_missed:
                    self.pending_objects.remove(pt_object)

    def _match_objects(self, bboxes: list[Bbox], used_indices: set, objects: List[TrackerObject]):
        # t_objest - tracked object
        for oid, t_object in enumerate(objects):
            dists = [distance.euclidean(t_object.centroid, bbox.centroid) for bbox in bboxes]
            min_dist = min(dists)
            idx = dists.index(min_dist) if dists else -1
            if min_dist < self.max_distance and abs(t_object.h - bboxes[idx].h) < self.height_thresh and t_object.type == bboxes[idx].type and idx not in used_indices:
                objects[oid].update(bboxes[idx])
                # print(f"Object matched with new detection at index {idx}, distance {min_dist}, velocity {t_object.velocity}")
                used_indices.add(idx)
            else:
                objects[oid].predict()
                if t_object.missed > self.max_missed:
                    objects.remove(t_object)

    def _add_new_objects(self, bboxes: list[Bbox], used_indices: set):
        for oid, bbox in enumerate(bboxes):
            if oid not in used_indices:
                # Only allow creation of new objects at the edges (first and last 200 pixels)
                if bbox.cx < 200 or bbox.cx > (1900 - 200):
                    self.pending_objects.append(TrackerObject(bbox, missed=0, detected_count=1, blackout_zones=self.blackout_zones))
                    used_indices.add(oid)

    def _merge_objects(self, bboxes: list[Bbox], used_indices: set):
        for bbox1 in bboxes:
            for bbox2 in bboxes:
                if bbox1 != bbox2:
                    if (bbox1.x < bbox2.cx < bbox1.end_corner()[0]) and \
                        ((bbox1.y < bbox2.cy < bbox1.end_corner()[1]) or \
                        (abs(bbox1.y - bbox2.end_corner()[1]) < 10)):# and \
                        #abs(bbox1.velocity - bbox2.velocity) < 5)):
                        # Merge contours of t_object and pt_object
                            combined_cnt = np.vstack((bbox1.cnt, bbox2.cnt))
                            bboxes.append(Bbox(combined_cnt))
                            bboxes.remove(bbox1)
                            bboxes.remove(bbox2)
                            break
                    if (abs(bbox1.y - bbox2.y) < 10 or abs(bbox1.end_corner()[1] - bbox2.end_corner()[1]) < 10) and \
                       (abs(bbox1.cx - bbox2.cx) < 50 or \
                       (abs(bbox1.cx - bbox2.cx) < 600 and (any(abs(bbox2.x - zone[1]) < 20 for zone in self.blackout_zones)))):
                            combined_cnt = np.vstack((bbox1.cnt, bbox2.cnt))
                            bboxes.append(Bbox(combined_cnt))
                            bboxes.remove(bbox1)
                            bboxes.remove(bbox2)
                            break

        return bboxes

    def update(self, bboxes: list[Bbox]):
        used_indices = set()
        # If no bboxes, predict all existing objects
        if not bboxes:
            self._predict_all()
            return

        # Merge objects
        bboxes = self._merge_objects(bboxes, used_indices)


        # Match existing objects to new detections
        # self._match_objects(bboxes, used_indices, self.objects)
        # self._match_objects(bboxes, used_indices, self.pending_objects)
        self._match_confirmed_objects(bboxes, used_indices)
        self._match_pending_objects(bboxes, used_indices)


        for pt_object in self.pending_objects:
            if pt_object.detected_count >= self.min_frames_detected and abs(pt_object.velocity) > 5:
                self.objects.append(pt_object)
                self.pending_objects.remove(pt_object)
                self.next_id += 1

        # Add new objects
        self._add_new_objects(bboxes, used_indices)


    def get_counts(self):
        counts = {Type.R2L_car.name: 0, Type.R2L_truck.name: 0, Type.TRAM.name: 0, Type.L2R_car.name: 0, Type.L2R_truck.name: 0, Type.PED.name: 0}
        
        for obj in self.objects:
            if (obj.coming_side == Side.LEFT and obj.x > FRAME_SHAPE[1] - 20) or \
               (obj.coming_side == Side.RIGHT and obj.end_corner()[0] < 20):
                counts[obj.type.name] += 1
                self.objects.remove(obj)
                
        return counts

    def draw(self, frame):
        # Draw confirmed objects
        for t_object in self.objects:
            cv.rectangle(frame, t_object.start_corner(), t_object.end_corner(), t_object.color, 2)
            # Draw centroid and ID
            cv.circle(frame, (t_object.cx, t_object.cy), 5, t_object.color, -1)
            cv.putText(frame, f"{t_object.type}, {t_object.coming_side.name}, {t_object.velocity}", 
              (t_object.cx + 10, t_object.cy), cv.FONT_HERSHEY_SIMPLEX, 1.5, t_object.color, 1)

        # Draw pending objects with dashed lines or different style
        for pt_object in self.pending_objects:
            color = tuple(max(0, c-100) for c in pt_object.color)  # Dimmer color
            # Draw dashed bounding box
            cv.rectangle(frame, pt_object.start_corner(), pt_object.end_corner(), color, 1)
            # Draw centroid
            cv.circle(frame, (pt_object.cx, pt_object.cy), 3, color, -1)
            cv.putText(frame, f"? ({pt_object.detected_count}/{self.min_frames_detected})", 
                      (pt_object.cx + 10, pt_object.cy), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw recovery zone
        for blackout_zone in self.blackout_zones:
            cv.rectangle(frame, (blackout_zone[0], 1), (blackout_zone[1], frame.shape[1]), (255, 255, 255), 2)
        print(f"Drawing {len(self.objects)} confirmed objects and {len(self.pending_objects)} pending objects")


class ContourTracker:
    def __init__(self, max_missed=50, blackout_zones=[[800, 1200], [1400, 1800]], max_distance=200, height_thresh=10, min_frames_detected=3):
        self.objects = {}      # object_id: {'bbox': Bbox, 'missed': int, 'detected_count': int, 'velocity': int}
        self.next_id = 0
        self.max_missed = max_missed
        self.blackout_zones = blackout_zones
        self.max_distance = max_distance
        self.height_thresh = height_thresh
        self.min_frames_detected = min_frames_detected
        self.pending_objects = {}  # Tracks objects not yet meeting min_frames_detected

    def update(self, cnts):
        # detected_centroids = [(b.cx, b.cy) for b in cnts]
        # used_indices = set()
        # new_objects = {}
        for cnt in cnts:
            # Create a Bbox object for each contour
            bbox = Bbox(cnt)

            if bbox.x < 5 or bbox.x + bbox.w > FRAME_SHAPE[1] - 5:
                pass
        print(f"Updating ContourTracker with {len(cnts)} contours")
        self.objects = cnts

    def draw(self, frame):
        # Draw confirmed objects
        for contour in self.objects:
            cv.drawContours(frame, [contour], -1, (0, 255, 0), 2)