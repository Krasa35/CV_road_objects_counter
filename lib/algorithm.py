import cv2 as cv


class FrameDifference():
    def __init__(self):
        self.prev_frame_static = None
        self.prev_frame_dynamic = None
        self.mode = 0

    def apply(self, frame, threshold=30, update_background=False, two_background=False):
        if self.prev_frame_static is not None and self.prev_frame_dynamic is not None:
            if update_background:
                diff = cv.absdiff(self.prev_frame_dynamic, frame)
            elif two_background:
                diff1 = cv.absdiff(self.prev_frame_static, frame)
                diff2 = cv.absdiff(self.prev_frame_dynamic, frame)
                diff = cv.bitwise_or(diff1, diff2)
            else:
                diff = cv.absdiff(self.prev_frame_static, frame)
            _, img_output = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)
        if self.prev_frame_static is None:
            self.prev_frame_static = frame
        self.prev_frame_dynamic = frame
        self.mode = 1 if update_background else 2 if two_background else 3
        return img_output if 'img_output' in locals() else frame
        

    def getBackgroundModel(self):
        if self.prev_frame_static is None or self.prev_frame_dynamic is None:
            return None
        if self.mode == 1:
            return self.prev_frame_dynamic
        elif self.mode == 2:
            return self.prev_frame_static
        elif self.mode == 3:
            return self.prev_frame_static
        