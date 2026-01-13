
import numpy as np
import cv2

class Exercise:
    def __init__(self, detector):
        self.detector = detector
        self.count = 0
        self.dir = 0 # 0 for going down/up, 1 for completing rep
        self.feedback = ""

    def calculate_angle(self, img, p1, p2, p3):
        return self.detector.findAngle(img, p1, p2, p3)

class DumbbellCurl(Exercise):
    def __init__(self, detector):
        super().__init__(detector)
        # Separate state for each arm
        self.dirs = {'Right': 0, 'Left': 0}
        self.reps = {'Right': 0, 'Left': 0} # Internal counts

    def update_arm(self, angle, side):
        if angle is None:
            return
            
        # Logic
        if angle < 80: # Full curl up (relaxed from 50)
            if self.dirs[side] == 0:
                self.reps[side] += 0.5
                self.dirs[side] = 1
                self.feedback = f"Up ({side})"

        if angle > 140: # Arm fully extended (relaxed from 150)
            if self.dirs[side] == 1:
                self.reps[side] += 0.5
                self.dirs[side] = 0
                self.feedback = f"Down ({side})"

    def update(self, img):
        # Calculate angles for both
        angle_r = self.calculate_angle(img, 12, 14, 16)
        angle_l = self.calculate_angle(img, 11, 13, 15)
        
        # Update states independently
        self.update_arm(angle_r, 'Right')
        self.update_arm(angle_l, 'Left')

        # Total count
        self.count = self.reps['Right'] + self.reps['Left']
        
        # If feedback is stale/empty, show detailed stats
        # We'll just append stats to feedback if it's generic
        if "Up" not in self.feedback and "Down" not in self.feedback:
             self.feedback = f"L: {int(self.reps['Left'])} R: {int(self.reps['Right'])}"
             
        # Add visual debug for angles
        cv2.putText(img, f"R: {int(angle_r) if angle_r else 'N'} L: {int(angle_l) if angle_l else 'N'}", 
                    (900, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        return self.count, self.feedback

class Squat(Exercise):
    def update(self, img):
        # Right Leg: 24 (Hip), 26 (Knee), 28 (Ankle)
        # Left Leg: 23, 25, 27
        
        # Calculate for both
        angle_r = self.calculate_angle(img, 24, 26, 28)
        angle_l = self.calculate_angle(img, 23, 25, 27)
        
        angle = None
        
        if angle_r is not None and angle_l is not None:
             angle = min(angle_r, angle_l)
        elif angle_r is not None:
             angle = angle_r
        elif angle_l is not None:
             angle = angle_l
             
        if angle is None:
             return self.count, self.feedback
        
        # Squat logic
        if angle <= 100: # Squat depth
            if self.dir == 0:
                self.count += 0.5
                self.dir = 1
                self.feedback = "Good Depth!"
        
        if angle >= 150: # Standing up
            if self.dir == 1:
                self.count += 0.5
                self.dir = 0
                self.feedback = "Stand straight"
            else:
                 # Only show count when not transitioning
                 pass

        # If feedback is stale, show count
        if self.feedback == "Good Depth!" and angle > 100:
             self.feedback = f"Squats: {int(self.count)}"
        if self.feedback == "Stand straight" and angle < 150:
             self.feedback = f"Squats: {int(self.count)}"
             
        if self.feedback == "":
            self.feedback = f"Squats: {int(self.count)}"

        return self.count, self.feedback
