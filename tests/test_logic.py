
import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pose_estimator import PoseDetector
from exercises import DumbbellCurl, Squat

# Mock class to simulate PoseDetector behavior without MediaPipe overhead for logic testing
class MockDetector:
    def __init__(self):
        self.lmList = []
    
    def findAngle(self, img, p1, p2, p3):
        # Helper to get coords and visibility
        def get_data(idx):
             # Search lmList for [id, x, y, v]
             for lm in self.lmList:
                 if lm[0] == idx:
                     # Check if it has 4 elements, else default visibility=1.0 for legacy tests
                     if len(lm) > 3:
                         return lm[1], lm[2], lm[3]
                     else:
                         return lm[1], lm[2], 1.0 
             return 0, 0, 0.0

        x1, y1, v1 = get_data(p1)
        x2, y2, v2 = get_data(p2)
        x3, y3, v3 = get_data(p3)

        # if v1 < 0.5 or v2 < 0.5 or v3 < 0.5:
        #      return None

        angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) -
                             np.arctan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        return angle

class TestExerciseLogic(unittest.TestCase):
    def setUp(self):
        self.detector = MockDetector()
        
    def test_curl_counting(self):
        curl = DumbbellCurl(self.detector)
        
        # Simulate arm straight (180 degrees) -> 0% curl
        # Landmarks: Shoulder (12), Elbow (14), Wrist (16)
        # Straight down: (300, 100), (300, 200), (300, 300) -> Angle ~ 180 (or close to 160+ range)
        self.detector.lmList = [
            [12, 300, 100, 1.0], [14, 300, 200, 1.0], [16, 300, 300, 1.0], # Right (Straight)
            [11, 100, 100, 1.0], [13, 100, 200, 1.0], [15, 100, 300, 1.0]  # Left (Straight)
        ]
        # Initial State
        count, feedback = curl.update(None)
        # Should be 0% curl, dir should become 0 (it starts at 0)
        self.assertEqual(count, 0)
        
        # Simulate Curl Up (Angle ~ 30 degrees) (Threshold < 80)
        # Right Arm curled (0 deg), Left arm straight (180 deg)
        self.detector.lmList = [
            [12, 300, 100, 1.0], [14, 300, 200, 1.0], [16, 300, 120, 1.0], # Right (Curled)
            [11, 100, 100, 1.0], [13, 100, 200, 1.0], [15, 100, 300, 1.0]  # Left (Straight)
        ]
        
        count, feedback = curl.update(None)
        self.assertEqual(count, 0.5)
        # Check Right Arm State
        self.assertEqual(curl.dirs['Right'], 1)
        self.assertEqual(curl.dirs['Left'], 0)
        self.assertIn("Up (Right)", feedback)
        
        # Move DOWN (Relax) -> Angle > 140
        self.detector.lmList = [
            [12, 300, 100, 1.0], [14, 300, 200, 1.0], [16, 300, 300, 1.0], # Right (Straight)
            [11, 100, 100, 1.0], [13, 100, 200, 1.0], [15, 100, 300, 1.0]  # Left (Straight)
        ]
        
        count, feedback = curl.update(None)
        self.assertEqual(count, 1.0) # Completed 1 rep
        self.assertEqual(curl.dirs['Right'], 0)

    def test_squat_logic(self):
        squat = Squat(self.detector)
        
        # Standing (180 deg) > 150
        self.detector.lmList = [
            [24, 300, 100, 1.0], [26, 300, 200, 1.0], [28, 300, 300, 1.0], # Right
            [23, 100, 100, 1.0], [25, 100, 200, 1.0], [27, 100, 300, 1.0]  # Left
        ] 
        
        squat.update(None)
        self.assertEqual(squat.dir, 0)
        
        # Squat Down (Angle < 100)
        self.detector.lmList = [
            [24, 400, 200, 1.0], [26, 300, 200, 1.0], [28, 300, 300, 1.0], # Right (Squat)
            [23, 0, 200, 1.0],   [25, 100, 200, 1.0], [27, 100, 300, 1.0]  # Left (Squat mirror)
        ]
        
        squat.update(None)
        self.assertEqual(squat.dir, 1) # Went down
        self.assertIn("Good Depth", squat.feedback)
        
        # Stand Up (Angle > 150)
        self.detector.lmList = [
            [24, 300, 100, 1.0], [26, 300, 200, 1.0], [28, 300, 300, 1.0],
            [23, 100, 100, 1.0], [25, 100, 200, 1.0], [27, 100, 300, 1.0]
        ]
        squat.update(None)
        self.assertEqual(squat.count, 1.0) # Rep complete

if __name__ == '__main__':
    unittest.main()
