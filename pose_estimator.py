
import cv2
import mediapipe as mp
import math

class PoseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # Note: model_complexity=1 is default, can be 0, 1, or 2.
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, 
                                     model_complexity=1,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # cx, cy are pixel coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Append visibility too
                self.lmList.append([id, cx, cy, lm.visibility])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        # p1, p2, p3 are indices of the landmarks (e.g. 11, 13, 15 for arm)
        
        # Ensure lmList is populated
        if len(self.lmList) == 0:
            return None
            
        # Check visibility
        v1 = self.lmList[p1][3]
        v2 = self.lmList[p2][3]
        v3 = self.lmList[p3][3]
        
        # Threshold for visibility (disabled for debugging)
        # if v1 < 0.3 or v2 < 0.3 or v3 < 0.3:
        #    return None

        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]
        x3, y3 = self.lmList[p3][1:3]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            # Display angle
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle


# Import Task API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GPUPoseDetector(PoseDetector):
    def __init__(self, model_path='pose_landmarker_full.task', detectionCon=0.5, trackCon=0.5):
        # We don't call super().__init__ because we don't want legacy Pose
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose # Used for connection constants
        
        base_options = python.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.GPU)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
            output_segmentation_masks=False)
            
        try:
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            print("GPU PoseLandmarker created successfully.")
        except Exception as e:
            print(f"Failed to create GPU delegate: {e}. Falling back to CPU.")
            base_options = python.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.CPU)
            options.base_options = base_options
            self.landmarker = vision.PoseLandmarker.create_from_options(options)

        self.lmList = []
        self.timestamp_ms = 0

    def findPose(self, img, draw=True):
        # Create MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        
        # Create monotonic timestamp
        self.timestamp_ms += 33 # Approx 30 FPS increment
        
        self.results = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        
        if self.results.pose_landmarks:
            if draw:
                # task API drawing utils are different or we can map back to legacy?
                # Legacy draw_landmarks expects NormalizedLandmarkList (proto)
                # Task API returns Python objects. mpDraw might fail.
                # We can manually draw or construct a proto-like object.
                # For simplicity, let's manually draw connections since we have coordinates in findPosition?
                # Or just iterate and draw lines.
                pass 
                
        # To support drawing, we can do it in findPosition or just skip extensive drawing for performance 
        # (User wants GPU for speed).
        # But we really should draw the skeleton.
        if draw and self.results.pose_landmarks:
             for landmarks in self.results.pose_landmarks:
                 # Draw connections
                 for connection in self.mpPose.POSE_CONNECTIONS:
                     start_idx = connection[0]
                     end_idx = connection[1]
                     if start_idx < len(landmarks) and end_idx < len(landmarks):
                         # Get coords
                         start = landmarks[start_idx]
                         end = landmarks[end_idx]
                         h, w, c = img.shape
                         x1, y1 = int(start.x * w), int(start.y * h)
                         x2, y2 = int(end.x * w), int(end.y * h)
                         cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                         cv2.circle(img, (x1, y1), 3, (255, 0, 0), cv2.FILLED)
                         cv2.circle(img, (x2, y2), 3, (255, 0, 0), cv2.FILLED)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            # Task API returns list of lists. We detect one person usually.
            myHand = self.results.pose_landmarks[0]
            for id, lm in enumerate(myHand):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy, lm.visibility])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

def main():
    # Simple test code
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
