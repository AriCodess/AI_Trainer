
import cv2
import time
import argparse
from pose_estimator import PoseDetector
from exercises import DumbbellCurl, Squat

def main():
    parser = argparse.ArgumentParser(description="AI Personal Trainer")
    parser.add_argument("--video", type=str, default=None, help="Path to video file. Leave empty for webcam.")
    parser.add_argument("--exercise", type=str, default="curl", choices=["curl", "squat"], help="Type of exercise to analyze.")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration.")
    args = parser.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)

    if args.gpu:
        print("Initializing GPU Pose Detector...")
        from pose_estimator import GPUPoseDetector
        detector = GPUPoseDetector(detectionCon=0.7, trackCon=0.7)
    else:
        detector = PoseDetector(detectionCon=0.7, trackCon=0.7)
    
    if args.exercise == "curl":
        exercise = DumbbellCurl(detector)
        print("Starting Dumbbell Curl Analysis...")
    elif args.exercise == "squat":
        exercise = Squat(detector)
        print("Starting Squat Analysis...")
        
    pTime = 0
    
    while True:
        success, img = cap.read()
        if not success:
            print("Video capture ended or failed.")
            break
            
        img = cv2.resize(img, (1280, 720))
        img = detector.findPose(img, draw=True)
        lmList = detector.findPosition(img, draw=False)
        
        if len(lmList) != 0:
            count, feedback = exercise.update(img)
            
            # Draw Feedback Panel
            cv2.rectangle(img, (0, 0), (350, 150), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f"Reps: {int(count)}", (50, 60), 
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.putText(img, str(feedback), (10, 130), 
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, f"FPS: {int(fps)}", (1100, 50), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        cv2.imshow("AI Personal Trainer", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
