
import mediapipe as mp
print("MediaPipe file:", mp.__file__)
print("Dir(mp):", dir(mp))
try:
    print("Solutions:", mp.solutions)
except AttributeError as e:
    print("Error accessing solutions:", e)
