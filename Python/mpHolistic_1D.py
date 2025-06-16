import cv2
import numpy as np
import mediapipe as mp
from src import ExtendedKalmanFilter_1D

WIDTH = 640
HEIGHT = 480
FPS = 30
WINNAME = "mediaipe_EKF"
DT = 1/FPS
MIN_DEPTH = 0.1  # minimum depth (10 cm)
MAX_DEPTH = 10.0  # maximum depth (10 meters)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# webcam input:
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # windows use Direct Show cv2.CAP_DSHOW
cv2.namedWindow(WINNAME, cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(WINNAME, WIDTH, HEIGHT)

ekf = ExtendedKalmanFilter_1D(
  x = np.zeros(shape=(6, 1)), # [px, py, pz, vx, vy, vz]
  Q = np.eye(6) * 0.01,       # process noise
  R = np.eye(3) * 0.2,        # trust model
  damping = 0.05
)

ekf_left_hand_x = [ExtendedKalmanFilter_1D(x=np.zeros(shape=(2, 1)), Q=np.eye(2) * 0.01, R=np.eye(1) * 0.2, damping=0.05) for _ in range(21)]

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
  ) as holistic:
  while cap.isOpened():
    success, frame = cap.read()

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    frame.flags.writeable = True
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Recolor frame
    results = holistic.process(rgb_frame) # Make Detection

    h, w, channels = frame.shape

    if results.left_hand_landmarks:
      for i, landmark in enumerate(results.left_hand_landmarks.landmark):
        x = landmark.x
        y = landmark.y
        z = landmark.z

        # Prepare measurement
        measured = np.array([[np.float32(x)]])

        # Predict next state
        ekf_left_hand_x[i].x[0][0] = landmark.x
        ekf_left_hand_x[i].predict(dt = DT)
        
        # Correct with actual measurement vector z
        corrected_x, convariance_P = ekf_left_hand_x[i].update(measurement = measured)

        # value from kalman
        pred_x = corrected_x[0, 0]

        print(f"Raw_x [{i}]: {landmark.x}, pred_x [{i}]: {pred_x}")

        results.left_hand_landmarks.landmark[i].x = pred_x

      mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
      )

    cv2.imshow(WINNAME, frame)

    if cv2.waitKey(5) & 0xFF == 27:
      break
    elif cv2.getWindowProperty(WINNAME, cv2.WND_PROP_VISIBLE) < 1:
      break

cap.release()
cv2.destroyAllWindows()