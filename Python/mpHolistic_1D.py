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

# ekf_left_hand = [ExtendedKalmanFilter(x = np.zeros(shape=(6, 1)), Q = np.eye(6) * 0.05, R = np.eye(3) * 0.05, damping = 0.1) for _ in range(21)]

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

    # if results.pose_world_landmarks:
    #   mypose_world_landmarks = results.pose_world_landmarks.landmark
    #   mp_drawing.draw_landmarks(
    #     image=frame,
    #     landmark_list=results.pose_landmarks,
    #     connections=mp_holistic.POSE_CONNECTIONS,
    #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    #   )

    # if results.face_landmarks:
    #   mp_drawing.draw_landmarks(
    #     image=frame,
    #     landmark_list=results.face_landmarks,
    #     connections=mp_holistic.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    #   )
    #   mp_drawing.draw_landmarks(
    #     image=frame,
    #     landmark_list=results.face_landmarks,
    #     landmark_drawing_spec=None,
    #     connections=mp_holistic.FACEMESH_TESSELATION,
    #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    #   )
      
    if results.left_hand_landmarks:
      for i, landmark in enumerate(results.left_hand_landmarks.landmark):
        x = landmark.x * w
        y = landmark.y * h
        z = landmark.z

        # Prepare measurement
        measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])

        # Predict next state
        ekf_left_hand[i].x[0][0] = landmark.x
        ekf_left_hand[i].x[1][0] = landmark.y
        ekf_left_hand[i].x[2][0] = landmark.z
        ekf_left_hand[i].predict(dt = DT)
        
        # Correct with actual measurement vector z
        corrected_x, convariance_P = ekf_left_hand[i].update(measurement = measured)

        # value from kalman
        pred_x, pred_y, pred_z = corrected_x[0, 0], corrected_x[1, 0], corrected_x[2, 0]

        print(f"Raw_x [{i}]: {landmark.x}, pred_x [{i}]: {pred_x}")
        print(f"Raw_y [{i}]: {landmark.y}, pred_y [{i}]: {pred_y}")
        print(f"Raw_z [{i}]: {landmark.z}, pred_z [{i}]: {pred_z}")

        results.left_hand_landmarks.landmark[i].x = pred_x
        results.left_hand_landmarks.landmark[i].y = pred_y
        results.left_hand_landmarks.landmark[i].z = z        # I don't use pred_z because something weird

      mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
      )

    # if results.right_hand_landmarks:
    #   wrist_test = results.right_hand_landmarks.landmark[0]
    #   x = wrist_test.x * w
    #   y = wrist_test.y * h
    #   z = wrist_test.z

    #   # Prepare measurement
    #   measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])
      
    #   # Predict next state
    #   ekf.x[0][0] = wrist_test.x
    #   ekf.x[1][0] = wrist_test.y
    #   ekf.x[2][0] = wrist_test.z
    #   ekf.predict(dt = DT)

    #   # Correct with actual measurement vector z
    #   corrected_x, convariance_P = ekf.update(measurement = measured)

    #   # value from kalman
    #   pred_x, pred_y, pred_z = corrected_x[0, 0], corrected_x[1, 0], corrected_x[2, 0]

    #   # print(f"Raw_x: {wrist_test.x}, pred_x: {pred_x}")
    #   # print(f"Raw_y: {wrist_test.y}, pred_y: {pred_y}")
    #   # print(f"Raw_z: {wrist_test.z}, pred_z: {pred_z}")

    #   results.right_hand_landmarks.landmark[0].x = pred_x
    #   results.right_hand_landmarks.landmark[0].y = pred_y
    #   results.right_hand_landmarks.landmark[0].z = z        # I don't use pred_z because something weird

    #   mp_drawing.draw_landmarks(
    #       image=frame,
    #       landmark_list=results.right_hand_landmarks,
    #       connections=mp_holistic.HAND_CONNECTIONS,
    #       landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    #   )

    cv2.imshow(WINNAME, frame)

    if cv2.waitKey(5) & 0xFF == 27:
      break
    elif cv2.getWindowProperty(WINNAME, cv2.WND_PROP_VISIBLE) < 1:
      break

cap.release()
cv2.destroyAllWindows()