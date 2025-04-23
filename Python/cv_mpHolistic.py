import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Init KalmanFilter
'''
x y z is position
dx dy dz is velocities
'''
def create_kf(smoothness: float = 0.03) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(dynamParams=6, measureParams=3) # 6 dynamic parameters: [x, y, z, dx, dy, dz], 3 measurement parameters: [x, y, z]
    
    # Measurement matrix: extract position (x, y, z) from state vector
    kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0], # x
            [0, 1, 0, 0, 0, 0], # y
            [0, 0, 1, 0, 0, 0], # z
        ], dtype=np.float32
    )

    # Transition matrix: constant velocity model for 3D
    kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],  # x' = x + dx
            [0, 1, 0, 0, 1, 0],  # y' = y + dy
            [0, 0, 1, 0, 0, 1],  # z' = z + dz
            [0, 0, 0, 1, 0, 0],  # dx' = dx
            [0, 0, 0, 0, 1, 0],  # dy' = dy
            [0, 0, 0, 0, 0, 1]   # dz' = dz
        ], dtype=np.float32
    )

    # Process noise: tweak the multiplier to tune smoothness
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * smoothness

    return kf

kf_face = [create_kf() for _ in range(478)]
kf_pose = [create_kf() for _ in range(33)]
kf_left_hand = [create_kf() for _ in range(21)]
kf_right_hand = [create_kf() for _ in range(21)]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # windows use Direct Show cv2.CAP_DSHOW
winName = "KalmanMediaPipe"
cv2.namedWindow(winName, cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(winName, 640, 480)

with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
    ) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        h, w, channels = frame.shape

        if results.face_landmarks:
            for i, landmark in enumerate(results.face_landmarks.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = int(landmark.z)

                # Update Kalman filter for the current landmark
                measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])
                kf_face[i].correct(measured)
                prediction = kf_face[i].predict()

                pred_x, pred_y, pred_z = int(prediction[0]), int(prediction[1]), int(prediction[2])
                results.face_landmarks.landmark[i].x = pred_x / w
                results.face_landmarks.landmark[i].y = pred_y / h
                results.face_landmarks.landmark[i].z = pred_z

            # draw Face Contours
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.face_landmarks,
                connections=mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            # draw Face Tesselation
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.face_landmarks,
                landmark_drawing_spec=None,
                connections=mp_holistic.FACEMESH_TESSELATION,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                # Get the (x, y) position of the landmark
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = int(landmark.z)

                # Update Kalman filter for the current landmark
                measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])
                kf_pose[i].correct(measured)
                prediction = kf_pose[i].predict()

                pred_x, pred_y, pred_z = int(prediction[0]), int(prediction[1]), int(prediction[2])
                results.pose_landmarks.landmark[i].x = pred_x / w
                results.pose_landmarks.landmark[i].y = pred_y / h
                results.pose_landmarks.landmark[i].z = pred_z
            
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.pose_landmarks,
                connections=mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        if results.right_hand_landmarks:
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                # Get the (x, y) position of the landmark
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = int(landmark.z)

                # Update Kalman filter for the current landmark
                measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])
                kf_right_hand[i].correct(measured)
                prediction = kf_right_hand[i].predict()

                pred_x, pred_y, pred_z = int(prediction[0]), int(prediction[1]), int(prediction[2])
                results.right_hand_landmarks.landmark[i].x = pred_x / w
                results.right_hand_landmarks.landmark[i].y = pred_y / h
                results.right_hand_landmarks.landmark[i].z = pred_z

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.right_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        if results.left_hand_landmarks:
            for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                # Get the (x, y) position of the landmark
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = int(landmark.z)

                # Update Kalman filter for the current landmark
                measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])
                kf_left_hand[i].correct(measured)
                prediction = kf_left_hand[i].predict()

                pred_x, pred_y, pred_z = int(prediction[0]), int(prediction[1]), int(prediction[2])
                results.left_hand_landmarks.landmark[i].x = pred_x / w
                results.left_hand_landmarks.landmark[i].y = pred_y / h
                results.left_hand_landmarks.landmark[i].z = pred_z

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.left_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        cv2.imshow(winName, frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break
        elif cv2.getWindowProperty(winName, cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()