import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Recolor image
    results = holistic.process(image) # Make Detection

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # print(image.shape) # probably witdth*height of camera or something

    if results.pose_world_landmarks:
      mypose_world_landmarks = results.pose_world_landmarks.landmark
      mp_drawing.draw_landmarks(
        image=image,
        landmark_list=results.pose_landmarks,
        connections=mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
      )

    if results.face_landmarks:
      myface_landmarks = results.face_landmarks
      # draw Face Contours
      mp_drawing.draw_landmarks(
        image=image,
        landmark_list=results.face_landmarks,
        connections=mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
      )
      # draw Face Tesselation
      mp_drawing.draw_landmarks(
        image=image,
        landmark_list=results.face_landmarks,
        landmark_drawing_spec=None,
        connections=mp_holistic.FACEMESH_TESSELATION,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
      )
      
    if results.left_hand_landmarks:
      myleft_landmarks = results.left_hand_landmarks
      mp_drawing.draw_landmarks(
        image=image,
        landmark_list=results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
      )

    if results.right_hand_landmarks:
      myright_landmarks = results.right_hand_landmarks
      mp_drawing.draw_landmarks(
        image=image,
        landmark_list=results.right_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
      )
    


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
    elif cv2.getWindowProperty('MediaPipe Holistic', cv2.WND_PROP_VISIBLE) < 1:
      break
cap.release()