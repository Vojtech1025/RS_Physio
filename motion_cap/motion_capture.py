import cv2
import pyrealsense2 as rs
import numpy as np
import mediapipe as mp
from enum import Enum
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 



class StreamConfig(Enum):
    COLOR = 0
    DEPTH = 1


class Feature(Enum):
    POSE = 0
    HAND = 1
    FACE = 2


# global dictionary with amount of landmarks for each feature
LANDMARKS = {
    Feature.POSE: 33,
    Feature.HAND: 21,
    Feature.FACE: 468
}


class MotionCapture:
    def __init__(self, stream_config={StreamConfig.COLOR:True, StreamConfig.DEPTH:True}, feature_config={Feature.HAND:True}, width=640, height=480, fps=30):
        self.stream_config = stream_config
        self.feature_config = feature_config
        self.W = width
        self.H = height
        self.FPS = fps

        # PREPARE CAMERA
        print("[INFO] Creating pipeline...", flush=True)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # Configure depth and color streams
        if stream_config[StreamConfig.COLOR]:
            print("[INFO] Setting up color stream...", flush=True)
            self.config.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, self.FPS)
        if stream_config[StreamConfig.DEPTH]:
            print("[INFO] Setting up depth stream...", flush=True)
            self.config.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16, self.FPS)

        print("[INFO] Starting streaming...", flush=True)
        self.pipeline.start(self.config)
        print("[INFO] Camera ready.", flush=True)

        # PREPARE MEDIAPIPE
        self.mpDraw = mp.solutions.drawing_utils

        # the output will be written to output.avi
        self.out = cv2.VideoWriter(
            'output.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            15.,
            (self.W, self.H))

    def GetImgs(self, color_frame, depth_frame):
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image

    def GetFrames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        return color_frame, depth_frame

    def Run(self, recording=False):
        # This function is used to run the motion capture to detect all features chosen in the config
  
        # PREPARE MEDIAPIPE
        if self.feature_config[Feature.HAND]:
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

        if self.feature_config[Feature.POSE]:
            self.mpPose = mp.solutions.pose
            self.pose = self.mpPose.Pose(static_image_mode=False, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)

        if self.feature_config[Feature.FACE]:
            self.mpFaceMesh = mp.solutions.face_mesh
            self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # CAPTURE LOOP
        while True:
            # Get frames
            color_frame, depth_frame = self.GetFrames()
            # Get images
            color_image, depth_image = self.GetImgs(color_frame, depth_frame)

            # We need to use two different images for processing and for output, because the processing should be done on the original image
            # and the output should be done on the image with the drawn features
            # If we let the processing be done on the image with already drawn features, it might corrupt the new found features
            imgRGB = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            imgOut = color_image.copy()

            # Draw supporting points into corners of the image
            cv2.putText(imgOut, "[80, 20]", (80, 20), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)
            cv2.putText(imgOut, "[20, 60]", (20, 60), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)
            cv2.putText(imgOut, "[20, 20]", (20, 20), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)

            # Detect HANDS
            if self.feature_config[Feature.HAND]:
                # Find the hand and its landmarks
                hands, imgOut = self.FindHands(imgRGB=imgRGB, imgOut=imgOut)

            # Detect POSE
            if self.feature_config[Feature.POSE]:
                # Find the pose and its landmarks
                imgOut = self.FindPose(imgRGB=imgRGB, imgOut=imgOut)
                
            # Detect FACE
            if self.feature_config[Feature.FACE]:
                # Find the face and its landmarks
                imgOut, faces = self.FindFace(imgRGB=imgRGB, imgOut=imgOut)

            # Write the output video
            if recording:
                self.out.write(imgOut.astype('uint8'))
            # Display
            cv2.imshow("Image", imgOut)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.pipeline.stop()

    def FindHands(self, imgRGB, imgOut, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = imgRGB.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)
                    # put idx number to each landmark
                    # cv2.putText(img, "[" + str(px) + ", " + str(py) + "]", (px+5, py+5), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(imgOut, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(imgOut, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(imgOut, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if draw:
            return allHands, imgOut
        else:
            return allHands

    def FindPose(self, imgRGB, imgOut, draw=True):
        """
        Finds the pose of the object in the image.
        :param img: Image to find the pose in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(imgOut, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return imgOut

    def FindFace(self, imgRGB, imgOut, draw=True):
        """
        Finds the face in the image.
        :param img: Image to find the face in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        self.results = self.faceMesh.process(imgRGB)
        faces = []

        drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(imgOut, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = imgOut.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return imgOut, faces


    # Simple Train Linear model.
    def RecordBodyMotion(self, class_name:str, at_first = False):
        num_coords = 0

        if self.feature_config[Feature.POSE]:
            num_coords += LANDMARKS[Feature.POSE] # 33 landmarks on body

        if self.feature_config[Feature.HAND]:
            num_coords += LANDMARKS[Feature.HAND]*2 # 21 landmarks on each hand

        if self.feature_config[Feature.FACE]:
            num_coords += LANDMARKS[Feature.FACE] # 468 landmarks on face

        if at_first:
            landmarks = ['class']
            for val in range(1, num_coords+1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
            try:
                os.remove('coords.csv') # remove file
            except:
                pass
            with open('coords.csv', mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)

        cap = cv2.VideoCapture(1)
        mp_drawing = mp.solutions.drawing_utils # Drawing helpers
        mp_holistic = mp.solutions.holistic # Mediapipe Solutions
        # Initiate holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
                
                # Make Detections
                results = holistic.process(image) # face + pose

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Export coordinates
                try:
                    row = []
                    # Extract Pose landmarks
                    if self.feature_config[Feature.POSE]:
                        pose = results.pose_landmarks.landmark
                        # Concate rows
                        row += list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                    # Extract Face landmarks
                    if self.feature_config[Feature.FACE]:
                        face = results.face_landmarks.landmark
                        # Concate rows
                        row += list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                    
                    # Extract Right hand landmarks
                    if self.feature_config[Feature.HAND]:
                        hands = results.right_hand_landmarks.landmark
                        # Concate rows
                        row += list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hands]).flatten())

                    # Extract Left hand landmarks
                    if self.feature_config[Feature.HAND]:
                        hands = results.left_hand_landmarks.landmark
                        # Concate rows
                        row += list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hands]).flatten())

                    # Append class name 
                    row.insert(0, class_name)
                    
                    # Export to CSV
                    with open('coords.csv', mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(row) 
                    
                except Exception as e:
                    print(str(e))
                    pass

                
                # 1. Draw face landmarks
                if self.feature_config[Feature.FACE]:
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                        )
                
                # 2. Right hand
                if self.feature_config[Feature.HAND]:
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                        )

                # 3. Left Hand
                if self.feature_config[Feature.HAND]:
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        )

                # 4. Pose Detections
                if self.feature_config[Feature.POSE]:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                                
                cv2.imshow('Raw Webcam Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def TrainBody(self):
        df = pd.read_csv('coords.csv')
        X = df.drop('class', axis=1) # features
        y = df['class'] # target value

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

        pipelines = {
            'lr':make_pipeline(StandardScaler(), LogisticRegression()),
            'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
            'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        }

        fit_models = {}
        for algo, pipeline in pipelines.items():
            model = pipeline.fit(self.X_train, self.y_train)
            fit_models[algo] = model

        for algo, model in fit_models.items():
            yhat = model.predict(self.X_test)
            print(algo, accuracy_score(self.y_test, yhat))

        try:
            os.remove('body_language.pkl') # remove file
        except:
            pass
        with open('body_language.pkl', 'wb') as f:
            pickle.dump(fit_models['rf'], f)

    def TestBody(self):
        with open('body_language.pkl', 'rb') as f:
            model = pickle.load(f)

        cap = cv2.VideoCapture(1)
        mp_drawing = mp.solutions.drawing_utils # Drawing helpers
        mp_holistic = mp.solutions.holistic # Mediapipe Solutions
        # Initiate holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
                
                # Make Detections
                results = holistic.process(image)
                
                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
          
                # Export coordinates
                try:
                    row = []
                    # Extract Pose landmarks
                    if self.feature_config[Feature.POSE]:
                        pose = results.pose_landmarks.landmark
                        # Concate rows
                        row += list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                    # Extract Face landmarks
                    if self.feature_config[Feature.FACE]:
                        face = results.face_landmarks.landmark
                        # Concate rows
                        row += list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                    
                    # Extract Right hand landmarks
                    if self.feature_config[Feature.HAND]:
                        hands = results.right_hand_landmarks.landmark
                        # Concate rows
                        row += list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hands]).flatten())

                    # Extract Left hand landmarks
                    if self.feature_config[Feature.HAND]:
                        hands = results.left_hand_landmarks.landmark
                        # Concate rows
                        row += list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hands]).flatten())
                    
                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    # print(body_language_class, body_language_prob)
                    
                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))
                    
                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Get status box
                    cv2.rectangle(image, (0,0), (100, 60), (245, 117, 16), -1)
                    
                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                except Exception as e:
                    # print(str(e))
                    pass


                # 1. Draw face landmarks
                if self.feature_config[Feature.FACE]:
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                        )
                
                # 2. Right hand
                if self.feature_config[Feature.HAND]:
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                        )

                # 3. Left Hand
                if self.feature_config[Feature.HAND]:
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        )

                # 4. Pose Detections
                if self.feature_config[Feature.POSE]:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                                
                cv2.imshow('Raw Webcam Feed', image)
                self.out.write(image.astype('uint8'))

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()




class Benchmark:
    def __init__(self, feature_config={Feature.HAND:True}, width=640, height=480, fps=30):
        self.feature_config = feature_config
        self.W = width
        self.H = height
        self.FPS = fps

        # PREPARE MEDIAPIPE
        self.mpDraw = mp.solutions.drawing_utils

        self.pipeline_D435 = rs.pipeline()
        self.config_D435 = rs.config()
        self.config_D435.enable_device('215322075176')
        self.config_D435.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16, self.FPS)
        self.config_D435.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, self.FPS)

        self.pipeline_L515 = rs.pipeline()
        self.config_L515 = rs.config()
        self.config_L515.enable_device('f1382219')
        self.config_L515.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16, self.FPS)
        self.config_L515.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, self.FPS)

        align_to = rs.stream.color
        self.alignL515 = rs.align(align_to)
        self.alignD435 = rs.align(align_to)

    def FindHands(self, imgRGB, imgOut, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = imgRGB.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)
                    # put idx number to each landmark
                    # cv2.putText(img, "[" + str(px) + ", " + str(py) + "]", (px+5, py+5), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(imgOut, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(imgOut, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(imgOut, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if draw:
            return allHands, imgOut
        else:
            return allHands

    def FindPose(self, imgRGB, imgOut, draw=True):
        """
        Finds the pose of the object in the image.
        :param img: Image to find the pose in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(imgOut, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return imgOut

    def FindFace(self, imgRGB, imgOut, draw=True):
        """
        Finds the face in the image.
        :param img: Image to find the face in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        self.results = self.faceMesh.process(imgRGB)
        faces = []

        drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(imgOut, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = imgOut.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return imgOut, faces


    def PutText(self, data:dict, imgOut, depth_frame):
        if data:
            thumb = 4 # index of thumb in lmList
            for hand in data:
                if hand:
                    landmarks = hand["lmList"]  # List of 21 Landmark points

                    for ld in landmarks:
                        # put idx number to each landmark
                        cv2.putText(imgOut, "[" + str(ld[0]) + ", " + str(ld[1]) + "]", (ld[0]+5, ld[1]+5), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)

                    if landmarks:
                        # We need to check if the landmark is out of the image
                        thumb_x = landmarks[thumb][0]
                        thumb_y = landmarks[thumb][1]
                        if thumb_x >= 0 and thumb_x < self.W and thumb_y >= 0 and thumb_y < self.H:
                            # print(self.W, "x", self.H , " ", landmarks[thumb] , " - ", thumb_x, " ", thumb_y ,flush=True)
                            depth = rs.depth_frame.get_distance(depth_frame, thumb_x, thumb_y)
                            cv2.putText(imgOut, str(depth) + " m", (landmarks[thumb][0]+10, landmarks[thumb][1]+10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)

    def Run(self, deviceL515=True, deviceD435=True):
        # Start streaming from both cameras
        if deviceD435:
            self.pipeline_D435.start(self.config_D435)
        if deviceL515:
            self.pipeline_L515.start(self.config_L515)

        if self.feature_config[Feature.HAND]:
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        while True:
            # Camera L515
            if deviceL515:
                # This call waits until a new coherent set of frames is available on a device
                framesL515 = self.pipeline_L515.wait_for_frames()
                
                #Aligning color frame to depth frame
                aligned_frames =  self.alignL515.process(framesL515)
                depth_frame = aligned_frames.get_depth_frame()
                aligned_color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not aligned_color_frame: continue
                
                depth_color_frame = rs.colorizer().colorize(depth_frame)
                depth_image = np.asanyarray(depth_color_frame.get_data())
                color_image = np.asanyarray(aligned_color_frame.get_data())
                
                if self.feature_config[Feature.HAND]:
                    hands, depth_image = self.FindHands(color_image, depth_image, draw=True)
                    hands, color_image = self.FindHands(color_image, color_image, draw=True)
                    self.PutText(hands, color_image, depth_frame)
                
                # Display
                cv2.imshow('L515_color', color_image)
                cv2.imshow('L515_depth', depth_image)

            # Camera D435
            if deviceD435:
                # This call waits until a new coherent set of frames is available on a device
                framesD435 = self.pipeline_D435.wait_for_frames()
                
                #Aligning color frame to depth frame
                aligned_frames =  self.alignD435.process(framesD435)
                depth_frame = aligned_frames.get_depth_frame()
                aligned_color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not aligned_color_frame: continue

                depth_color_frame = rs.colorizer().colorize(depth_frame)
                depth_image = np.asanyarray(depth_color_frame.get_data())
                color_image = np.asanyarray(aligned_color_frame.get_data())

                if self.feature_config[Feature.HAND]:
                    hands, depth_image = self.FindHands(color_image, depth_image, draw=True)
                    hands, color_image = self.FindHands(color_image, color_image, draw=True)
                    self.PutText(hands, color_image, depth_frame)

                # Display
                cv2.imshow('D435_color', color_image)
                cv2.imshow('D435_depth', depth_image)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.pipeline_D435.stop()
        self.pipeline_L515.stop()
        cv2.destroyAllWindows()




if __name__ == "__main__":
    capture = MotionCapture(feature_config={Feature.HAND:True, Feature.POSE:False, Feature.FACE:False})
    capture.Run()

    # capture.RecordBodyMotion("Arms Up", at_first=False)
    # capture.TrainBody()
    # capture.TestBody()

    # benchmark = Benchmark()
    # benchmark.Run(deviceL515=True, deviceD435=False)

