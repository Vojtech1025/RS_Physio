import cv2
import pyrealsense2 as rs
import numpy as np
import mediapipe as mp
from enum import Enum


class StreamConfig(Enum):
    COLOR = 0
    DEPTH = 1


class Feature(Enum):
    POSE = 0
    HAND = 1
    FACE = 2


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

    def Run(self):
        # This function is used to run the motion capture to detect all features chosen in the config
  
        # PREPARE MEDIAPIPE
        if self.feature_config[Feature.HAND]:
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        if self.feature_config[Feature.POSE]:
            self.mpPose = mp.solutions.pose
            self.pose = self.mpPose.Pose(static_image_mode=False, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        if self.feature_config[Feature.FACE]:
            self.mpFaceMesh = mp.solutions.face_mesh
            self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

                if hands:
                    thumb = 4 # index of thumb in lmList
                    for hand in hands:
                        if hand:
                            landmarks = hand["lmList"]  # List of 21 Landmark points

                            for ld in landmarks:
                                # put idx number to each landmark
                                cv2.putText(imgOut, "[" + str(ld[0]) + ", " + str(ld[0]) + "]", (ld[0]+5, ld[1]+5), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)

                            if landmarks:
                                # We need to check if the landmark is out of the image
                                thumb_x = landmarks[thumb][0]
                                thumb_y = landmarks[thumb][1]
                                if thumb_x >= 0 and thumb_x < self.W and thumb_y >= 0 and thumb_y < self.H:
                                    # print(self.W, "x", self.H , " ", landmarks[thumb] , " - ", thumb_x, " ", thumb_y ,flush=True)
                                    depth = rs.depth_frame.get_distance(depth_frame, thumb_x, thumb_y)
                                    cv2.putText(imgOut, str(depth) + " m", (landmarks[thumb][0]+10, landmarks[thumb][1]+10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)

            # Detect POSE
            if self.feature_config[Feature.POSE]:
                # Find the pose and its landmarks
                imgOut = self.FindPose(imgRGB=imgRGB, imgOut=imgOut)
                
            # Detect FACE
            if self.feature_config[Feature.FACE]:
                # Find the face and its landmarks
                imgOut, faces = self.FindFace(imgRGB=imgRGB, imgOut=imgOut)


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



if __name__ == "__main__":
    capture = MotionCapture(feature_config={Feature.HAND:True, Feature.POSE:True, Feature.FACE:False})
    capture.Run()
