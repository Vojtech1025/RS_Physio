import cv2
from cvzone.PoseModule import PoseDetector
from cvzone.HandTrackingModule import HandDetector
import pyrealsense2 as rs
import numpy as np
import mediapipe as mp

class Detection:
    def __init__(self):
        self.W = 640
        self.H = 480
        self.FPS = 30

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, self.FPS)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.FPS)

        print("[INFO] Starting streaming...", flush=True)
        self.pipeline.start(self.config)
        print("[INFO] Camera ready.", flush=True)

        # the output will be written to output.avi
        self.out = cv2.VideoWriter(
            'output/output.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            15.,
            (640,480))

    def PoseDetection(self):
        detector = PoseDetector()
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # detect people in the image
            img = detector.findPose(img=color_image)
            lmList, bboxInfo = detector.findPosition(img)

            if bboxInfo: # if there is a person
                center = bboxInfo["center"]
                cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

            # Write the output video 
            self.out.write(img.astype('uint8'))
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cv2.destroyAllWindows()
        self.pipeline.stop()

    def HandDetection(self):
        detector = HandDetector(detectionCon=0.8, maxHands=6)
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Find the hand and its landmarks
            hands, hand_img = detector.findHands(img=color_image)  # with draw
            # hands = detector.findHands(img, draw=False)  # without draw

            if hands:
                thumb = 4 # index of thumb in lmList
                for hand in hands:
                    if hand:
                        landmarks = hand["lmList"]  # List of 21 Landmark points

                        for ld in landmarks:
                            # put idx number to each landmark
                            cv2.putText(hand_img, "[" + str(ld[0]) + ", " + str(ld[0]) + "]", (ld[0]+5, ld[1]+5), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)

                        if landmarks:
                            # We need to check if the landmark is out of the image
                            thumb_x = landmarks[thumb][0]
                            thumb_y = landmarks[thumb][1]
                            if thumb_x >= 0 and thumb_x < self.W and thumb_y >= 0 and thumb_y < self.H:
                                # print(self.W, "x", self.H , " ", landmarks[thumb] , " - ", thumb_x, " ", thumb_y ,flush=True)
                                depth = rs.depth_frame.get_distance(depth_frame, thumb_x, thumb_y)
                                cv2.putText(hand_img, str(depth) + " m", (landmarks[thumb][0]+10, landmarks[thumb][1]+10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)


            # Draw supporting points into corners of the image
            cv2.putText(hand_img, "[20, 60]", (20, 60), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)
            cv2.putText(hand_img, "[20, 20]", (20, 20), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)

            # Write the output video 
            self.out.write(hand_img.astype('uint8'))

            # Display
            cv2.imshow("Image", hand_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.pipeline.stop()

    def PoseHandDetection(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                # Convert images to numpy arrays
                image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # Make Detections
                results = holistic.process(image)

                # 1. Draw face landmarks
                # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                #                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                #                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                #                         )
                
                # 2. Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                        )

                # 3. Left Hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        )

                # 4. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )


                # Write the output video 
                self.out.write(image.astype('uint8'))
                cv2.imshow("Image", image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # When everything done, release the capture
            cv2.destroyAllWindows()
            self.pipeline.stop()



if __name__ == "__main__":
    detection = Detection()
    # detection.HandDetection()
    # detection.PoseDetection()
    detection.PoseHandDetection()
