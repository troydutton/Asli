# Data manipulation
import string
import numpy as np
# Image processing
import cv2
# Hand Tracking
import handDetector
# Model
import tensorflow as tf

INTERNAL_DISPLAY = 0
EXIT_KEY = 27
BLACK_COLOR = (0, 0, 0)
ALPHABET = list(string.ascii_lowercase)



def main():
    # Initialize hand detection module
    detector = handDetector.handDetector(detection_confidence=.6, max_hands=1, tracking_confidence=.6, model_complexity=1)

    # Load model
    model = tf.keras.models.load_model("model.h5")

    # Initialize video feed
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # When we start, we have no last prediction
    last_prediction = None

    while (cap.isOpened()):
        success, img = cap.read()

        if not success:
            print("Frame not loaded")
            break

        # Get hand landmarks
        results = detector.detectHands(img, False)

        if results.multi_hand_landmarks:
            #Iterate through all detected hands
            for i, hand_lmks in enumerate(results.multi_hand_landmarks):
                # multi_handedness -> {'classification': [{'index': 1, 'score': 0.92601216, 'label': 'Right'}]}
                handedness = results.multi_handedness[i].classification[0].label

                topLeft = [99999, 99999]
                botRight = [0, 0]
                for id in detector.id2num.keys():
                    pos = detector.getLandmarkPos(img, hand_lmks, id)

                    if (pos != None):
                        if (pos[0] < topLeft[0]):
                            topLeft[0] = pos[0]
                        if (pos[1] < topLeft[1]):
                            topLeft[1] = pos[1]
                        if (pos[0] > botRight[0]):
                            botRight[0] = pos[0]
                        if (pos[1] > botRight[1]):
                            botRight[1] = pos[1]
                            
                w = botRight[0] - topLeft[0]
                h = botRight[1] - topLeft[1]

                if (w > h):
                    topLeft[1] -= int((w - h) / 2)
                    botRight[1] += int((w - h) / 2)
                elif (w < h):
                    topLeft[0] -= int((h - w) / 2)
                    botRight[0] += int((h - w) / 2)         

                w = botRight[0] - topLeft[0]
                h = botRight[1] - topLeft[1]

                topLeft[0] -= int(w * .1)
                topLeft[1] -= int(h * .1)
                botRight[0] += int(w * .1)
                botRight[1] += int(h * .1)

                w = botRight[0] - topLeft[0]
                h = botRight[1] - topLeft[1]

                hand_img = img[topLeft[1]: topLeft[1] + h, topLeft[0]:topLeft[0] + w]

                gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)

                gray = cv2.resize(gray, (28, 28))
                cv2.rectangle(img, topLeft, botRight, BLACK_COLOR, 2)

            cur_prediction = ALPHABET[model.predict(np.array(gray).reshape((-1, 28, 28, 1)).astype(np.uint8)).argmax()]

            if (last_prediction is not None):
                if (last_prediction != cur_prediction):
                    last_prediction = cur_prediction
                    print(cur_prediction)
            else:
                last_prediction = cur_prediction

            if (len(gray) > 0):
                cv2.imshow("Hand", gray)
                    
        cv2.imshow("Image", img)
        if (cv2.waitKey(1) & 0xFF == EXIT_KEY):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()