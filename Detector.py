import cv2 # Image procesing
import mediapipe as mp # Hand tracking model

MAX_HANDS = 2
MODEL_COMPLEXITY = 0
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
LMK_NAME_TO_ID = {
"WRIST": 0,
"THUMB_CMC": 1,
"THUMB_MCP": 2,
"THUMB_IP": 3,
"THUMB_TIP": 4,
"INDEX_FINGER_MCP": 5,
"INDEX_FINGER_PIP": 6,
"INDEX_FINGER_DIP": 7,
"INDEX_FINGER_TIP": 8,
"MIDDLE_FINGER_MCP": 9,
"MIDDLE_FINGER_PIP": 10,
"MIDDLE_FINGER_DIP": 11,
"MIDDLE_FINGER_TIP": 12,
"RING_FINGER_MCP": 13,
"RING_FINGER_PIP": 14,
"RING_FINGER_DIP": 15,
"RING_FINGER_TIP": 16,
"PINKY_MCP": 17,
"PINKY_PIP": 18,
"PINKY_DIP": 19,
"PINKY_TIP": 20
}

mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, MAX_HANDS, MODEL_COMPLEXITY, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE)

def detectHands(img, draw_landmarks=False):
    """Detect and optionally display hand landmarks in an image."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    img_rgb.flags.writeable = False
    results = hands.process(img_rgb) # Detect hand landmarks in image

    # If requested, draw the hand landmarks onto the image
    if draw_landmarks and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
            img, 
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(), 
            mp_styles.get_default_hand_connections_style()
            )

    return results

def getLandmarkPos(img, hand_landmarks, landmark_ame, draw_landmark=False, color = (0, 255, 0)):
    """Returns the pixel coordinates of a landmark in an image.
    Valid landmark identifiers: 
    WRIST, 
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP, 
    INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_DIP, INDEX_FINGER_TIP, 
    MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP, MIDDLE_FINGER_TIP, 
    RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_DIP, RING_FINGER_TIP, 
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, 
    """
    h, w, _ = img.shape # Get the image dimensions
    lmk = hand_landmarks.landmark[LMK_NAME_TO_ID[landmark_ame]] # Select the landmark by id
    pos = mp_draw._normalized_to_pixel_coordinates(lmk.x, lmk.y, w, h) # Obtain pixel coords of landmark

    # If requested, draw a circle at the landmark's position onto the image
    if draw_landmark and pos:
        cv2.circle(img, (pos[0], pos[1]), 9, color, cv2.FILLED)

    return pos