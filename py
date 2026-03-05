import cv2
import mediapipe as mp
import numpy as np
import math

# -----------------------
# Setup
# -----------------------
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Cannot open camera")
    exit()

frame = cv2.flip(frame, 1)
h, w, _ = frame.shape

drawing = np.zeros((h, w, 3), dtype=np.uint8)  # canvas

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# -----------------------
# State
# -----------------------
prev_pos = None
color = (255, 0, 0)  # initial blue
brush_size = 8
drawing_enabled = False  # only draw when pinched

# -----------------------
# Buttons
# -----------------------
color_buttons = [
    (10, 10, 70, 70, (255, 0, 0)),
    (80, 10, 140, 70, (0, 255, 0)),
    (150, 10, 210, 70, (0, 0, 255)),
    (220, 10, 280, 70, (0, 255, 255))
]

eraser_button = (290, 10, 350, 70, (0, 0, 0))
clear_button = (360, 10, 430, 70, (255, 255, 255))
brush_buttons = [
    (440, 10, 470, 40, 4),
    (480, 10, 510, 40, 8),
    (520, 10, 550, 40, 16)
]

# -----------------------
# Helpers
# -----------------------
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def check_hover(finger_pos):
    global color, brush_size, drawing
    x, y = finger_pos

    # Colors
    for x1, y1, x2, y2, c in color_buttons:
        if x1 < x < x2 and y1 < y < y2:
            color = c

    # Eraser
    x1, y1, x2, y2, c = eraser_button
    if x1 < x < x2 and y1 < y < y2:
        color = (0, 0, 0)

    # Clear
    x1, y1, x2, y2, c = clear_button
    if x1 < x < x2 and y1 < y < y2:
        drawing[:] = 0

    # Brush sizes
    for x1, y1, x2, y2, size in brush_buttons:
        if x1 < x < x2 and y1 < y < y2:
            brush_size = size

# -----------------------
# Main loop
# -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_pos = None
    drawing_enabled = False

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Index tip
        ix, iy = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)
        finger_pos = (ix, iy)

        # Thumb tip
        tx, ty = int(hand.landmark[4].x * w), int(hand.landmark[4].y * h)
        thumb_pos = (tx, ty)

        # Pinch detection
        if distance(finger_pos, thumb_pos) < 40:
            drawing_enabled = True
        else:
            prev_pos = None

        # Check hover for buttons
        check_hover(finger_pos)

        # Draw
        if drawing_enabled and prev_pos is not None:
            cv2.line(drawing, prev_pos, finger_pos, color, brush_size)
        if drawing_enabled:
            prev_pos = finger_pos

    else:
        prev_pos = None

    # -----------------------
    # Draw UI buttons
    # -----------------------
    for x1, y1, x2, y2, c in color_buttons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Eraser
    x1, y1, x2, y2, c = eraser_button
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.putText(frame, 'E', (x1+15, y1+45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Clear
    x1, y1, x2, y2, c = clear_button
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.putText(frame, 'C', (x1+15, y1+45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Brush sizes
    for x1, y1, x2, y2, size in brush_buttons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200,200,200), -1)
        cv2.putText(frame, str(size), (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)

    # -----------------------
    # Overlay drawing
    # -----------------------
    overlay = cv2.addWeighted(frame, 0.5, drawing, 0.5, 0)
    cv2.imshow("Finger Paint AR", overlay)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
