import cv2
import mediapipe as mp
import time
import math
import numpy as np

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.6, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

sib = cv2.imread('d1.png', cv2.IMREAD_UNCHANGED)
scale_percent = 65  # percent of original size
width = int(sib.shape[1] * scale_percent / 100)
height = int(sib.shape[0] * scale_percent / 100)
dim = (width, height)
sib = cv2.resize(sib, dim, interpolation=cv2.INTER_AREA)
BLUE = [255, 255, 255]
sib = cv2.copyMakeBorder(sib, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=BLUE)

sibW, sibH, sibC = sib.shape


def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

    return background


def angle_of_line(x1, y1, x2, y2):
    return math.degrees(math.atan2(-y2+y1, x2 - x1))


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    maxSide = int(3.4*max(image.shape[0],image.shape[1]))
    print(image.shape)
    print(maxSide)
    result = cv2.warpAffine(image, rot_mat, (maxSide,maxSide), flags=cv2.INTER_LINEAR)
    return result


while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id in [4, 8, 12, 16, 20]:
                    lm_prev = handLms.landmark[id - 1]
                    px, py = int(lm_prev.x * w), int(lm_prev.y * h)
                    angleOfRotation = angle_of_line(cx, cy, px, py)+90
                    # print(cx, cy, px, py)
                    # print(angleOfRotation)
                    rImg = rotate_image(sib, angleOfRotation)
                    tx, ty = int(cx - sibW / 2), int(cy - sibH / 2)
                    tx = max(0, tx)
                    ty = max(0, ty)
                    overlay_transparent(img, rImg, tx, ty)
                if id > 0:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
