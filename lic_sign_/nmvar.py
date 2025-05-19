import cv2
import numpy as np


finger_to_letter = {
    1: 'A',
    2: 'B',
    3: 'C',
    4: 'D',
    5: 'E',
    6: 'F'
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:300, 100:300]

    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    letter = ""
    if contours:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        if len(cnt) >= 5:
            hull = cv2.convexHull(cnt, returnPoints=False)

            if hull is not None and len(hull) > 3:
                try:
                    defects = cv2.convexityDefects(cnt, hull)

                    if defects is not None:
                        finger_count = 0
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])

                            a = np.linalg.norm(np.array(end) - np.array(start))
                            b = np.linalg.norm(np.array(far) - np.array(start))
                            c = np.linalg.norm(np.array(end) - np.array(far))
                            angle = np.arccos((b * 2 + c * 2 - a ** 2) / (2 * b * c)) * 57

                            if angle <= 90:
                                finger_count += 1
                                cv2.circle(roi, far, 5, [0, 255, 0], -1)

                        
                        letter = finger_to_letter.get(finger_count + 1, "Unknown")

                        cv2.putText(frame, f'Sign: {letter}', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

                except Exception as e:
                    print("Error:", e)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()