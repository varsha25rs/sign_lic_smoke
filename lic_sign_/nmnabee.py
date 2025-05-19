import cv2
import numpy as np

def detect_fire_smoke(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fire color range (tune as needed)
    lower_fire = np.array([0, 50, 50])
    upper_fire = np.array([35, 255, 255])
    fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)

    # Smoke detection (grayish, low saturation)
    lower_smoke = np.array([0, 0, 100])
    upper_smoke = np.array([180, 50, 200])
    smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

    # Combine masks
    combined_mask = cv2.bitwise_or(fire_mask, smoke_mask)

    # Find contours for visualization
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # filter small areas
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'Fire/Smoke', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    return frame

# Start video capture
cap = cv2.VideoCapture(0)  # Use video file path or 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = detect_fire_smoke(frame)
    cv2.imshow("Fire & Smoke Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()