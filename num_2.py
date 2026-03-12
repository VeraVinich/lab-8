import cv2
import numpy as np


def track_marker():
    cap = cv2.VideoCapture(0)
    frame_size = (640, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 🔴 Красный цвет (два диапазона, т.к. красный на границе HSV)
        lower_red1 = np.array([0, 150, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            r = int(r)

            cv2.circle(frame, center, r, (0, 255, 0), 2)
            cv2.circle(frame, center, 3, (0, 0, 255), -1)

        cv2.imshow('Marker Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    track_marker()