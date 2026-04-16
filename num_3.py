import cv2
import numpy as np
import math


def track_marker_with_area():
    cap = cv2.VideoCapture(0)
    frame_size = (640, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 150, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            c = max(contours, key=cv2.contourArea)
            moments = cv2.moments(c)

            if moments['m00'] != 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
                center = (center_x, center_y)

                cv2.circle(frame, center, 5, (255, 0, 0), -1)

                frame_height, frame_width = frame.shape[:2]
                frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
                distance = math.hypot(center_x - frame_center_x, center_y - frame_center_y)

                cv2.putText(
                    frame,
                    f'{int(distance)} px',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow('Marker Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    track_marker_with_area()
