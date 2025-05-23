import cv2
from cvzone.HandTrackingModule import HandDetector

webcam = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, image = webcam.read()

    if not success:
        break

    hands, image_with_hands = detector.findHands(image)

    total_fingers = 0  

    if hands:
        for hand in hands:
            # Get raised fingers for each hand
            fingers = detector.fingersUp(hand)
            num_fingers = fingers.count(1) 
            total_fingers += num_fingers  

        cv2.putText(image_with_hands, f'Fingers raised: {total_fingers}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    image_with_hands = cv2.resize(image_with_hands, (800, 600))

    # Show the image with detected hands
    cv2.imshow("Hand Tracker", image_with_hands)

    if cv2.waitKey(1) != -1:
        break

webcam.release()
cv2.destroyAllWindows()
