import cv2
import numpy as np

cap = cv2.VideoCapture("test.mp4")
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True )
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
kernel_for_morph_filter = np.ones((3,3) , np.uint8 ) * 255

while True :
    ret, frame = cap.read()

    height, width, _ = frame.shape
    print(height,width)
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Image Processing Pipeline for frame to make visually better.

    # a ) CLAHE
    frame_grayscale = clahe.apply(frame_grayscale)

    # b ) LoG ( Laplacian of Gaussian, weighted LoG )
    frame_grayscale_bright_R = cv2.GaussianBlur(frame_grayscale, (5,5) , 0 )
    frame_grayscale_bright_R = cv2.Laplacian(frame_grayscale_bright_R, cv2.CV_16S , ksize=5)
    frame_grayscale_dark_R = cv2.bitwise_not(frame_grayscale_bright_R)

    k1 = 0.87 ; k2 = 0.13
    k1_times_R_dark = cv2.multiply(frame_grayscale_dark_R, np.array([k1]))
    k2_times_R_bright = cv2.multiply(frame_grayscale_bright_R, np.array([k2]))
    frame_grayscale = cv2.add(k1_times_R_dark, k2_times_R_bright)

    # Normalization and converting the frame to 0 - 255
    #frame_grayscale = np.clip(frame_grayscale, 0, 255)
    min_value_of_frame = np.min(frame_grayscale)
    max_value_of_frame = np.max(frame_grayscale)
    frame_grayscale = (255 * ( (frame_grayscale - min_value_of_frame) / ( max_value_of_frame - min_value_of_frame) )).astype(np.uint8)

    mask = object_detector.apply(frame_grayscale)
    print("Mask is : ", mask)

    cv2.imshow("Grayscale Frame", frame_grayscale)
    cv2.imshow("Vanilla Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 113 : # if key equals 'q' char 
        break

cap.release()
cv2.destroyAllWindows()