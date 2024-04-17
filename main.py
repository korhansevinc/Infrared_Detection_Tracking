import cv2
import numpy as np
# print(f"OpenCV Version is : {cv2.__version__ }")

cap = cv2.VideoCapture("small_target_test2.mp4")
object_detector = cv2.createBackgroundSubtractorMOG2()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))

while True:
    ret, frame = cap.read()

    height, width, _ = frame.shape
    print(height,width)
    frame_vanilla = frame
    # Image Processing Pipeline for frame  to make visually better
    # a) Contrast Limiting Adaptive Histogram Equalization (CLAHE) in RGB Img
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayFrame = clahe.apply(grayFrame)
    # b, g, r = cv2.split(frame)
    # clahe_b = clahe.apply(b)
    # clahe_g = clahe.apply(g)
    # clahe_r = clahe.apply(r)

    # frame = cv2.merge([clahe_b, clahe_g, clahe_r])

    # b) Morphological Filters.

    kernel_for_morph_filter = np.ones((5,5), np.uint8)
    opened_img = cv2.morphologyEx(grayFrame, cv2.MORPH_OPEN, kernel_for_morph_filter)
    closed_img = cv2.morphologyEx(grayFrame, cv2.MORPH_CLOSE, kernel_for_morph_filter)

    # desired_k_for_loop = 3
    # for _ in range(desired_k_for_loop):
    #     opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel_for_morph_filter)
    #     closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel_for_morph_filter)

    frame = closed_img

    # c) LoG (Laplacian of Gaussian) and - LoG
    frame_grayscale = cv2.GaussianBlur(frame, (5,5), 0)
    frame_grayscale = cv2.Laplacian(frame_grayscale, cv2.CV_16S, ksize=5)
    
    frame_grayscale_bright_R = cv2.GaussianBlur(frame, (5,5), 0)
    frame_grayscale_bright_R  = cv2.Laplacian(frame_grayscale_bright_R , cv2.CV_16S, ksize=5)

    frame_grayscale_dark_R = cv2.GaussianBlur(frame, (5,5), 0)
    frame_grayscale_dark_R = cv2.Laplacian(frame_grayscale_dark_R, cv2.CV_16S, ksize=5) 
    # k1, k2 are variables which can adjust the response values of different targets.
    
    k1 = 0.6 ; k2 = 0.4
    k1_times_R_dark= cv2.multiply(frame_grayscale_dark_R , np.array([k1]))
    k2_times_R_bright = cv2.multiply(frame_grayscale_bright_R , np.array([k2]))
    
    frame_grayscale = cv2.add(k1_times_R_dark , k2_times_R_bright)
    # Normalization and converting to 0-255
    min_value_of_frame = np.min(frame_grayscale)
    max_value_of_frame = np.max(frame_grayscale)
    frame_grayscale = (255 * ( (frame_grayscale - min_value_of_frame) / (max_value_of_frame - min_value_of_frame) )).astype(np.uint8)


    roi = frame_grayscale[0:320, :]
    
    # Object Detection
    mask = object_detector.apply(frame_grayscale)
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    for contour in contours :
        area = cv2.contourArea(contour)
        if area < 40 and area > 3:
            cv2.drawContours(frame_grayscale, [contour], -1, (255), 1)


    print("ROI : ", roi)
    print(" Frame Grayscale : ", frame_grayscale)
    print(" MASK : ", mask)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame_grayscale)
    cv2.imshow("Mask", mask)
    cv2.imshow("CLAHE Img ", frame)
    cv2.imshow("Vanilla Frame" ,frame_vanilla)
    key = cv2.waitKey(300)
    print(key)
    if key == 113: # Press q to quit.
        break
cap.release()
cv2.destroyAllWindows()
