import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from vidstab import VidStab
# print(f"OpenCV Version is : {cv2.__version__ }")

def match_regions(image, T, region_size, target_size):
    height, width = image.shape

    marked_pixels = []

    for y in range(region_size//2, height - region_size//2):
        for x in range(region_size//2, width - region_size//2):
            # Background mean hesabı :
            background_region = image[y - region_size//2: y + region_size//2 + 1, x - region_size//2: x + region_size//2 + 1]
            background_mean = np.mean(background_region)
            
            # Target mean hesabı :
            target_region = background_region[target_size//2: target_size//2 + 1, target_size//2: target_size//2 + 1]
            target_mean = np.mean(target_region)

            if target_mean - background_mean > T :
                marked_pixels.append((y,x))
            else:
                if image[y,x] < 254:
                    image[y,x] = 0
    
    for pixel in marked_pixels:
        x,y = pixel
        image[x-1 : x+2 , y-1 : y+2] = 255
    
    return image

def process_region(image, y , x , T , region_size , target_size):
    # Background mean hesabı :
    background_region = image[y - region_size//2: y + region_size//2 + 1, x - region_size//2: x + region_size//2 + 1]
    background_mean = np.mean(background_region)
    
    # Target mean hesabı :
    target_region = background_region[target_size//2: target_size//2 + 1, target_size//2: target_size//2 + 1]
    target_mean = np.mean(target_region)
    if target_mean - background_mean > T :
        return y,x
    else:
        if image[y,x] < 254:
            image[y,x] = 0
        return None
    

def match_regions_faster(image, T, region_size, target_size):
    height, width = image.shape

    marked_pixels = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for y in range(region_size//2 , height - region_size//2):
            for x in range(region_size//2 , width - region_size//2):
                futures.append(executor.submit(process_region, image, y , x , T , region_size , target_size))
    
    for pixel in marked_pixels:
        y,x = pixel
        image[y-1:y+2, x-1:x+2] = 255
    
    return image


#cap = cv2.VideoCapture("test1.mp4")
cap = cv2.VideoCapture("test1.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
kernel_for_morph_filter = np.ones((1,1), np.uint8)

#stabilizer = VidStab()

while True:
    ret, frame = cap.read()

    height, width, _ = frame.shape
    print(height,width) 
    #frame = stabilizer.stabilize_frame(input_frame=frame, border_size=50)
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Image Enhancement Pipeline
    
    # a) CLAHE
    #frame_grayscale = clahe.apply(frame_grayscale)
    
    # b) LoG ( Laplacian of Gaussian) ( weighted LoG )
    frame_grayscale_bright_R = cv2.GaussianBlur(frame_grayscale, (7,7), 0)
    frame_grayscale_bright_R  = cv2.Laplacian(frame_grayscale_bright_R , cv2.CV_16S, ksize=5)
    frame_grayscale_dark_R = cv2.bitwise_not(frame_grayscale_bright_R)

    k1 = 0.77
    k2 = 0.23
    k1_times_R_dark= cv2.multiply(frame_grayscale_dark_R , np.array([k1]))
    k2_times_R_bright = cv2.multiply(frame_grayscale_bright_R , np.array([k2]))

    frame_grayscale = cv2.add(k1_times_R_dark, k2_times_R_bright)

    # Normalization and converting to 0-255

    # METHOD 1 : 
    # min_value_of_frame = np.min(frame_grayscale)
    # max_value_of_frame = np.max(frame_grayscale)
    # frame_grayscale = (255 * ( (frame_grayscale - min_value_of_frame) / (max_value_of_frame - min_value_of_frame) )).astype(np.uint8)
    
    # METHOD 2 :
    frame_grayscale = np.clip(frame_grayscale, 0, 255)
    min_value_of_frame = np.min(frame_grayscale)
    max_value_of_frame = np.max(frame_grayscale)
    frame_grayscale = (255 * ( (frame_grayscale - min_value_of_frame) / (max_value_of_frame - min_value_of_frame) )).astype(np.uint8)
    #print(frame_grayscale)
    

    # Extract Region of Interest
    #roi_rgb = frame[0:200, 0:320] # small_target_test2
    #roi_gray = frame_grayscale[0:200,0:320] # small_target_test2

    roi_rgb = frame[0:380, 0:600] # test1
    roi_gray = frame_grayscale[0:380, 0:600] # test1

    # THRESHOLD = 80
    # BACKGROUND_SIZE = 9 # 9x9
    # TARGET_SIZE = 3 # 3x3
    #roi_gray = match_regions(roi_gray,THRESHOLD,BACKGROUND_SIZE,TARGET_SIZE)

    edges = cv2.Canny(roi_gray,150,245)

    # Object Detection
    mask = object_detector.apply(roi_gray)
    # _, mask = cv2.threshold(mask,254, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours :
        area = cv2.contourArea(contour)
        print("Area", area)
        if area > 5 and area < 80 :
            x_area, y_area, w_area, h_area = cv2.boundingRect(contour)
            if w_area * h_area < 50 and w_area * h_area > 15 :
                cv2.rectangle(roi_rgb, (x_area,y_area), (x_area + w_area, y_area + h_area), (0,255,0),1)
    




    cv2.imshow("Vanilla Frame", frame)
    cv2.imshow("Grayscale Frame", frame_grayscale) 
    cv2.imshow("Mask", mask)
    cv2.imshow("Region Of Interest RGB", roi_rgb)
    cv2.imshow("Region Of Interest Gray", roi_gray)
    cv2.imshow("Canny Edges", edges)

    key = cv2.waitKey(0)
    if key == 113:
        break


cap.release()
cv2.destroyAllWindows()    