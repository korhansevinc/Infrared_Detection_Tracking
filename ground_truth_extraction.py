import cv2
import numpy as np


start_x, start_y, end_x, end_y = -1, -1, -1, -1
selected = False

def draw_bbox(event, x,y, flags, param):
    global startx, start_y, end_x, end_y, selected

    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        selected = False
        print("Alindi 1 !")
        print("Start x,y : ",start_x,start_y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        end_x, end_y = x,y
        selected = True
        print("Alindi 2 !")
        print("End x,y : ",end_x,end_y)

# Loading Image
image = cv2.imread("Screenshot from test1.mp4.png")
clone = image.copy()

cv2.namedWindow("Select Bbox")
cv2.setMouseCallback("Select Bbox", draw_bbox)

while True :
    cv2.imshow("Select Bbox", image)
    height, width, _ = image.shape
    print(height,width) 
    frame_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_for_canny_without_Gaussian = frame_grayscale
    # Image Enhancement Pipeline
    ''' Contrast Enhancement'''
    # a) CLAHE
    # frame_grayscale = clahe.apply(frame_grayscale)
    ''' Weighted LoG and Normalization'''
    # b) LoG ( Laplacian of Gaussian) ( weighted LoG )
    frame_grayscale_bright_R = cv2.GaussianBlur(frame_grayscale, (5,5), 0)
    frame_grayscale_bright_R  = cv2.Laplacian(frame_grayscale_bright_R , cv2.CV_16S, ksize=5)
    frame_grayscale_dark_R = cv2.bitwise_not(frame_grayscale_bright_R)

    # k1 = 0.27
    # k2 = 0.73
    k1 = 0.74
    k2 = 0.26
    k1_times_R_dark= cv2.multiply(frame_grayscale_dark_R , np.array([k1]))
    k2_times_R_bright = cv2.multiply(frame_grayscale_bright_R , np.array([k2]))

    frame_grayscale = cv2.add(k1_times_R_dark, k2_times_R_bright)

    frame_grayscale = np.clip(frame_grayscale, 0, 255)
    min_value_of_frame = np.min(frame_grayscale)
    max_value_of_frame = np.max(frame_grayscale)
    frame_grayscale = (255 * ( (frame_grayscale - min_value_of_frame) / (max_value_of_frame - min_value_of_frame) )).astype(np.uint8)



    selected_region = frame_grayscale[175:183, 301:309]
    cv2.imwrite("ground_truth.png", selected_region)
    cv2.imwrite("Grayscale Image.png",frame_grayscale)
    break
cv2.destroyAllWindows()
