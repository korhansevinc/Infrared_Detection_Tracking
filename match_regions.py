import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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



