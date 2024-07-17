import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from skimage.metrics import structural_similarity as compare_ssim

# To print out key values in calculations : DEBUG
DEBUG = 0
# To enable tracking pipeline : TRACKING_PIPELINE_TOGGLE_SWITCH
TRACKING_PIPELINE_TOGGLE_SWITCH = 0

# Güncel kodda kullanılmayan fonksiyonlar var. bunlar, pipeline değiştiğinde denemek amacıyla kullanılabileceğinden kaldırılmadı.
# 

class DetectedObject:
    # bbox : x,y -> sol üst nokta , w, h
    def __init__(self, id, bbox, frame, currentTile):
        self.id = id
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        self.bbox = bbox 
        self.last_bbox = bbox 
        self.speed = 0
        self.total_dist = 0
        self.current_Tile = currentTile
    
    def update(self, frame, new_bbox):

        similarity = self.calculate_similarity(new_bbox)
        if similarity < 5 : 

            success, bbox = self.tracker.update(frame)
            if success :
                # New formula for velocity calculation.
                x1,y1,w1,h1 = self.last_bbox
                last_center_x = int(x1 + w1 / 2) 
                last_center_y = int(y1 + h1 / 2)

                new_center_x = int(bbox[0] + bbox[2] / 2)
                new_center_y = int(bbox[1] + bbox[3] / 2)

                speed_x = new_center_x - last_center_x
                speed_y = new_center_y - last_center_y

                dist = np.sqrt(speed_x ** 2 + speed_y ** 2)
                self.speed = dist
                self.total_dist += dist
                self.last_bbox = self.bbox
                self.bbox = bbox
            return success, bbox, self.speed
        else:
            return True, self.last_bbox, self.speed

    # Deprecieted.
    def calculate_speed(self):
        x1,y1,w1,h1 = self.last_bbox
        last_center = np.array([x1,y1])
        x2,y2,w2,h2 = self.bbox
        center = np.array([x2,y2])
    
        distance = np.linalg.norm(center - last_center)
        if DEBUG :
            print("Current center is : ", center)
            print("Last center is  :", center)
            print("Updating the distance : ", distance)
        # Distance : Belirli frame aralığındaki pixel-wise değişim.
        self.speed = distance 

    def update_bbox(self, bbox):
        self.last_bbox = self.bbox
        self.bbox = bbox

    def calculate_similarity(self, new_bbox):
        x1,y1,w1,h1 = self.last_bbox
        x2,y2,w2,h2 = new_bbox
        return np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 + (np.sqrt(w1) - np.sqrt(w2) )**2 + ( np.sqrt(h1) - np.sqrt(h2) )**2 )
    


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


# TILE COUNT değiştiğinde bu fonksiyon geçersiz kalacaktır.
def initialize_weights_for_tiles():
    # Degrees are from 0 - 100
    static_weights = [6, 9, 9, 6, 6, 9, 9, 6, 6, 6, 6, 6, 0.5, 1, 1, 0.5]
    return static_weights


def create_tiles(image):
    height, width = image.shape[:2]
    TILE_COUNT =16

    tile_width = width // 4
    tile_height = height // 4

    tiles = []

    tile_indices = [ (i * tile_height,  (i+1) * tile_height, j * tile_width, (j+1) * tile_width) for i in range(4) for j in range(4)]

    for idx, (start_y, end_y, start_x, end_x) in enumerate(tile_indices):
        tile = image[start_y : end_y , start_x : end_x]
        tiles.append(tile)

    return tiles


def show_tiles(tiles):
    for i , tile in enumerate(tiles):
        cv2.imshow('Tile {}'.format(i+1), tile)


def key_point_extraction_and_elimination(frame_grayscale, tileNumber):
        orb = cv2.ORB.create()
        # block_size = 3
        # k_size = 5
        # harris_k = 0.04
        # harris_corner_score = cv2.cornerHarris(frame_grayscale, block_size, k_size, harris_k)
        ''' Keypoint detection and Elimination'''
        # ORB
       #print("Type of frame is : ", type(frame_grayscale))
        keypoints, descriptors = orb.detectAndCompute(frame_grayscale, None)
        #print("Keypoints for tile : ", keypoints)
        # # 1 ) Eliminating (Thresholding) for Harris Corner Score
        
        # threshold_score_harris_corner = 0.12 * harris_corner_score.max()
        # selected_keypoints_after_harris_elimination = [kp for kp in keypoints if harris_corner_score[int(kp.pt[1]) , int(kp.pt[0])] > threshold_score_harris_corner]
        
        # 2 ) Eliminating (Thresholding) for Response Scores

        responses_of_kp = [kp.response for kp in keypoints]
        
        threshold_kp_response_high_boundry = 0.0223
        threshold_kp_response_low_boundry = 0.008
        selected_keypoints_after_response_elimination = [kp for kp,response in zip(keypoints,responses_of_kp) if response < threshold_kp_response_high_boundry] 
        selected_keypoints_after_response_elimination = [kp for kp,response in zip(selected_keypoints_after_response_elimination,responses_of_kp) if response > threshold_kp_response_low_boundry ] 


        # 3 ) Eliminating (Thresholding) for Keypoint Size
        
        sizes_of_kp_radius = [kp.size for kp in selected_keypoints_after_response_elimination]
        threshold_size_kp_radius = 32
        selected_keypoints_after_radius_thresholding = [kp for kp, size in zip(selected_keypoints_after_response_elimination, sizes_of_kp_radius) if size < threshold_size_kp_radius]
        
        
        # Eliminating Through Keypoints Count In a Tile
        keypoint_coordinates = [ (kp,int(kp.pt[0]), int(kp.pt[1]), tileNumber) for kp in selected_keypoints_after_radius_thresholding]

    

        return keypoint_coordinates, (len(keypoint_coordinates))


def calculate_dynamic_weights(keypoint_count, total_keypoint_count):
    if total_keypoint_count != 0:
        return  (1 - (keypoint_count / total_keypoint_count))
    return 1


def calculate_tiles_weight(Wt): # e^t_i / sum(e^ts)
    exponential_weights = np.exp(Wt)
    exponential_weights_sum = np.sum(exponential_weights)
    normalized_weights = exponential_weights / exponential_weights_sum
    return normalized_weights


def count_pixels_with_value(image, value):
    mask = np.where(image == value , 1 , 0)
    count = np.sum(mask)
    return count


def count_ratio(count, w_area, h_area):
    return count / (w_area * h_area)


def check_pixel_range(image, lower_bound, upper_bound, area, THRESHOLD):

    mask = np.logical_and(image >= lower_bound, image <= upper_bound)
    count = np.sum(mask)
    if count / area > THRESHOLD:
        return True
    return False


def non_maxima_suppression(boxes, OVERLAP_THRESHOLD):
    if len(boxes) == 0:
        return []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # For the biggest bboxs : best bboxes.
    pick = []

    while len(idxs) > 0 :
        # Pick the biggest scored box and append to pick list.
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Finding the occlusion (overlapping parts.)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.maximum(x2[i], x2[idxs[:last]])
        yy2 = np.maximum(y2[i], y2[idxs[:last]])

        # Computing the area of occlusion:
        w = np.maximum(0, xx2-xx1 + 1)
        h = np.maximum(0, yy2-yy1 + 1)

        # Computing the ratio of overlapping:
        overlap = (w * h) / areas[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap> OVERLAP_THRESHOLD)[0])))
    
    return pick


def pixel_wise_filter(tile_img,pixels_255_coords):
    
    filtered_coords = []
    for x,y in pixels_255_coords:
        patch = tile_img[y-1:y+2, x-1:x+2]
        count_of_255 = np.count_nonzero(patch == 255)
        if count_of_255 > 4:
            filtered_coords.append((x,y))   


    filtered_coords_np = np.array([(x,y,x,y) for x,y in filtered_coords])
    pick = non_maxima_suppression(filtered_coords_np, 0.30)
    filtered_coordinates = [filtered_coords[i] for i in pick]
    
    return filtered_coordinates


def key_pixel_filter(image, pixels_255_coords, THRESHOLD_FOR_KEY_PIXEL_FILTER = 75):
    filtered_coordinates = []

    for x,y in pixels_255_coords:
        target_area = image[y-2:y+3, x-2:x+3]
        target_area_mean = np.mean(target_area)

        background_area = image[y-6:y+7, x-6:x+7]
        background_area_mean = np.mean(background_area)

        difference = abs(target_area_mean - background_area_mean)
        if difference > THRESHOLD_FOR_KEY_PIXEL_FILTER:
            filtered_coordinates.append((x,y))
    
    if len(filtered_coordinates)< 1:
        return []
    
    filtered_coordinates_np = np.array([(x,y,x,y) for x, y in filtered_coordinates])
    pick = non_maxima_suppression(filtered_coordinates_np, 0.32)
    filtered_coordinates = [filtered_coordinates[i] for i in pick]

    return filtered_coordinates


def pixel_wise_patch_filter(image):
    filtered_coordinates = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y,x] == 255:
                if(y-1 >=0 and y+1 < image.shape[0] and x-1 >=0 and x+1 < image.shape[1]):
                    patch = image[y-1:y+2, x-1: x+2]
                    if np.all(patch == 255):
                        filtered_coordinates.append((x,y))
    filtered_coordinates_np = np.array([(x,y,x,y) for x, y in filtered_coordinates])
    pick = non_maxima_suppression(filtered_coordinates_np, 0.32)
    filtered_coordinates = [filtered_coordinates[i] for i in pick]

    return filtered_coordinates


def find_connected_regions(image, THRESHOLD_AREA=9):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    connectivity = 8
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    labels = output[1]

    centroid_coordinates = []
    # for i in range(1, len(stats)):
    #     if stats[i, cv2.CC_STAT_AREA] > THRESHOLD_AREA :   
    #         center = (int(centroids[i][0]), int(centroids[i][1]))
    #         centroid_coordinates.append(center)

    for label in range(1, np.max(labels) + 1):
        if np.any(image[labels == label] == 255):
            indices = np.argwhere(labels == label)
            center = np.mean(indices,axis=0, dtype=np.int32)
            centroid_coordinates.append(center[::-1])

    return centroid_coordinates


def distance_of_boxes(box1,box2):
    center1 = (box1[0], box1[1])
    center2 = (box2[0], box2[1])
    result = np.sqrt(  ((center1[0] - center2[0])**2) + ((center1[1] - center2[1])**2))
    return result


def is_close_enough(box1, box2, THRESHOLD_IS_CLOSE= 5):
    return distance_of_boxes(box1, box2) < THRESHOLD_IS_CLOSE


def mean_box_fusion(boxes):
    threshold = 4
    if len(boxes) <=1:
        return boxes
    min_distance = float('inf')
    merge_pair = None 

    for i in range(len(boxes)):
        for j in range(i+1 , len(boxes)):
            dist = distance_of_boxes(boxes[i], boxes[j])
            if dist < min_distance and dist < threshold :
                min_distance = dist 
                merge_pair = (i,j)
    
    if merge_pair is None:
        return None
    
    i,j = merge_pair
    merged_x = int((boxes[i][0] + boxes[j][0]) / 2)
    merged_y = int((boxes[i][1] + boxes[j][1]) / 2)
    merged_width = boxes[i][2] if boxes[i][2] > boxes[j][2] else boxes[j][2]
    merged_height = boxes[i][3] if boxes[i][3] > boxes[j][3] else boxes[j][3]
    merged_tile_num = boxes[i][4]

    del boxes[j]
    del boxes[i]
    boxes.append( (merged_x, merged_y, merged_width, merged_height,merged_tile_num))

    return boxes

# New Merge Function
def merge_all(boxes):
    i = 0
    while True :
        merged_boxes = mean_box_fusion(boxes)
        if merged_boxes is None:
            break
        boxes = merged_boxes
        i+=1
        if i > 50 :
            break
    return boxes
    

# It merge boxes.
# Depreciated. Use mean_box_fusion and merge_all instead.
# Eski merge fonksiyonu.
def merge_boxes(boxes):
    merged_boxes = []
    merged_indices = set()
    for i in range(boxes):
        if i not in merged_indices:
            merged_indices.add(i)
            merged=False
            for j in range(i+1, len(boxes)):
                if j not in merged_indices and is_close_enough(boxes[i], boxes[j], 5):
                    merged_indices.add(j)
                    merged=True
                    center_x = ( boxes[i][0] + boxes[j][0] ) / 2
                    center_y = ( boxes[i][1] + boxes[j][1] ) / 2
                    merged_boxes.append( (center_x, center_y))
            if not merged:
                merged_boxes.append((boxes[i][0], boxes[i][1]))
    return merged_boxes


def filter_by_keyfeatures_of_plane(image):
    KERNEL_TO_FIND = np.array([[0,1,0], [1,1,1], [0,1,0]] , dtype=np.uint8)
    kernel_height, kernel_width = KERNEL_TO_FIND.shape
    image_height, image_width = image.shape

    matching_pixels = []

    for y in range(image_height - kernel_height + 1):
        for x in range(image_width - kernel_width + 1):
            patch = image[y:y + kernel_height , x: x + kernel_width]
            if np.array_equal(patch * KERNEL_TO_FIND, KERNEL_TO_FIND):
                matching_pixels.append((x + kernel_width // 2, y + kernel_height // 2 ))
    

    filtered_pixels = []
    processed_indices = set()

    for i in range(len(matching_pixels)):
        if i not in processed_indices :
            x1,y1 = matching_pixels[i]
            for j in range(i+1, len(matching_pixels)):
                if j not in processed_indices :
                    x2,y2 = matching_pixels[j]
                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if distance < 8 :
                        filtered_pixels.append((x1 + x2) // 2, (y1 + y2) // 2)
                        processed_indices.add(i)
                        processed_indices.add(j)
    
    matching_pixels = [matching_pixels[i] for i in range(len(matching_pixels)) if i not in processed_indices]
    return matching_pixels + filtered_pixels


def track_objects(frame, obj, tracker):
    ok, bbox = tracker.update(frame)
    if ok:
        obj.update_bbox(bbox)
        obj.calculate_speed()
        x,y,w,h = bbox
        p1 = (int(x - w//2) , int(y - h//2))
        p2 = (int(x + w//2),  int(y + h//2))
      #  cv2.rectangle(frame, p1,p2 , (0, 255, 0), 1)
        cv2.putText(frame, f"ID: {obj.id}", (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Speed: {obj.speed:.2f}", (p1[0], p1[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


def calculate_iou(box1, box2):
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2

    x1_min, y1_min, x1_max, y1_max = x1 - w1 // 2, y1 - h1 // 2, x1 + w1 // 2, y1 + h1 // 2
    x2_min, y2_min, x2_max, y2_max = x2 - w2 // 2, y2 - h2 // 2, x2 + w2 // 2, y2 + h2 // 2

    intersection_xmin = max(x1_min, x2_min)
    intersection_ymin = max(y1_min, y2_min)
    intersection_xmax = min(x1_max, x2_max)
    intersection_ymax = min(y1_max, y2_max)

    if intersection_xmax <= intersection_xmin or intersection_ymax <= intersection_ymin:
        return 0.0
    
    #intersection
    intersection_area = (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)

    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - intersection_area 
    iou = intersection_area / union_area
    return iou 


def check_changes(prev_objects, curr_objects):
    anchor_objects = []
    for prev_obj in prev_objects:
        for curr_obj in curr_objects:
            if prev_obj.id == curr_obj.id:
                anchor_objects.append(curr_obj)
                break

    return anchor_objects

def calculate_distance(obj1, obj2):
    x1,y1,w1,h1 = obj1.bbox
    x2,y2,w2,h2 = obj2.bbox

    p1_x, p1_y = int(x1 - w1//2) , int(y1 - h1//2)
    p2_x, p2_y = int(x2 + w2//2),  int(y2 + h2//2)
    
    return  np.sqrt( (p1_x - p2_x)**2 + (p1_y - p2_y)**2)


def calculate_total_distances(objects):
    total_distances = {}
    for obj1 in objects:
        total_distance = 0
        for obj2 in objects:
            if obj1.id != obj2.id :
                dist = calculate_distance(obj1,obj2)
                total_distance += dist
        total_distances[obj1.id] = total_distance
    
    return total_distances


def find_matching_objects(anchor_objects, prev_objects,curr_objects):
    anchor_matching_prev_frame = [prev_obj for prev_obj in prev_objects if prev_obj.id in [anchor_obj.id for anchor_obj in anchor_objects]]
    anchor_matching_curr_frame = [curr_obj for curr_obj in curr_objects if curr_obj.id in [anchor_obj.id for anchor_obj in anchor_objects]]

    total_distances_prev = {}
    total_distances_next = {}

    for anchor_obj in anchor_objects:
        total_distance_prev = 0
        total_distance_next = 0

        for obj in anchor_matching_prev_frame:
            if anchor_obj.id != obj.id :
                total_distance_prev += calculate_distance(anchor_obj, obj)
        
        total_distances_prev[anchor_obj.id] = total_distance_prev

        for obj in anchor_matching_curr_frame:
            if anchor_obj.id != obj.id :
                total_distance_next += calculate_distance(anchor_obj, obj)

        total_distances_next[anchor_obj.id] = total_distance_next

    return anchor_matching_prev_frame, anchor_matching_curr_frame, total_distances_prev, total_distances_next


def calculate_keypoint_similarity(image1, image2):
    if len(image2.shape) <= 1 or len(image1.shape) <= 1 :
        return 0

    if len(image2.shape)==2 and (image2.shape[0]== 1 or image2.shape[1] == 1):
        return 0


    x,y = image1.shape
    x2,y2 = image2.shape
    if x <1 and y < 1 and y2< 1 and x2 < 1:
        return 0
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    
    # brute force matcher :
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)
    # distances = [match.distance for match in matches]
    # similarity = np.median(distances) if distances else 0
    
    return len(matches)


def normalize_score(score):
    min_val = np.min(score)
    max_val = np.max(score)
    normalize_score = (score - min_val) / (max_val - min_val)
    return normalize_score


def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr



def calculate_similarities(image1, image2):
    if image2.shape[0] == 8 and image2.shape[1] == 8 :
        mse_score = np.mean((image1 - image2) **2)
        ssim = compare_ssim(image1,image2)
        
        cross_corr = np.mean(np.correlate(image1.flatten(), image2.flatten()))

        # hist1 = cv2.calcHist([image1],[0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        # hist2 = cv2.calcHist([image2],[0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        # hist_comp = cv2.compareHist(hist1,hist2,cv2.HISTCMP_CHISQR)

        # Normalize scores : 
        # mse_norm = np.clip(0,1)
        # ssim_norm = normalize_score(ssim)
        # cross_corr_norm = normalize_score(cross_corr)
    # hist_comp_norm = normalize_score(hist_comp)

        return  mse_score, ssim, cross_corr
    else :
        return 0,0,0


def calculate_weighted_score(normalized_mses, normalized_ssims, normalized_ccorrs):
    scores = []
    ALPHA = 0.3
    BETA = 0.6
    GAMA = 0.1
    for i in range(len(normalized_mses)):
        result =  BETA * normalized_ssims[i] - ALPHA * normalized_mses[i] +  GAMA * normalized_ccorrs[i]     
        scores.append(result)
    
    return scores



def main():
    ground_truth_image = cv2.imread("ground_truth.png")
    ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)
    cap = cv2.VideoCapture("test1.mp4")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #kernel_for_morph_filter = np.ones((5,5), np.uint8)
    orb = cv2.ORB.create()
    TILE_COUNT = 4 # 4x4 = 16 için 4 verildi, buradan tile sayisi oynanabilir. 


    obj_id = 0
    detected_objects = []

    prev_frame_tracked_objects = []
    total_distances_curr = None
    total_distances_prev = None


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame is None:
            continue
        bboxes = []
        

        # 1 - Object Detection

        height, width, _ = frame.shape
        if DEBUG :
            print(height,width) 

        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Image Enhancement Pipeline
        ''' Contrast Enhancement'''
        # a) CLAHE
        #frame_grayscale_another = clahe.apply(frame_grayscale)
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
        ''' Canny Edge Detection'''
        # parameters for Canny Edge Detector
        # low_T = 150
        # high_T = 245
        
        # edges_canny = cv2.Canny(frame_grayscale, low_T, high_T)
        # mask = object_detector.apply(frame_grayscale)
        # contours_of_canny,_ = cv2.findContours(edges_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        

        # # # False Positive artışı ve Bulut içinde detection :
        # for contour in contours_of_canny :
        #     area = cv2.contourArea(contour)
        #     if area > 5 and area < 80: 
        #         x_area ,y_area ,w_area ,h_area = cv2.boundingRect(contour)
        #         if w_area * h_area < 80 and w_area * h_area > 15 :
        #             cv2.rectangle(frame_grayscale, (x_area, y_area), (x_area + w_area, y_area + h_area), (255,255,255), 1)
        #             center_x = int(x_area + w_area // 2)
        #             center_y  = int(y_area + h_area // 2)
        #             bboxes.append((center_x, center_y, w_area, h_area, 44))

        ''' Keypoint detection and Elimination'''
     
        # 16 TILES ALGORTIHM:
     
        # ORB

        keypoints, descriptors = orb.detectAndCompute(frame_grayscale, None)
        if DEBUG :
            print("Descriptors : ", descriptors)
        
        # OPTIONAL PART:
        # print("Type of GREAT frame is : ", type(frame_grayscale))
        # # 1 ) Eliminating (Thresholding) for Harris Corner Score
        
        # threshold_score_harris_corner = 0.12 * harris_corner_score.max()
        # selected_keypoints_after_harris_elimination = [kp for kp in keypoints if harris_corner_score[int(kp.pt[1]) , int(kp.pt[0])] > threshold_score_harris_corner]
        
        # # 2 ) Eliminating (Thresholding) for Response Scores

        # responses_of_kp = [kp.response for kp in selected_keypoints_after_harris_elimination]
        
        # threshold_kp_response_high_boundry = 0.0223
        # threshold_kp_response_low_boundry = 0.008
        # selected_keypoints_after_response_elimination = [kp for kp,response in zip(selected_keypoints_after_harris_elimination,responses_of_kp) if response < threshold_kp_response_high_boundry] 
        # selected_keypoints_after_response_elimination = [kp for kp,response in zip(selected_keypoints_after_response_elimination,responses_of_kp) if response > threshold_kp_response_low_boundry ] 


        # # 3 ) Eliminating (Thresholding) for Keypoint Size
        
        # sizes_of_kp_radius = [kp.size for kp in selected_keypoints_after_response_elimination]
        # threshold_size_kp_radius = 32
        # selected_keypoints_after_radius_thresholding = [kp for kp, size in zip(selected_keypoints_after_response_elimination, sizes_of_kp_radius) if size < threshold_size_kp_radius]
        
        
        # Eliminating Through Keypoints Count In a Tile
        keypoint_coordinates = [ (kp,int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

        image_height, image_width = frame_grayscale.shape
        tile_size_x = image_width // TILE_COUNT
        tile_size_y = image_height // TILE_COUNT
        tile_keypoint_counts = [0] * (TILE_COUNT * TILE_COUNT)

        # To Print out the tiles 
        # img = np.zeros( (image_height, image_width, 3) , dtype=np.uint8)
        # for i in range(TILE_COUNT):
        #     for j in range(TILE_COUNT):
        #         x_start = int(i * tile_size_x)
        #         y_start = int(j * tile_size_y)
        #         x_end = int((i + 1) * tile_size_x)
        #         y_end = int((i + 1) * tile_size_y)
        #         cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0,0,255), 1)

      #  cv2.imshow("Yeni Resim", img)


        for kp,x,y in keypoint_coordinates:
            tile_x = min(x // tile_size_x, TILE_COUNT -1)
            tile_y = min(y // tile_size_y, TILE_COUNT - 1)
            tile_index = tile_y * TILE_COUNT + tile_x
            tile_keypoint_counts[tile_index] += 1
        total_keypoint_count = sum(tile_keypoint_counts)

        if DEBUG :
            print("Total keypoint count is : ", total_keypoint_count)

        if DEBUG :
            for i , count in enumerate(tile_keypoint_counts):
                print(f"Bölge {i}: {count} keypoints içerir.")

        static_weights = initialize_weights_for_tiles()
        dynamic_weights = [calculate_dynamic_weights(count, total_keypoint_count) for count in tile_keypoint_counts]
        if DEBUG :
            print("Dynamic Weight'ler : ", dynamic_weights)

        static_weights_np = np.array(static_weights)
        dynamic_weights_np = np.array(dynamic_weights)
        Wt = static_weights_np * dynamic_weights_np # Wt = Ws x Wd
        if DEBUG :
            print("Wt : ",Wt)

        weights_of_tiles = calculate_tiles_weight(Wt) * 3 # 4 is for 4 x 4 tiles to normalize it to 0 - 1  from 0 - 0.25.
        if DEBUG :
            print("Weights of Tiles : ")
            for num in weights_of_tiles:
                print("{:.10f}".format(num))


        # Low and High Thresholding the Weights of Tiles.
        THRESHOLD_LOW_WEIGHT = 0.01
        THRESHOLD_HIGH_WEIGHT = 0.57
        
        # Yüzey veya Arkaplan olarak saptanmış Tile'lardır. Uçak vb. cisimlerin olma olasılığı en düşük olan Tile'lardır.
        background_detected_tiles = []
        # Bir uçak, uçak benzeri, kuş veya diğer türlü saptanabilecek havada duran/ yol alan cisimlerin tespit edilmesinin
        # yüksek olasılıklı olduğu Tile'lar.
        possible_detection_tiles = []
        # Bulut içerebilen, kirlilik içerebilen tile'lardır. Ekstra işlem gerektirir.
        grey_detection_tiles = []

        for i, weight in enumerate(weights_of_tiles) :   
            if weight <  THRESHOLD_LOW_WEIGHT :
                background_detected_tiles.append(i)
            elif weight > THRESHOLD_LOW_WEIGHT and weight < THRESHOLD_HIGH_WEIGHT :
                grey_detection_tiles.append(i)
            elif weight > THRESHOLD_HIGH_WEIGHT :
                possible_detection_tiles.append(i)
            else :
                if DEBUG :
                    print("Error occured for the current tile. Going for the next one.")
        if DEBUG :
            print("Background tiles : ", background_detected_tiles)
            print("Possible Detection Tiles : ", possible_detection_tiles)
            print("Grey Detection Tiles : ", grey_detection_tiles)
        
       
       
        # Background subtraction and object detection for possible tiles :
        p = 0
        PATCH_THRESHOLD = 70
        GREATER_PATCH_THRESHOLD = 10

        DETECTED_ON_POSSIBLE_TILES = 0

        tile_width = width // 4
        tile_height = height // 4
        for tile_num in possible_detection_tiles:
            row = tile_num // 4
            col = tile_num % 4
            tile_x_start = col * tile_width
            tile_x_end = (col + 1) * tile_width
            tile_y_start = row * tile_height
            tile_y_end = (row + 1) * tile_height

            tile_img = frame_grayscale[tile_y_start:tile_y_end, tile_x_start: tile_x_end]
            contours, _ = cv2.findContours(tile_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            

            contour_centers = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"]) 
                else:
                    cX, cY = 0,0
                contour_centers.append((cX, cY, contour))


            detected = 0
            for cX,cY,contour in contour_centers:

                if cX != 0 and cY != 0:
                    if tile_img[cY, cX] > 200 :
                        patch = tile_img[cY - 1 : cY + 1,  cX-1: cX+1]
                        patch_mean = np.mean(patch)

                        if patch_mean > PATCH_THRESHOLD and np.count_nonzero(patch == 255) >= 2: # mean threshold'dan yüksekse ve patch_mean'teki 255 sayısı THRESHOLD dan fazlaysa :
                            greater_patch = tile_img[cY - 19 : cY + 19 , cX - 19 : cX + 19]
                            greater_patch_mean =np.mean(greater_patch)
                            if greater_patch_mean < GREATER_PATCH_THRESHOLD:
                                area = cv2.contourArea(contour)
                                if area > 5 and area < 200 : 
                                    if DEBUG :
                                        print("Patch mean is  :", patch_mean)
                                        print("Greater Patch img mean is : ", greater_patch_mean)

                                    x_area, y_area, w_area, h_area = cv2.boundingRect(contour)
                                    #cv2.rectangle(tile_img, (x_area,y_area), (x_area + w_area, y_area + h_area), (255,255,255),1)
                                    center_x = ( x_area + w_area//2 ) + tile_x_start
                                    center_y = ( y_area + h_area//2 ) + tile_y_start

                                    bboxes.append((center_x, center_y, w_area, h_area, tile_num))
                                    detected = 1
                                    DETECTED_ON_POSSIBLE_TILES = 1

            
            # detectlenmeyenler için de pixel-wise işlem yapılması gerekiyor burada : ona  göre yine rectangle çizdireceğiz : 
                else:
                    copy_tile = tile_img.copy()
                    edg_tile_img = copy_tile[copy_tile < 65] = 0
                    edg_tile_img = cv2.Canny(copy_tile, 200, 230)
                    coords_from_canny = np.argwhere(edg_tile_img == 255)
                    coords_from_tile = np.argwhere(tile_img == 255)
                    coords = np.concatenate((coords_from_canny, coords_from_tile))
                    if len(coords) > 0 :
                        mean_x = int(np.mean(coords[:,1]))
                        mean_y = int(np.mean(coords[:,0]))
                        patch = tile_img[mean_y - 1 : mean_y + 1 , mean_x - 1: mean_x + 1]
                        patch_mean = np.mean(patch)
                        if patch_mean > PATCH_THRESHOLD and np.count_nonzero(patch == 255) >= 2:
                            greater_patch = tile_img[cY - 4 : cY + 4 , cX - 4 : cX + 4]
                            greater_patch_mean =np.mean(greater_patch)
                            if greater_patch_mean < GREATER_PATCH_THRESHOLD:
                                
                                rectangle_width = 5
                                rectangle_height = 5

                                top_left = (mean_x - rectangle_width // 2, mean_y - rectangle_height // 2)
                                bottom_right = (mean_x + rectangle_width // 2, mean_y + rectangle_height //2)
                                #cv2.rectangle(tile_img, top_left, bottom_right, (255,255,255),1)
                                center_x = mean_x + tile_x_start
                                center_y = mean_y + tile_y_start
                                bboxes.append((center_x, center_y, rectangle_width, rectangle_height, tile_num))
                                DETECTED_ON_POSSIBLE_TILES = 1
                                detected = 1

            if detected == 0 :
                a_copy_tile = tile_img.copy()
                a_copy_tile = cv2.Canny(a_copy_tile, 170, 230)
                contours, _ = cv2.findContours(a_copy_tile, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours :
                    area = cv2.contourArea(contour)
                    if area > 7 and area < 70 :
                        x_area, y_area, w_area, h_area = cv2.boundingRect(contour)
                        if w_area * h_area < 70 :
                            roi = tile_img[y_area: y_area + h_area, x_area: x_area + w_area]
                            max_of_roi = np.max(roi)
                            if max_of_roi > 250:
                                #cv2.rectangle(tile_img, (x_area,y_area), (x_area + w_area, y_area + h_area), (255,255,255),1)
                                center_x = ( x_area + w_area//2 ) + tile_x_start
                                center_y = ( y_area + h_area//2 ) + tile_y_start
                                bboxes.append((center_x, center_y, w_area, h_area,tile_num))
                                DETECTED_ON_POSSIBLE_TILES = 1
                                detected = 1
                            
            # LAST FILTER PART 1 :
            # Pixel-wise filtering in cloud : Using the LoG effect on clouds with NMS.
            #if detected == 0 :
            pixels_255_coords = np.where(tile_img == 255)
            pixels_255_coords = list(zip(pixels_255_coords[1] , pixels_255_coords[0]))
            filtered_coordinates = pixel_wise_filter(tile_img,pixels_255_coords)
            for x,y in filtered_coordinates:
                #cv2.rectangle(tile_img, (x-2 , y-2), (x+2 , y+2), (255,255,255), 1)
                center_x = x + tile_x_start
                center_y = y + tile_y_start
                w_area = 5
                h_area = 5
                bboxes.append((center_x, center_y, w_area, h_area, tile_num))
                DETECTED_ON_POSSIBLE_TILES = 1

            # LAST FILTER PART 2 :
            # Pixel-wise filtering in cloud : Using the mean difference formula with NMS.
            #if detected == 0 :
            THRESHOLD_FOR_KEY_PIXEL_FILTER = 75
            filtered_coordinates_2 = key_pixel_filter(tile_img, pixels_255_coords, THRESHOLD_FOR_KEY_PIXEL_FILTER)
            for x,y in filtered_coordinates_2:
                #cv2.rectangle(tile_img, (x-2 , y-2), (x+2 , y+2), (255,255,255), 1)       
                center_x = x + tile_x_start
                center_y = y + tile_y_start
                w_area = 5
                h_area = 5
                bboxes.append((center_x, center_y, w_area, h_area, tile_num)) 
                DETECTED_ON_POSSIBLE_TILES = 1

           
            # FPS'i çok düşürdüklerinden ve accuracy'i yeterince arttırmadıklarından yorum satırına alındılar.
            # matching_pixels = filter_by_keyfeatures_of_plane(tile_img)
            # for x,y in matching_pixels :
            #     cv2.rectangle(tile_img, (x-3,y-3), (x+3,y+3), (255,255,255), 1)
                

            # LAST FILTER PART 3 : FALSE POSITIVE ARTIŞI, OBJECT'in detect'inde artış.
            # Finding connected bright regions.
            THRESHOLD_REGION = 12
            centroid_coordinates = find_connected_regions(tile_img, THRESHOLD_REGION)
            for center in centroid_coordinates:
                x,y = center 
                #cv2.rectangle(tile_img, (x-3,y-3), (x+3,y+3), (255,255,255), 1)
                center_x = x + tile_x_start
                center_y = y + tile_y_start
                w_area = 7
                h_area = 7
                bboxes.append((center_x, center_y, w_area, h_area, tile_num))
                DETECTED_ON_POSSIBLE_TILES = 1

            if DETECTED_ON_POSSIBLE_TILES == 0 :
                low_T = 150
                high_T = 245
                
                edges_canny = cv2.Canny(tile_img, low_T, high_T)
                contours_of_canny,_ = cv2.findContours(edges_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # # False Positive artışı ve Bulut içinde detection :
                for contour in contours_of_canny :
                    area = cv2.contourArea(contour)
                    if area > 5 and area < 80: 
                        x_area ,y_area ,w_area ,h_area = cv2.boundingRect(contour)
                        if w_area * h_area < 80 and w_area * h_area > 24 :
                           # cv2.rectangle(frame_grayscale, (x_area, y_area), (x_area + w_area, y_area + h_area), (255,255,255), 1)
                            center_x = int(x_area + w_area // 2)
                            center_y  = int(y_area + h_area // 2)
                            bboxes.append((tile_x_start + center_x, tile_y_start + center_y, w_area, h_area, tile_num))

                #cv2.imshow(f"{tile_num}_Tile_Img", tile_img)
                p +=1

        # GREY TILES :
        if DETECTED_ON_POSSIBLE_TILES == 0 :
            for tile_num in grey_detection_tiles :
                row = tile_num // 4
                col = tile_num % 4
                tile_x_start = col * tile_width
                tile_x_end = (col + 1) * tile_width
                tile_y_start = row * tile_height
                tile_y_end = (row + 1) * tile_height

                tile_img = frame_grayscale[tile_y_start:tile_y_end, tile_x_start: tile_x_end]


                # # # False Positive artışı ve Bulut içinde detection :
                low_T = 150
                high_T = 245
                
                edges_canny = cv2.Canny(tile_img, low_T, high_T)
                contours_of_canny,_ = cv2.findContours(edges_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours_of_canny :
                    area = cv2.contourArea(contour)
                    if area > 5 and area < 80: 
                        x_area ,y_area ,w_area ,h_area = cv2.boundingRect(contour)
                        if w_area * h_area < 80 and w_area * h_area > 24 :
                           # cv2.rectangle(frame_grayscale, (x_area, y_area), (x_area + w_area, y_area + h_area), (255,255,255), 1)
                            center_x = int(x_area + w_area // 2)
                            center_y  = int(y_area + h_area // 2)
                            bboxes.append((tile_x_start + center_x, tile_y_start + center_y, w_area, h_area, tile_num))



                contours, _ = cv2.findContours(tile_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                detected = 0
                for contour in contours :
                    area = cv2.contourArea(contour)
                    if area > 10 and area < 70 :
                        x_area, y_area, w_area, h_area = cv2.boundingRect(contour)
                        roi = tile_img[y_area: y_area + h_area, x_area: x_area + w_area]
                        mean_of_roi = np.mean(roi)

                        count_of_zero = count_pixels_with_value(roi,0)
                        ratio_of_zeros_to_area = count_ratio(count_of_zero, w_area, h_area)  

                        if ratio_of_zeros_to_area < 0.30 :
                            if mean_of_roi > 90:
                                lower_bound = 80
                                upper_bound = 200
                                THRESHOLD_PR = 0.28
                                is_Range = check_pixel_range(roi, lower_bound, upper_bound, w_area * h_area , THRESHOLD_PR)
                                if not is_Range:
                                    if w_area * h_area < 70 and w_area * h_area > 20 :
                                        #cv2.rectangle(tile_img, (x_area,y_area), (x_area + w_area, y_area + h_area), (255,255,255),1)  
                                        center_x = ( x_area + w_area // 2 ) + tile_x_start
                                        center_y = ( y_area + h_area // 2 ) + tile_y_start
                                        bboxes.append((center_x, center_y, w_area, h_area,tile_num))
                                        detected = 1
                                    
                        else:
                            if mean_of_roi > 70:
                                max_of_roi = np.max(roi)
                                if max_of_roi > 253 :
                                    if w_area * h_area < 70 and w_area * h_area > 20 :

                                        #cv2.rectangle(tile_img, (x_area,y_area), (x_area + w_area, y_area + h_area), (255,255,255),1)  
                                        center_x = ( x_area + w_area // 2 ) + tile_x_start
                                        center_y = ( y_area + h_area // 2 ) + tile_y_start
                                        bboxes.append((center_x, center_y, w_area, h_area, tile_num))
                                        detected = 1
                

                # LAST FILTER PART 1 :
                # Pixel-wise filtering in cloud : Using the LoG effect on clouds with NMS.
                pixels_255_coords = np.where(tile_img == 255)
                pixels_255_coords = list(zip(pixels_255_coords[1] , pixels_255_coords[0]))
                filtered_coordinates = pixel_wise_filter(tile_img,pixels_255_coords)
                for x,y in filtered_coordinates:
                    #cv2.rectangle(tile_img, (x-2 , y-2), (x+2 , y+2), (255,255,255), 1)
                    center_x = x + tile_x_start
                    center_y = y + tile_y_start
                    w_area = 5
                    h_area = 5
                    bboxes.append((center_x, center_y, w_area, h_area, tile_num))

                

                # LAST FILTER PART 2 :
                # Pixel-wise filtering in cloud : Using the mean difference formula with NMS.
                THRESHOLD_FOR_KEY_PIXEL_FILTER = 75
                filtered_coordinates_2 = key_pixel_filter(tile_img, pixels_255_coords, THRESHOLD_FOR_KEY_PIXEL_FILTER)
                for x,y in filtered_coordinates_2:
                    #cv2.rectangle(tile_img, (x-2 , y-2), (x+2 , y+2), (255,255,255), 1)
                    center_x = x + tile_x_start
                    center_y = y + tile_y_start
                    w_area = 5
                    h_area = 5
                    bboxes.append((center_x, center_y, w_area, h_area, tile_num))


                # matching_pixels = filter_by_keyfeatures_of_plane(tile_img)
                # for x,y in matching_pixels :
                #     cv2.rectangle(tile_img, (x-3,y-3), (x+3,y+3), (255,255,255), 1)
                    

                # LAST FILTER PART 3 : FALSE POSITIVE ARTIŞI, OBJECT'in detect'inde artış.
                # Finding connected bright regions.
                THRESHOLD_REGION = 12
                centroid_coordinates = find_connected_regions(tile_img, THRESHOLD_REGION)
                for center in centroid_coordinates:
                    x,y = center 
                    #cv2.rectangle(tile_img, (x-3,y-3), (x+3,y+3), (255,255,255), 1)
                    center_x = x + tile_x_start
                    center_y = y + tile_y_start
                    w_area = 7
                    h_area = 7
                    bboxes.append((center_x, center_y, w_area, h_area, tile_num))
                #cv2.imshow(f"{tile_num}_Tile_Img", tile_img)
        
        
        mses = []
        ssims = []
        ccorrs = []
        for box in bboxes :
            x,y,w,h,tile_number = box 
            # selected region : patch to be compared to ground truth.
            selected_region = frame_grayscale[x-4:x+4, y-4:y+4]
            # Calculate Similarity between : ground_truth_image and selected_region.
            mse, ssim , ccorr = calculate_similarities(ground_truth_image, selected_region)
            mses.append(mse)
            ssims.append(ssim)
            ccorrs.append(ccorr)

        normalized_mses = normalize(mses)
        normalized_ssims = normalize(ssims)
        normalized_ccorrs = normalize(ccorrs)

        scores = calculate_weighted_score(normalized_mses, normalized_ssims, normalized_ccorrs)
        best_bboxes = []
        for i,box in enumerate(bboxes):
            best_bboxes.append((box,scores[i]))
        if DEBUG :    
            print(best_bboxes)

        sorted_best_bboxes = sorted(best_bboxes, key=lambda x:x[1], reverse=True)
        bboxes = sorted_best_bboxes[:50]
        bboxes = [[item[0][0], item[0][1],item[0][2],item[0][3],item[0][4]] for item in bboxes]
        bboxes = merge_all(bboxes)  

        if DEBUG :
            print("###########################")
            print("The len of bboxes : ", len(bboxes))
            print("###########################")
        
        for box in bboxes:
            x,y,w,h,tile_number = box
            cv2.rectangle(frame_grayscale, (int(x-w//2), int(y-h//2)), (int(x+w//2), int(y+h//2)), (255,255,255), 1 )
        
    #     # 2 - Object Tracking :
        if TRACKING_PIPELINE_TOGGLE_SWITCH :    
            current_frame_tracked_objects = []
            #bboxes = merge_all(bboxes)  
            print("The boxes are : ")
            for box in bboxes:
                
                if DEBUG :
                    print(box)
                    print("####")
                x,y,w,h,tile_number = box
                cv2.circle(frame_grayscale, (x,y), 10, 255, 1) # -> To see if its correct : Drawing circle around the detected boxes.
                # box to bbox convertion for Tracking Algorithm : x,y -> centers but Tracker uses top left.
                # converting :
                found = False
                bbox = (int(x - w //2) , int(y - h // 2), w ,h)
                for obj in detected_objects :
                    if obj.calculate_similarity(bbox) < 6 : # 4- 7
                        found = True
                        success, new_bbox, speed = obj.update(frame,bbox)
                        obj.current_Tile = tile_number
                        if success:
                            cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),  (0,0,255),1)
                            cv2.putText(frame, f"ID: {obj.id}", (bbox[0], bbox[1]-10) , cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0 , 0 , 255), 1)
                            if DEBUG :
                                print(f"Velocity for obj {obj.id} : {speed} ")
                                cv2.putText(frame, f"Speed: {obj.speed:.4f}", (bbox[0], bbox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) , 1)
                                cv2.putText(frame, f"Total Dist: {obj.total_dist:.4f}", (bbox[0], bbox[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) , 1)
                        
                            if obj not in current_frame_tracked_objects:
                                current_frame_tracked_objects.append(obj)

                        break
                if not found :
                    detected_object = DetectedObject(obj_id, bbox, frame, tile_number)
                    cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),  (0,255,0),1)
                    detected_objects.append(detected_object)
                    obj_id +=1

            if DEBUG :
                print("Prev frame tracked : ", prev_frame_tracked_objects)
                print("Curr frame tracked : ", current_frame_tracked_objects)

            
    
            if prev_frame_tracked_objects :
                anchor_objects = check_changes(prev_frame_tracked_objects, current_frame_tracked_objects)
                if DEBUG :
                    print("anchor_objects are : ")
                    for a in anchor_objects :
                        print("ID: ", a.id ," Object :" , a)
                    print("#####################")

                # total distances current hesapla :
                total_distances_curr = calculate_total_distances(anchor_objects)

                anchor_matching_prev, anchor_matching_curr, total_distances_prev, total_distances_next = find_matching_objects(anchor_objects, prev_frame_tracked_objects,current_frame_tracked_objects)
                
                if DEBUG :
                    print("Total Distances Prev : ")
                    print(total_distances_prev)
                    print("Total Distances Next : ")
                    print(total_distances_next)
                
            # prev distance curr ve total distance curr kıyası yap : 
            if DEBUG :
                if total_distances_curr :
                    print("Total Distance Curr : ")
                    print(total_distances_curr)
                if total_distances_prev :
                    print("Prev Distance Curr : ")
                    print(total_distances_prev)
            
            diff_dict = {}
            if total_distances_curr and total_distances_prev :
                for key, val in total_distances_curr.items():
                    for key2, val2 in total_distances_prev.items():
                        if key == key2 :
                            diff_dict[key] = val2 - val
                            break
            if DEBUG :
                if diff_dict :
                    print("Diff Dict : ")
                    print(diff_dict)
                        

            img_with_kp = cv2.drawKeypoints(frame_grayscale, keypoints, None, flags = cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

            prev_frame_tracked_objects = current_frame_tracked_objects
            if total_distances_curr :
                total_distances_prev = total_distances_curr

    # TRACKING ENDS.


        cv2.imshow("Grayscale Frame", frame_grayscale)
        cv2.imshow("Vanilla Frame", frame)
        if DEBUG :
            cv2.imshow("ORB", img_with_kp)
            cv2.imshow("Canny Edges", edges_canny)


        key = cv2.waitKey(0)

        if key == 113: # Press q to quit.
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


