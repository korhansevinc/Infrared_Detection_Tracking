import numpy as np

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

    del boxes[j]
    del boxes[i]
    boxes.append( (merged_x, merged_y, merged_width, merged_height))

    return boxes

def merge_all(boxes):
    while True :
        merged_boxes = mean_box_fusion(boxes)
        if merged_boxes is None:
            break
        boxes = merged_boxes
    return boxes
    
    # Another method :

    # merged_boxes = []
    # merged_indices = set()
    # for i in range(len(boxes)):
    #     if i not in merged_indices:
    #         merged_indices.add(i)
    #         merged=False
    #         for j in range(i+1, len(boxes)):
    #             if j not in merged_indices and is_close_enough(boxes[i], boxes[j], 5):
    #                 merged_indices.add(j)
    #                 merged=True
    #                 center_x = ( boxes[i][0] + boxes[j][0] ) / 2
    #                 center_y = ( boxes[i][1] + boxes[j][1] ) / 2
    #                 merged_boxes.append( (center_x, center_y))
    #         if not merged:
    #             merged_boxes.append((boxes[i][0], boxes[i][1]))
    # return merged_boxes

boxes = [ (10,10,24,44), (13,14,27,24),(15,15,23,26), (50,50,32,30), (100,100,26,25) ,  (12,13,28,23), (14,15,20,22),(15,16,19,21), (6,8,42,20)]
merged_boxes = merge_all(boxes)
print("Merged boxes : ", merged_boxes)
