import cv2
import numpy as np
import math

def draw_bbox(img, bbox, thickness=3, color=(255, 0, 0)):
    canvas = img.copy()
    cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return canvas

def draw_mask(img, mask, thickness=3, color=(255, 0, 0)):
    def _get_edge(mask, thickness=3):
        dtype = mask.dtype
        x=cv2.Sobel(np.float32(mask),cv2.CV_16S,1,0, ksize=thickness) 
        y=cv2.Sobel(np.float32(mask),cv2.CV_16S,0,1, ksize=thickness)
        absX=cv2.convertScaleAbs(x)
        absY=cv2.convertScaleAbs(y)  
        edge = cv2.addWeighted(absX,0.5,absY,0.5,0)
        return edge.astype(dtype)
    
    img = img.copy()
    canvas = np.zeros(img.shape, img.dtype) + color
    img[mask > 0] = img[mask > 0] * 0.8 + canvas[mask > 0] * 0.2
    edge = _get_edge(mask, thickness)
    img[edge > 0] = img[edge > 0] * 0.2 + canvas[edge > 0] * 0.8
    return img

def draw_skeleton(img, kpt, connection=None, colors=None, bbox=None):
    kpt = np.array(kpt, dtype=np.int32).reshape(-1, 3)
    npart = kpt.shape[0]
    canvas = img.copy()

    if npart==17: # coco
        part_names = ['nose', 
                      'left_eye', 'right_eye', 'left_ear', 'right_ear', 
                      'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                      'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
                      'left_knee', 'right_knee', 'left_ankle', 'right_ankle'] 
        #visible_map = {1: 'vis', 
        #               2: 'not_vis', 
        #               3: 'missing'}
        visible_map = {2: 'vis', 
                       1: 'not_vis', 
                       0: 'missing'}
        map_visible = {value: key for key, value in visible_map.items()}
        if connection is None:
            connection = [[16, 14], [14, 12], [17, 15], 
                          [15, 13], [12, 13], [6, 12], 
                          [7, 13], [6, 7], [6, 8], 
                          [7, 9], [8, 10], [9, 11], 
                          [2, 3], [1, 2], [1, 3], 
                          [2, 4], [3, 5], [4, 6], [5, 7]]
    elif npart==19: # ochuman
        part_names = ["right_shoulder", "right_elbow", "right_wrist",
                     "left_shoulder", "left_elbow", "left_wrist",
                     "right_hip", "right_knee", "right_ankle",
                     "left_hip", "left_knee", "left_ankle",
                     "head", "neck"] + \
                     ['right_ear', 'left_ear', 'nose', 'right_eye', 'left_eye']
        visible_map = {0: 'missing', 
                       1: 'vis', 
                       2: 'self_occluded', 
                       3: 'others_occluded'}
        map_visible = {value: key for key, value in visible_map.items()}
        if connection is None:
            connection = [[16, 19], [13, 17], [4, 5],
                         [19, 17], [17, 14], [5, 6],
                         [17, 18], [14, 4], [1, 2],
                         [18, 15], [14, 1], [2, 3],
                         [4, 10], [1, 7], [10, 7],
                         [10, 11], [7, 8], [11, 12], [8, 9],
                         [16, 4], [15, 1]] # TODO
            
    
    if colors is None:
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], 
                 [255, 255, 0], [170, 255, 0], [85, 255, 0], 
                 [0, 255, 0], [0, 255, 85], [0, 255, 170], 
                 [0, 255, 255], [0, 170, 255], [0, 85, 255], 
                 [0, 0, 255], [85, 0, 255], [170, 0, 255],
                 [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    elif type(colors[0]) not in [list, tuple]:
        colors = [colors]
    
    idxs_draw = np.where(kpt[:, 2] != map_visible['missing'])[0]
    if len(idxs_draw)==0:
        return img
    
    if bbox is None:
        bbox = [np.min(kpt[idxs_draw, 0]), np.min(kpt[idxs_draw, 1]),
                np.max(kpt[idxs_draw, 0]), np.max(kpt[idxs_draw, 1])] # xyxy
    
    Rfactor = math.sqrt((bbox[2]-bbox[0]) * (bbox[3]-bbox[1])) / math.sqrt(img.shape[0] * img.shape[1])
    Rpoint = int(min(10, max(Rfactor*10, 4)))
    Rline = int(min(10, max(Rfactor*5, 2)))
    #print (Rfactor, Rpoint, Rline)
    
    for idx in idxs_draw:
        x, y, v = kpt[idx, :]
        cv2.circle(canvas, (x, y), Rpoint, colors[idx%len(colors)], thickness=-1)
        
        if v==2:
            cv2.rectangle(canvas, (x-Rpoint-1, y-Rpoint-1), (x+Rpoint+1, y+Rpoint+1), 
                          colors[idx%len(colors)], 1)
        elif v==3:
            cv2.circle(canvas, (x, y), Rpoint+2, colors[idx%len(colors)], thickness=1)

    for idx in range(len(connection)):
        idx1, idx2 = connection[idx]
        y1, x1, v1 = kpt[idx1-1]
        y2, x2, v2 = kpt[idx2-1]
        if v1 == map_visible['missing'] or v2 == map_visible['missing']:
            continue
        mX = (x1+x2)/2.0
        mY = (y1+y2)/2.0
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        angle = math.degrees(math.atan2(x1 - x2, y1 - y2))
        polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), Rline), int(angle), 0, 360, 1)
        cur_canvas = canvas.copy()
        cv2.fillConvexPoly(cur_canvas, polygon, colors[idx%len(colors)])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
    return canvas