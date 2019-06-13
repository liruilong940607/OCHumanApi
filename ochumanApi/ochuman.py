import json
import cv2
import os
import numpy as np
import ochumanApi.mask as mask_util
from ochumanApi.vis import draw_bbox, draw_mask, draw_skeleton

def annToMask(segm, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    def _annToRLE(segm, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_util.frPyObjects(segm, height, width)
            rle = mask_util.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = mask_util.frPyObjects(segm, height, width)
        else:
            # rle
            rle = segm
        return rle

    rle = _annToRLE(segm, height, width)
    mask = mask_util.decode(rle)
    return mask


def Poly2Mask(ploys):
    #{'outer', 'inner', 'height', 'width'}
    if len(ploys['outer']) > 0:
        mask = annToMask(ploys['outer'], ploys['height'], ploys['width'])
        if len(ploys['inner']) > 0:
            mask_inner = annToMask(ploys['inner'], ploys['height'], ploys['width'])
            mask[mask_inner>0] = 0
    else:
        mask = np.zeros((ploys['height'], ploys['width']), dtype=np.uint8)
    return mask
    
    
class OCHuman():
    def __init__(self, AnnoFile, Filter=None):
        self.dataset = json.load(open(AnnoFile, 'r'))
        self._filter = Filter
        
        self.keypoint_names = self.dataset['keypoint_names']
        self.keypoint_visible = self.dataset['keypoint_visible']
        
        self.images = {}
        self.imgIds = []
        if Filter==None:
            for imgItem in self.dataset['images']:
                self.imgIds.append(imgItem['image_id'])
                self.images[imgItem['image_id']] = imgItem
        else:
            for imgItem in self.dataset['images']:                
                if self._filter in ['kpt&segm', 'segm&kpt']:
                    annos = [anno for anno in imgItem['annotations'] \
                             if anno['keypoints'] and anno['segms']]
                elif self._filter in ['kpt|segm', 'segm|kpt']:
                    annos = [anno for anno in imgItem['annotations'] \
                             if anno['keypoints'] or anno['segms']]
                elif self._filter in ['kpt']:
                    annos = [anno for anno in imgItem['annotations'] \
                             if anno['keypoints']]
                elif self._filter in ['segm']:
                    annos = [anno for anno in imgItem['annotations'] \
                             if anno['segms']]
                if len(annos)>0:
                    imgItem['annotations'] = annos
                    self.imgIds.append(imgItem['image_id'])
                    self.images[imgItem['image_id']] = imgItem
        

    def getImgIds(self):
        return self.imgIds
    
    def loadImgs(self, imgIds=None):
        return [self.images[image_id] for image_id in imgIds] 
    
    def visImg(self, ImgDir, image_id):
        data = self.loadImgs(imgIds=[image_id])[0]
        img = cv2.imread(os.path.join(ImgDir, data['file_name']))
        
        colors = [[255, 0, 0], 
                 [255, 255, 0],
                 [0, 255, 0],
                 [0, 255, 255], 
                 [0, 0, 255], 
                 [255, 0, 255]]
        
        for i, anno in enumerate(data['annotations']):
            bbox = anno['bbox']
            kpt = anno['keypoints']
            segm = anno['segms']
            #print (anno['links'])
            max_iou = anno['max_iou']

            img = draw_bbox(img, bbox, thickness=3, color=colors[i%len(colors)])
            if segm is not None:
                mask = Poly2Mask(segm)
                img = draw_mask(img, mask, thickness=3, color=colors[i%len(colors)])
            if kpt is not None:
                img = draw_skeleton(img, kpt, connection=None, colors=colors[i%len(colors)], bbox=bbox)
            
        
        return img
    
    def toCocoFormart(self, subset='all', maxIouRange=(0., 1.), save_dir=None):
        # maxIouRange: Moderate [0.5, 0.75), Hard [0.75, 1.]
        assert self._filter in ['kpt&segm', 'segm&kpt']
        assert subset in ['all', 'val', 'test']
        
        # total 4731 Image; 8110 Persons.
        # val set: first 2500 Images
        # test set: last 2231 Images
        
        dataset_json = {'annotations': [], 
                        'categories': [{'supercategory': 'person', 
                                          'id': 1, 
                                          'name': 'person', 
                                          'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], 
                                                       [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], 
                                                       [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]],
                                          'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
                                                        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                                                        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 
                                                        'right_knee', 'left_ankle', 'right_ankle']
                                         }], 
                        'images': []
                       }
        annotation = {'image_id': None, 
                      'area': None, 
                      'num_keypoints': 0, 
                      'iscrowd': 0, 
                      'id': None, 
                      'category_id': 1, 
                      'keypoints': [], 
                      'segmentation': [[]], 
                      'bbox': []
                     }
        image = {'id': None, 
                 'file_name': '0.jpg', 
                 'height': None, 
                 'width': None}

        if subset == 'all':
            imgIds = self.imgIds
        elif subset == 'val':
            imgIds = self.imgIds[0: 2500]
        elif subset == 'test':
            imgIds = self.imgIds[2500:]
            
        IMGID = 1 # v4 modified
        ANNOID = 1 # v4 modified
        for image_id in imgIds:
            data = self.loadImgs(imgIds=[image_id])[0]
            file_name = data['file_name']
            width = data['width']
            height = data['height']
            
            image = {'id': IMGID, 
                     'file_name': file_name, 
                     'height': height, 
                     'width': width}
    
            total_anno = 0
            for i, anno in enumerate(data['annotations']):
                bbox = anno['bbox']
                kpt = anno['keypoints']
                segm = anno['segms']
                #print (anno['links'])
                max_iou = anno['max_iou']
                if max_iou < maxIouRange[0] or max_iou >= maxIouRange[1]:
                    continue
                
                ## coco box: xyxy -> xywh
                x1, y1, x2, y2 = bbox
                bboxCoco = [x1, y1, x2-x1, y2-y1]
                area = (x2-x1)*(y2-y1)
                
                ## coco kpt: vis 2, not vis 1, missing 0.
                # 'keypoint_visible': {'missing': 0, 'vis': 1, 'self_occluded': 2, 'others_occluded': 3},
                kptDef = self.dataset['keypoint_names']
                kptDefCoco = ['nose', 
                      'left_eye', 'right_eye', 'left_ear', 'right_ear', 
                      'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                      'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
                      'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
                kptCoco = []
                num_keypoints = 0
                for i in range(len(kptDefCoco)):
                    idx = kptDef.index(kptDefCoco[i])
                    x, y, v= kpt[idx*3: idx*3+3]
                    if v == 1 or v == 2:
                        v = 2
                        num_keypoints += 1
                    elif v == 3:
                        v = 1
                        num_keypoints += 1
                    kptCoco += [x, y, v]
                assert len(kptCoco) == 17 * 3
                
                ## coco segm
                mask = Poly2Mask(segm)
                maskencode = mask_util.encode(np.asfortranarray(mask))
                maskencode['counts'] = maskencode['counts'].decode('ascii')
                segmCoco = maskencode
                
                annotation = {'image_id': IMGID, 
                              'area': area, 
                              'num_keypoints': num_keypoints, 
                              'iscrowd': 0, 
                              'id': ANNOID, 
                              'category_id': 1, 
                              'keypoints': kptCoco, 
                              'segmentation': segmCoco, 
                              'bbox': bboxCoco
                             }
                total_anno += 1
                dataset_json['annotations'].append(annotation)
                ANNOID += 1
                
            if total_anno > 0:
                dataset_json['images'].append(image)
                IMGID += 1
                
        print ('convert OCHuman to COCO format done.', 
               'total %d persons within'%len(dataset_json['annotations']), 
               '%d images.'%len(dataset_json['images']))
        
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open('%s/ochuman_coco_format_%s_range_%.2f_%.2f.json'%(save_dir, 
                                                                        subset, 
                                                                        maxIouRange[0], 
                                                                        maxIouRange[1]), 'w') as f:
                json.dump(dataset_json, f)
        else:
            return dataset_json
        
