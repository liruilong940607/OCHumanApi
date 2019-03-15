# OCHuman(Occluded Human) Dataset Api

Dataset proposed in 'Pose2Seg: Human Instance Segmentation Without Detection'[[ProjectPage]](http://www.liruilong.cn/Pose2Seg/index.html)[[arXiv]](https://arxiv.org/abs/1803.10683) @ CVPR2019. 

<div align="center">

<img src="figures/dataset.jpg" width="1000px"/>

<p> Samples of OCHuman Dataset</p>

</div>

This dataset focus on heavily occluded human with comprehensive annotations including bounding-box, humans pose and instance mask. This dataset contains 13360 elaborately annotated human instances within 5081 images. With average 0.573 MaxIoU of each person, OCHuman is the most complex and challenging dataset related to human. Through this dataset, we want to emphasize occlusion as a challenging problem for researchers to study.


## Statistics

All the instances in this dataset are annotated by bounding-box. While not all of them have the
keypoint/mask annotation. If you want to compare your results to our paper, please use the subset
that contains both keypoint and mask annotations.

|          | bbox  | keypoint | mask | keypoint&mask | bbox&keypoint&mask|
| ------   | ----- | ----- | ----- | ----- | ----- |
| #Images  | 5081  | 5081  | 4731  | 4731  | 4731  |
| #Persons | 13360 | 10375 | 8110  | 8110  | 8110  |
| #mMaxIou | 0.573 | 0.670 | 0.669 | 0.669 | 0.669 |

**Note**: 
- *MaxIoU* measures the severity of an object being occluded, which means the max IoU with other same category objects in a single image.
- All instances in OCHuman with kpt/mask annotations are suffered by heavy occlusion. (MaxIou > 0.5)

## Download Links

- [Images (667MB)](https://cg.cs.tsinghua.edu.cn/people/~lrl/OCHuman/image.zip)
- [Annotations](https://cg.cs.tsinghua.edu.cn/people/~lrl/OCHuman/ochuman.json)

Here we also provide the coco style annotations [[Annotations COCO Style (val)]](https://cg.cs.tsinghua.edu.cn/people/~lrl/OCHuman/ochuman_coco_format_val_range_0.00_1.00.json)[[Annotations COCO Style (test)]](https://cg.cs.tsinghua.edu.cn/people/~lrl/OCHuman/ochuman_coco_format_test_range_0.00_1.00.json) so that you can run evaluation using cocoEval toolbox.

## Install API 
```
git clone https://github.com/liruilong940607/OCHuman
cd OCHuman
make install
```

## How to use
See [Demo.ipynb](git remote add origin https://github.com/liruilong940607/OCHumanApi/Demo.ipynb)
