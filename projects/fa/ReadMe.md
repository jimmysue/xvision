# Face Alignment

## 数据集

为了训出精准鲁邦的人脸对齐模型, 需要足够多且多样的数据, 这里汇总了现在学术界及竞赛用到的一些公开的数据集:

### IBUG 数据集

由四个子集组成, 外加一个检测框标注

- i-bug: https://ibug.doc.ic.ac.uk/download/annotations/ibug.zip
- afw: https://ibug.doc.ic.ac.uk/download/annotations/afw.zip
- helen: https://ibug.doc.ic.ac.uk/download/annotations/helen.zip
- lfpw: https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip
- bounding box annotations: https://ibug.doc.ic.ac.uk/media/uploads/competitions/bounding_boxes.zip

数据集组织如下:
```
unzip ibug.zip -d ibug
mv ibug/image_092\ _01.jpg ibug/image_092_01.jpg
mv ibug/image_092\ _01.pts ibug/image_092_01.pts

unzip afw.zip -d afw
unzip helen.zip -d helen
unzip lfpw.zip -d lfpw
unzip bounding_boxes.zip ; mv Bounding\ Boxes Bounding_Boxes
```

## [WFLW](https://wywu.github.io/projects/LAB/WFLW.html)

WFLW 数据集总共标注了10000个人脸, 其中7500作为训练集,2500作为测试. 图像来自WIDER FACE 人脸检测数据集

## [JDAI-CV/lapa-dataset](https://github.com/JDAI-CV/lapa-dataset)

数据用于人脸解析比赛, 其中包含了人脸106点标注

## [JD Grand Challenge of 106-p Facial Landmark Localization](https://github.com/facial-landmarks-localization-challenge/facial-landmarks-localization-challenge.github.io/blob/master/Corrected_landmark.zip?raw=true)
