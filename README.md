# X-Vision

## Face Detection

### Wider Face Performance on Validation Set
 
 | Method             | Easy      | Medium    | Hard      |
 | ------------------ | --------- | --------- | --------- |
 | [Slim(Linzaer)][1] | 0.765     | 0.662     | **0.385** |
 | [Slim(biubiu6)][2] | 0.795     | 0.683     | 0.345     |
 | Slim(SoftMatching) | **0.799** | **0.710** | 0.370     |
 | [RFB(Linzaer)][1]  | 0.784     | 0.688     | **0.418** |
 | [RFB(biubiu6)][2]  | 0.814     | 0.710     | 0.363     |
 | RFB(SoftMatching)  | **0.818** | **0.740** | 0.395     |

> test with long size 320

## TODO

### Facial Relative Recoginition Algorithms

- [x] face detection
- [ ] face aligment **WIP**
- [ ] face reconstruction
- [ ] face recognition

### Image Recognition 

- [ ] Image Aesthetics Assessment
- [ ] ...



[1]: Ultra-Light-Fast-Generic-Face-Detector-1MB
[2]: https://github.com/biubug6/Face-Detector-1MB-with-landmark
[google cloud]: https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS
[baidu cloud]: https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ
[WiderFace-Evaluation]: https://github.com/wondervictor/WiderFace-Evaluation