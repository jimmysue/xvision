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