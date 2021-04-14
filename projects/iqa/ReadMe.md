# Image Quality Assessment
## NIMA

This is a PyTorch implementation of the paper [NIMA: Neural IMage Assessment][NIMA] ([accepted at IEEE Transactions on Image Processing][TIP]) by Hossein Talebi and Peyman Milanfar. You can learn more from this post at [Google Research Blog][Google Research Blog].

### Implementation Details

- Use `OneCycleLR` learning rate scheduler instead
- Make data memmap to speed up training
- Vectorize `emd_loss` calculation

### Acknowledgement

This is implementation largely references [kentsyx/Neural-IMage-Assessment][kentsyx]


[NIMA]: https://arxiv.org/abs/1709.05424
[TIP]: https://ieeexplore.ieee.org/document/8352823
[Google Research Blog]: https://research.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html
[kentsyx]: https://github.com/kentsyx/Neural-IMage-Assessment