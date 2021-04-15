# Image Quality Assessment
## NIMA

This is a PyTorch implementation of the paper [NIMA: Neural IMage Assessment][NIMA] ([accepted at IEEE Transactions on Image Processing][TIP]) by Hossein Talebi and Peyman Milanfar. You can learn more from this post at [Google Research Blog][Google Research Blog].

### Implementation Details

- Use `OneCycleLR` learning rate scheduler instead
- Make data memmap to speed up training
- Vectorize `emd_loss` calculation

### Performance

The final evaluation emd loss ~0.07, which is slightly better than [kentsyx/Neural-IMage-Assessment][kentsyx]

```shell
2021-04-15 05:02:03,917 | step: [21952/22400] img_s: 838.68 train: [lr: 0.000020        loss: 0.0686 (0.0683)] eval:[loss: 0.0707 (0.0699)]
2021-04-15 05:06:35,449 | step: [22176/22400] img_s: 850.87 train: [lr: 0.000012        loss: 0.0680 (0.0683)] eval:[loss: 0.0706 (0.0699)]
2021-04-15 05:11:08,993 | step: [22400/22400] img_s: 844.05 train: [lr: 0.000010        loss: 0.0678 (0.0683)] eval:[loss: 0.0707 (0.0700)]
```

### Acknowledgement

This is implementation largely references [kentsyx/Neural-IMage-Assessment][kentsyx]


[NIMA]: https://arxiv.org/abs/1709.05424
[TIP]: https://ieeexplore.ieee.org/document/8352823
[Google Research Blog]: https://research.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html
[kentsyx]: https://github.com/kentsyx/Neural-IMage-Assessment
