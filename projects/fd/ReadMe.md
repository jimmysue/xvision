# Face Detection by Soft Matching

## Wider Face Performance
 
 | Method             | Easy      | Medium    | Hard      |
 | ------------------ | --------- | --------- | --------- |
 | [Slim(Linzaer)][1] | 0.765     | 0.662     | **0.385** |
 | [Slim(biubiu6)][2] | 0.795     | 0.683     | 0.345     |
 | Slim(SoftMatching) | **0.801** | **.719**  | .378      |
 | [RFB(Linzaer)][1]  | 0.784     | 0.688     | **0.418** |
 | [RFB(biubiu6)][2]  | 0.814     | 0.710     | 0.363     |
 | RFB(SoftMatching)  | **0.818** | **0.740** | 0.395     |

> test with long size 320

## How 

### Installation

```shell
# 1. clone the repo
git clone https://github.com/jimmysue/xvision  && cd xvision  
cd xvision
# 2. add xvision to search python search path
export PYTHONPATH=.:$PYTHONPATH  
# 3. install requirements
pip install -r requirements.txt
```

### Prepare Dataset

We use data provided by [biubiu6][2], you can download the annotation file from:
- [google cloud][google cloud] 
- [baidu cloud][], with password: ruck

Details about the data please refer to [Face-Detector-1MB-with-landmark][2]

### Train 

```shell
python projects/fd/train.py --workdir workspace/fd  \  # change the workdir as you want
                            --model.name  Slim or RFB
```

### Test

1. predict 
```shell
python projects/fd/test.py --workdir workspace/fd \   # training workdir
                           --model.name Slim          # same with training
```
Predict results will be save in `workspace/fd/result`

2. evaluation

Before evaluation, you shold download ground truth from [evaluation code and validation results](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip)

```
python projects/fd/evaluation.py -p workspace/fd/result -g path/to/goundtruth/
```

ps: The evaluation script is a copy from [WiderFace-Evaluation][WiderFace-Evaluation], 
and I change the box iou  with torchvision implementation, Which will cause slightly performance derease. 


## References

- [Ultra-Light-Fast-Generic-Face-Detector-1MB][1]
- [Face-Detector-1MB-with-landmark][2]


[1]: Ultra-Light-Fast-Generic-Face-Detector-1MB
[2]: https://github.com/biubug6/Face-Detector-1MB-with-landmark
[google cloud]: https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS
[baidu cloud]: https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ
[WiderFace-Evaluation]: https://github.com/wondervictor/WiderFace-Evaluation