# Face Detection by Soft Matching

## Wider Face Performance
 
 Method|Easy|Medium|Hard
------|--------|----------|--------
[Slim(Linzaer)][1]|0.765     |0.662       |0.385
[Slim(biubiu6)][2]|0.795     |0.683       |0.345
[Slim(SoftMatching)|**0.799**     |**0.710**       | **0.370**

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
### Preparation 

- download dataset

We using data provide by [biubiu6]()


```shell

```


## References

- [Ultra-Light-Fast-Generic-Face-Detector-1MB][1]
- [Face-Detector-1MB-with-landmark][2]


[1]: Ultra-Light-Fast-Generic-Face-Detector-1MB
[2]: https://github.com/biubug6/Face-Detector-1MB-with-landmark
[google cloud]: https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS
[baidu cloud]: https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ