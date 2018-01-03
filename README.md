# ResNeXt.pytorch
Reproduces ResNet-V3 (Aggregated Residual Transformations for Deep Neural Networks) with pytorch.

- [x] Trains on Cifar10 and Cifar100
- [x] Upload Cifar Training Curves
- [x] Upload Cifar Trained Models
- [ ] Train Imagenet

## Usage
To train on Cifar-10 using 2 gpu:

```bash
python train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots --log ./logs --ngpu 2 --learning_rate 0.05 -b 128
```
It should reach *~3.65%* on Cifar-10, and *~17.77%* on Cifar-100.


After train phase, you can check saved model.

**Thanks to [@AppleHolic](https://github.com/AppleHolic) we have now a test script:**

To test on Cifar-10 using 2 gpu:
```bash
python test.py ~/DATASETS/cifar.python cifar10 --ngpu 2 --load ./snapshots/model.pytorch --test_bs 128 
```


## Configurations
From [the original paper](https://arxiv.org/pdf/1611.05431.pdf):

| cardinality |  base_width  | parameters |  Error cifar10   |   error cifar100  | default |
|:-----------:|:------------:|:----------:|:----------------:|:-----------------:|:-------:|
|      8      |      64      |    34.4M   |       3.65       |       17.77       |    x    |
|      16     |      64      |    68.1M   |       3.58       |       17.31       |         |

**Update:** ``widen_factor`` has been disentangled from ``base_width`` because it was confusing. Now widen factor is set to consant 4, and ``base_width`` is the same as in the original paper.

## Trained models and curves
[Link](https://mega.nz/#F!wbJXDS6b!YN3hCDi1tT3SdNFrLPm7mA) to trained models corresponding to the following curves:

**Update:** several commits have been pushed after training the models in Mega, so it is recommended to revert to ``e10c37d8cf7a958048bc0f58cd86c3e8ac4e707d``

![CIFAR-10](https://github.com/prlz77/ResNeXt.pytorch/blob/master/cifar10/cifar-10.jpg)
![CIFAR-100](https://github.com/prlz77/ResNeXt.pytorch/blob/master/cifar100/cifar-100.jpg)

## Other frameworks
* [torch (@facebookresearch)](https://github.com/facebookresearch/ResNeXt). (Original) Cifar and Imagenet
* [caffe (@terrychenism)](https://github.com/terrychenism/ResNeXt). Imagenet
* [MXNet (@dmlc)](https://github.com/dmlc/mxnet/tree/master/example/image-classification#imagenet-1k). Imagenet

## Cite
```
@article{xie2016aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  journal={arXiv preprint arXiv:1611.05431},
  year={2016}
}
```
