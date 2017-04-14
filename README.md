# ResNeXt.pytorch
Reproduces ResNet-V3 (Aggregated Residual Transformations for Deep Neural Networks) with pytorch.

- [x] Trains on Cifar10 and Cifar100
- [ ] Upload Cifar Training Curves
- [ ] Upload Cifar Trained Models
- [ ] Train Imagenet

## Usage
To train on Cifar-10 using 2 gpu:

```bash
python train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots --log ./logs --ngpu 2 --learning_rate 0.05 -b 128
```
It should reach *~3.65%* on Cifar-10, and *~17.77%* on Cifar-100.

## Configurations
From [the original paper](https://arxiv.org/pdf/1611.05431.pdf):

| cardinality | widen_factor | parameters | accuracy cifar10 | accuracy cifar100 | default |
|:-----------:|:------------:|:----------:|:----------------:|:-----------------:|:-------:|
|      8      |       4      |    34.4M   |       3.65       |       17.77       |    x    |
|      16     |      64      |    68.1M   |       3.58       |       17.31       |         |


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
