# 基于PaddleSeg的手机屏幕瑕疵检测

手机屏幕瑕疵检测是对手机屏幕破损、划痕的位置、大小等问题的检测

下图为手机屏幕划痕的图像与标注图像：

<img src='https://ai-studio-static-online.cdn.bcebos.com/9e78af02edaf46318cce40969cee9658b77eb042de874a029fd838487dc581cc' width='40%' height='40%'>

<img src='https://ai-studio-static-online.cdn.bcebos.com/c60fe6fae0b54cc7bb55b2e90681a9cee560280323ff46358f831d10a45839d3' width='38%' height='38%'>


### 项目流程

- 使用HRNet进行手机屏幕瑕疵检测，得到更精细化的识别效果进而计算瑕疵面积。
- 使用数据增强手段，扩充次品的样本量，提升分割效果。
- 使用针对类别不均衡问题效果的损失函数，我们应用dice_loss、bce_loss提高二分类分割精度。
- 考虑到检测的实时性等实际问题，选择MobileNet作为DeepLabv3+的BackBone。


## PaddleSeg

PaddleSeg是基于PaddlePaddle开发的端到端图像分割开发套件，覆盖了DeepLabv3+, U-Net, ICNet, PSPNet, HRNet, Fast-SCNN等主流分割网络。通过模块化的设计，以配置化方式驱动模型组合，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用

###  特点

- **丰富的数据增强**

基于百度视觉技术部的实际业务经验，内置10+种数据增强策略，可结合实际业务场景进行定制组合，提升模型泛化能力和鲁棒性。

- **模块化设计**

支持U-Net, DeepLabv3+, ICNet, PSPNet, HRNet, Fast-SCNN六种主流分割网络，结合预训练模型和可调节的骨干网络，满足不同性能和精度的要求；选择不同的损失函数如Dice Loss, Lovasz Loss等方式可以强化小目标和不均衡样本场景下的分割精度。

- **高性能**

PaddleSeg支持多进程I/O、多卡并行等训练加速策略，结合飞桨核心框架的显存优化功能，可大幅度减少分割模型的显存开销，让开发者更低成本、更高效地完成图像分割训练。

- **工业级部署**

全面提供**服务端**和**移动端**的工业级部署能力，依托飞桨高性能推理引擎和高性能图像处理实现，开发者可以轻松完成高性能的分割模型部署和集成。通过[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)，可以在移动设备或者嵌入式设备上完成轻量级、高性能的人像分割模型部署。

- **产业实践案例**

PaddleSeg提供丰富地产业实践案例，如[人像分割](./contrib/HumanSeg)、[工业表计检测](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/contrib#%E5%B7%A5%E4%B8%9A%E8%A1%A8%E7%9B%98%E5%88%86%E5%89%B2)、[遥感分割](./contrib/RemoteSensing)、[人体解析](contrib/ACE2P)，[工业质检](https://aistudio.baidu.com/aistudio/projectdetail/184392)等产业实践案例，助力开发者更便捷地落地图像分割技术。


## 安装

### 1. 安装PaddlePaddle

版本要求
* PaddlePaddle >= 1.7.0
* Python >= 3.5+

由于图像分割模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleSeg.
```
pip install -U paddlepaddle-gpu
```
同时请保证您参考NVIDIA官网，已经正确配置和安装了显卡驱动，CUDA 9，cuDNN 7.3，NCCL2等依赖，其他更加详细的安装信息请参考：[PaddlePaddle安装说明](https://www.paddlepaddle.org.cn/install/doc/index)。

### 2. 下载PaddleSeg代码

```
git clone https://github.com/PaddlePaddle/PaddleSeg
```

### 3. 安装PaddleSeg依赖
通过以下命令安装python包依赖，请确保在该分支上至少执行过一次以下命令：
```
cd PaddleSeg
pip install -r requirements.txt
```

## 数据准备

- 解压数据集并移动至指定位置
- 由于原始数据过大，不适合做示例演示，现有的数据已经过处理，如有需要可留言。


## 关于标注数据

- **数据标注**
1. PaddleSeg采用单通道的标注图片，每一种像素值代表一种类别，像素标注类别需要从0开始递增，例如0，1，2，3表示有4种类别。
1. NOTE: 标注图像请使用PNG无损压缩格式的图片。标注类别最多为256类
1. PaddleSeg支持灰度标注同时也支持伪彩色标注

- **标注转换**

PaddleSeg支持灰度标注转换为伪彩色标注，如需转换成伪彩色标注图，可使用PaddleSeg自带的的转换工具：

```buildoutcfg
python pdseg/tools/gray2pseudo_color.py <dir_or_file> <output_dir>
```

|参数|用途|
|-|-|
|dir_or_file|指定灰度标注所在目录|
|output_dir|彩色标注图片的输出目录|


本项目中包含灰度标注与伪彩色标注两种标注：
- PaddleSeg/dataset/phone/Annotation_color为伪彩色标注
- PaddleSeg/dataset/phone/Annotations为灰度标注

## 训练阶段可视化

在训练的过程中可以使用 VisualDL 观察损失函数、准确率的变化曲线以及阶段性保存模型的预测结果。

本网页的地址：https://aistudio.baidu.com/bdvgpu/user/61916/698034/notebooks/698034.ipynb **将后续notebooks及其后面的地址删除掉，替换为visualdl**

替换后的地址：https://aistudio.baidu.com/bdvgpu/user/61916/698034/visualdl

## 模型选择与参数配置

1. 模型选择：根据自己的需求选择合适的模型进行训练。本文选择HRNet-W18作为训练模型
1. 预训练模型：pretrained_model/download_model.py中提供了相应的预训练模型下载地址，可以根据自己的需求在其中寻找相应的预训练模型，如不存在，可以按照同样的格式添加对应的模型名称与下载地址。
1. 参数配置：根据选择的模型修改相应的模型配置文件
1. 配置校验：在开始训练和评估之前，对配置和数据进行一次校验，确保数据和配置是正确的。使用下述命令下载预训练模型核启动校验流程：

```buildoutcfg
python pretrained_model/download_model.py hrnet_w18_bn_cityscapes
python pdseg/check.py --cfg ./configs/hrnet_optic.yaml
```

## 常用参数配置详细说明

`TRAIN.PRETRAINED_MODEL_DIR` 指定预训练模型路径

`MODEL.DEFAULT_NORM_TYPE` 指定norm的类型，此处提供bn和gn（默认）两种选择，分别指batch norm和group norm。

`BATCH_SIZE` 批处理大小

`--use_mpio` 是否开启多进程

`DATASET.NUM_CLASSES` 类别数（包括背景类别）

`TRAIN_CROP_SIZE` 训练时图像裁剪尺寸（宽，高）

`TRAIN.MODEL_SAVE_DIR` 模型保存路径

`TRAIN.SYNC_BATCH_NORM` 是否使用多卡间同步BatchNorm均值和方差，默认False

`SOLVER.LR` 初始学习率

`SOLVER.NUM_EPOCHS` 训练epoch数，正整数

`SOLVER.LR_POLICY` 学习率下降方法, 选项为poly、piecewise和cosine

`SOLVER.OPTIMIZER` 优化算法, 选项为sgd和adam

`MODEL.DEEPLAB.OUTPUT_STRIDE` DeepLabv3+网络中output stride设置，取值为16（默认）或8。取值为16时，网络规模较小，速度较快，当取8时，网络规模较大，精度较高，可根据实际需求进行选取。

`TEST.TEST_MODEL` 为测试模型路径

`EVAL_CROP_SIZE` 验证时图像裁剪尺寸（宽，高），具体的取值要求分如下情况：

- 当`AUG.AUG_METHOD`为unpadding时，`EVAL_CROP_SIZE`的宽高应不小于`AUG.FIX_RESIZE_SIZE`的宽和高。

- 当`AUG.AUG_METHOD`为stepscaling时，`EVAL_CROP_SIZE`的宽高应不小于原图中最大的宽和高。

- 当`AUG.AUG_METHOD`为rangscaling时，`EVAL_CROP_SIZE`的宽高应不小于缩放后图像中最大的宽和高。

## 开始训练

运行PaddleSeg/pdseg/train.py 可以直接训练模型：

* --cfg是yaml文件配置参数，许多参数都在相应的yaml文件中进行了配置，在configs有一些公开数据集的yaml文件。
* --use_gpu是指开启GPU进行训练，若是在gpu上运行，请开启参数--use_gpu。

**需要指出的是，关于参数的文件有两个，一个是在pdseg/utils/configs.py文件中，一个是通过--cfg传入的yaml文件，其中yaml文件参数配置优先级大于config.py文件。configs.py中对各个参数的含义做了明确的说明**

如果想要改变configs.py参数的配置，一种做法是设计一个yaml文件传给--cfg，另一种做法是在命令行直接对相应参数赋值，前者适合需要长期作出改变的情况，如更换数据集，后者适合临时对一些参数进行更改。
## 命令行FLAGS

[PaddleSeg各参数设置说明文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.5.0/docs/config.md#%E5%91%BD%E4%BB%A4%E8%A1%8Cflags)

|FLAG|用途|支持脚本|默认值|备注|
|-|-|-|-|-|
|--cfg|配置文件路径|ALL|None||
|--use_gpu|是否使用GPU进行训练|train/eval/vis|False||
|--use_mpio|是否使用多进程进行IO处理|train/eval|False|打开该开关会占用一定量的CPU内存，但是可以提高训练速度。</br> **NOTE：** windows平台下不支持该功能, 建议使用自定义数据初次训练时不打开，打开会导致数据读取异常不可见。 |
|--use_vdl|是否使用VisualDL记录训练数据|train|False||
|--log_steps|训练日志的打印周期（单位为step）|train|10||
|--debug|是否打印debug信息|train|False|IOU等指标涉及到混淆矩阵的计算，会降低训练速度|
|--vdl_log_dir &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|VisualDL的日志路径|train|None||
|--do_eval|是否在保存模型时进行效果评估   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|train|False||
|--vis_dir|保存可视化图片的路径|vis|"visual"||
## 模型评估
```buildoutcfg
python pdseg/train.py --use_gpu --cfg ./configs/hrnet_optic.yaml
```

## 结果可视化
```buildoutcfg
python pdseg/eval.py --use_gpu --cfg ./configs/hrnet_optic.yaml
```


## 选择MobileNet为DeepLabv3+的BackBone

在DeepLabv3+中，提供了4种backbone类型，分别是xception_41、xception_65（默认）、xception_71和mobilenetv2， 通过`MODEL.DEEPLAB.BACKBONE`参数指定。其中xception系列精度较高，但速度慢，mobilenetv2在精度方面会有损失，但速度则要快很多。

在精度与速度的选取方面，可通过设置DeplabV3+中是否保留aspp和decoder进行取舍。参数设置分别为：`MODEL.DEEPLAB.ENCODER_WITH_ASPP`，`MODEL.DEEPLAB.ENABLE_DECODER`，默认为True，当你更偏向于速度时，可设置为False.

## 下载预训练模型
考虑到检测的实时性等实际问题，选择MobileNet作为DeepLabv3+的BackBone，`MODEL.DEEPLAB.ENCODER_WITH_ASPP`，`MODEL.DEEPLAB.ENABLE_DECODER`，设置为False

```buildoutcfg
python pretrained_model/download_model.py deeplabv3p_mobilenetv2-1-0_bn_coco
```

## 配置校验

```buildoutcfg
python pdseg/train.py --use_gpu --cfg ./configs/deeplabv3p_mobilenetv2_cityscapes.yaml
```

## 模型评估

```buildoutcfg
python pdseg/eval.py --use_gpu --cfg ./configs/deeplabv3p_mobilenetv2_cityscapes.yaml
```

## 结果可视化

```buildoutcfg
python pdseg/vis.py --use_gpu --cfg ./configs/deeplabv3p_mobilenetv2_cityscapes.yaml
```
