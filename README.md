# 实时文本检测

## 背景
    - pse这种基于pixel的文本检测方法在追求准确性的前提下对于FPS的性能并不能达到很好的要求
    - 为了在保证一定准确性的同时并且保有一定的fps性能故进行本项目的研究与开发

## 分析
    - pixel-based方法,采用文本像素分割,具有很高的准确率，但对于小文本由于像素特征稀疏会有召回率低的问题,同时性能不太好
    - anchor-based方法,对于文本尺寸不敏感，有高的召回率但密集大角度的文本面临"anchor"匹配困难的问题，准确率低于基于像素的方法，但速度快
    - textboxes, textboxes++, seglink以及east对长文本极差; ctpn对于倾斜文本不行并且性能不好
    - pixel-link和PSENet这些基于像素link的算法一个是粘连问题，一个是性能比较差
    - pixel-anchor比较好的结合了两种算法的优势,小文本以及长的anchors由anchor-based方法进行；pixel-based方法删除小文本
      并且pixel-base提取的特征作为anchor-baesed的注意力机制。但问题是云从科技的方法并未被CVPR接收，因此未开源

## 思路
    - 本文结合pixel-anchor的思路以及psenet的启发

## 参考
    - pse
    - 其他github上相关二值化目标检测的一些项目, 这里不具体罗列了

## 安装

### 分离卷积
#### 未使用
    - 未找到合适的实现方式, tensorflow不提供, 而试验下来的包括自己写的都有些问题
    - 实际上根据一些二值化的目标检测论文, 改方法能够有效的提升检测精度, 可以考虑未来的改进思路

### 依赖库
    - pip install -r ./doc/requirements.txt -i https://pypi.douban.com/simple/



### 配置
#### 文件位置
    - config/config.py
        - cfg.TRAIN.IMG_DIR 训练数据图片目录
        - cfg.TRAIN.LABEL_DIR 训练数据标注目录
        - cfg.TRAIN.RESTORE　是否在上一个版本模型基础上继续训练
        - cfg.TRAIN.RESTORE_CKPT_PATH 上个版本预训练模型目录
        - cfg.TRAIN.PRETRAINED_MODEL_PATH 上个版本预训练模型文件名字
        - cfg.TRAIN.LEARNING_RATE 学习率
        - cfg.TRAIN.AUG_TOOL　重要的数据增强工具选项
            - 根据经验选取:
                - GaussianBlur 高斯模糊
                - BilateralBlur 水纹
                - ElasticTransformation 像素变换
                - PerspectiveTransform 透视变换
                - Affine_rot 角度旋转
                - Affine_scale 放缩
                - Fliplr 水平翻转
                - Pad 补边


### 数据说明
#### 数据格式
    - images: 目录用于存放图片数据
    - txt: 目录用于存放标注
        - 格式: 左上、右上、右下、左下, 语言, 文字

### VOC转ICDAR
    - voc2icdar.py
        - 根据情况修改代码, 并不通用


### 训练
    - 执行train.py即可


### 推理
    - 执行inference.py


### 转pb
    - 执行freeze_graph.py

### 数据增强
    - 使用python imgaug包进行数据增强
        - 具体效果可以通过执行test_data_aug.py进行查看
        - 对应的实际项目执行代码在utils/dataset/img_aug.py中




