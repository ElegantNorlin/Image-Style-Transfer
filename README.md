[toc]

## 项目简介

Image-Style-Transfer

基于Pytorch框架的图像风格迁移项目

该项目基于https://github.com/YaoShunyu19/style-transfer-pytorch

## 项目结构说明

```txt
├── .DS_Store-mac os系统文件，忽略或者删除即可
├── README.md-项目的介绍文件
├── StyleTransfer.py-风格迁移算法
├── img-存放风格参照、风格化图片
├── output-存放输出图片
├── main.py-项目的入口文件
├── requirements.txt-记载了项目所需的库以及版本
└── rgba to rgb.py-将四维向量的图片转换为三维向量
```

## 如何运行该项目？

项目运行之前需要先修改几个参数：

打开`main.py`

修改`content_img_path`和`style_img_path`。前者为你要进行风格化的图片；后者为风格照片（风格参照）。

默认输出路径为`./output/`

默认输出图片名称为`output.jpg`

### 1.克隆该项目到本地

```shell
git clone git@github.com:ElegantNorlin/Image-Style-Transfer.git
```

### 2.安装所需项目依赖

```shell
pip install -r requirements.txt
```

### 3.运行main.py文件

```shell
python main.py
```

## 示例

<div id="container" style="width:500px">
    <img src="https://github.com/ElegantNorlin/Image-Style-Transfer/blob/main/img/content.jpg?raw=true" alt="cat&dog"></img>
<img src="https://github.com/ElegantNorlin/Image-Style-Transfer/blob/main/output/output.jpg?raw=true" alt="cat&dog" width="600"></img>
</div>
