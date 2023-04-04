from StyleTransfer import *
from PIL import Image


"""
图像风格迁移
"""


if __name__ == '__main__':
    content_img_path = './img/mine.jpg'
    style_img_path = 'img/style2.jpg'
    content_img = Image.open(content_img_path)
    style_img = Image.open(style_img_path)
    StyleTransfer(content_img, style_img, save_path='./output/', file_name='output.jpg',epochs_num=1000)
