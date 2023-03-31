from PIL import Image
import numpy as np

# 图片路径
img_path = r"./img/style.jpg"
# 保存图片路径
save_path = r"./img/save.jpg"
img = Image.open(img_path)
temp_img = np.array(img).transpose(2, 0, 1)
img2 = None
if (temp_img.shape[0] == 4):
    print("输入图片为4通道图片，接下来将进行RGBA to RGB转换")
    img2 = img.convert("RGB")
    img2.save(save_path)
    print("RGBA to RGB 转换已完成")
elif (temp_img.shape[0] == 3):
    print("输入图片为RGB三通道，无需进行转换")
else:
    print("请问您输入的是什么怪物？")

del img,temp_img,img2