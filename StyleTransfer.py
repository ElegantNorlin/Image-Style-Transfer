import numpy as np
import torch
import torchvision
from torch import nn


def preprocess(img, image_shape, rgb_params):
    """
    对原始图像进行处理，将图像变为可训练的tensor
    :param img: 原始图像
    :param image_shape: 处理之后的图像尺寸
    :param rgb_params: rgb均值 方差
    :return: 处理之后的图像
    """
    rgb_mean, rgb_std = rgb_params
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),  # resize
        torchvision.transforms.ToTensor(),  # 转tensor
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])  # 标准化
    return transforms(img).unsqueeze(0)  # 增加一个维度


def postprocess(img, rgb_params):
    """
    将处理完的图像处理回去，将tensor变为图像
    :param img: 处理完的图像
    :param rgb_params: rgb均值 方差
    :return: 处理回去的图像，PIL Image型对象
    """
    rgb_mean, rgb_std = rgb_params
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))


def extract_features(net, X, layers):
    """
    提取内容图像和样式图像的特征
    :param net: 网络结构
    :param X: 内容图像tensor
    :param layers: 特征提取层序号
    :return: 特征
    """
    features = []
    for i in range(len(net)):
        X = net[i](X)
        if i in layers:
            features.append(X)
    return features


def get_contents(net, content_img, content_layers, image_shape, GPU, rgb_params):
    """
    对内容图像进行处理
    :param net: 网络结构
    :param content_img: 内容图像
    :param content_layers: 内容特征提取层
    :param image_shape: 图像尺寸
    :param GPU: 是否能用GPU进行训练
    :param rgb_params: rgb均值 方差
    :return: 样式图像tensor；样式图像的特征
    """
    content_X = preprocess(content_img, image_shape, rgb_params)
    if GPU:
        content_X = content_X.to('cuda')
    contents_Y = extract_features(net, content_X, content_layers)
    return content_X, contents_Y


def get_styles(net, style_img, style_layers, image_shape, GPU, rgb_params):
    """
    对样式图像进行处理
    :param net: 网络结构
    :param style_img: 风格图像
    :param style_layers: 风格特征提取层
    :param image_shape: 图像尺寸
    :param GPU: 是否能用GPU进行训练
    :param rgb_params: rgb均值 方差
    :return: 样式图像tensor；样式图像的特征
    """
    style_X = preprocess(style_img, image_shape, rgb_params)
    if GPU:
        style_X = style_X.to('cuda')
    styles_Y = extract_features(net, style_X, style_layers)
    return style_X, styles_Y


def content_loss(Y_hat, Y):
    """
    计算内容损失（均方误差）
    :param Y_hat: 生成的图像
    :param Y: 内容图像
    :return: 内容损失tensor
    """
    return torch.square(Y_hat - Y.detach()).mean()


def gram(X):
    """
    二阶统计信息 协方差矩阵 来表示样式信息
    :param X: 图像
    :return: 图像的协方差矩阵
    """
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)


def style_loss(Y_hat, gram_Y):
    """
    计算样式损失
    :param Y_hat: 生成的图像
    :param gram_Y: 提前算好的样式图像的协方差矩阵
    :return: 样式损失tensor
    """
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()


def tv_loss(Y_hat):
    """
    计算噪音损失（表示图像是否平滑）
    :param Y_hat: 生成的图像
    :return: 噪音损失
    """
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram, weights):
    """
    计算总的损失
    :param X: 处理之后的内容图像
    :param contents_Y_hat: 估计出的内容图像特征
    :param styles_Y_hat: 样式图像特征
    :param contents_Y: 内容图像特征
    :param styles_Y_gram: 样式信息
    :return: 各个损失分量和总的损失
    """
    # 分别计算内容损失、样式损失和总变差损失
    content_weight, style_weight, tv_weight = weights
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


class SynthesizedImage(nn.Module):
    # 要训练图像，用这个类来把图像的所有像素值当作权重进行跟踪
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


def get_inits(X, styles_Y, GPU, lr):
    """
    初始化输出图像的初始解
    :param X: 处理之后的内容图像
    :param styles_Y: 样式图像的特征
    :param GPU: 是否能用GPU进行训练
    :param lr: 学习率
    :return: 初始解图像；样式图像的协方差矩阵；优化器
    """
    gen_img = SynthesizedImage(X.shape)
    if GPU:
        gen_img = gen_img.to('cuda')
    gen_img.weight.data.copy_(X.data)
    optim = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, optim


def train(net, X, contents_Y, styles_Y, layers, params):
    """
    训练
    :param net: 网络结构
    :param X: 内容图像tensor
    :param contents_Y: 内容图像的特征
    :param styles_Y: 样式图像的特征
    :param layers: (内容提取层，风格提取层)
    :param params: (GPU，学习率，epochs数，学习率下降epochs数，各种损失的权重)
    :return: 样式迁移之后的图像（需要变换回PIL Image型对象）
    """
    GPU, lr, epochs_num, lr_decay_epoch, weights = params
    content_layers, style_layers = layers
    X, styles_Y_gram, optim = get_inits(X, styles_Y, GPU, lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_decay_epoch, 0.8)
    for epoch in range(epochs_num):
        optim.zero_grad()
        contents_Y_hat = extract_features(net, X, content_layers)
        styles_Y_hat = extract_features(net, X, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram, weights=weights)

        if epoch % 5 == 0:
            print('epoch: ' + str(epoch) +
                  '  contents loss: ' + str(round(sum(contents_l).item(), 3)) +
                  '  styles loss: ' + str(round(sum(styles_l).item(), 3)) +
                  '  tv loss: ' + str(round(tv_l.item(), 3)) +
                  '  loss: ' + str(round(l.item(), 3)))

        l.backward()
        optim.step()
        scheduler.step()
    return X


def StyleTransfer(content_img, style_img, save_path, file_name, epochs_num=100):
    GPU = False
    if torch.cuda.is_available():
        GPU = True
        print("通过GPU训练...")

    print("通过CPU训练...")

    content_img_tensor = torch.from_numpy(np.array(content_img))
    image_shape = (content_img_tensor.shape[0], content_img_tensor.shape[1])

    while image_shape[0] * image_shape[1] > 1e5:
        content_img = content_img.resize((int(0.9 * image_shape[1]), int(0.9 * image_shape[0])))
        content_img_tensor = torch.from_numpy(np.array(content_img))
        image_shape = (content_img_tensor.shape[0], content_img_tensor.shape[1])

    style_img = style_img.resize((image_shape[1], image_shape[0]))
    style_image_shape = (content_img_tensor.shape[0], content_img_tensor.shape[1])
    print('content picture size: ' + str(image_shape))
    print('style picture size: ' + str(style_image_shape))

    rgb_mean = torch.tensor([0.485, 0.456, 0.406])
    rgb_std = torch.tensor([0.229, 0.224, 0.225])

    pretrained_net = torchvision.models.vgg19(pretrained=True)

    style_layers, content_layers = [0, 2, 2, 2, 2, 5, 5, 5, 7, 7, 12, 15, 28, 34], [25]  # 2, 5, 7, 10, 12, 14, 34
    net = nn.Sequential(*[pretrained_net.features[i] for i in range(max(content_layers + style_layers) + 1)])
    if GPU:
        net = net.to('cuda')

    content_weight, style_weight, tv_weight = 1, 500, 0
    weights = content_weight, style_weight, tv_weight
    lr_decay_epoch = 50
    lr = 0.5

    content_X, contents_Y = get_contents(net, content_img, content_layers, image_shape, GPU, rgb_params=(rgb_mean, rgb_std))
    _, styles_Y = get_styles(net, style_img, style_layers, image_shape, GPU, rgb_params=(rgb_mean, rgb_std))
    output = train(net, content_X, contents_Y, styles_Y, (content_layers, style_layers), (GPU, lr, epochs_num, lr_decay_epoch, weights))
    output = postprocess(output, rgb_params=(rgb_mean, rgb_std))

    output.save(save_path + file_name)
