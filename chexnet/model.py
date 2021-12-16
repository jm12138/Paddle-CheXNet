import paddle
import paddle.nn as nn
from .densenet import DenseNet121


class CheXNet(nn.Layer):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, out_size, backbone_pretrained=True):
        super(CheXNet, self).__init__()
        self.densenet121 = DenseNet121(pretrained=backbone_pretrained)
        num_ftrs = self.densenet121.num_features
        self.densenet121.out = nn.Sequential(nn.Linear(num_ftrs, out_size),
                                             nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x


class CheXNetCAM(nn.Layer):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, out_size, backbone_pretrained=True):
        super(CheXNetCAM, self).__init__()
        self.out_size = out_size
        self.densenet121 = DenseNet121(pretrained=backbone_pretrained)
        num_ftrs = self.densenet121.num_features
        self.densenet121.batch_norm.register_forward_post_hook(
            self.hook_feature)
        self.densenet121.out = nn.Sequential(nn.Linear(num_ftrs, out_size),
                                             nn.Sigmoid())

    def forward(self, x):
        size = x.shape[2:]
        x = self.densenet121(x)
        cams = self.cam(self.feature, self.densenet121.out[0].weight, size)
        return x, cams

    @staticmethod
    @paddle.no_grad()
    def cam(feature_conv, weight_softmax, size):
        cams = paddle.einsum('bchw, cn -> bnhw', feature_conv, weight_softmax)
        cams = cams - paddle.min(cams, axis=[2, 3], keepdim=True)
        cams_img = cams / paddle.max(cams, axis=[2, 3], keepdim=True)
        cams_img = nn.functional.upsample(cams_img, size, mode='bilinear')
        cams_img = (255 * cams_img).cast(paddle.uint8)
        return cams_img

    def hook_feature(self, layer, input, output):
        self.feature = output
