import paddle.nn as nn
from densenet import DenseNet121


class CheXNet(nn.Layer):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(CheXNet, self).__init__()
        self.densenet121 = DenseNet121(pretrained=True)
        num_ftrs = self.densenet121.num_features
        self.densenet121.out = nn.Sequential(nn.Linear(num_ftrs, out_size),
                                             nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x
