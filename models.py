import torch
import torch.nn as nn
import torch.nn.functional as F

from model.wide_resnet50_2 import wide_resnet50_2


class STPM(nn.Module):
    def __init__(self):
        super(STPM, self).__init__()

        self.init_features()

        def hook_t(module, input, output):
            self.features.append(output)

        self.model = wide_resnet50_2(pretrained=True)
        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
        self.pool = nn.AvgPool2d(3, 1, 1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)

        self.criterion = nn.MSELoss(reduction='sum')

    def init_features(self):
        self.features = []

    def embedding_concat(self, x, y):
        # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z

    def forward(self, x_t, flag=0):
        self.init_features()
        _ = self.model(x_t)
        # print("features: {}, {}".format(self.features[0].sum(), self.features[1].sum()))

        self.features[0] = self.pool(self.features[0])
        self.features[1] = self.pool(self.features[1])

        if flag == 0:
            b0, c0, h0, w0 = self.features[0].size()
            b1, c1, h1, w1 = self.features[1].size()
            self.features[1] = F.interpolate(self.features[1], (h0, w0), mode="bilinear", align_corners=False)
            return torch.cat([self.features[0], self.features[1]], dim=1)
        elif flag == 1:
            return self.embedding_concat(self.features[0], self.features[1])


def test():
    model = STPM()
    input_x = torch.ones([1,3,224,224])
    y = model(input_x, 1)
    print(input_x.sum())


if __name__ == "__main__":
    test()
