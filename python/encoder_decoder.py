# CREATE ENCODERS AND DECODERS
import torch
import torch.nn as nn
import models.models as models

class Encoder(nn.Module):

    def __init__(self, depth):
        super(Encoder, self).__init__()

        assert(type(depth).__name__ == 'int' and 1 <= depth <= 5)
        self.depth = depth

        if depth == 1:
            self.model = models.vgg_normalised_conv1_1
            self.model.load_state_dict(torch.load("models/params/vgg_normalised_conv1_1.pth"))
        elif depth == 2:
            self.model = models.vgg_normalised_conv2_1
            self.model.load_state_dict(torch.load("models/params/vgg_normalised_conv2_1.pth"))
        elif depth == 3:
            self.model = models.vgg_normalised_conv3_1
            self.model.load_state_dict(torch.load("models/params/vgg_normalised_conv3_1.pth"))
        elif depth == 4:
            self.model = models.vgg_normalised_conv4_1
            self.model.load_state_dict(torch.load("models/params/vgg_normalised_conv4_1.pth"))
        elif depth == 5:
            self.model = models.vgg_normalised_conv5_1
            self.model.load_state_dict(torch.load("models/params/vgg_normalised_conv5_1.pth"))


    def forward(self, x):
        out = self.model(x)
        return out


class Decoder(nn.Module):
    def __init__(self, depth):
        super(Decoder, self).__init__()

        assert (type(depth).__name__ == 'int' and 1 <= depth <= 5)
        self.depth = depth

        if depth == 1:
            self.model = models.feature_invertor_conv1_1
            self.model.load_state_dict(torch.load("models/params/feature_invertor_conv1_1.pth"))
        elif depth == 2:
            self.model = models.feature_invertor_conv2_1
            self.model.load_state_dict(torch.load("models/params/feature_invertor_conv2_1.pth"))
        elif depth == 3:
            self.model = models.feature_invertor_conv3_1
            self.model.load_state_dict(torch.load("models/params/feature_invertor_conv3_1.pth"))
        elif depth == 4:
            self.model = models.feature_invertor_conv4_1
            self.model.load_state_dict(torch.load("models/params/feature_invertor_conv4_1.pth"))
        elif depth == 5:
            self.model = models.feature_invertor_conv5_1
            self.model.load_state_dict(torch.load("models/params/feature_invertor_conv5_1.pth"))

    def forward(self, x):
        out = self.model(x)
        return out