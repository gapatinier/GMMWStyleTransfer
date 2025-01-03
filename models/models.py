# MODEL ARCHITECTURES FOR VGG ENCODER/DECODER
import torch.nn as nn

vgg_normalised_conv1_1 = nn.Sequential( # Sequential,
	nn.Conv2d(3,3,(1, 1)),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(3,64,(3, 3)),
	nn.ReLU(),
)

feature_invertor_conv1_1 = nn.Sequential( # Sequential,
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,3,(3, 3)),
)

vgg_normalised_conv2_1 = nn.Sequential( # Sequential,
	nn.Conv2d(3,3,(1, 1)),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(3,64,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,64,(3, 3)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,128,(3, 3)),
	nn.ReLU(),
)

feature_invertor_conv2_1 = nn.Sequential( # Sequential,
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,64,(3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,64,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,3,(3, 3)),
)

vgg_normalised_conv3_1 = nn.Sequential( # Sequential,
	nn.Conv2d(3,3,(1, 1)),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(3,64,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,64,(3, 3)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,128,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,128,(3, 3)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,256,(3, 3)),
	nn.ReLU(),
)

feature_invertor_conv3_1 = nn.Sequential( # Sequential,
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,128,(3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,128,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,64,(3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,64,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,3,(3, 3)),
)

vgg_normalised_conv4_1 = nn.Sequential( # Sequential,
	nn.Conv2d(3,3,(1, 1)),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(3,64,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,64,(3, 3)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,128,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,128,(3, 3)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,512,(3, 3)),
	nn.ReLU(),
)

feature_invertor_conv4_1 = nn.Sequential( # Sequential,
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,256,(3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,128,(3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,128,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,64,(3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,64,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,3,(3, 3)),
)

vgg_normalised_conv5_1 = nn.Sequential( # Sequential,
	nn.Conv2d(3,3,(1, 1)),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(3,64,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,64,(3, 3)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,128,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,128,(3, 3)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,512,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
)

feature_invertor_conv5_1 = nn.Sequential( # Sequential,
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,256,(3, 3)),
	nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,128,(3, 3)),
	nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,128,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,64,(3, 3)),
	nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,64,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,3,(3, 3)),
)