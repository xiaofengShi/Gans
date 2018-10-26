import torchvision.models as models
from networks import ResnetGenerator,UnetGenerator,UNET
from torch.autograd import Variable
import torch as t
import torch.nn as nn


densenet = models.resnet152(pretrained=False)
resnet=ResnetGenerator(3,3)
# params = densenet.state_dict()
# for k,v in params.items():
#     print(k)    
     
# print('*'*100)
# paramsres = resnet.state_dict()
# for k,v in paramsres.items():
#     print(k)  


# unet = UnetGenerator(3, 3, 7, 64, nn.InstanceNorm2d, True)
# unet = UNET()
# input = Variable(t.randn(1, 3, 128, 128))
# out = unet(input)
# print('out:',out.size())
# for name, para in unet.named_parameters():
#     print(name,':',para.size())
