import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from modules.layers import *
from util import *

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', #org
    #'resnet50' : 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',# v2
    #'resnet50': "https://download.pytorch.org/models/resnet50-0676ba61.pth", # V1
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1,name=''):
    """3x3 convolution with padding"""
    name = name
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False,name=name)


def conv1x1(in_planes, out_planes, stride=1,name=''):
    """1x1 convolution"""
    name = name
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,name=name)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, name = ''):
        super(BasicBlock, self).__init__()
        self.name = name
        

        self.conv1 = conv3x3(inplanes, planes, stride,name = self.name + '.conv1')
        self.bn1 = BatchNorm2d(planes, name = self.name + '.bn1')
        self.conv2 = conv3x3(planes, planes, name= self.name + '.conv2')
        self.bn2 = BatchNorm2d(planes, name= self.name + '.bn2')
        self.downsample = downsample
        self.stride = stride

        self.relu1 = ReLU(inplace=True,name= self.name + '.relu1')
        self.relu2 = ReLU(inplace=True,name= self.name + '.relu2')

        self.add = Add()


    def forward(self, x,inference_pkg):
        if inference_pkg.remember_input_graph:
            inference_pkg.input_dict[self.name] = x.clone() # must remember what the block input was
        else:
            inference_pkg.input_dict[self.name] = x.clone().detach().cpu() 
        x1 = x
        x2 = x.clone()

        out = self.conv1(x1,inference_pkg)
        out = self.bn1(out,inference_pkg)
        out = self.relu1(out,inference_pkg)

        out = self.conv2(out,inference_pkg)
        out = self.bn2(out,inference_pkg)

        if self.downsample is not None:
            x2 = self.downsample(x2,inference_pkg)

        out = self.add.forward([out, x2],inference_pkg)
        out = self.relu2(out,inference_pkg)
        if inference_pkg.remember_input_graph:
            inference_pkg.output_dict[self.name] = out.clone() # must remember what the block input was
        else:
            inference_pkg.output_dict[self.name] = out.clone().detach().cpu() 
        return out



    def simprop2(self,o,inference_pkg):
        
        o = self.relu2.simprop2(o,inference_pkg)
        o1,o2 = self.add.simprop2(o,inference_pkg)

        if self.downsample is not None:
            o2 = self.downsample.simprop2(o2,inference_pkg)
        
        o1 = self.bn2.simprop2(o1,inference_pkg)
        o1 = self.conv2.simprop2(o1,inference_pkg)
        o1 = self.relu1.simprop2(o1,inference_pkg)

        o1 = self.bn1.simprop2(o1,inference_pkg)
        o1 = self.conv1.simprop2(o1,inference_pkg)
        
        block_input = inference_pkg.input_dict[self.name].to(inference_pkg.device)
        o1 = o1.simprop2_input(block_input,o1,inference_pkg)
        o2 = o2.simprop2_input(block_input,o2,inference_pkg)

        o1.simmap = o1.simmap + o2.simmap
        inference_pkg.simmaps[self.name] = o1.simmap.clone().detach().cpu()
        return o1

    
    
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, name = ''):
        super(Bottleneck, self).__init__()
        self.name = name

        self.conv1 = conv1x1(inplanes, planes,name= self.name + '.conv1')
        self.bn1 = BatchNorm2d(planes,name= self.name + '.bn1')
        self.conv2 = conv3x3(planes, planes, stride,name= self.name + '.conv2')
        self.bn2 = BatchNorm2d(planes,name= self.name + '.bn2')
        self.conv3 = conv1x1(planes, planes * self.expansion,name= self.name + '.conv3')
        self.bn3 = BatchNorm2d(planes * self.expansion,name= self.name + '.bn3')
        self.downsample = downsample
        self.stride = stride

        self.relu1 = ReLU(inplace=True,name= self.name + '.relu1')
        self.relu2 = ReLU(inplace=True,name= self.name + '.relu2')
        self.relu3 = ReLU(inplace=True,name= self.name + '.relu3')

        self.add = Add()


    def forward(self, x,inference_pkg):
        if inference_pkg.remember_input_graph:
            inference_pkg.input_dict[self.name] = x.clone() # must remember what the block input was
        else:
            inference_pkg.input_dict[self.name] = x.clone().detach().cpu() 
        out = self.conv1(x,inference_pkg)
        out = self.bn1(out,inference_pkg)
        out = self.relu1(out,inference_pkg)

        out = self.conv2(out,inference_pkg)
        out = self.bn2(out,inference_pkg)
        out = self.relu2(out,inference_pkg)

        out = self.conv3(out,inference_pkg)
        out = self.bn3(out,inference_pkg)

        if self.downsample is not None:
            x = self.downsample(x,inference_pkg)

        out = self.add.forward([out, x],inference_pkg)
        out = self.relu3(out,inference_pkg)
        if inference_pkg.remember_input_graph:
            inference_pkg.output_dict[self.name] = out.clone() # must remember what the block input was
        else:
            inference_pkg.output_dict[self.name] = out.clone().detach().cpu() 

        return out


    def simprop2(self,o,inference_pkg):

        block_input = inference_pkg.input_dict[self.name].to(inference_pkg.device)
        o = self.relu3.simprop2(o,inference_pkg)
        tot_simmap = o.simmap.clone()
        
        o1,o2 = self.add.simprop2(o,inference_pkg)
        if self.downsample is not None:
            _k = self.downsample[0].kernel_size
            _s = self.downsample[0].stride
            m = nn.Conv2d(1,1,_k,_s,bias=False).to(inference_pkg.device)
            torch.nn.init.ones_(m.weight)
            o2 = self.downsample.simprop2(o2,inference_pkg)
            
            o2 = o2.simprop2_input(block_input,o2,inference_pkg) # 


        else:
            o2 = o2.simprop2_input(block_input,o2,inference_pkg)
        o1 = self.bn3.simprop2(o1,inference_pkg)
        o1 = self.conv3.simprop2(o1,inference_pkg)
        o1 = self.relu2.simprop2(o1,inference_pkg) 

        o1 = self.bn2.simprop2(o1,inference_pkg)
        o1 = self.conv2.simprop2(o1,inference_pkg)
        o1 = self.relu1.simprop2(o1,inference_pkg)

        o1 = self.bn1.simprop2(o1,inference_pkg)
        o1 = self.conv1.simprop2(o1,inference_pkg)
        o1 = o1.simprop2_input(block_input,o1,inference_pkg)
        

        o1.simmap = o1.simmap + o2.simmap #
        inference_pkg.simmaps[self.name] = o1.simmap.clone().detach().cpu()
        if o.target_layer_name == self.name: # if we want to build ERF of specific hidden layer
            o1.simmap = o1.external_simmap
            return o1
        return o1

    def simprop2_layer1(self,o,inference_pkg,maxpool,relu):
        block_input = inference_pkg.input_dict[self.name].to(inference_pkg.device)
        o = self.relu3.simprop2(o,inference_pkg)
        tot_simmap = o.simmap.clone()
        
        o1,o2 = self.add.simprop2(o,inference_pkg)
        if self.downsample is not None:
            o2 = self.downsample.simprop2(o2,inference_pkg)
            o2 = maxpool.simprop2(o2,inference_pkg)
            o2 = relu.simprop2(o2,inference_pkg)


        else:
            pass

        o1 = self.bn3.simprop2(o1,inference_pkg)
        o1 = self.conv3.simprop2(o1,inference_pkg)
        o1 = self.relu2.simprop2(o1,inference_pkg)

        o1 = self.bn2.simprop2(o1,inference_pkg)
        o1 = self.conv2.simprop2(o1,inference_pkg)
        o1 = self.relu1.simprop2(o1,inference_pkg)

        o1 = self.bn1.simprop2(o1,inference_pkg)
        o1 = self.conv1.simprop2(o1,inference_pkg)
        o1 = maxpool.simprop2(o1,inference_pkg)
        o1 = relu.simprop2(o1,inference_pkg)
        
        
        o1.simmap = o1.simmap + o2.simmap
        inference_pkg.simmaps[relu.name] = o1.simmap.clone().detach().cpu()
        return o1



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,name=''):
        super(ResNet, self).__init__()
        
        
        self.is_decomposition = False
        self.name = name
        self.inplanes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False,name= self.name + '.conv1')
        self.bn1 = BatchNorm2d(64,name= self.name + '.bn1')
        self.relu = ReLU(inplace=True,name= self.name + '.relu')
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1,return_indices=True,name= self.name + '.maxpool')
        self.layer1 = self._make_layer(block, 64, layers[0],name= self.name + '.layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,name= self.name + '.layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,name= self.name + '.layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,name= self.name + '.layer4')
        self.avgpool = AdaptiveAvgPool2d((1, 1),name=self.name + '.avgpool')
        self.fc = Linear(512 * block.expansion, num_classes,name = self.name + '.fc')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1,name=''):
        name = name
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
                name = name + '.downsample'
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,name=name + '.0'))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,name=name + '.'+str(_)))

        return Sequential(*layers,name=name)

    def forward(self, x,inference_pkg):
        if inference_pkg.remember_input_graph:
            inference_pkg.input_dict['input'] = x.clone() # must remember what the block input was
        else:
            inference_pkg.input_dict['input'] = x.clone().detach().cpu() 
     
        x = self.conv1(x,inference_pkg)
        x = self.bn1(x,inference_pkg)
        x = self.relu(x,inference_pkg)
        x = self.maxpool(x,inference_pkg)

        x = self.layer1(x,inference_pkg)
        x = self.layer2(x,inference_pkg)
        x = self.layer3(x,inference_pkg)
        x = self.layer4(x,inference_pkg)

        x = self.avgpool(x,inference_pkg)
        x = x.view(x.size(0), -1)
        x = self.fc(x,inference_pkg)

        return x
    


    
    def get_pixel_cont(self,input,y,inference_pkg,contrastive_rule = lambda x:(x - x.mean()),rule = None):
        """
        y : vector. (N,C) or (N)
        """
        
        o = SimmapObj()
        if isinstance(y,int):
            N = input.shape[0]
            o.model_out = torch.zeros((N,1000)).to(inference_pkg.device)
            o.model_out[0,y] = 1
        elif isinstance(y,torch.Tensor):
            if len(y.shape) == 1:
                N = input.shape[0]
                o.model_out = torch.zeros((N,1000)).to(inference_pkg.device)
                o.model_out[:,y] = 1
            elif len(y.shape) == 2:
                N = input.shape[0]
                o.model_out = torch.zeros((N,1000)).to(inference_pkg.device)
                o.model_out[y] = 1



        o.cav_contrastive_rule = contrastive_rule
        simmap = self.simprop2(input,o,inference_pkg,rule=rule)
        return simmap

    def get_ERF(self,input,extern_simmap,inference_pkg,target_layer_name=None):
        """
        extern_simmap : (N,H,W) for target layer size. denote specifit H,W index as one. for example, if the layer output was (1,64,7,7), then the extern_simmap should be (1,7,7)
        and if want to find (5,6) coordinate ERF, then the extern_simmap should be (1,7,7) and the value of (5,6) should be 1, and the others should be 0.
        
        """

        if target_layer_name == None:
            raise 
        else:
            o = SimmapObj()
            o.target_layer_name = target_layer_name
            o.external_simmap =extern_simmap
            simmap = self.simprop2(input,o,inference_pkg)
        return simmap

    def simprop2(self,input,o,inference_pkg,rule = None):
        if inference_pkg.remember_input_graph:
            o = self.fc.simprop2(o,inference_pkg) 
            o = self.avgpool.simprop2(o,inference_pkg,rule = rule) 
            o = self.layer4.simprop2(o,inference_pkg)
            o = self.layer3.simprop2(o,inference_pkg)
            o = self.layer2.simprop2(o,inference_pkg)
            o = self.layer1.simprop2_layer1(o,inference_pkg,self.maxpool,self.relu)
            o = self.bn1.simprop2(o,inference_pkg)
            o = self.conv1.simprop2(o,inference_pkg)
            o = o.simprop2_input(input,o,inference_pkg)
        else:
            with torch.no_grad():
                o = self.fc.simprop2(o,inference_pkg)
                o = self.avgpool.simprop2(o,inference_pkg, rule = rule) 
                o = self.layer4.simprop2(o,inference_pkg)
                o = self.layer3.simprop2(o,inference_pkg)
                o = self.layer2.simprop2(o,inference_pkg)
                o = self.layer1.simprop2_layer1(o,inference_pkg,self.maxpool,self.relu)
                inference_pkg.simmaps[self.name + 'relu'] = o.simmap
                o = self.bn1.simprop2(o,inference_pkg)
                o = self.conv1.simprop2(o,inference_pkg)
                o = o.simprop2_input(input,o,inference_pkg)


        return o.simmap


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
