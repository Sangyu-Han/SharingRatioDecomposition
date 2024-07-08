import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from modules.layers import *
from util import *
from collections import defaultdict
from tqdm import tqdm
from torchray.attribution.grad_cam import grad_cam
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model = torchvision.models.vgg16(pretrained=True)
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True,name=''):
        super(VGG, self).__init__()
        self.name = name
        self.features = features
        self.avgpool = AdaptiveAvgPool2d((7, 7),name=name + '.avgpool')
        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(),
            Dropout(),
            Linear(4096, 4096),
            ReLU(),
            Dropout(),
            Linear(4096, num_classes), name=name + '.classifier'
        )
        self.ideal_feature_vector_maxpool = defaultdict(lambda : None)
        self.ideal_feature_vector_linear = defaultdict(lambda : None)
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x,inference_pkg):
        x = self.features(x,inference_pkg)
        inference_pkg.output_dict[self.name + '.features'] = x
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x,inference_pkg)
        return x


    def get_feature_contribution_true(self,out_f,target_out,inference_pkg):
        """
        feature[29] relu (14,14) point-wise feature vector가 원하는 클래스에 대한 각각의 기여 구하는 거
        """
        maxpool_final = self.features[-1]
        linear = self.classifier[0]
        relu = self.classifier[1]
        cos = nn.CosineSimilarity(dim=3, eps=1e-12)
        n,c_o,h_o,w_o = out_f.shape
        total_simmap = torch.zeros((n,h_o,w_o),device=inference_pkg.device)
        last_out = target_out
        
        h_mp = self.features[-1].kernel_size
        w_mp = self.features[-1].kernel_size
        for _h_mp in range(h_mp):
            for _w_mp in range(w_mp):
                stimulus_maxpool = torch.zeros_like(out_f,device=inference_pkg.device)
                stimulus_maxpool[:,:,_h_mp::h_mp,_w_mp::w_mp] = out_f[:,:,_h_mp::h_mp,_w_mp::w_mp]

                out_maxpool = maxpool_final.decomp_forward(stimulus_maxpool,inference_pkg)
                n,c,h,w = out_maxpool.shape
                stimulus_linear = torch.zeros((n,h,w,c,h,w),device = inference_pkg.device)
                for _h in range(h):
                    for _w in range(w):
                        stimulus_linear[:,_h,_w,:,_h,_w] = out_maxpool[:,:,_h,_w]
                stimulus_linear = stimulus_linear.view(n,h,w,-1)
                out = linear.decomp_forward(stimulus_linear,inference_pkg,bias_shrink=h_mp*w_mp*h*w) #out.shape = (N,H,W,1000). last_out.shape = (N,1000)


                last_out = last_out.view(n,1,1,4096)
                
                    #calculate simmap for given position
                simmap = cos(out,last_out) * torch.norm(out,dim=3) / (torch.norm(last_out,dim=3) + 1e-12)
                total_simmap[:,_h_mp::h_mp,_w_mp::w_mp] = simmap
        
        return total_simmap

    def get_pixel_cont(self,input,y,inference_pkg,contrastive_rule = lambda x:(x - x.mean()),per_channel=False):
        if inference_pkg.remember_input_graph == True:
            return self._get_pixel_cont_simple(input,inference_pkg,contrastive_rule = contrastive_rule,idx=y,per_channel=per_channel)
        else:
            with torch.no_grad():
                return self._get_pixel_cont_simple(input,inference_pkg,contrastive_rule = contrastive_rule,idx=y,per_channel=per_channel)


    def get_pixel_cont_simple(self,input,inference_pkg,contrastive_rule = lambda x:(x - x.mean()),idx=None,per_channel=False):
        if inference_pkg.remember_input_graph == True:
            return self._get_pixel_cont_simple(input,inference_pkg,contrastive_rule = contrastive_rule,idx=idx,per_channel=per_channel)
        else:
            with torch.no_grad():
                return self._get_pixel_cont_simple(input,inference_pkg,contrastive_rule = contrastive_rule,idx=idx,per_channel=per_channel)

    def _get_pixel_cont_simple(self,input,inference_pkg,contrastive_rule = lambda x:(x - x.mean()),idx=None,per_channel=False):
        

       

        
        o = SimmapObj()
        o.cav_contrastive_rule = contrastive_rule
        out_f= inference_pkg.output_dict[self.features[-2].name].to(inference_pkg.device)
        last_out = inference_pkg.output_dict[self.classifier[0].name].clone().detach().to(inference_pkg.device)
        if idx == None:
            
            o.simmap = self.get_feature_contribution_true(out_f,last_out,inference_pkg)
        else:
            idx = int(idx)
            _input= input.clone().detach()
            _input.requires_grad = True
            with torch.autograd.set_grad_enabled(True):
                simmap_contrast = grad_cam(model.to(inference_pkg.device), _input, idx, saliency_layer='features.28').squeeze(0)
            o.simmap = simmap_contrast

        inference_pkg.simmaps[self.features[-2].name] = o.simmap.clone().cpu().detach()
        o.simmap = o.cav_contrastive_rule(o.simmap)

        #simprop2 start from features.29
        layer_list = []
        for i in range(len(self.features)):
            if i < len(self.features)-1: # exclude last maxpool layer, since it is not a activation layer
                layer_list.append(self.features[i])
        features_dummy = Sequential(*layer_list,name=self.name + '.features')

        o = features_dummy.simprop2(o,inference_pkg)
        if per_channel:
            simmap = cal_simmap_input_cr_per_neuron(input,o,inference_pkg)
        else:
            simmap = cal_simmap_cr(input,o,inference_pkg)


        return simmap
       

    def get_ERF(self,input,extern_simmap,inference_pkg,target_layer_name=None):
        """
        extern_simmap : target layer의 H,W를 가진 (N,H,W) simmap. 
        
        """

        if target_layer_name == None:
            raise 
        else:
            o = SimmapObj()
            o.target_layer_name = target_layer_name
            o.external_simmap =extern_simmap
            o.simmap = torch.zeros((1,14,14),device=inference_pkg.device) 
            layer_list = []
        for i in range(len(self.features)):
            if i < len(self.features)-1: 
                layer_list.append(self.features[i])
        features_dummy = Sequential(*layer_list,name=self.name + '.features')

        o = features_dummy.simprop2(o,inference_pkg)
        simmap = cal_simmap_cr(input,o,inference_pkg)
        return simmap


def make_layers(cfg, batch_norm=False,name=''):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [MaxPool2d(kernel_size=2, stride=2,return_indices=True)]
        else:
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), ReLU(inplace=True)]
            else:
                layers += [conv2d, ReLU(inplace=True)]
            in_channels = v
    return Sequential(*layers,name=name + '.features')


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False,name = 'vgg11', **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'],name=name),name=name,**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False,name='vgg13', **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'],name=name),name=name,**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True),name=name, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False,name='vgg16', **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'],name=name),name=name,**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))

    return model


def vgg16_bn(pretrained=False,name='vgg16_bn', **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True,name=name),name=name, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False,name='vgg19', **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'],name=name),name=name,**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model
