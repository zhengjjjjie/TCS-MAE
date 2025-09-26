"""
author: zhengj
"""

from typing import Optional, Union, List
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationHead,
    SegmentationModel
)
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.base import initialization as init
import torch.nn as nn

from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

class ProjectHead(nn.Sequential):
    def __init__(self, in_channels, out_dims, pooling='avg', dropout=None, activation=None):
        if pooling not in ('max', 'avg'):
            raise ValueError("pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        
        pool = nn.AdaptiveAvgPool2d(1) if pooling=='avg' else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout=nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, out_dims, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, linear, activation)

class UnetAE(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",  
        decoder_channels: List[int] = (512, 256, 128, 128, 128),
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,  
        aux_params: Optional[List[dict]] = None     
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=None,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None,
        )

        self.reconstruction_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.encoder_project_heads = nn.ModuleList()

        if aux_params is not None:
            #从最深层到最浅层
            for k, aux_param in enumerate(aux_params):
                self.encoder_project_heads.append(ProjectHead(in_channels=self.encoder.out_channels[-k-1], **aux_param))

            
        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.reconstruction_head)

    def forward(self, x):
        self.check_input_shape(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        recons = self.reconstruction_head(decoder_output)

        if len(self.encoder_project_heads)>0:
            encoder_project_outputs = []
            for k, project_head in enumerate(self.encoder_project_heads):
                k_embed = project_head(features[-k-1])
                encoder_project_outputs.append(k_embed)
            return recons, encoder_project_outputs
        
        return recons



if __name__ == '__main__':
    import copy
    import torch
    input = torch.rand(24, 3, 256, 256)  # batch_size, channels, w, h

    project_heads = []
    project_outdims =  [128, 256, 512]
    for outdim in project_outdims:
        project_heads.append(dict(
            pooling='max',
            dropout=None,
            activation=None,
            out_dims= outdim
        ))
    
    model = UnetAE(encoder_name='mit_b2',classes=1, activation='sigmoid', aux_params= project_heads)
    input = input.to('cuda')
    model = model.to('cuda')
    output, prject_embeds = model(input)
    print(output.shape)

    for project_embed in prject_embeds:
        print(project_embed.shape)
    
    '''for i in range(len(decoders)):
        print('decoders',i, decoders[i].shape)'''
    '''recon, embeddings, logit_scale = model(input)
    print(type(embeddings))
    print(embeddings.shape)  # 输出(24,16384), 24:batch_size, 16384:向量维度
    print(type(recon))
    print(recon.shape)  # 输出(24,1,256,256)'''

