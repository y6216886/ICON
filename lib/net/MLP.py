# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import pytorch_lightning as pl
from termcolor import colored
import torch.nn.functional as F
from torch import einsum

    
class MLP(pl.LightningModule):
    def __init__(self, filter_channels, name=None, res_layers=[], norm='group', last_op=None,mode=None, args=None):

        super(MLP, self).__init__()
        filter_channels[-1]=2
        self.args=args
        self.len_channels=len(filter_channels)
        if args.mlp_first_dim!=0:
            filter_channels[0]=args.mlp_first_dim
        print(colored("I have modified mlp filter channles{}".format(filter_channels),"red"))
        # if args.uncertainty:
        #     print("uncertainty")
        #     filter_channels[-1]+=1  #We follow the authors’ suggestion and train the network to predict the log of the observation noise scalar, s, for numerical stability.
        self.filters = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_layers = res_layers
        self.norm = norm
        self.last_op = last_op
        self.mode = mode
        self.name = name

        self.activate = nn.LeakyReLU(inplace=True)
        assert [self.args.mlpSe, self.args.mlpSev1, self.args.mlpSemax].count(True) in [0,1], "mlp se strategy cannot be embodied simultaneously"
        self.se_start_channel=self.args.se_start_channel
        self.se_end_channel=self.args.se_end_channel
        assert self.se_end_channel <self.len_channels
        assert self.se_start_channel >=0
        if self.args.mlpSe: ##this strategy yields best results, while not surpasses baseline yet. 
            self.se_conv = nn.ModuleList()
            for filters_nums_ in filter_channels[self.se_start_channel:self.se_end_channel]:
                # self.se_conv.append(SpatialSELayer(filters_nums_))  #1449 gpu memory for bs 2
                # self.se_conv.append(ChannelSELayer(filters_nums_))  #1457 gpu memory for bs 2
                self.se_conv.append(SCSEModule(filters_nums_, self.args.se_reduction, self.args))
        elif self.args.mlpSev1:
            self.se_conv = nn.ModuleList()
            for filters_nums_ in filter_channels[self.se_start_channel:self.se_end_channel]:
                # self.se_conv.append(SpatialSELayer(filters_nums_))  #1449 gpu memory for bs 2
                self.se_conv.append(ChannelSELayer(filters_nums_, self.args.se_reduction))  #1457 gpu memory for bs 2
        elif self.args.mlpSemax:
            self.se_conv_spatial = nn.ModuleList()
            self.se_conv_channel = nn.ModuleList()
            for filters_nums_ in filter_channels[self.se_start_channel:self.se_end_channel]:
                self.se_conv_spatial.append(SpatialSELayer(filters_nums_))  #1449 gpu memory for bs 2
                self.se_conv_channel.append(ChannelSELayer(filters_nums_)) 
        
        for l in range(0, len(filter_channels) - 1):
                if l in self.res_layers:
                    self.filters.append(
                        nn.Conv1d(filter_channels[l] + filter_channels[0], filter_channels[l + 1], 1)
                    )
                else:
                    self.filters.append(nn.Conv1d(filter_channels[l], filter_channels[l + 1], 1))

                if l != len(filter_channels) - 2:
                    if norm == 'group':
                        self.norms.append(nn.GroupNorm(32, filter_channels[l + 1]))
                    elif norm == 'batch':
                        self.norms.append(nn.BatchNorm1d(filter_channels[l + 1]))
                    elif norm == 'instance':
                        self.norms.append(nn.InstanceNorm1d(filter_channels[l + 1]))
                    elif norm == 'weight':
                        self.filters[l] = nn.utils.weight_norm(self.filters[l], name='weight')
                        # print(self.filters[l].weight_g.size(),
                        #       self.filters[l].weight_v.size())
        self.len_filter=len(self.filters)
        self.filters_fine = nn.ModuleList()
        self.norms_fine = nn.ModuleList()
        filter_channels_fine = [16, 512, 256, 128, 1]
        for l in range(0, len(filter_channels_fine) - 1):
            if l in self.res_layers:
                self.filters_fine.append(
                    nn.Conv1d(filter_channels_fine[l] + filter_channels_fine[0],
                              filter_channels_fine[l + 1], 1))
            else:
                self.filters_fine.append(
                    nn.Conv1d(filter_channels_fine[l], filter_channels_fine[l + 1], 1))
            if l != len(filter_channels_fine) - 2:
                self.norms_fine.append(nn.BatchNorm1d(filter_channels_fine[l + 1]))

    
#     def forward_se(self, feature, clip_feature=None): ##todo fuse clip feature into
#         '''
#         feature may include multiple view inputs
#         args:
#             feature: [B, C_in, N]
#         return:
#             [B, C_out, N] prediction
#         '''
#         y = feature
#         if self.args.use_clip: clip_feature=clip_feature.unsqueeze(-1).repeat(1,1,8000)
#         tmpy = feature
#         # len_=len(self.filters)
#         j=0
#         for i, f in enumerate(self.filters):
#             ####se net
#             if self.args.mlpSe or self.args.mlpSev1:
#                 if i in range(self.se_start_channel,self.se_end_channel):
#                     y=self.se_conv[j](y) 
#                     j+=1
#             elif self.args.mlpSemax:
#                 if i in range(self.se_start_channel,self.se_end_channel):
#                     y_spa=self.se_conv_spatial[j](y) ##
#                     y_cha=self.se_conv_channel[j](y) ##
#                     y=torch.max(y_spa, y_cha)
#                     j+=1
#             #####
#             if self.args.use_clip and i in self.clip_fuse_layer:
#                 input=torch.cat([y, clip_feature], 1) if i not in self.res_layers else torch.cat([y, tmpy, clip_feature], 1)
#                 if self.args.dropout!=0 and self.training and i>0: y= self.dropout(y)
#                 y = f(input)
#             else: 
#                 input=y if i not in self.res_layers else torch.cat([y, tmpy], 1)
#                 if self.args.dropout!=0 and self.training and i>0: 
#                     y= self.dropout(y)
#                 y = f(input)

#             ###activation
#             if i != len(self.filters) - 1:
#                 if self.norm not in ['batch', 'group', 'instance']:
#                     y = self.activate(y)
#                 else:
#                     y = self.activate(self.norms[i](y))
#             ###
# ##bug do not activate the last channel

#         if self.last_op is not None:
#             y = self.last_op(y)
#         if self.args.uncertainty:
#             y_uncertainty=self.mlp_uncertainty(y)
#             return torch.cat([y,y_uncertainty],dim=1)
#         return y
    
#     def forward_vol_attention(self, feature, vol_feat=None): ##todo fuse clip feature into
#         '''
#         feature may include multiple view inputs
#         args:
#             feature: [B, C_in, N]
#         return:
#             [B, C_out, N] prediction
#         '''
#         y = feature 
#         # vol_feat=vol_feat.detach()
#         tmpy = feature
#         # len_=len(self.filters)
#         j=0
#         # breakpoint()
#         for i, f in enumerate(self.filters):
#             ####se net
#             if self.args.mlpSe or self.args.mlpSev1:
#                 if i in range(self.se_start_channel,self.se_end_channel):
#                     y=self.se_conv[j](y, vol_feat) 
#                     j+=1
#             elif self.args.mlpSemax:
#                 if i in range(self.se_start_channel,self.se_end_channel):
#                     y_spa=self.se_conv_spatial[j](y) ##
#                     y_cha=self.se_conv_channel[j](y) ##
#                     y=torch.max(y_spa, y_cha)
#                     j+=1

#             input=y if i not in self.res_layers else torch.cat([y, tmpy], 1)
#             if self.args.dropout!=0 and self.training and i>0: 
#                 y= self.dropout(y)
#             y = f(input)

#             ###activation
#             if i != len(self.filters) - 1:
#                 if self.norm not in ['batch', 'group', 'instance']:
#                     y = self.activate(y)
#                 else:
#                     y = self.activate(self.norms[i](y))
#             ###
# ##bug do not activate the last channel

#         if self.last_op is not None:
#             y = self.last_op(y)
#         if self.args.uncertainty:
#             y_uncertainty=self.mlp_uncertainty(y)
#             return torch.cat([y,y_uncertainty],dim=1)
#         return y

    def forward_dif(self, feature, vol_feat=None): ##todo fuse clip feature into
        '''
        feature may include multiple view inputs
        args:
            feature: [B, C_in, N]
        return:
            [B, C_out, N] prediction
        '''
        y = feature 
        # vol_feat=vol_feat.detach()
        tmpy = feature
        # len_=len(self.filters)
        j=0
        # breakpoint()
        for i, f in enumerate(self.filters):
            ####se net
            if self.args.mlpSe or self.args.mlpSev1:
                if i in range(self.se_start_channel,self.se_end_channel):
                    y=self.se_conv[j](y, vol_feat) 
                    j+=1
            elif self.args.mlpSemax:
                if i in range(self.se_start_channel,self.se_end_channel):
                    y_spa=self.se_conv_spatial[j](y) ##
                    y_cha=self.se_conv_channel[j](y) ##
                    y=torch.max(y_spa, y_cha)
                    j+=1

            input=y if i not in self.res_layers else torch.cat([y, tmpy], 1)
            if self.args.dropout!=0 and self.training and i>0: 
                y= self.dropout(y)
            y = f(input)

            ###activation
            if i != len(self.filters) - 1:
                if self.norm not in ['batch', 'group', 'instance']:
                    y = self.activate(y)
                else:
                    y = self.activate(self.norms[i](y))
            ###
##bug do not activate the last channel

        mu_0, sigma_0 = torch.split(y, 1, dim=1) # (B, C=1, N), (B, C=1, N)
        sigma_0 = F.softplus(sigma_0)
        sigma_0 += 1e-8
        q_distribution = torch.distributions.Normal(mu_0, sigma_0)
        z = q_distribution.rsample()

        if self.mode != 'test':
            feat_fine = torch.cat([tmpy, z, mu_0, sigma_0], 1)  # (B, C=13+2+1, N)
        else:
            feat_fine = torch.cat([tmpy, mu_0, mu_0, sigma_0], 1)
        fine_y = feat_fine
        tmp_fine_y = feat_fine

        for i, f in enumerate(self.filters_fine):
            fine_y = f(fine_y if i not in self.res_layers else torch.cat([fine_y, tmp_fine_y], 1))
            if i != len(self.filters_fine) - 1:
                if self.norm not in ['batch', 'group', 'instance']:
                    fine_y = self.activate(fine_y)
                else:
                    fine_y = self.activate(self.norms_fine[i](fine_y)) # (B, C=1, N)

        if self.last_op is not None:
            fine_y = self.last_op(fine_y) # (B, C=1, N)
        else:
            fine_y = fine_y
        if self.mode != 'test':
            return fine_y, mu_0, sigma_0
        else:
            return fine_y
    
    def forward(self,x, vol_feat=None):
        if self.args.smpl_attention:
            return self.forward_dif(x, vol_feat)
        else:
            return self.forward_se(x)

    



    

class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv1d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a= input_tensor.size()


 
        out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor



class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2) ###cannot mix features of different points toghther

        # channel excitation
        # fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_1 = self.fc1(squeeze_tensor)
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16, args=None):
        super().__init__()

        self.args=args
        assert (self.args.sse or self.args.cse)==True
        if self.args.cse:
                self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        if self.args.sse:
            # breakpoint()
            # self.sSE = nn.Sequential(nn.Conv1d(in_channels, 1, 1), nn.Sigmoid())
            self.sSE = nn.Sequential(nn.Conv1d(in_channels, out_channels=1, kernel_size=int(self.args.kernel_pad_num[0]), padding=int(self.args.kernel_pad_num[1]), padding_mode=self.args.mlp_pad_mode), nn.Sigmoid())
            """
            padding_mode=['replicate', 'circular', 'zeros', 'reflect']
            kernel_size=3, padding=1, dilation=1
            kernel_size=5, padding=2, dilation=1
            kernel_size=7, padding=3, dilation=1
            """
        if self.args.smpl_attention:
            self.smpl_attention=SI_MODULE(points_channel=in_channels, args=self.args)
            
            

    def forward(self, x, vol_feat=None):
        y=0
        if self.args.smpl_attention:

            sse=self.smpl_attention(x, vol_feat) ##it is better to draw the attetion of the network to the whole feature map from voxelized smpl model, rather than a single point
            # y+= x * sse
            # breakpoint()
            y+=einsum('bcp, bip-> bcp', x, sse)
            # sse =sse.permute(0,2,1)
            return y
        else:
            if self.args.cse:
                cse=self.cSE(x)
                y+=x * cse
            if self.args.sse:
                sse=self.sSE(x) ##it is better to draw the attetion of the network to the whole feature map from voxelized smpl model, rather than a single point
                y+= x * sse
            return y

class SI_MODULE(nn.Module):
    def __init__(self, points_channel,  args=None):
        super().__init__()
        self.sigmoid=nn.Sigmoid()
        self.fuse_net=fuse_net(vol_dim=args.pamir_vol_dim, points_channel=points_channel)
        # self.fuse_net=Attention(points_channel)
    def forward(self, point_feature, vol_feature): # (bs, c, h, w, d) (bs, num_p, c)
        fused_feature=self.fuse_net(point_feature, vol_feature)
        # fused_feature=self.fuse_net(point_feature)
        # breakpoint()
        weights=self.sigmoid(fused_feature)
        return weights



# class fuse_net(nn.Module):
#     #fuse a 2d feature of several points with the size of (channel dimension size, number of points ) and a 4d feature of a voxelization feature with size (channel dimension, height, width, depth) into a 2d feature
#     def __init__(self, vol_dim=3, points_channel=13, args=None):
#         super().__init__()
#         self.vox_encoder=nn.Conv3d(in_channels=vol_dim, out_channels=points_channel, kernel_size=1)
#         # self.fc1=nn.Linear(points_channel, points_channel*3)
#         self.adaptive_pool=nn.AdaptiveAvgPool3d(1)
#         self.scale=points_channel**-0.5
#         self.bn1=nn.BatchNorm3d(points_channel)
#         self.relu=nn.LeakyReLU(0.2, inplace=True)
#         # self.fc1=nn.Linear(points_channel*2, points_channel)

#     def forward(self, point_feature, vol_feature):
#         vol_feature=self.vox_encoder(vol_feature)
#         vol_feature = self.bn1(vol_feature)
#         vol_feature = self.relu(vol_feature)
#         vol_feature=self.adaptive_pool(vol_feature)
#         vol_feature=vol_feature.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1)
#         # breakpoint()
#         fuse_feat=einsum('bic, bcp-> bip', vol_feature, point_feature)*self.scale
#         # fuse_feat=torch.matmul(vol_feature, point_feature)  # (bs, 1, c) * (bs, c, num_p) = (bs, 1, num_p)

#         # fuse_feat=torch.cat([fuse_feat, point_feature], dim=1)
#         # fuse_feat=self.fc1(fuse_feat)
#         # breakpoint()
#         print(fuse_feat.mean(), fuse_feat.std())
#         return fuse_feat


from einops import rearrange, repeat
class Attention(nn.Module):              
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        x=x.permute(0,2,1)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)           # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)          # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class fuse_net(nn.Module):
    #fuse a 2d feature of several points with the size of (channel dimension size, number of points ) and a 4d feature of a voxelization feature with size (channel dimension, height, width, depth) into a 2d feature
    def __init__(self, vol_dim=3, points_channel=13, args=None):
        super().__init__()
        self.vox_encoder=nn.Conv3d(in_channels=vol_dim, out_channels=points_channel, kernel_size=1)
        # self.fc1=nn.Linear(points_channel, points_channel*3)
        self.adaptive_pool=nn.AdaptiveAvgPool3d(1)
        # self.scale=points_channel**-0.5
        self.bn1=nn.BatchNorm3d(points_channel)
        self.relu=nn.LeakyReLU(0.2, inplace=True)
        # self.fc1=nn.Linear(points_channel*2, points_channel)

    def forward(self, point_feature, vol_feature): 
        vol_feature=self.vox_encoder(vol_feature) 
        vol_feature = self.bn1(vol_feature)
        vol_feature = self.relu(vol_feature)
        vol_feature=self.adaptive_pool(vol_feature)
        vol_feature=vol_feature.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1)
        # breakpoint()
        fuse_feat=einsum('bic, bcp-> bip', vol_feature, point_feature)

        return fuse_feat


if __name__=="__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    class args_():
        def __init__(self) -> None:
            self.test_code=True
            self.mlp_first_dim=12
            self.mlpSev1=False
            self.mlpSe=True
            self.mlpSemax=False
            self.uncertainty=False
            self.use_clip=False
            self.dropout=0
            self.se_start_channel=1
            self.se_end_channel=4
            self.se_reduction=16
            self.cse=False
            self.sse=True
            self.kernel_pad_num=[3,1]
            self.mlp_pad_mode="zeros"
            self.smpl_attention=True
    args_=args_()
    net=MLP(filter_channels=[12,128,256,128,1], res_layers=[2,4] ,args=args_).cuda()
    # net=MLP(filter_channels=[12,128,256,128,1], args=args_).cuda()
    vol_feature=torch.randn(2,3,32,32,32).cuda()
    input=torch.randn(2,12,8000).cuda()
    print(net(input, vol_feature).size())
    # print(1)
    # point_feature=torch.randn(2,8000,16).cuda()
    # vol_feature=torch.randn(2,3,32,32,32).cuda()
    # SI_MODULE=SI_MODULE(points_channel=16).cuda()
    # output=SI_MODULE(point_feature, vol_feature)
    # print(output.max(), output.min())


    # fuse_net=fuse_net(3).cuda()
    # point_feature=torch.randn(2,8000,16).cuda()
    # vol_feature=torch.randn(2,3,32,32,32).cuda()
    # print(fuse_net(point_feature, vol_feature).size())

    # fuse_net=Attention(16).cuda()
    # point_feature=torch.randn(2,8000,16).cuda()
    # print(fuse_net(point_feature).size())
