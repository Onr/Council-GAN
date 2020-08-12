"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

##################################################################################
# Discriminator
##################################################################################
class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params, cuda_device='cuda:0'):
        super(MsImageDis, self).__init__()
        self.prev_real_input = None
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.cuda_device = cuda_device

        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net().cuda(self.cuda_device))

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        x = x.cuda(self.cuda_device)
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(self.cuda_device), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(self.cuda_device), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            elif self.gan_type == 'RelativisticAverageHingeGAN':
                self.prev_real_input = input_real  # save it for the gen train later
                # difference between real and fake:
                r_f_diff = out1 - torch.mean(out0, dim=0, keepdim=True).repeat(10, 1, 1, 1)
                # difference between fake and real samples
                f_r_diff = out0 - torch.mean(out1, dim=0, keepdim=True).repeat(10, 1, 1, 1)
                # return the loss
                loss += (torch.mean(torch.nn.ReLU()(1 - r_f_diff))
                         + torch.mean(torch.nn.ReLU()(1 + f_r_diff)))

            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake, input_real=None):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2)  # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(self.cuda_device), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            elif self.gan_type == 'RelativisticAverageHingeGAN':
                if input_real is not None:
                    outs1 = self.forward(input_real)
                elif self.prev_real_input is not None:
                    outs1 = self.forward(self.prev_real_input)
                else:
                    assert 0, "try using cal_gan_loss with RelativisticAverageHingeGAN but did not provid input_real"
                out1 = outs1[it]
                # difference between real and fake:
                r_f_diff = out1 - torch.mean(out0, dim=0, keepdim=True).repeat(10, 1, 1, 1)
                # difference between fake and real samples
                f_r_diff = out0 - torch.mean(out1, dim=0, keepdim=True).repeat(10, 1, 1, 1)
                loss = torch.mean(torch.nn.ReLU()(1 + r_f_diff)) + torch.mean(torch.nn.ReLU()(1 - f_r_diff))

            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


##################################################################################
# Council Discriminator
##################################################################################
class MsImageDisCouncil(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params, cuda_device='cuda:0'):
        super(MsImageDisCouncil, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.cuda_device = cuda_device
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        # cnn_x += [Conv2dBlock(input_dim=2 * self.input_dim, output_dim=dim, kernel_size=4, stride=2, padding=1, norm='none', activation=self.activ, pad_type=self.pad_type)] # original
        cnn_x += [Conv2dBlock(input_dim=2 * self.input_dim, output_dim=dim, kernel_size=3, stride=1, padding=1, norm='none', activation=self.activ, pad_type=self.pad_type)] # ON
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, dim, 1, 1, 0)]
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x).cuda(self.cuda_device)
        return cnn_x

    def forward(self, x, x_input):
        x = x.cuda(self.cuda_device)
        x_input = x_input.cuda(self.cuda_device)
        outputs = []
        for model in self.cnns:
            model_input = torch.cat((x, x_input), 1)
            outputs.append(model(model_input))
            x = self.downsample(x)
            x_input = self.downsample(x_input)
        return outputs

    def calc_dis_loss(self, input_fake, input_real, input):
        # calculate the loss to train D
        outs0 = self.forward(input_fake, input)
        outs1 = self.forward(input_real, input)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(self.cuda_device), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(self.cuda_device), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))

            elif self.gan_type == 'RelativisticAverageHingeGAN':
                self.prev_real_input = input_real  # save it for the gen train later
                self.prev_input = input  # save it for the gen train later
                # difference between real and fake:
                r_f_diff = out1 - torch.mean(out0, dim=0, keepdim=True).repeat(10, 1, 1, 1)
                # difference between fake and real samples
                f_r_diff = out0 - torch.mean(out1, dim=0, keepdim=True).repeat(10, 1, 1, 1)
                loss = torch.mean(torch.nn.ReLU()(1 + r_f_diff)) + torch.mean(torch.nn.ReLU()(1 - f_r_diff))
                # return the loss
                loss += (torch.mean(torch.nn.ReLU()(1 - r_f_diff))
                         + torch.mean(torch.nn.ReLU()(1 + f_r_diff)))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake, input, input_real=None):
        # calculate the loss to train G
        outs0 = self.forward(input_fake, input)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(self.cuda_device), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            elif self.gan_type == 'RelativisticAverageHingeGAN':
                if input_real is not None:
                    outs1 = self.forward(input_real)
                elif self.prev_real_input is not None:
                    outs1 = self.forward(self.prev_real_input, self.prev_input)
                else:
                    assert 0, "try using cal_gan_loss with RelativisticAverageHingeGAN but did not provid input_real"

                out1 = outs1[it]
                # difference between real and fake:
                r_f_diff = out1 - torch.mean(out0, dim=0, keepdim=True).repeat(10, 1, 1, 1)
                # difference between fake and real samples
                f_r_diff = out0 - torch.mean(out1, dim=0, keepdim=True).repeat(10, 1, 1, 1)
                loss = torch.mean(torch.nn.ReLU()(1 + r_f_diff)) + torch.mean(torch.nn.ReLU()(1 - f_r_diff))
                loss = torch.mean(torch.nn.ReLU()(1 + r_f_diff)) + torch.mean(torch.nn.ReLU()(1 - f_r_diff))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss



##################################################################################
# Generator
##################################################################################

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params, cuda_device='cuda:0'):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        self.n_downsample = params['n_downsample']
        n_res = params['n_res']
        self.activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        self.do_my_style = params['do_my_style']
        self.cuda_device = cuda_device

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=self.activ, pad_type=pad_type).cuda(self.cuda_device)

        # content encoder
        self.enc_content = ContentEncoder(self.n_downsample, n_res, input_dim, dim, 'in', self.activ, pad_type=pad_type).cuda(self.cuda_device)

        if self.do_my_style:
            self.dec = Decoder_V2_atten(n_upsample=self.n_downsample, n_res=n_res, dim=self.enc_content.output_dim + style_dim, output_dim=input_dim, res_norm='in', activ=self.activ, pad_type=pad_type, num_of_mask_dim_to_add=params['num_of_mask_dim_to_add']).cuda(self.cuda_device)
        else:
            self.dec = Decoder_V2_atten(self.n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=self.activ, pad_type=pad_type, num_of_mask_dim_to_add=params['num_of_mask_dim_to_add']).cuda(self.cuda_device)

        # MLP to generate AdaIN parameters or adding sytle my way
        if self.do_my_style:
            self.mlp = MLP(input_dim=style_dim, output_dim=style_dim, dim=mlp_dim, n_blk=3, norm='none',
                           activ=self.activ).cuda(self.cuda_device)
        else:
            self.mlp = MLP(input_dim=style_dim, output_dim=self.get_num_adain_params(self.dec), dim=mlp_dim, n_blk=3,
                           norm='none', activ=self.activ).cuda(self.cuda_device)

    def forward(self, images, return_mask=False):
        # reconstruct an image
        images = images.cuda(self.cuda_device)
        content, style_fake = self.encode(images)
        if return_mask:
            images_recon, mask = self.decode(content=content, style=style_fake, images=images, return_mask=return_mask)
            return images_recon, mask
        else:
            images_recon = self.decode(content=content, style=style_fake, images=images, return_mask=return_mask)
            return images_recon

    def forward(self, images, style, return_mask=False):
        # reconstruct an image
        content, _ = self.encode(images)
        if return_mask:
            images_recon, mask = self.decode(content=content, style=style, images=images, return_mask=return_mask)
            return images_recon, mask
        else:
            images_recon = self.decode(content=content, style=style, images=images, return_mask=return_mask)
            return images_recon


    def encode(self, images):
        images = images.cuda(self.cuda_device)
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style, images, return_mask=False):
        content, style, images = content.cuda(self.cuda_device), style.cuda(self.cuda_device), images.cuda(self.cuda_device)
        # decode content and style codes to an image
        if self.do_my_style:
            style_to_add = self.mlp(style)
            style_to_add = style_to_add.repeat(content.shape[2], content.shape[3], 1, 1)
            style_to_add = style_to_add.transpose(0, 2).transpose(3, 1)
            content = torch.cat((content, style_to_add), 1)
        else:
            adain_params = self.mlp(style)
            self.assign_adain_params(adain_params, self.dec)
        if return_mask:
            images, mask = self.dec(content, images, return_mask)
            return images, mask
        else:
            images = self.dec(content, images)
            return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        start_ind = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, start_ind: start_ind + m.num_features]
                std = adain_params[:, start_ind + m.num_features: start_ind + 2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                start_ind += 2*m.num_features

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params



    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


##################################################################################
# Encoder and Decoders
##################################################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)] # ORIGINAL
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)




class Decoder_V2_atten(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero', num_of_mask_dim_to_add=1):
        super(Decoder_V2_atten, self).__init__()
        self.num_of_mask_dim_to_add = num_of_mask_dim_to_add # 3  # 2
        self.model = []
        self.output_dim = output_dim
        self.mask_s = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2)]
            self.model += [Conv2dBlock(input_dim=dim, output_dim=dim // 2, kernel_size=3, stride=1, padding=1, norm='adain',
                                       activation=activ, pad_type=pad_type)]
            dim //= 2
            self.model += [Conv2dBlock(input_dim=dim, output_dim=dim, kernel_size=3, stride=1, padding=1, norm='adain',
                                       activation=activ, pad_type=pad_type)]


        self.model += [Conv2dBlock(input_dim=dim, output_dim=dim, kernel_size=1, stride=1, padding=0, norm='none', activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(input_dim=dim, output_dim=dim, kernel_size=1, stride=1, padding=0, norm='none', activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(input_dim=dim, output_dim=(output_dim*self.num_of_mask_dim_to_add+self.num_of_mask_dim_to_add), kernel_size=1, stride=1, padding=0, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, im_in, return_mask=False):
        new_x = self.model(x)
        self.mask_s = ((torch.tanh(10 * new_x[:, (-1 * self.num_of_mask_dim_to_add):, :, :]) + 1) / 2)
        new_im = im_in
        curr_ind = 0
        for k in range(self.num_of_mask_dim_to_add):
            new_im_o = new_x[:, curr_ind:self.output_dim * (k + 1), :, :] ##
            curr_ind = self.output_dim * (k + 1)  ##
            mask = self.mask_s[:, k, :, :].unsqueeze(1).repeat(1, new_im.shape[1], 1, 1)
            new_im = (1 - mask) * new_im + mask * new_im_o

        if return_mask:
            if self.mask_s.shape[1] != 3:
                mask_s_prePixTot = torch.sum(self.mask_s, 1).unsqueeze(1).repeat(1, 3, 1, 1) / self.mask_s.shape[1]
                self.mask_s = mask_s_prePixTot
            return new_im, self.mask_s

        return new_im


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        # relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
