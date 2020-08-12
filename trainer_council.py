"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, MsImageDisCouncil
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import random
import threading
from multiprocessing.pool import ThreadPool
import warnings
from collections import deque
import numpy as np
import torchvision.transforms.functional as TF
from scipy import ndimage

class Council_Trainer(nn.Module):
    def __init__(self, hyperparameters, cuda_device='cuda:0'):
        super(Council_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.council_size = hyperparameters['council']['council_size']
        self.council_size_conf = self.council_size
        self.gen_a2b_s = []
        self.gen_b2a_s = []
        self.dis_a2b_s = []
        self.dis_b2a_s = []
        self.do_dis_council = hyperparameters['council_w'] != 0
        self.do_ads_council_loss = hyperparameters['council_abs_w'] != 0
        self.numberOfCouncil_dis_relative_iteration_conf = hyperparameters['council']['numberOfCouncil_dis_relative_iteration']
        self.discriminetro_less_style_by_conf = hyperparameters['council']['discriminetro_less_style_by']
        self.cuda_device = cuda_device

        # all varible with '_conf' at the end will be saved and displayed in tensorboard logs
        self.recon_x_w_conf = hyperparameters['recon_x_w']
        self.recon_c_w_conf = hyperparameters['recon_c_w']
        self.recon_s_w_conf = hyperparameters['recon_s_w']
        self.recon_x_cyc_w_conf = hyperparameters['recon_x_cyc_w']
        self.gan_w_conf = hyperparameters['gan_w']
        self.vgg_w_conf = hyperparameters['vgg_w']
        self.abs_beginning_end_w_conf = hyperparameters['abs_beginning_end']
        self.flipOnOff_On_iteration_conf = hyperparameters['council']['flipOnOff_On_iteration']
        self.flipOnOff_Off_iteration_conf = hyperparameters['council']['flipOnOff_Off_iteration']
        self.flipOnOff_Off_iteration_conf = hyperparameters['council']['flipOnOff_start_with']
        self.council_abs_w_conf = hyperparameters['council_abs_w']
        self.council_w_conf = hyperparameters['council_w']
        self.council_start_at_iter_conf = hyperparameters['council']['council_start_at_iter']
        self.focus_loss_start_at_iter_conf = hyperparameters['focus_loss']['focus_loss_start_at_iter']
        self.mask_zero_or_one_w_conf = hyperparameters['mask_zero_or_one_w']
        self.mask_zero_or_one_center_conf = hyperparameters['focus_loss']['mask_zero_or_one_center']
        self.mask_zero_or_one_epsilon_conf = hyperparameters['focus_loss']['mask_zero_or_one_epsilon']
        self.mask_total_w_conf = hyperparameters['mask_total_w']
        self.mask_tv_w_conf = hyperparameters['mask_tv_w']
        self.batch_size_conf = hyperparameters['batch_size']
        self.do_w_loss_matching = hyperparameters['do_w_loss_matching']
        self.do_w_loss_matching_focus = hyperparameters['focus_loss']['do_w_loss_matching_focus']
        self.los_matching_hist_size_conf = hyperparameters['loss_matching_hist_size']
        self.do_a2b_conf = hyperparameters['do_a2b']
        self.do_b2a_conf = hyperparameters['do_b2a']
        self.w_match_b2a_conf = 1
        self.w_match_a2b_conf = 1
        self.w_match_focus_a2b_conf = 1
        self.w_match_focus_b2a_conf = 1
        self.w_match_focus_zero_one_a2b_conf = 1
        self.w_match_focus_zero_one_b2a_conf = 1

        if self.do_a2b_conf:
            self.los_hist_gan_a2b_s = []
            self.los_hist_council_a2b_s = []
            self.los_hist_focus_a2b_s = []
            self.los_hist_focus_zero_one_a2b_s = []
        if self.do_b2a_conf:
            self.los_hist_gan_b2a_s = []
            self.los_hist_council_b2a_s = []
            self.los_hist_focus_b2a_s = []
            self.los_hist_focus_zero_one_b2a_s = []

        for ind in range(self.council_size):
            if self.do_a2b_conf:
                self.los_hist_gan_a2b_s.append(deque(np.ones(self.los_matching_hist_size_conf)))
                self.los_hist_council_a2b_s.append(deque(np.ones(self.los_matching_hist_size_conf)))
                self.los_hist_focus_a2b_s.append(deque(np.ones(self.los_matching_hist_size_conf)))
                self.los_hist_focus_zero_one_a2b_s.append(deque(np.ones(self.los_matching_hist_size_conf)))

            if self.do_b2a_conf:
                self.los_hist_gan_b2a_s.append(deque(np.ones(self.los_matching_hist_size_conf)))
                self.los_hist_council_b2a_s.append(deque(np.ones(self.los_matching_hist_size_conf)))
                self.los_hist_focus_b2a_s.append(deque(np.ones(self.los_matching_hist_size_conf)))
                self.los_hist_focus_zero_one_b2a_s.append(deque(np.ones(self.los_matching_hist_size_conf)))

        self.do_council_loss = None

        if self.do_dis_council:
            self.dis_council_a2b_s = []
            self.dis_council_b2a_s = []

        # defining all the networks
        for i in range(self.council_size):
            if self.do_a2b_conf:
                self.gen_a2b_s.append(
                    AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'], cuda_device=self.cuda_device))  # auto-encoder for domain a2b
                self.dis_a2b_s.append(
                    MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'], cuda_device=self.cuda_device))  # discriminator for domain a2b
                if self.do_dis_council:
                    self.dis_council_a2b_s.append(
                        MsImageDisCouncil(hyperparameters['input_dim_a'],
                                          hyperparameters['dis'], cuda_device=self.cuda_device))  # council discriminator for domain a2b
            if self.do_b2a_conf:
                self.gen_b2a_s.append(
                    AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'], cuda_device=self.cuda_device))  # auto-encoder for domain b
                self.dis_b2a_s.append(
                    MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'], cuda_device=self.cuda_device))  # discriminator for domain b
                if self.do_dis_council:
                    self.dis_council_b2a_s.append(#
                        MsImageDisCouncil(hyperparameters['input_dim_b'],
                                          hyperparameters['dis'], cuda_device=self.cuda_device))  # discriminator for domain b

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        if self.do_a2b_conf:
            self.gen_a2b_s = nn.ModuleList(self.gen_a2b_s)
            self.dis_a2b_s = nn.ModuleList(self.dis_a2b_s)
            if self.do_dis_council:
                self.dis_council_a2b_s = nn.ModuleList(self.dis_council_a2b_s)
        if self.do_b2a_conf:
            self.gen_b2a_s = nn.ModuleList(self.gen_b2a_s)
            self.dis_b2a_s = nn.ModuleList(self.dis_b2a_s)
            if self.do_dis_council:
                self.dis_council_b2a_s = nn.ModuleList(self.dis_council_b2a_s)

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda(self.cuda_device)
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda(self.cuda_device)
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params_s = []
        gen_params_s = []
        self.dis_opt_s = []
        self.gen_opt_s = []
        self.dis_scheduler_s = []
        self.gen_scheduler_s = []
        if self.do_dis_council:
            dis_council_params_s = []
            self.dis_council_opt_s = []
            self.dis_council_scheduler_s = []
        for i in range(self.council_size):
            dis_parms = []
            gen_parms = []
            dis_council_parms = []
            if self.do_a2b_conf:
                dis_parms += list(self.dis_a2b_s[i].parameters())
                gen_parms += list(self.gen_a2b_s[i].parameters())
                if self.do_dis_council:
                    dis_council_parms += list(self.dis_council_a2b_s[i].parameters())
            if self.do_b2a_conf:
                dis_parms += list(self.dis_b2a_s[i].parameters())
                gen_parms += list(self.gen_b2a_s[i].parameters())
                if self.do_dis_council:
                    dis_council_parms += list(self.dis_council_b2a_s[i].parameters())
            dis_params_s.append(dis_parms)
            gen_params_s.append(gen_parms)
            if self.do_dis_council:
                dis_council_params_s.append(dis_council_parms)
            self.dis_opt_s.append(torch.optim.Adam([p for p in dis_params_s[i] if p.requires_grad],
                                                   lr=lr, betas=(beta1, beta2),
                                                   weight_decay=hyperparameters['weight_decay']))
            self.gen_opt_s.append(torch.optim.Adam([p for p in gen_params_s[i] if p.requires_grad],
                                                   lr=lr, betas=(beta1, beta2),
                                                   weight_decay=hyperparameters['weight_decay']))
            if self.do_dis_council:
                self.dis_council_opt_s.append(torch.optim.Adam([p for p in dis_council_params_s[i] if p.requires_grad],
                                                               lr=lr, betas=(beta1, beta2),
                                                               weight_decay=hyperparameters['weight_decay']))
            self.dis_scheduler_s.append(get_scheduler(self.dis_opt_s[i], hyperparameters))
            self.gen_scheduler_s.append(get_scheduler(self.gen_opt_s[i], hyperparameters))
            if self.do_dis_council:
                self.dis_council_scheduler_s.append(get_scheduler(self.dis_council_opt_s[i], hyperparameters))

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        for i in range(self.council_size):
            if self.do_a2b_conf:
                self.gen_a2b_s[i].apply(weights_init(hyperparameters['init']))
                self.dis_a2b_s[i].apply(weights_init('gaussian'))
                if self.do_dis_council:
                    self.dis_council_a2b_s[i].apply(weights_init('gaussian'))
            if self.do_b2a_conf:
                self.gen_b2a_s[i].apply(weights_init(hyperparameters['init']))
                self.dis_b2a_s[i].apply(weights_init('gaussian'))
                if self.do_dis_council:
                    self.dis_council_b2a_s[i].apply(weights_init('gaussian'))

        # Load VGG model if needed
        self.vgg = None
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def recon_criterion_v2_color(self, input, target):
        loss_1 = torch.mean(torch.abs(input - target))
        loss_2 = torch.mean(torch.pow((input - target), 2))
        if loss_1 > loss_2:
            return loss_1
        return loss_2

    def recon_criterion_v3_gray_scale(self, input, target):
        loss_1 = torch.mean(torch.abs(torch.sum(input, 1) - torch.sum(target, 1)))
        loss_2 = torch.mean(torch.pow((torch.sum(input, 1) - torch.sum(target, 1)), 2))
        if loss_1 > loss_2:
            return loss_1
        return loss_2

    def council_basic_criterion_gray_scale(self, input, target):
        return torch.mean(torch.abs(torch.sum(input, 1) - torch.sum(target, 1)))

    def council_basic_criterion_with_color(self, input, target):
        return torch.mean(torch.abs(input - target))

    def mask_zero_one_criterion(self, mask, center=0.5, epsilon=0.01):
        return torch.sum(1 / (torch.abs(mask - center) + epsilon)) / mask.numel()

    def mask_small_criterion(self, mask):
        assert self.hyperparameters['focus_loss']['mask_small_use_abs'] or self.hyperparameters['focus_loss']['mask_small_use_square'], 'at leas one small mask loss should be true, mask_small_use_abs or mask_small_use_square'
        loss = 0
        if self.hyperparameters['focus_loss']['mask_small_use_abs']:
            loss += self.mask_small_criterion_abs(mask)
        if self.hyperparameters['focus_loss']['mask_small_use_square']:
            loss += self.mask_small_criterion_square(mask)
        return loss

    def mask_small_criterion_square(self, mask):
        return (torch.sum(mask) / mask.numel()) ** 2

    def mask_small_criterion_abs(self, mask):
        return torch.abs((torch.sum(mask))) / mask.numel()

    def mask_criterion_TV(self, mask):
        return (torch.sum(torch.abs(mask[:, :, 1:, :]-mask[:, :, :-1, :])) + \
               torch.sum(torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]))) / mask.numel()

    def forward(self, x_a, s_t=None, x_b=None, s_a=None, s_b=None):
        self.eval()
        if s_t is not None:
            s_a = s_t
            s_b = s_t
        if self.do_a2b_conf:
            s_b = self.s_b if s_b is None else s_b
            s_b = Variable(s_b)
            x_ab_s = []
        if self.do_b2a_conf:
            s_a = self.s_a if s_a is None else s_a
            s_a = Variable(s_a)
            x_ba_s = []
        for i in range(self.council_size):
            if self.do_a2b_conf:
                c_a, s_a_fake = self.gen_a2b.encode(x_a)
                x_ab_s.append(self.gen_a2b.decode(c_a, s_b, x_a))
            if self.do_b2a_conf:
                x_b = x_a if x_b is None else x_b
                c_b, s_b_fake = self.gen_b2a_s[i].encode(x_b)
                x_ba_s.append(self.gen_b2a_s[i].decode(c_b, s_a, x_b))

        if self.do_a2b_conf and self.do_b2a_conf:
            return x_ab_s, x_ba_s
        elif self.do_b2a_conf:
            return x_ba_s
        return x_ab_s

    def gen_update(self, x_a, x_b, hyperparameters, iterations=0):
        self.hyperparameters = hyperparameters
        for gen_opt in self.gen_opt_s:
            gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda(self.cuda_device))
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(self.cuda_device))
        c_a_s = []
        s_a_prime_s = []
        c_b_s = []
        s_b_prime_s = []
        x_a_recon_s = []
        x_b_recon_s = []
        x_ba_s = []
        x_ab_s = []
        c_b_recon_s = []
        s_a_recon_s = []
        c_a_recon_s = []
        s_b_recon_s = []
        x_aba_s = []
        x_bab_s = []
        mask_ba_s = []
        mask_ab_s = []
        self.loss_gen_mask_zero_one_ba_s = []
        self.loss_gen_mask_zero_one_ab_s = []
        self.loss_gen_mask_total_ba_s = []
        self.loss_gen_mask_total_ab_s = []
        self.loss_gen_mask_TV_ab_s = []
        self.loss_gen_mask_TV_ba_s = []
        self.loss_gen_recon_x_a_s = []
        self.loss_gen_recon_x_b_s = []
        self.loss_gen_recon_s_a_s = []
        self.loss_gen_recon_s_b_s = []
        self.loss_gen_recon_c_a_s = []
        self.loss_gen_recon_c_b_s = []
        self.loss_gen_cycrecon_x_a_s = []
        self.loss_gen_cycrecon_x_b_s = []
        self.loss_gen_beginning_end_a_ab_s = []
        self.loss_gen_beginning_end_b_ba_s = []
        self.loss_gen_adv_a2b_s = []
        self.loss_gen_adv_b2a_s = []
        self.loss_gen_vgg_a_s = []
        self.loss_gen_vgg_b_s = []
        self.loss_gen_total_s = []
        self.council_w_conf = hyperparameters['council_w'] if hyperparameters['iteration'] > hyperparameters['council']['council_start_at_iter'] else 0
        self.mask_zero_or_one_w_conf = hyperparameters['mask_zero_or_one_w'] if hyperparameters['iteration'] > hyperparameters['focus_loss']['focus_loss_start_at_iter'] else 0
        self.mask_total_w_conf = hyperparameters['mask_total_w'] if hyperparameters['iteration'] > hyperparameters['focus_loss']['focus_loss_start_at_iter'] else 0
        self.mask_tv_w_conf = hyperparameters['mask_tv_w'] if hyperparameters['iteration'] > hyperparameters['focus_loss']['focus_loss_start_at_iter'] else 0

        for i in range(self.council_size):
            # encode
            if self.do_a2b_conf:
                c_a, s_a_prime = self.gen_a2b_s[i].encode(x_a)
                c_a_s.append(c_a)
                s_a_prime_s.append(s_a_prime)
            if self.do_b2a_conf:
                c_b, s_b_prime = self.gen_b2a_s[i].encode(x_b)
                c_b_s.append(c_b)
                s_b_prime_s.append(s_b_prime)

            # decode (within domain)
            if hyperparameters['recon_x_w'] != 0:
                if not self.do_a2b_conf and not self.do_b2a_conf:
                    print('cant do recon_x loss if not both do_a2b and b2a set to true')
                else:
                    x_a_recon_s.append(self.gen_b2a_s[i].decode(c_a_s[i], s_a_prime_s[i], x_a))
                    x_b_recon_s.append(self.gen_a2b_s[i].decode(c_b_s[i], s_b_prime_s[i], x_b))

            # decode (cross domain)
            if self.do_a2b_conf:
                x_ab_s.append(self.gen_a2b_s[i].decode(c_a_s[i], s_b, x_a))
            if self.do_b2a_conf:
                x_ba_s.append(self.gen_b2a_s[i].decode(c_b_s[i], s_a, x_b))

            if hyperparameters['mask_zero_or_one_w'] != 0 or hyperparameters['mask_total_w'] != 0:
                if self.do_a2b_conf:
                    mask_ab_s.append(self.gen_a2b_s[i].dec.mask_s)
                if self.do_b2a_conf:
                    mask_ba_s.append(self.gen_b2a_s[i].dec.mask_s)

            # encode again
            if hyperparameters['recon_s_w'] != 0 or hyperparameters['recon_c_w'] != 0 or hyperparameters['recon_x_cyc_w'] != 0:
                if not self.do_a2b_conf and not self.do_b2a_conf:
                    print('cant do recon_s and recon_c loss if not both do_a2b and b2a set to true')
                else:
                    c_b_recon, s_a_recon = self.gen_a2b_s[i].encode(x_ba_s[i])
                    c_a_recon, s_b_recon = self.gen_b2a_s[i].encode(x_ab_s[i])
                    c_b_recon_s.append(c_b_recon)
                    s_a_recon_s.append(s_a_recon)
                    c_a_recon_s.append(c_a_recon)
                    s_b_recon_s.append(s_b_recon)

            # decode again (if needed)
            if hyperparameters['recon_x_cyc_w'] != 0:
                if not self.do_a2b_conf and not self.do_b2a_conf:
                    print('cant do recon_x_cyc loss if not both do_a2b and b2a set to true')
                else:
                    x_aba_s.append(
                        self.gen_b2a.decode(c_a_recon_s[i], s_a_prime_s[i], x_a) if hyperparameters['recon_x_cyc_w'] > 0 else None)
                    x_bab_s.append(
                        self.gen_a2b.decode(c_b_recon_s[i], s_b_prime_s[i], x_b) if hyperparameters['recon_x_cyc_w'] > 0 else None)

            self.loss_gen_total_s.append(0)
            if hyperparameters['do_a2b']:
                self.loss_gen_mask_TV_ab_s.append(0)
                self.loss_gen_mask_total_ab_s.append(0)
            if hyperparameters['do_b2a']:
                self.loss_gen_mask_TV_ba_s.append(0)
                self.loss_gen_mask_total_ba_s.append(0)

            # masks should contain ones or zeros
            if hyperparameters['iteration'] > hyperparameters['focus_loss']['focus_loss_start_at_iter'] and (hyperparameters['mask_zero_or_one_w'] != 0 or hyperparameters['mask_total_w'] != 0 ):

                if hyperparameters['mask_zero_or_one_w'] != 0:
                    if hyperparameters['do_a2b']:
                        self.loss_gen_mask_zero_one_ab_s.append(self.mask_zero_one_criterion(mask_ab_s[i], center=hyperparameters['focus_loss']['mask_zero_or_one_center'], epsilon=hyperparameters['focus_loss']['mask_zero_or_one_epsilon']))
                    if hyperparameters['do_b2a']:
                        self.loss_gen_mask_zero_one_ba_s.append(self.mask_zero_one_criterion(mask_ba_s[i], center=hyperparameters['focus_loss']['mask_zero_or_one_center'], epsilon=hyperparameters['focus_loss']['mask_zero_or_one_epsilon']))

                    if self.do_w_loss_matching_focus:
                        if hyperparameters['do_a2b']:
                            self.los_hist_focus_zero_one_a2b_s[i].append(self.loss_gen_mask_zero_one_ab_s[i].detach().cpu().numpy())
                            self.los_hist_focus_zero_one_a2b_s[i].popleft()
                            self.w_match_focus_zero_one_a2b_conf = np.mean(self.los_hist_gan_a2b_s[i]) / np.mean(self.los_hist_focus_zero_one_a2b_s[i])
                            self.loss_gen_mask_zero_one_ab_s[i] *= self.w_match_focus_zero_one_a2b_conf
                            self.loss_gen_total_s[i] += hyperparameters['mask_zero_or_one_w'] * self.loss_gen_mask_zero_one_ab_s[i].cuda(self.cuda_device)
                        if hyperparameters['do_b2a']:
                            self.los_hist_focus_zero_one_b2a_s[i].append(self.loss_gen_mask_zero_one_ba_s[i].detach().cpu().numpy())
                            self.los_hist_focus_zero_one_b2a_s[i].popleft()
                            self.w_match_focus_zero_one_b2a_conf = np.mean(self.los_hist_gan_b2a_s[i]) / np.mean(self.los_hist_focus_zero_one_b2a_s[i])
                            self.loss_gen_mask_zero_one_ba_s[i] *= self.w_match_focus_zero_one_b2a_conf
                            self.loss_gen_total_s[i] += hyperparameters['mask_zero_or_one_w'] * self.loss_gen_mask_zero_one_ba_s[i].cuda(self.cuda_device)
                    else:
                        if hyperparameters['do_a2b']:
                            self.loss_gen_total_s[i] += hyperparameters['mask_zero_or_one_w'] * self.loss_gen_mask_zero_one_ab_s[i].cuda(self.cuda_device)
                        if hyperparameters['do_b2a']:
                            self.loss_gen_total_s[i] += hyperparameters['mask_zero_or_one_w'] * self.loss_gen_mask_zero_one_ba_s[i].cuda(self.cuda_device)

                # masks should as small as possible to leave original domain with little changes
                if hyperparameters['mask_total_w'] != 0:
                    if hyperparameters['do_a2b']:
                        self.loss_gen_mask_total_ab_s[i] += self.mask_small_criterion(mask_ab_s[i])
                    if hyperparameters['do_b2a']:
                        self.loss_gen_mask_total_ba_s[i] += self.mask_small_criterion(mask_ba_s[i])

                # TV loss on the mask
                if hyperparameters['mask_tv_w'] != 0:
                    if hyperparameters['do_a2b']:
                        self.loss_gen_mask_TV_ab_s[i] += self.mask_criterion_TV(mask_ab_s[i])
                        self.loss_gen_total_s[i] += hyperparameters['mask_tv_w'] * self.loss_gen_mask_TV_ab_s[i]
                    if hyperparameters['do_b2a']:
                        self.loss_gen_mask_TV_ba_s[i] += self.mask_criterion_TV(mask_ba_s[i])
                        self.loss_gen_total_s[i] += hyperparameters['mask_tv_w'] * self.loss_gen_mask_TV_ba_s[i]

                if self.do_w_loss_matching_focus:
                    if hyperparameters['do_a2b']:
                        self.los_hist_focus_a2b_s[i].append(self.loss_gen_mask_total_ab_s[i].detach().cpu().numpy())
                        self.los_hist_focus_a2b_s[i].popleft()
                        self.w_match_focus_a2b_conf = np.mean(self.los_hist_gan_a2b_s[i]) / np.mean(self.los_hist_focus_a2b_s[i])
                        self.loss_gen_mask_total_ab_s[i] *= self.w_match_focus_a2b_conf
                        self.loss_gen_total_s[i] += hyperparameters['mask_total_w'] * self.loss_gen_mask_total_ab_s[i].cuda(self.cuda_device)
                    if hyperparameters['do_b2a']:
                        self.los_hist_focus_b2a_s[i].append(self.loss_gen_mask_total_ab_s[i].detach().cpu().numpy())
                        self.los_hist_focus_b2a_s[i].popleft()
                        self.w_match_focus_b2a_conf = np.mean(self.los_hist_gan_b2a_s[i]) / np.mean(self.los_hist_focus_b2a_s[i])
                        self.loss_gen_mask_total_ba_s[i] *= self.w_match_focus_b2a_conf
                        self.loss_gen_total_s[i] += hyperparameters['mask_total_w'] * self.loss_gen_mask_total_ba_s[i].cuda(self.cuda_device)

                else:
                    if hyperparameters['do_a2b']:
                        self.loss_gen_total_s[i] += hyperparameters['mask_total_w'] * self.loss_gen_mask_total_ab_s[i].cuda(self.cuda_device)
                    if hyperparameters['do_b2a']:
                        self.loss_gen_total_s[i] += hyperparameters['mask_total_w'] * self.loss_gen_mask_total_ba_s[i].cuda(self.cuda_device)


            # reconstruction loss
            if hyperparameters['recon_x_w'] != 0 and self.do_a2b_conf and self.do_b2a_conf:
                self.loss_gen_recon_x_a_s.append(self.recon_criterion(x_a_recon_s[i], x_a))
                self.loss_gen_recon_x_b_s.append(self.recon_criterion(x_b_recon_s[i], x_b))
                self.loss_gen_total_s[i] += hyperparameters['recon_x_w'] * (
                        self.loss_gen_recon_x_a_s[i].cuda(self.cuda_device) + self.loss_gen_recon_x_b_s[i].cuda(self.cuda_device))
            if hyperparameters['recon_s_w'] != 0 and self.do_a2b_conf and self.do_b2a_conf:
                self.loss_gen_recon_s_a_s.append(self.recon_criterion(s_a_recon_s[i], s_a))
                self.loss_gen_recon_s_b_s.append(self.recon_criterion(s_b_recon_s[i], s_b))
                self.loss_gen_total_s[i] += hyperparameters['recon_s_w'] * (
                        self.loss_gen_recon_s_a_s[i].cuda(self.cuda_device) + self.loss_gen_recon_s_b_s[i].cuda(self.cuda_device))
            if hyperparameters['recon_c_w'] != 0 and self.do_a2b_conf and self.do_b2a_conf:
                self.loss_gen_recon_c_a_s.append(self.recon_criterion(c_a_recon_s[i], c_a_s[i]))
                self.loss_gen_recon_c_b_s.append(self.recon_criterion(c_b_recon_s[i], c_b_s[i]))
                self.loss_gen_total_s[i] += hyperparameters['recon_c_w'] * (
                        self.loss_gen_recon_c_a_s[i].cuda(self.cuda_device) + self.loss_gen_recon_c_b_s[i].cuda(self.cuda_device))
            if hyperparameters['recon_x_cyc_w'] != 0 and self.do_a2b_conf and self.do_b2a_conf:
                self.loss_gen_cycrecon_x_a_s.append(
                    self.recon_criterion(x_aba_s[i], x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0)
                self.loss_gen_cycrecon_x_b_s.append(
                    self.recon_criterion(x_bab_s[i], x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0)
                self.loss_gen_total_s[i] += hyperparameters['recon_x_cyc_w'] * (
                        self.loss_gen_cycrecon_x_a_s[i].cuda(self.cuda_device) + self.loss_gen_cycrecon_x_b_s[i].cuda(self.cuda_device))
            if hyperparameters['abs_beginning_end'] != 0 and self.abs_beginning_end_w_conf > 0.005:
                if hyperparameters['do_a2b']:
                    self.loss_gen_beginning_end_a_ab_s.append(
                        self.recon_criterion_v2_color(x_ab_s[i], x_a) if hyperparameters['abs_beginning_end'] > 0 or hyperparameters['abs_beginning_end_minimume'] > 0 else 0)
                else:
                    self.loss_gen_beginning_end_a_ab_s.append(0)
                if hyperparameters['do_b2a']:
                    self.loss_gen_beginning_end_b_ba_s.append(
                        self.recon_criterion_v2_color(x_ba_s[i], x_b) if hyperparameters['abs_beginning_end'] > 0 or hyperparameters['abs_beginning_end_minimume'] > 0 else 0)
                else:
                    self.loss_gen_beginning_end_b_ba_s.append(0)

                self.abs_beginning_end_w_conf = hyperparameters['abs_beginning_end'] * (hyperparameters['abs_beginning_end_less_by'] ** iterations)
                self.abs_beginning_end_w_conf = max(self.abs_beginning_end_w_conf, hyperparameters['abs_beginning_end_minimume'])

                if hyperparameters['do_a2b']:
                    self.loss_gen_total_s[i] += self.abs_beginning_end_w_conf * self.loss_gen_beginning_end_a_ab_s[i].cuda(self.cuda_device)
                if hyperparameters['do_b2a']:
                    self.loss_gen_total_s[i] += self.abs_beginning_end_w_conf * self.loss_gen_beginning_end_b_ba_s[i].cuda(self.cuda_device)

            # GAN loss
            if hyperparameters['gan_w'] != 0:
                i_dis = i
                if hyperparameters['gen']['useRandomDis']:
                    i_dis = np.random.randint(self.council_size)

                if hyperparameters['do_a2b']:
                    x_ab_s_curr = x_ab_s[i] if not hyperparameters['dis']['do_Dis_only_gray'] else torch.sum(x_ab_s[i], 1).unsqueeze(1).repeat(1, hyperparameters['input_dim_b'], 1, 1) / hyperparameters['input_dim_b']
                    loss_gen_adv_a2b = self.dis_a2b_s[i_dis].calc_gen_loss(x_ab_s_curr)
                else:
                    loss_gen_adv_a2b = 0

                if hyperparameters['do_b2a']:
                    x_ba_s_curr = x_ba_s[i] if not hyperparameters['dis']['do_Dis_only_gray'] else torch.sum(x_ba_s[i], 1).unsqueeze(1).repeat(1, hyperparameters['input_dim_a'], 1, 1) / hyperparameters['input_dim_a']
                    loss_gen_adv_b2a = self.dis_b2a_s[i_dis].calc_gen_loss(x_ba_s_curr)
                else:
                    loss_gen_adv_b2a = 0

                self.loss_gen_adv_a2b_s.append(loss_gen_adv_a2b)
                self.loss_gen_adv_b2a_s.append(loss_gen_adv_b2a)

                if self.do_w_loss_matching:
                    if hyperparameters['do_a2b']:
                        self.los_hist_gan_a2b_s[i].append(loss_gen_adv_a2b.detach().cpu().numpy())
                        self.los_hist_gan_a2b_s[i].popleft()
                    if hyperparameters['do_b2a']:
                        self.los_hist_gan_b2a_s[i].append(loss_gen_adv_b2a.detach().cpu().numpy())
                        self.los_hist_gan_b2a_s[i].popleft()

                if hyperparameters['do_a2b']:
                    self.loss_gen_total_s[i] += hyperparameters['gan_w'] * self.loss_gen_adv_a2b_s[i].cuda(self.cuda_device)
                if hyperparameters['do_b2a']:
                    self.loss_gen_total_s[i] += hyperparameters['gan_w'] * self.loss_gen_adv_b2a_s[i].cuda(self.cuda_device)

            # domain-invariant perceptual loss
            if hyperparameters['vgg_w'] != 0:
                self.loss_gen_vgg_a_s.append(
                    self.compute_vgg_loss(self.vgg, x_ba_s[i], x_b) if hyperparameters['vgg_w'] > 0 else 0)
                self.loss_gen_vgg_b_s.append(
                    self.compute_vgg_loss(self.vgg, x_ab_s[i], x_a) if hyperparameters['vgg_w'] > 0 else 0)
                self.loss_gen_total_s[i] += hyperparameters['vgg_w'] * (
                        self.loss_gen_vgg_a_s[i].cuda(self.cuda_device) + self.loss_gen_vgg_b_s[i].cuda(self.cuda_device))

        # Council loss
        onOffCycle = hyperparameters['council']['flipOnOff_On_iteration'] + hyperparameters['council'][
            'flipOnOff_Off_iteration']
        currIterCyc = hyperparameters['iteration'] % onOffCycle
        if hyperparameters['council']['flipOnOff_start_with']:
            startCyc = hyperparameters['council']['flipOnOff_On_iteration']
        else:
            startCyc = hyperparameters['council']['flipOnOff_Off_iteration']

        self.do_council_loss = hyperparameters['council']['flipOnOff_start_with'] if (currIterCyc < startCyc) \
            else not hyperparameters['council']['flipOnOff_start_with']

        if not hyperparameters['council']['flipOnOff']:
            self.do_council_loss = True
        if hyperparameters['iteration'] < hyperparameters['council']['council_start_at_iter']:
            self.do_council_loss = False
        self.council_loss_ba_s = []
        self.council_loss_ab_s = []
        for i in range(self.council_size):
            if (hyperparameters['council_w'] != 0 or hyperparameters['council_abs_w'] != 0) and self.do_council_loss and self.council_size > 1:
                # if i == 0:
                #     print('do council loss: True')
                if self.do_a2b_conf:
                    self.council_loss_ab_s.append(0)
                if self.do_b2a_conf:
                    self.council_loss_ba_s.append(0)

                if self.do_dis_council:  # do council discriminator
                    if hyperparameters['do_a2b']:
                        dis_council_loss_ab = self.dis_council_a2b_s[i].calc_gen_loss(x_ab_s[i], x_a)
                    else:
                        dis_council_loss_ab = 0
                    if hyperparameters['do_b2a']:
                        dis_council_loss_ba = self.dis_council_b2a_s[i].calc_gen_loss(x_ba_s[i], x_b)
                    else:
                        dis_council_loss_ba = 0
                    if self.do_w_loss_matching:
                        if hyperparameters['do_a2b']:
                            self.los_hist_council_a2b_s[i].append(dis_council_loss_ab.detach().cpu().numpy())
                            self.los_hist_council_a2b_s[i].popleft()
                            self.w_match_a2b_conf = np.mean(self.los_hist_gan_a2b_s[i]) / np.mean(self.los_hist_council_a2b_s[i])
                            dis_council_loss_ab *= self.w_match_a2b_conf
                        if hyperparameters['do_b2a']:
                            self.los_hist_council_b2a_s[i].append(dis_council_loss_ba.detach().cpu().numpy())
                            self.los_hist_council_b2a_s[i].popleft()
                            self.w_match_b2a_conf = np.mean(self.los_hist_gan_b2a_s[i]) / np.mean(self.los_hist_council_b2a_s[i])
                            dis_council_loss_ba *= self.w_match_b2a_conf

                    if hyperparameters['do_a2b']:
                        dis_council_loss_ab *= hyperparameters['council_w']
                        self.council_loss_ab_s[i] += dis_council_loss_ab
                    if hyperparameters['do_b2a']:
                        dis_council_loss_ba *= hyperparameters['council_w']
                        self.council_loss_ba_s[i] += dis_council_loss_ba

                if hyperparameters['council_abs_w'] != 0 and self.council_size > 1:  # ads loss without discriminetor
                    tmp = list(range(0, i)) + list(range(i + 1, self.council_size))
                    comper_to_i = random.choice(tmp)
                    if hyperparameters['council_abs_gray_scale']:
                        if hyperparameters['do_a2b']:
                            abs_council_loss_ab = hyperparameters['council_abs_w'] * self.council_basic_criterion_gray_scale(x_ab_s[i], x_ab_s[comper_to_i].detach())
                        else:
                            abs_council_loss_ab = 0
                        if hyperparameters['do_b2a']:
                            abs_council_loss_ba = hyperparameters['council_abs_w'] * self.council_basic_criterion_gray_scale(x_ba_s[i], x_ba_s[comper_to_i].detach())
                        else:
                            abs_council_loss_ba = 0
                    else:
                        if hyperparameters['do_a2b']:
                            abs_council_loss_ab = hyperparameters['council_abs_w'] * self.council_basic_criterion_with_color(x_ab_s[i], x_ab_s[comper_to_i].detach())
                        else:
                            abs_council_loss_ab = 0
                        if hyperparameters['do_b2a']:
                            abs_council_loss_ba = hyperparameters['council_abs_w'] * self.council_basic_criterion_with_color(x_ba_s[i], x_ba_s[comper_to_i].detach())
                        else:
                            abs_council_loss_ba = 0
                    if self.do_a2b_conf:
                        self.council_loss_ab_s[i] += abs_council_loss_ba.cuda(self.cuda_device)
                    if self.do_b2a_conf:
                        self.council_loss_ba_s[i] += abs_council_loss_ab.cuda(self.cuda_device)

                if hyperparameters['do_a2b']:
                    self.loss_gen_total_s[i] += self.council_loss_ab_s[i].cuda(self.cuda_device)
                if hyperparameters['do_b2a']:
                        self.loss_gen_total_s[i] += self.council_loss_ba_s[i].cuda(self.cuda_device)

            else:
                if self.do_a2b_conf:
                    self.council_loss_ab_s.append(0)
                if self.do_b2a_conf:
                    self.council_loss_ba_s.append(0)

            # backpropogation
            self.loss_gen_total_s[i].backward()
            self.gen_opt_s[i].step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a=None, x_b=None, s_a=None, s_b=None, council_member_to_sample_vec=None, return_mask=True):
        self.eval()
        if self.do_a2b_conf:
            x_a_s = []
            s_b = self.s_b if s_b is None else s_b
            s_b1 = Variable(s_b)
            s_b2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda(self.cuda_device))
            x_a_recon, x_ab1, x_ab2, x_ab1_mask = [], [], [], []
        if self.do_b2a_conf:
            x_b_s = []
            s_a = self.s_a if s_a is None else s_a
            s_a1 = Variable(s_a)
            s_a2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(self.cuda_device))
            x_b_recon, x_ba1, x_ba2, x_ba1_mask = [], [], [], []

        council_member_to_sample_vec = range(self.council_size) if council_member_to_sample_vec is None else council_member_to_sample_vec
        x_size = x_a.size(0) if x_a is not None else x_b.size(0)
        for i in range(x_size):
            for j in council_member_to_sample_vec:
                if self.do_b2a_conf:
                    x_b_s.append(x_b[i].unsqueeze(0))
                    c_b, s_b_fake = self.gen_b2a_s[j].encode(x_b[i].unsqueeze(0))
                    if not return_mask:
                        x_b_recon.append(self.gen_b2a_s[j].decode(c_b, s_b_fake, x_b[i].unsqueeze(0)))
                        x_ba1.append(self.gen_b2a_s[j].decode(c_b, s_a1[i].unsqueeze(0), x_b[i].unsqueeze(0)))
                        x_ba2.append(self.gen_b2a_s[j].decode(c_b, s_a2[i].unsqueeze(0), x_b[i].unsqueeze(0)))
                    else:
                        x_ba1_tmp, x_ba1_mask_tmp = self.gen_b2a_s[j].decode(c_b, s_a1[i].unsqueeze(0), x_b[i].unsqueeze(0), return_mask=return_mask)
                        x_ba1_mask.append(x_ba1_mask_tmp)
                        x_ba1.append(x_ba1_tmp)
                        x_ba2.append(self.gen_b2a_s[j].decode(c_b, s_a2[i].unsqueeze(0), x_b[i].unsqueeze(0)))
                if self.do_a2b_conf:
                    x_a_s.append(x_a[i].unsqueeze(0))
                    c_a, s_a_fake = self.gen_a2b_s[j].encode(x_a[i].unsqueeze(0))
                    if not return_mask:
                        x_a_recon.append(self.gen_a2b_s[j].decode(c_a, s_a_fake, x_a[i].unsqueeze(0)))
                        x_ab1.append(self.gen_a2b_s[j].decode(c_a, s_b1[i].unsqueeze(0), x_a[i].unsqueeze(0)))
                        x_ab2.append(self.gen_a2b_s[j].decode(c_a, s_b2[i].unsqueeze(0), x_a[i].unsqueeze(0)))
                    else:
                        x_ab1_tmp, x_ab1_mask_tmp = self.gen_a2b_s[j].decode(c_a, s_b1[i].unsqueeze(0), x_a[i].unsqueeze(0), return_mask=return_mask)
                        do_double = False
                        if do_double:
                            c_a_double, s_a_fake = self.gen_a2b_s[j].encode(x_ab1_tmp)
                            x_ab1_tmp, x_ab1_mask_tmp = self.gen_a2b_s[j].decode(c_a_double, s_b1[i].unsqueeze(0),
                                                                               x_ab1_tmp,
                                                                               return_mask=return_mask)

                        x_ab1_mask.append(x_ab1_mask_tmp)
                        x_ab1.append(x_ab1_tmp)
                        x_ab2.append(self.gen_a2b_s[j].decode(c_a, s_b2[i].unsqueeze(0), x_a[i].unsqueeze(0)))

        if self.do_b2a_conf:
            x_b_s = torch.cat(x_b_s)
            x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
            if not return_mask:
                x_b_recon = torch.cat(x_b_recon)
            else:
                x_ba1_mask = torch.cat(x_ba1_mask)
        if self.do_a2b_conf:
            x_a_s = torch.cat(x_a_s)
            x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
            if not return_mask:
                x_a_recon = torch.cat(x_a_recon)
            else:
                x_ab1_mask = torch.cat(x_ab1_mask)

        self.train()

        do_diff = False
        if do_diff:
            if self.do_a2b_conf:
                x_ab1 = x_a_s - x_ab1
                x_ab2 = x_a_s - x_ab2
            if self.do_b2a_conf:
                x_ba1 = x_b_s - x_ba1
                x_ba2 = x_b_s - x_ba2

        if not return_mask:
            if self.do_a2b_conf and self.do_b2a_conf:
                return x_a_s, x_a_recon, x_ab1, x_ab2, x_b_s, x_b_recon, x_ba1, x_ba2
            if self.do_a2b_conf:
                return x_a_s, x_a_recon, x_ab1, x_ab2, None, None, None, None
            if self.do_b2a_conf:
                return None, None, None, None, x_b_s, x_b_recon, x_ba1, x_ba2
        else:
            if self.do_a2b_conf and self.do_b2a_conf:
                return x_a_s, x_ab1_mask, x_ab1, x_ab2, x_b_s, x_ba1_mask, x_ba1, x_ba2
            if self.do_a2b_conf:
                return x_a_s, x_ab1_mask, x_ab1, x_ab2, None, None, None, None
            if self.do_b2a_conf:
                return None, None, None, None, x_b_s, x_ba1_mask, x_ba1, x_ba2

    def dis_update(self, x_a=None, x_b=None, hyperparameters=None):
        x_a_dis = x_a if not hyperparameters['dis']['do_Dis_only_gray'] else torch.sum(x_a.detach(), 1).unsqueeze(1).repeat(1, hyperparameters['input_dim_a'], 1, 1) / hyperparameters['input_dim_a']
        x_b_dis = x_b if not hyperparameters['dis']['do_Dis_only_gray'] else torch.sum(x_b.detach(), 1).unsqueeze(1).repeat(1, hyperparameters['input_dim_b'], 1, 1) / hyperparameters['input_dim_b']
        for dis_opt in self.dis_opt_s:
            dis_opt.zero_grad()
        if self.do_a2b_conf:
            s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(self.cuda_device))
            self.loss_dis_a2b_s = []
        if self.do_b2a_conf:
            s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda(self.cuda_device))
            self.loss_dis_b2a_s = []
        self.loss_dis_total_s = []
        for i in range(self.council_size):
            i_gen = i
            if hyperparameters['dis']['useRandomGen']:
                i_gen = np.random.randint(self.council_size)

            # encode
            if hyperparameters['do_a2b']:
                c_a, _ = self.gen_a2b_s[i_gen].encode(x_a)
            if hyperparameters['do_b2a']:
                c_b, _ = self.gen_b2a_s[i_gen].encode(x_b)

            # decode (cross domain)
            if hyperparameters['do_a2b']:
                x_ab = self.gen_a2b_s[i_gen].decode(c_a, s_b, x_a)
                x_ab = x_ab if not hyperparameters['dis']['do_Dis_only_gray'] else torch.sum(x_ab.detach(), 1).unsqueeze(1).repeat(1, hyperparameters['input_dim_b'], 1, 1) / hyperparameters['input_dim_b']

            if hyperparameters['do_b2a']:
                x_ba = self.gen_b2a_s[i_gen].decode(c_b, s_a, x_b)
                x_ba = x_ba if not hyperparameters['dis']['do_Dis_only_gray'] else torch.sum(x_ba.detach(), 1).unsqueeze(1).repeat(1, hyperparameters['input_dim_a'], 1, 1) / hyperparameters['input_dim_a']

            # D loss
            if hyperparameters['do_a2b']:
                self.loss_dis_a2b_s.append(self.dis_a2b_s[i].calc_dis_loss(x_ab.detach(), x_b_dis))
            if hyperparameters['do_b2a']:
                self.loss_dis_b2a_s.append(self.dis_b2a_s[i].calc_dis_loss(x_ba.detach(), x_a_dis))

            self.loss_dis_total_s.append(0)
            if hyperparameters['do_a2b']:
                self.loss_dis_total_s[i] += hyperparameters['gan_w'] * self.loss_dis_a2b_s[i]
            if hyperparameters['do_b2a']:
                self.loss_dis_total_s[i] += self.loss_dis_b2a_s[i]

            self.loss_dis_total_s[i].backward()
            self.dis_opt_s[i].step()

    def dis_council_update(self, x_a=None, x_b=None, hyperparameters=None):

        if self.council_size <= 1 or hyperparameters['council']['numberOfCouncil_dis_relative_iteration'] == 0:
            print('no council discriminetor is needed (council size <= 1 or numberOfCouncil_dis_relative_iteration == 0)')
            return
        onOffCycle = hyperparameters['council']['flipOnOff_On_iteration'] + hyperparameters['council'][
            'flipOnOff_Off_iteration']
        currIterCyc = hyperparameters['iteration'] % onOffCycle
        if hyperparameters['council']['flipOnOff_start_with']:
            startCyc = hyperparameters['council']['flipOnOff_On_iteration']
        else:
            startCyc = hyperparameters['council']['flipOnOff_Off_iteration']

        self.do_council_loss = hyperparameters['council']['flipOnOff_start_with'] if (currIterCyc < startCyc) \
            else not hyperparameters['council']['flipOnOff_start_with']
        if not hyperparameters['council']['flipOnOff']:
            self.do_council_loss = hyperparameters['council']['flipOnOff_start_with']

        if not self.do_council_loss or hyperparameters['council_w'] == 0 or hyperparameters['iteration'] < hyperparameters['council']['council_start_at_iter']:
            return

        for dis_council_opt in self.dis_council_opt_s:
            dis_council_opt.zero_grad()

        if self.do_b2a_conf:
            s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda(self.cuda_device))
        if self.do_a2b_conf:
            s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(self.cuda_device))
        if hyperparameters['council']['discriminetro_less_style_by'] != 0:
            if self.do_b2a_conf:
                s_a_less = s_a * hyperparameters['council']['discriminetro_less_style_by']
            if self.do_a2b_conf:
                s_b_less = s_b * hyperparameters['council']['discriminetro_less_style_by']

        self.loss_dis_council_a2b_s = []
        self.loss_dis_council_b2a_s = []
        self.loss_dis_council_total_s = []
        c_a_s = []
        c_b_s = []
        x_ba_s = []
        x_ab_s = []
        x_ba_s_less = []
        x_ab_s_less = []

        for i in range(self.council_size):
            # encode
            if hyperparameters['do_a2b']:
                c_a, _ = self.gen_a2b_s[i].encode(x_a)
                c_a_s.append(c_a)
            if hyperparameters['do_b2a']:
                c_b, _ = self.gen_b2a_s[i].encode(x_b)
                c_b_s.append(c_b)

            # decode (cross domain)
            if hyperparameters['do_a2b']:
                x_ab = self.gen_a2b_s[i].decode(c_a, s_b, x_a)
                x_ab_s.append(x_ab)
            if hyperparameters['do_b2a']:
                x_ba = self.gen_b2a_s[i].decode(c_b, s_a, x_b)
                x_ba_s.append(x_ba)

            if hyperparameters['council']['discriminetro_less_style_by'] != 0:
                # decode (cross domain) less_style_by
                if hyperparameters['do_a2b']:
                    x_ab_less = self.gen_a2b_s[i].decode(c_a, s_b_less, x_a)
                    x_ab_s_less.append(x_ab_less)

                if hyperparameters['do_b2a']:
                    x_ba_less = self.gen_b2a_s[i].decode(c_b, s_a_less, x_b)
                    x_ba_s_less.append(x_ba_less)

        if self.do_a2b_conf:
            comper_x_ab_s = x_ab_s if hyperparameters['council']['discriminetro_less_style_by'] == 0 else x_ab_s_less
        if self.do_b2a_conf:
            comper_x_ba_s = x_ba_s if hyperparameters['council']['discriminetro_less_style_by'] == 0 else x_ba_s_less

        for i in range(self.council_size):
            self.loss_dis_council_a2b_s.append(0)
            self.loss_dis_council_b2a_s.append(0)
            index_to_chose_from = list(range(0, i)) + list(range(i + 1, self.council_size))
            for k in range(hyperparameters['council']['numberOfCouncil_dis_relative_iteration']):
                if k == self.council_size:
                    break
                if len(index_to_chose_from) == 0:
                    index_to_chose_from = list(range(0, i)) + list(range(i + 1, self.council_size)) # reinitilize the indexes to chose from if numberOfCouncil_dis_relative_iteration is biger then thr number of council members
                index_to_comper = random.choice(index_to_chose_from)
                index_to_chose_from.remove(index_to_comper)

                # D loss
                if hyperparameters['do_a2b']:
                    self.loss_dis_council_a2b_s[i] += self.dis_council_a2b_s[i].calc_dis_loss(x_ab_s[i].detach(), comper_x_ab_s[index_to_comper].detach(), x_a)  # original
                if hyperparameters['do_b2a']:
                    self.loss_dis_council_b2a_s[i] += self.dis_council_b2a_s[i].calc_dis_loss(x_ba_s[i].detach(), comper_x_ba_s[index_to_comper].detach(), x_b)  # original

            self.loss_dis_council_total_s.append(0)
            if hyperparameters['do_a2b']:
                self.loss_dis_council_total_s[i] += hyperparameters['council_w'] * self.loss_dis_council_a2b_s[i] / hyperparameters['council']['numberOfCouncil_dis_relative_iteration']
            if hyperparameters['do_b2a']:
                self.loss_dis_council_total_s[i] += hyperparameters['council_w'] * self.loss_dis_council_b2a_s[i] / hyperparameters['council']['numberOfCouncil_dis_relative_iteration']

            self.loss_dis_council_total_s[i].backward()
            self.dis_council_opt_s[i].step()

    def update_learning_rate(self):
        for dis_scheduler in self.dis_scheduler_s:
            if dis_scheduler is not None:
                dis_scheduler.step()
        for gen_scheduler in self.gen_scheduler_s:
            if gen_scheduler is not None:
                gen_scheduler.step()
        if not self.do_dis_council:
            return
        for dis_council_scheduler in self.dis_council_scheduler_s:
            if dis_council_scheduler is not None:
                dis_council_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        iterations = 0
        # Load generators
        for i in range(self.council_size):
            last_model_name = get_model_list(checkpoint_dir, "gen_" + str(i))
            if last_model_name is not None:
                last_model_name = last_model_name.replace('a2b_gen_', 'gen_').replace('b2a_gen_', 'gen_')
                print('loading: ' + last_model_name)
                if self.do_a2b_conf:
                    state_dict = torch.load(last_model_name.replace('gen_', 'a2b_gen_'), map_location=torch.device(self.cuda_device))
                    self.gen_a2b_s[i].load_state_dict(state_dict['a2b'])
                if self.do_b2a_conf:
                    state_dict = torch.load(last_model_name.replace('gen_', 'b2a_gen_'), map_location=torch.device(self.cuda_device))
                    self.gen_b2a_s[i].load_state_dict(state_dict['b2a'])
                iterations = int(last_model_name[-11:-3])
            else:
                warnings.warn('Failed to find gen checkpoint, did not load model')

            # Load discriminators
            last_model_name = get_model_list(checkpoint_dir, "dis_" + str(i))
            if last_model_name is not None:
                last_model_name = last_model_name.replace('a2b_dis_', 'dis_').replace('b2a_dis_', 'dis_')
                print('loading: ' + last_model_name)
                if self.do_a2b_conf:
                    state_dict = torch.load(last_model_name.replace('dis_', 'a2b_dis_'), map_location=torch.device(self.cuda_device))
                    self.dis_a2b_s[i].load_state_dict(state_dict['a2b'])
                if self.do_b2a_conf:
                    state_dict = torch.load(last_model_name.replace('dis_', 'b2a_dis_'), map_location=torch.device(self.cuda_device))
                    self.dis_b2a_s[i].load_state_dict(state_dict['b2a'])
            else:
                warnings.warn('Failed to find dis checkpoint, did not load model')
            # Load council discriminators
            if self.do_dis_council:
                try:
                    last_model_name = get_model_list(checkpoint_dir, "dis_council_" + str(i))
                    print('loading: ' + last_model_name)
                    if last_model_name is not None:
                        last_model_name = last_model_name.replace('a2b_dis_council_', 'dis_council_').replace('b2a_dis_council_', 'dis_council_')

                        if self.do_a2b_conf:
                            state_dict = torch.load(last_model_name.replace('dis_council_', 'a2b_dis_council_'), map_location=torch.device(self.cuda_device))
                            self.dis_council_a2b_s[i].load_state_dict(state_dict['a2b'])
                        if self.do_b2a_conf:
                            state_dict = torch.load(last_model_name.replace('dis_council_', 'b2a_dis_council_'), map_location=torch.device(self.cuda_device))
                            self.dis_council_b2a_s[i].load_state_dict(state_dict['b2a'])
                    else:
                        warnings.warn('Failed to find dis checkpoint, did not load model')
                except:
                    warnings.warn('some council discriminetor FAILED to load')

            # Load optimizers
            try:
                state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer_' + str(i) + '.pt'), map_location=torch.device(self.cuda_device))
                self.dis_opt_s[i].load_state_dict(state_dict['dis'])
                self.gen_opt_s[i].load_state_dict(state_dict['gen'])
                if self.do_dis_council:
                    self.dis_council_opt_s[i].load_state_dict(state_dict['dis_council'])

                # Reinitilize schedulers
                self.dis_scheduler_s[i] = get_scheduler(self.dis_opt_s[i], hyperparameters, iterations)
                self.gen_scheduler = get_scheduler(self.gen_opt_s[i], hyperparameters, iterations)
                if self.do_dis_council:
                    self.dis_council_scheduler_s[i] = get_scheduler(self.dis_council_opt_s[i], hyperparameters, iterations)
            except:
                warnings.warn('some optimizer FAILED to load ')
        if iterations > 0 :
            print('Resume from iteration %d' % iterations)
        else:
            warnings.warn('FAILED TO RESUME STARTED FROM 0')
        return iterations

    def save(self, snapshot_dir, iterations):
        for i in range(self.council_size):

            # Save generators, discriminators, and optimizers
            gen_name = os.path.join(snapshot_dir, 'gen_' + str(i) + '_%08d.pt' % (iterations + 1))
            dis_name = os.path.join(snapshot_dir, 'dis_' + str(i) + '_%08d.pt' % (iterations + 1))
            if self.do_dis_council:
                dis_council_name = os.path.join(snapshot_dir, 'dis_council_' + str(i) + '_%08d.pt' % (iterations + 1))
            opt_name = os.path.join(snapshot_dir, 'optimizer_' + str(i) + '.pt')
            if self.do_a2b_conf:
                torch.save({'a2b': self.gen_a2b_s[i].state_dict()}, gen_name.replace('gen_', 'a2b_gen_'))
                torch.save({'a2b': self.dis_a2b_s[i].state_dict()}, dis_name.replace('dis_', 'a2b_dis_'))
            if self.do_b2a_conf:
                torch.save({'b2a': self.gen_b2a_s[i].state_dict()}, gen_name.replace('gen_', 'b2a_gen_'))
                torch.save({'b2a': self.dis_b2a_s[i].state_dict()}, dis_name.replace('dis_', 'b2a_dis_'))
            if self.do_dis_council:
                if self.do_a2b_conf:
                    torch.save({'a2b': self.dis_council_a2b_s[i].state_dict()}, dis_council_name.replace('dis_council_', 'a2b_dis_council_'))
                if self.do_b2a_conf:
                    torch.save({'b2a': self.dis_council_b2a_s[i].state_dict()},  dis_council_name.replace('dis_council_', 'b2a_dis_council_'))
                torch.save({'gen': self.gen_opt_s[i].state_dict(), 'dis': self.dis_opt_s[i].state_dict(),
                            'dis_council': self.dis_council_opt_s[i].state_dict()}, opt_name)
            else:
                torch.save({'gen': self.gen_opt_s[i].state_dict(), 'dis': self.dis_opt_s[i].state_dict()}, opt_name)

#%%
