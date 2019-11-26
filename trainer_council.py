"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, MsImageDisCouncil, VAEGen, MINEnet
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

# from simple_classifier import ClassifierTest

class Council_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(Council_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.council_size = hyperparameters['council']['council_size']
        self.council_size_conf = self.council_size
        self.gen_a_s = []
        self.gen_b_s = []
        self.dis_a_s = []
        self.dis_b_s = []
        self.do_dis_council = hyperparameters['council_w'] != 0
        self.do_ads_council_loss = hyperparameters['council_abs_w'] != 0
        self.numberOfCouncil_dis_relative_iteration_conf = hyperparameters['council']['numberOfCouncil_dis_relative_iteration']  # self.council_size - 1 # todo chagne to difrent number maybe not self.council size
        self.discriminetro_less_style_by_conf = hyperparameters['council']['discriminetro_less_style_by']

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
        self.start_On_iteration_conf = hyperparameters['council']['start_On_iteration']

        self.council_abs_w_conf = hyperparameters['council_abs_w']
        self.council_w_conf = hyperparameters['council_w']
        self.mask_zero_or_one_w_conf = hyperparameters['mask_zero_or_one_w']
        self.mask_total_w_conf = hyperparameters['mask_total_w']
        self.batch_size_conf = hyperparameters['batch_size']
        self.do_w_loss_matching = hyperparameters['do_w_loss_matching']
        self.los_matching_hist_size_conf = hyperparameters['loss_matching_hist_size']
        self.do_a2b_conf = hyperparameters['do_a2b']
        self.do_b2a_conf = hyperparameters['do_b2a']
        self.w_match_a_conf = 1
        self.w_match_b_conf = 1

        if self.do_a2b_conf:
            self.los_hist_gan_a2b_s = []
            self.los_hist_council_a2b_s = []
        if self.do_b2a_conf:
            self.los_hist_gan_b2a_s = []
            self.los_hist_council_b2a_s = []
        for ind in range(self.council_size):
            if self.do_a2b_conf:
                self.los_hist_gan_a2b_s.append(deque(np.ones(self.los_matching_hist_size_conf)))
                self.los_hist_council_a2b_s.append(deque(np.ones(self.los_matching_hist_size_conf)))
            if self.do_b2a_conf:
                self.los_hist_gan_b2a_s.append(deque(np.ones(self.los_matching_hist_size_conf)))
                self.los_hist_council_b2a_s.append(deque(np.ones(self.los_matching_hist_size_conf)))



        self.do_mine_test = False
        self.do_council_loss = None

        if self.do_dis_council:
            self.dis_council_b_s = []
            self.dis_council_a_s = []

        # defining all the networks
        for i in range(self.council_size):
            if self.do_a2b_conf:
                self.gen_a2b_s.append(
                    AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen']))  # auto-encoder for domain a2b
                self.dis_a2b_s.append(
                    MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis']))  # discriminator for domain a2b
                if self.do_dis_council:
                    self.dis_council_a2b_s.append(
                        MsImageDisCouncil(hyperparameters['input_dim_a'],
                                          hyperparameters['dis']))  # council discriminator for domain a2b
            if self.do_b2a_conf:
                self.gen_b2a_s.append(
                    AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen']))  # auto-encoder for domain b
                self.dis_b2a_s.append(
                    MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis']))  # discriminator for domain b
                if self.do_dis_council:
                    self.dis_council_b2a_s.append(#
                        MsImageDisCouncil(hyperparameters['input_dim_b'],
                                          hyperparameters['dis']))  # discriminator for domain b




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
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

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

        dis_parms = []
        gen_parms = []
        dis_council_parms = []
        for i in range(self.council_size):
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

    def mask_zero_one_criterion(self, mask):
        return torch.sum(1 / (torch.abs(mask - 0.5) + 0.01)) / mask.numel()

    def mask_small_criterion(self, mask):
        # return torch.sum(mask) / mask.numel()
        return torch.sum(mask) ** 2 / mask.numel()

    def forward(self, x_a, x_b=None, s_a=None, s_b=None):
        self.eval()
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
                c_b, s_b_fake = self.gen_b2a.encode(x_b)
                x_ba_s.append(self.gen_b2a.decode(c_b, s_a, x_b))

        self.train()
        if self.do_a2b_conf and self.do_b2a_conf:
            return x_ab_s, x_ba_s
        elif self.do_b2a_conf:
            return x_ba_s
        return x_ab_s


    def gen_update(self, x_a, x_b, hyperparameters, iterations=0):
        for gen_opt in self.gen_opt_s:
            gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
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
        self.loss_gen_adv_a_s = []
        self.loss_gen_adv_b_s = []
        self.loss_gen_vgg_a_s = []
        self.loss_gen_vgg_b_s = []
        self.loss_gen_total_s = []



        for i in range(self.council_size):
            # encode
            c_a, s_a_prime = self.gen_a_s[i].encode(x_a)
            c_b, s_b_prime = self.gen_b_s[i].encode(x_b)
            c_a_s.append(c_a)
            s_a_prime_s.append(s_a_prime)
            c_b_s.append(c_b)
            s_b_prime_s.append(s_b_prime)
            # decode (within domain)
            if hyperparameters['recon_x_w'] != 0:
                x_a_recon_s.append(self.gen_a_s[i].decode(c_a_s[i], s_a_prime_s[i], x_a))
                x_b_recon_s.append(self.gen_b_s[i].decode(c_b_s[i], s_b_prime_s[i], x_b))
            # decode (cross domain)
            x_ba_s.append(self.gen_a_s[i].decode(c_b_s[i], s_a, x_b))
            x_ab_s.append(self.gen_b_s[i].decode(c_a_s[i], s_b, x_a))
            if hyperparameters['mask_zero_or_one_w'] != 0 or hyperparameters['mask_total_w'] != 0:
                mask_ba_s.append(self.gen_a_s[i].dec.mask_s)
                mask_ab_s.append(self.gen_b_s[i].dec.mask_s)
            # encode again
            if hyperparameters['recon_s_w'] != 0 or hyperparameters['recon_c_w'] != 0 or hyperparameters[
                'recon_x_cyc_w'] != 0:
                c_b_recon, s_a_recon = self.gen_a_s[i].encode(x_ba_s[i])
                c_a_recon, s_b_recon = self.gen_b_s[i].encode(x_ab_s[i])
                c_b_recon_s.append(c_b_recon)
                s_a_recon_s.append(s_a_recon)
                c_a_recon_s.append(c_a_recon)
                s_b_recon_s.append(s_b_recon)
            # decode again (if needed)
            if hyperparameters['recon_x_cyc_w'] != 0:
                x_aba_s.append(
                    self.gen_a.decode(c_a_recon_s[i], s_a_prime_s[i], x_a) if hyperparameters['recon_x_cyc_w'] > 0 else None)
                x_bab_s.append(
                    self.gen_b.decode(c_b_recon_s[i], s_b_prime_s[i], x_b) if hyperparameters['recon_x_cyc_w'] > 0 else None)

            self.loss_gen_total_s.append(0)

            # masks should be make up of ones or zeros
            if hyperparameters['mask_zero_or_one_w'] != 0:
                self.loss_gen_mask_zero_one_ba_s.append(self.mask_zero_one_criterion(mask_ba_s[i]))
                self.loss_gen_mask_zero_one_ab_s.append(self.mask_zero_one_criterion(mask_ab_s[i]))
                self.loss_gen_total_s[i] += hyperparameters['mask_zero_or_one_w'] * (
                        self.loss_gen_mask_zero_one_ba_s[i] + self.loss_gen_mask_zero_one_ab_s[i])

            # masks should as small as posible to leave to original domain intacet
            if hyperparameters['mask_total_w'] != 0:
                self.loss_gen_mask_total_ba_s.append(self.mask_small_criterion(mask_ba_s[i]))
                self.loss_gen_mask_total_ab_s.append(self.mask_small_criterion(mask_ab_s[i]))
                self.loss_gen_total_s[i] += hyperparameters['mask_total_w'] * (
                        self.loss_gen_mask_total_ba_s[i] + self.loss_gen_mask_total_ab_s[i])


            # reconstruction loss
            if hyperparameters['recon_x_w'] != 0:
                self.loss_gen_recon_x_a_s.append(self.recon_criterion(x_a_recon_s[i], x_a))
                self.loss_gen_recon_x_b_s.append(self.recon_criterion(x_b_recon_s[i], x_b))
                self.loss_gen_total_s[i] += hyperparameters['recon_x_w'] * (
                        self.loss_gen_recon_x_a_s[i] + self.loss_gen_recon_x_b_s[i])
            if hyperparameters['recon_s_w'] != 0:
                self.loss_gen_recon_s_a_s.append(self.recon_criterion(s_a_recon_s[i], s_a))
                self.loss_gen_recon_s_b_s.append(self.recon_criterion(s_b_recon_s[i], s_b))
                self.loss_gen_total_s[i] += hyperparameters['recon_s_w'] * (
                        self.loss_gen_recon_s_a_s[i] + self.loss_gen_recon_s_b_s[i])
            if hyperparameters['recon_c_w'] != 0:
                self.loss_gen_recon_c_a_s.append(self.recon_criterion(c_a_recon_s[i], c_a_s[i]))
                self.loss_gen_recon_c_b_s.append(self.recon_criterion(c_b_recon_s[i], c_b_s[i]))
                self.loss_gen_total_s[i] += hyperparameters['recon_c_w'] * (
                        self.loss_gen_recon_c_a_s[i] + self.loss_gen_recon_c_b_s[i])
            if hyperparameters['recon_x_cyc_w'] != 0:
                self.loss_gen_cycrecon_x_a_s.append(
                    self.recon_criterion(x_aba_s[i], x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0)
                self.loss_gen_cycrecon_x_b_s.append(
                    self.recon_criterion(x_bab_s[i], x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0)
                self.loss_gen_total_s[i] += hyperparameters['recon_x_cyc_w'] * (
                        self.loss_gen_cycrecon_x_a_s[i] + self.loss_gen_cycrecon_x_b_s[i])
            if hyperparameters['abs_beginning_end'] != 0 and self.abs_beginning_end_w_conf > 0.005:
                if hyperparameters['do_a2b']:
                    self.loss_gen_beginning_end_a_ab_s.append(
                        self.recon_criterion_v2_color(x_ab_s[i], x_a) if hyperparameters['abs_beginning_end'] > 0 or hyperparameters['abs_beginning_end_minimume'] > 0 else 0)
                        # self.recon_criterion_v3_gray_scale(x_ab_s[i], x_a) if hyperparameters['abs_beginning_end'] > 0 else 0)
                else:
                    self.loss_gen_beginning_end_a_ab_s.append(0)
                if hyperparameters['do_b2a']:
                    self.loss_gen_beginning_end_b_ba_s.append(
                        self.recon_criterion_v2_color(x_ba_s[i], x_b) if hyperparameters['abs_beginning_end'] > 0 or hyperparameters['abs_beginning_end_minimume'] > 0 else 0)
                        # self.recon_criterion_v3_gray_scale(x_ba_s[i], x_b) if hyperparameters['abs_beginning_end'] > 0 else 0)
                else:
                    self.loss_gen_beginning_end_b_ba_s.append(0)
                self.abs_beginning_end_w_conf = hyperparameters['abs_beginning_end'] * (hyperparameters['abs_beginning_end_less_by'] ** iterations)  # TODO move
                self.abs_beginning_end_w_conf = max(self.abs_beginning_end_w_conf, hyperparameters['abs_beginning_end_minimume'])

                if hyperparameters['do_a2b']:
                    self.loss_gen_total_s[i] += self.abs_beginning_end_w_conf * self.loss_gen_beginning_end_a_ab_s[i]
                if hyperparameters['do_b2a']:
                    self.loss_gen_total_s[i] += self.abs_beginning_end_w_conf * self.loss_gen_beginning_end_b_ba_s[i]

            # GAN loss
            if hyperparameters['gan_w'] != 0:
                i_dis = i
                if hyperparameters['gen']['useRandomDis']:
                    i_dis = np.random.randint(self.council_size)

                if hyperparameters['do_a2b']:
                    x_ab_s_curr = x_ab_s[i] if not hyperparameters['dis']['do_Dis_only_gray'] else torch.sum(x_ab_s[i], 1).unsqueeze(1).repeat(1, hyperparameters['input_dim_b'], 1, 1) / hyperparameters['input_dim_b']
                    loss_gen_adv_b = self.dis_b_s[i_dis].calc_gen_loss(x_ab_s_curr)
                else:
                    loss_gen_adv_b = 0

                if hyperparameters['do_b2a']:
                    x_ba_s_curr = x_ba_s[i] if not hyperparameters['dis']['do_Dis_only_gray'] else torch.sum(x_ba_s[i], 1).unsqueeze(1).repeat(1, hyperparameters['input_dim_a'], 1, 1) / hyperparameters['input_dim_a']
                    loss_gen_adv_a = self.dis_a_s[i_dis].calc_gen_loss(x_ba_s_curr)
                else:
                    loss_gen_adv_a = 0

                self.loss_gen_adv_b_s.append(loss_gen_adv_b)
                self.loss_gen_adv_a_s.append(loss_gen_adv_a)

                if self.do_w_loss_matching:
                    if hyperparameters['do_a2b']:
                        self.los_hist_gan_b_s[i].append(loss_gen_adv_b.detach().cpu().numpy())
                        self.los_hist_gan_b_s[i].popleft()
                    if hyperparameters['do_b2a']:
                        self.los_hist_gan_a_s[i].append(loss_gen_adv_a.detach().cpu().numpy())
                        self.los_hist_gan_a_s[i].popleft()

                if hyperparameters['do_a2b']:
                    self.loss_gen_total_s[i] += hyperparameters['gan_w'] * self.loss_gen_adv_b_s[i]
                if hyperparameters['do_b2a']:
                    self.loss_gen_total_s[i] += hyperparameters['gan_w'] * self.loss_gen_adv_a_s[i]


            # domain-invariant perceptual loss
            if hyperparameters['vgg_w'] != 0:
                self.loss_gen_vgg_a_s.append(
                    self.compute_vgg_loss(self.vgg, x_ba_s[i], x_b) if hyperparameters['vgg_w'] > 0 else 0)
                self.loss_gen_vgg_b_s.append(
                    self.compute_vgg_loss(self.vgg, x_ab_s[i], x_a) if hyperparameters['vgg_w'] > 0 else 0)
                self.loss_gen_total_s[i] += hyperparameters['vgg_w'] * (
                        self.loss_gen_vgg_a_s[i] + self.loss_gen_vgg_b_s[i])

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
            # self.do_council_loss = hyperparameters['council']['flipOnOff_start_with']
            self.do_council_loss = True

        self.council_loss_ba_s = []
        self.council_loss_ab_s = []
        for i in range(self.council_size):
            if (hyperparameters['council_w'] != 0 or hyperparameters['council_abs_w'] != 0) and self.do_council_loss and self.council_size > 1:
                # if i == 0:
                #     print('do council loss: True')
                self.council_loss_ab_s.append(0)
                self.council_loss_ba_s.append(0)


                if self.do_dis_council:  # do council discriminator
                    if hyperparameters['do_a2b']:
                        dis_council_loss_ab = self.dis_council_b_s[i].calc_gen_loss(x_ab_s[i], x_a)
                    else:
                        dis_council_loss_ab = 0
                    if hyperparameters['do_b2a']:
                        dis_council_loss_ba = self.dis_council_a_s[i].calc_gen_loss(x_ba_s[i], x_b)
                    else:
                        dis_council_loss_ba = 0
                    if self.do_w_loss_matching:
                        if hyperparameters['do_a2b']:
                            self.los_hist_council_b_s[i].append(dis_council_loss_ab.detach().cpu().numpy())
                            self.los_hist_council_b_s[i].popleft()
                        if hyperparameters['do_b2a']:
                            self.los_hist_council_a_s[i].append(dis_council_loss_ba.detach().cpu().numpy())
                            self.los_hist_council_a_s[i].popleft()

                        self.w_match_a_conf = np.mean(self.los_hist_gan_a_s[i]) / np.mean(self.los_hist_council_a_s[i])
                        self.w_match_b_conf = np.mean(self.los_hist_gan_b_s[i]) / np.mean(self.los_hist_council_b_s[i])
                        dis_council_loss_ab *= self.w_match_b_conf
                        dis_council_loss_ba *= self.w_match_a_conf
                    dis_council_loss_ab *= hyperparameters['council_w']
                    dis_council_loss_ba *= hyperparameters['council_w']
                    if hyperparameters['do_a2b']:
                        self.council_loss_ab_s[i] += dis_council_loss_ab
                    if hyperparameters['do_b2a']:
                        self.council_loss_ba_s[i] += dis_council_loss_ba


                            # print('dis_council_loss_ab: ' + str(dis_council_loss_ab.item()))
                        # print('dis_council_loss_ba: ' + str(dis_council_loss_ba.item()))

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
                    self.council_loss_ba_s[i] += abs_council_loss_ab
                    self.council_loss_ab_s[i] += abs_council_loss_ba
                    # print('abs_council_loss_ab: ' + str(abs_council_loss_ab.item()))
                    # print('abs_council_loss_ba: ' + str(abs_council_loss_ba.item()))
                if hyperparameters['council']['useJudge'] and i == 0:
                    if hyperparameters['do_a2b']:
                        self.loss_gen_total_s[i] = self.council_loss_ab_s[i]  # only update based on the council loss
                    if hyperparameters['do_b2a']:
                        self.loss_gen_total_s[i] = self.council_loss_ba_s[i]
                else:
                    if hyperparameters['do_a2b']:
                        self.loss_gen_total_s[i] += self.council_loss_ab_s[i]
                    if hyperparameters['do_b2a']:
                        self.loss_gen_total_s[i] += self.council_loss_ba_s[i]


            else:
                self.council_loss_ba_s.append(0)
                self.council_loss_ab_s.append(0)
                # if i == 0:
                #     print('do council loss: False')
            # backpropogation
            self.loss_gen_total_s[i].backward()
            self.gen_opt_s[i].step()

            # if type(self.council_loss_ba_s[i]) is not int:
            #     if i == 0:
            #         print('**********************************************')
            #     print('=======================')
            #     print('gan_w * loss_gen_adv_a_s[' + str(i) + '] = ' + str(
            #         (hyperparameters['gan_w'] * self.loss_gen_adv_a_s[i]).item()))
            #     print('gan_w * loss_gen_adv_b_s[' + str(i) + '] = ' + str(
            #         (hyperparameters['gan_w'] * self.loss_gen_adv_b_s[i]).item()))
            #     print('council_w * council_loss_ba_s[' + str(i) + '] = ' + str(
            #         (hyperparameters['gan_w'] * self.council_loss_ba_s[i]).item()))
            #     print('council_w * council_loss_ab_s[' + str(i) + '] = ' + str(
            #         (hyperparameters['gan_w'] * self.council_loss_ab_s[i]).item()))
            #     print('=======================')

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a=None, x_b=None, s_a=None, s_b=None, council_member_to_sample_vec=None, return_mask=True):
        # council_member_to_sample_vec = list of indexes of council members to sample with
        self.eval()
        if self.do_a2b_conf:
            x_a_s = []
            s_b = self.s_b if s_b is None else s_b
            s_b1 = Variable(s_b)
            s_b2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
            x_a_recon, x_ab1, x_ab2, x_ab1_mask = [], [], [], []
        if self.do_b2a_conf:
            x_b_s = []
            s_a = self.s_a if s_a is None else s_a
            s_a1 = Variable(s_a)
            s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
            x_b_recon, x_ba1, x_ba2, x_ba1_mask = [], [], [], []

        council_member_to_sample_vec = range(self.council_size) if council_member_to_sample_vec is None else council_member_to_sample_vec
        for i in range(x_a.size(0)):
            for j in council_member_to_sample_vec:
                if self.do_b2a_conf:
                    x_b_s.append(x_b[i].unsqueeze(0))
                    c_b, s_b_fake = self.gen_b_s[j].encode(x_b[i].unsqueeze(0))
                    if not return_mask:
                        x_b_recon.append(self.gen_b_s[j].decode(c_b, s_b_fake, x_b[i].unsqueeze(0)))
                        x_ba1.append(self.gen_a_s[j].decode(c_b, s_a1[i].unsqueeze(0), x_b[i].unsqueeze(0)))
                        x_ba2.append(self.gen_a_s[j].decode(c_b, s_a2[i].unsqueeze(0), x_b[i].unsqueeze(0)))
                    else:
                        x_ba1_tmp, x_ba1_mask_tmp = self.gen_a_s[j].decode(c_b, s_a1[i].unsqueeze(0), x_b[i].unsqueeze(0), return_mask=return_mask)
                        x_ba1_mask.append(x_ba1_mask_tmp)
                        x_ba1.append(x_ba1_tmp)
                        x_ba2.append(self.gen_a_s[j].decode(c_b, s_a2[i].unsqueeze(0), x_b[i].unsqueeze(0)))
                if self.do_a2b_conf:
                    x_a_s.append(x_a[i].unsqueeze(0))
                    c_a, s_a_fake = self.gen_a_s[j].encode(x_a[i].unsqueeze(0))
                    if not return_mask:
                        x_a_recon.append(self.gen_a_s[j].decode(c_a, s_a_fake, x_a[i].unsqueeze(0)))
                        x_ab1.append(self.gen_b_s[j].decode(c_a, s_b1[i].unsqueeze(0), x_a[i].unsqueeze(0)))
                        x_ab2.append(self.gen_b_s[j].decode(c_a, s_b2[i].unsqueeze(0), x_a[i].unsqueeze(0)))
                    else:
                        x_ab1_tmp, x_ab1_mask_tmp = self.gen_b_s[j].decode(c_a, s_b1[i].unsqueeze(0), x_a[i].unsqueeze(0), return_mask=return_mask)
                        do_double = False
                        if do_double:
                            c_a_double, s_a_fake = self.gen_a_s[j].encode(x_ab1_tmp)
                            x_ab1_tmp, x_ab1_mask_tmp = self.gen_b_s[j].decode(c_a_double, s_b1[i].unsqueeze(0),
                                                                               x_ab1_tmp,
                                                                               return_mask=return_mask)

                        x_ab1_mask.append(x_ab1_mask_tmp)
                        x_ab1.append(x_ab1_tmp)
                        x_ab2.append(self.gen_b_s[j].decode(c_a, s_b2[i].unsqueeze(0), x_a[i].unsqueeze(0)))

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
            if self.do_b2b_conf:
                return None, None, None, None, x_b_s, x_ba1_mask, x_ba1, x_ba2


    def dis_update(self, x_a, x_b, hyperparameters):
        x_b_dis = x_b if not hyperparameters['dis']['do_Dis_only_gray'] else torch.sum(x_b.detach(), 1).unsqueeze(1).repeat(1, hyperparameters['input_dim_b'], 1, 1) / hyperparameters['input_dim_b']
        x_a_dis = x_a if not hyperparameters['dis']['do_Dis_only_gray'] else torch.sum(x_a.detach(), 1).unsqueeze(1).repeat(1, hyperparameters['input_dim_a'], 1, 1) / hyperparameters['input_dim_a']

        for dis_opt in self.dis_opt_s:
            dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        self.loss_dis_a_s = []
        self.loss_dis_b_s = []
        self.loss_dis_total_s = []
        for i in range(self.council_size):
            if hyperparameters['council']['useJudge'] and i == 0:  # regular discriminator is not needed for the judge
                self.loss_dis_a_s.append(0)
                self.loss_dis_b_s.append(0)
                self.loss_dis_total_s.append(0)
                continue
            i_gen = i
            if hyperparameters['dis']['useRandomGen']:
                i_gen = np.random.randint(self.council_size)
            # encode
            if hyperparameters['do_a2b']:
                c_a, _ = self.gen_a_s[i_gen].encode(x_a)
            if hyperparameters['do_b2a']:
                c_b, _ = self.gen_b_s[i_gen].encode(x_b)
            # decode (cross domain)
            if hyperparameters['do_a2b']:
                x_ab = self.gen_b_s[i_gen].decode(c_a, s_b, x_a)
            if hyperparameters['do_b2a']:
                x_ba = self.gen_a_s[i_gen].decode(c_b, s_a, x_b)

            # D loss
            if hyperparameters['do_a2b']:
                self.loss_dis_b_s.append(self.dis_b_s[i].calc_dis_loss(x_ab.detach(), x_b_dis))
            else:
                self.loss_dis_b_s.append(0)
            if hyperparameters['do_b2a']:
                self.loss_dis_a_s.append(self.dis_a_s[i].calc_dis_loss(x_ba.detach(), x_a_dis))
            else:
                self.loss_dis_a_s.append(0)

            self.loss_dis_total_s.append(
                hyperparameters['gan_w'] * self.loss_dis_a_s[i] + hyperparameters['gan_w'] * self.loss_dis_b_s[i])
            self.loss_dis_total_s[i].backward()
            self.dis_opt_s[i].step()

    def dis_council_update(self, x_a, x_b, hyperparameters):
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

        if not self.do_council_loss or hyperparameters['council_w'] == 0:
            return
        for dis_council_opt in self.dis_council_opt_s:
            dis_council_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        if hyperparameters['council']['discriminetro_less_style_by'] != 0:
            s_a_less = s_a * hyperparameters['council']['discriminetro_less_style_by']
            s_b_less = s_b * hyperparameters['council']['discriminetro_less_style_by']


        self.loss_dis_council_b_s = []
        self.loss_dis_council_a_s = []
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
                c_a, _ = self.gen_a_s[i].encode(x_a)
                c_a_s.append(c_a)
            if hyperparameters['do_b2a']:
                c_b, _ = self.gen_b_s[i].encode(x_b)
                c_b_s.append(c_b)

            # decode (cross domain)
            if hyperparameters['do_a2b']:
                x_ab = self.gen_b_s[i].decode(c_a, s_b, x_a)
                x_ab_s.append(x_ab)
            if hyperparameters['do_b2a']:
                x_ba = self.gen_a_s[i].decode(c_b, s_a, x_b)
                x_ba_s.append(x_ba)

            if hyperparameters['council']['discriminetro_less_style_by'] != 0:
                # decode (cross domain) less_style_by
                if hyperparameters['do_a2b']:
                    x_ab_less = self.gen_b_s[i].decode(c_a, s_b_less, x_a)
                    x_ab_s_less.append(x_ab_less)

                if hyperparameters['do_b2a']:
                    x_ba_less = self.gen_a_s[i].decode(c_b, s_a_less, x_b)
                    x_ba_s_less.append(x_ba_less)

        comper_x_ba_s, comper_x_ab_s = (x_ba_s, x_ab_s)\
            if hyperparameters['council']['discriminetro_less_style_by'] == 0 else \
            (x_ba_s_less, x_ab_s_less)


        for i in range(self.council_size):
            self.loss_dis_council_a_s.append(0)
            self.loss_dis_council_b_s.append(0)
            index_to_chose_from = list(range(0, i)) + list(range(i + 1, self.council_size))
            for k in range(hyperparameters['council']['numberOfCouncil_dis_relative_iteration']): # TODO maybe chagne so the same net/images wont be chossen twice
                if k == self.council_size:
                    break
                if len(index_to_chose_from) == 0:
                    index_to_chose_from = list(range(0, i)) + list(range(i + 1, self.council_size)) # reinitilize the indexes to chose from if numberOfCouncil_dis_relative_iteration is biger then thr number of council members
                index_to_comper = random.choice(index_to_chose_from)
                index_to_chose_from.remove(index_to_comper)

                # D loss
                if hyperparameters['do_a2b']:
                    self.loss_dis_council_b_s[i] += self.dis_council_b_s[i].calc_dis_loss(x_ab_s[i].detach(), comper_x_ab_s[index_to_comper].detach(), x_a)  # original
                if hyperparameters['do_b2a']:
                    self.loss_dis_council_a_s[i] += self.dis_council_a_s[i].calc_dis_loss(x_ba_s[i].detach(), comper_x_ba_s[index_to_comper].detach(), x_b)  # original


                # self.loss_dis_council_a_s.append(
                #     self.dis_council_a_s[i].calc_dis_loss(comper_x_ba_s[index_to_comper].detach(), x_ba_s[i].detach(), x_b))  # TMP
                # self.loss_dis_council_b_s.append(
                #     self.dis_council_b_s[i].calc_dis_loss(comper_x_ab_s[index_to_comper].detach(), x_ab_s[i].detach(), x_a))  # TMP

            self.loss_dis_council_total_s.append(
                hyperparameters['council_w'] * self.loss_dis_council_a_s[i] / hyperparameters['council']['numberOfCouncil_dis_relative_iteration']
                + hyperparameters['council_w'] * self.loss_dis_council_b_s[i] / hyperparameters['council']['numberOfCouncil_dis_relative_iteration'])
            self.loss_dis_council_total_s[i].backward()
            self.dis_council_opt_s[i].step()


#############################################################################################
    def mine_test_net_update(self, x_a, x_b, hyperparameters): #stoped here
        if self.council_size <= 1:
            print('no council discriminetor is needed (council size <= 1)')
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

        if not self.do_council_loss:
            return
        for dis_council_opt in self.dis_council_opt_s:
            dis_council_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        if hyperparameters['council']['discriminetro_less_style_by'] != 0:
            s_a_less = s_a * hyperparameters['council']['discriminetro_less_style_by']
            s_b_less = s_b * hyperparameters['council']['discriminetro_less_style_by']

        self.loss_dis_council_b_s = []
        self.loss_dis_council_a_s = []
        self.loss_dis_council_total_s = []
        c_a_s = []
        c_b_s = []
        x_ba_s = []
        x_ab_s = []
        x_ba_s_less = []
        x_ab_s_less = []

        for i in range(self.council_size):
            # encode
            c_a, _ = self.gen_a_s[i].encode(x_a)
            c_b, _ = self.gen_b_s[i].encode(x_b)
            c_a_s.append(c_a)
            c_b_s.append(c_b)
            # decode (cross domain)
            x_ba = self.gen_a_s[i].decode(c_b, s_a, x_b)
            x_ab = self.gen_b_s[i].decode(c_a, s_b, x_a)
            x_ab_s.append(x_ab)
            x_ba_s.append(x_ba)
            if hyperparameters['council']['discriminetro_less_style_by'] != 0:
                # decode (cross domain) less_style_by
                x_ba_less = self.gen_a_s[i].decode(c_b, s_a_less, x_b)
                x_ab_less = self.gen_b_s[i].decode(c_a, s_b_less, x_a)
                x_ab_s_less.append(x_ab_less)
                x_ba_s_less.append(x_ba_less)

        comper_x_ba_s, comper_x_ab_s = (x_ba_s, x_ab_s) \
            if hyperparameters['council']['discriminetro_less_style_by'] == 0 else \
            (x_ba_s_less, x_ab_s_less)

        for i in range(self.council_size):
            self.loss_dis_council_a_s.append(0)
            self.loss_dis_council_b_s.append(0)
            index_to_chose_from = list(range(0, i)) + list(range(i + 1, self.council_size))
            for k in range(hyperparameters['council'][
                               'numberOfCouncil_dis_relative_iteration']):  # TODO maybe chagne so the same net/images wont be chossen twice
                if k == self.council_size:
                    break
                if len(index_to_chose_from) == 0:
                    index_to_chose_from = list(range(0, i)) + list(range(i + 1,
                                                                         self.council_size))  # reinitilize the indexes to chose from if numberOfCouncil_dis_relative_iteration is biger then thr number of council members
                index_to_comper = random.choice(index_to_chose_from)
                index_to_chose_from.remove(index_to_comper)
                # D loss
                self.loss_dis_council_a_s[i] += \
                    self.dis_council_a_s[i].calc_dis_loss(x_ba_s[i].detach(), comper_x_ba_s[index_to_comper].detach(),
                                                          x_b)  # original
                self.loss_dis_council_b_s[i] += \
                    self.dis_council_b_s[i].calc_dis_loss(x_ab_s[i].detach(), comper_x_ab_s[index_to_comper].detach(),
                                                          x_a)  # original

                # self.loss_dis_council_a_s.append(
                #     self.dis_council_a_s[i].calc_dis_loss(comper_x_ba_s[index_to_comper].detach(), x_ba_s[i].detach(), x_b))  # TMP
                # self.loss_dis_council_b_s.append(
                #     self.dis_council_b_s[i].calc_dis_loss(comper_x_ab_s[index_to_comper].detach(), x_ab_s[i].detach(), x_a))  # TMP

            self.loss_dis_council_total_s.append(
                hyperparameters['gan_w'] * self.loss_dis_council_a_s[i] / hyperparameters['council'][
                    'numberOfCouncil_dis_relative_iteration']
                + hyperparameters['gan_w'] * self.loss_dis_council_b_s[i] / hyperparameters['council'][
                    'numberOfCouncil_dis_relative_iteration'])
            self.loss_dis_council_total_s[i].backward()
            self.dis_council_opt_s[i].step()

    #############################################################################################

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
                print(last_model_name)
                state_dict = torch.load(last_model_name)
                self.gen_a_s[i].load_state_dict(state_dict['a'])
                self.gen_b_s[i].load_state_dict(state_dict['b'])
                iterations = int(last_model_name[-11:-3])
            else:
                warnings.warn('Failed to find gen checkpoint, did not load model')

            # Load discriminators
            last_model_name = get_model_list(checkpoint_dir, "dis_" + str(i))
            if last_model_name is not None:
                state_dict = torch.load(last_model_name)
                self.dis_a_s[i].load_state_dict(state_dict['a'])
                self.dis_b_s[i].load_state_dict(state_dict['b'])
            else:
                warnings.warn('Failed to find dis checkpoint, did not load model')
            # Load council discriminators
            if self.do_dis_council:
                try:
                    last_model_name = get_model_list(checkpoint_dir, "dis_council_" + str(i))
                    if last_model_name is not None:
                        state_dict = torch.load(last_model_name)
                        self.dis_council_a_s[i].load_state_dict(state_dict['a'])
                        self.dis_council_b_s[i].load_state_dict(state_dict['b'])
                    else:
                        warnings.warn('Failed to find dis checkpoint, did not load model')
                except:
                    warnings.warn('some council discriminetor FAILED to load')
            # Load optimizers
            try:
                state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer_' + str(i) + '.pt'))
                self.dis_opt_s[i].load_state_dict(state_dict['dis'])
                self.gen_opt_s[i].load_state_dict(state_dict['gen'])
                if self.do_dis_council:
                    self.dis_council_opt_s[i].load_state_dict(state_dict['dis_council'])
                # Reinitilize schedulers
                self.dis_scheduler_s[i] = get_scheduler(self.dis_opt_s[i], hyperparameters, iterations)
                self.gen_scheduler = get_scheduler(self.gen_opt_s[i], hyperparameters, iterations)
                if self.do_dis_council:
                    self.dis_council_scheduler_s[i] = get_scheduler(self.dis_council_opt_s[i], hyperparameters,
                                                                    iterations)
            except:
                warnings.warn('some optimizer FAILED to load')
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
            torch.save({'a': self.gen_a_s[i].state_dict(), 'b': self.gen_b_s[i].state_dict()}, gen_name)
            torch.save({'a': self.dis_a_s[i].state_dict(), 'b': self.dis_b_s[i].state_dict()}, dis_name)
            if self.do_dis_council:
                torch.save({'a': self.dis_council_a_s[i].state_dict(), 'b': self.dis_council_b_s[i].state_dict()},
                           dis_council_name)
                torch.save({'gen': self.gen_opt_s[i].state_dict(), 'dis': self.dis_opt_s[i].state_dict(),
                            'dis_council': self.dis_council_opt_s[i].state_dict()}, opt_name)
            else:
                torch.save({'gen': self.gen_opt_s[i].state_dict(), 'dis': self.dis_opt_s[i].state_dict()}, opt_name)


class UNIT_council_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_council_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)



#%%


