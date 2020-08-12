"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04
from trainer_council import Council_Trainer
from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from data import ImageFolder
import numpy as np
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import torch
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--input_folder', type=str, help="input image folder")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--output_path', type=str, default='outputs', help="outputs path")

parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b 0 for b2a")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--output_only', action='store_true', help="whether only save the output images or also save the input images")
parser.add_argument('--num_of_images_to_test', type=int, default=10000, help="number of images to sample")

data_name = 'out'

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']
council_size = config['council']['council_size']


# Setup model and data loader
image_names = ImageFolder(opts.input_folder, transform=None, return_paths=True)
if not 'new_size_a' in config.keys():
    config['new_size_a'] = config['new_size']
is_data_A = opts.a2b
data_loader = get_data_loader_folder(opts.input_folder, 1, False,\
                                     new_size=config['new_size_a'] if 'new_size_a' in config.keys() else config['new_size'],\
                                     crop=False, config=config, is_data_A=is_data_A)


style_dim = config['gen']['style_dim']
trainer = Council_Trainer(config)
only_one = False
if 'gen_' in opts.checkpoint[-21:]:
    state_dict = torch.load(opts.checkpoint)
    try:
        if opts.a2b:
            trainer.gen_a2b_s[0].load_state_dict(state_dict['a2b'])
        else:
            trainer.gen_b2a_s[0].load_state_dict(state_dict['b2a'])
    except:
       print('opts.a2b should be set to ' + str(not opts.a2b) + ' , Or config file could be wrong')
       opts.a2b = not opts.a2b
       if opts.a2b:
           trainer.gen_a2b_s[0].load_state_dict(state_dict['a2b'])
       else:
           trainer.gen_b2a_s[0].load_state_dict(state_dict['b2a'])
            
    council_size = 1
    only_one = True
else:
    for i in range(council_size):
        try:
            if opts.a2b:
                tmp_checkpoint = opts.checkpoint[:-8] + 'a2b_gen_' + str(i) + '_' + opts.checkpoint[-8:] + '.pt'
                state_dict = torch.load(tmp_checkpoint, map_location=trainer.gen_a2b_s[i].cuda_device)
                trainer.gen_a2b_s[i].load_state_dict(state_dict['a2b'])
            else:
                tmp_checkpoint = opts.checkpoint[:-8] + 'b2a_gen_' + str(i) + '_' + opts.checkpoint[-8:] + '.pt'
                state_dict = torch.load(tmp_checkpoint, map_location=trainer.gen_b2a_s[i].cuda_device)
                trainer.gen_b2a_s[i].load_state_dict(state_dict['b2a'])
        except:
            print('opts.a2b should be set to ' + str(not opts.a2b) + ' , Or config file could be wrong')

            opts.a2b = not opts.a2b
            if opts.a2b:
                tmp_checkpoint = opts.checkpoint[:-8] + 'a2b_gen_' + str(i) + '_' + opts.checkpoint[-8:] + '.pt'
                state_dict = torch.load(tmp_checkpoint, map_location=trainer.gen_a2b_s[i].cuda_device)
                trainer.gen_a2b_s[i].load_state_dict(state_dict['a2b'])
            else:
                tmp_checkpoint = opts.checkpoint[:-8] + 'b2a_gen_' + str(i) + '_' + opts.checkpoint[-8:] + '.pt'
                state_dict = torch.load(tmp_checkpoint, map_location=trainer.gen_b2a_s[i].cuda_device)
                trainer.gen_b2a_s[i].load_state_dict(state_dict['b2a'])
            


trainer.cuda()
trainer.eval()

encode_s = []
decode_s = []
if opts.a2b:
    for i in range(council_size):
        encode_s.append(trainer.gen_a2b_s[i].encode)  # encode function
        decode_s.append(trainer.gen_a2b_s[i].decode)  # decode function
else:
    for i in range(council_size):
        encode_s.append(trainer.gen_b2a_s[i].encode)  # encode function
        decode_s.append(trainer.gen_b2a_s[i].decode)  # decode function


# creat testing images
num_of_images_to_test = opts.num_of_images_to_test
seed = 1
curr_image_num = -1
for i, (images, names) in tqdm(enumerate(zip(data_loader, image_names)), total=num_of_images_to_test):
    if curr_image_num == num_of_images_to_test:
        break
    curr_image_num += 1
    k = np.random.randint(council_size)
    style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
    print(names[1])
    images = Variable(images.cuda(), volatile=True)

    content, _ = encode_s[k](images)
    seed += 1
    torch.random.manual_seed(seed)
    style = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
    for j in range(opts.num_style):
        s = style[j].unsqueeze(0)
        outputs = decode_s[k](content, s, images)



        basename = os.path.basename(names[1])
        output_folder = os.path.join(opts.output_path, 'test_res')
        if only_one:
            path = os.path.join(output_folder, opts.checkpoint[-11:-3] + "_%02d" % j, data_name + '_out_' + str(curr_image_num) + '_' + str(j) + '.jpg')
            path_all_in_one = os.path.join(output_folder, opts.checkpoint[-11:-3] + '_all_in_1', data_name + '_out_' + str(curr_image_num) + '_' + str(j) + '.jpg')

        else:
            path = os.path.join(output_folder, opts.checkpoint[-8:] + "_%02d" % j, data_name + '_out_' + str(curr_image_num) + '_' + str(j) + '.jpg')
            path_all_in_one = os.path.join(output_folder,  opts.checkpoint[-8:] + '_all_in_1', data_name + '_out_' + str(curr_image_num) + '_' + str(j) + '.jpg')

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        vutils.save_image(outputs.data, path, padding=0, normalize=True)
        do_all_in_one = True
        if do_all_in_one:
            if not os.path.exists(os.path.dirname(path_all_in_one)):
                os.makedirs(os.path.dirname(path_all_in_one))
            vutils.save_image(outputs.data, path_all_in_one, padding=0, normalize=True)
    if not opts.output_only:
        # also save input images
        output_folder = os.path.join(output_folder, 'input')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        vutils.save_image(images.data, os.path.join(output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
