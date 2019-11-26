"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from scipy.stats import entropy
from torch import nn
import datetime


from trainer_council import Council_Trainer

from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer, \
    load_inception

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
import threading
import torchvision.utils as vutils
import math
from scipy.stats import binom
from tqdm import tqdm
import time




parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/glasses_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.outputs', help="outputs path")
parser.add_argument("--resume", action="store_true")

opts = parser.parse_args()


# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path


# FOR REPRODUCIBILITY
torch.manual_seed(config['random_seed'])
np.random.seed(config['random_seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Setup model and data loader
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

train_display_images_a = torch.stack([train_loader_a[0].dataset[np.random.randint(train_loader_a[0].__len__())] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([train_loader_b[0].dataset[np.random.randint(train_loader_b[0].__len__())] for i in range(display_size)]).cuda()
test_display_images_a = torch.stack([test_loader_a[0].dataset[np.random.randint(test_loader_a[0].__len__())] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_loader_b[0].dataset[np.random.randint(test_loader_b[0].__len__())] for i in range(display_size)]).cuda()

trainer = Council_Trainer(config)
trainer.cuda()


# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path, model_name)
checkpoint_directory, image_directory, log_directory = prepare_sub_folder(output_directory)

config_backup_folder = os.path.join(output_directory, 'config_backup')
if not os.path.exists(config_backup_folder):
    os.mkdir(config_backup_folder)
shutil.copy(opts.config, os.path.join(config_backup_folder, 'config_backup_ ' + str(datetime.datetime.now())[:19] + '.yaml'))  # copy config file to output folder


# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0


def launchTensorBoard(port=6006):
    import os
    os.system('tensorboard --logdir=' + log_directory + '--port=' + str(port) + ' > /dev/null 2>/dev/null')
    print('tensorboard board launched at http://127.0.0.1:' + str(port))
    return
if config['misc']['start_tensor_board']:
    port = config['misc']['start_tensor_board port']
    t_tensorBoard = threading.Thread(target=launchTensorBoard, args=([port]))  # launches TensorBoard in a diffrent thread
    t_tensorBoard.start()
train_writer = tensorboardX.SummaryWriter(log_directory, purge_step=iterations)



t = time.time()
dis_iter = 1
while True:
    tmp_train_loader_a, tmp_train_loader_b = (train_loader_a[0], train_loader_b[0])

    for it, (images_a, images_b) in enumerate(zip(tmp_train_loader_a, tmp_train_loader_b)):

        trainer.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        print("Iteration: " + str(iterations + 1) + "/" + str(max_iter) + " Elapsed time " + str(time.time()-t)[:5])
        t = time.time()

        if iterations > max_iter:
            sys.exit('Finish training')




        # Main training code
        config['iteration'] = it
        numberOf_dis_relative_iteration = config['dis']['numberOf_dis_relative_iteration'] if config['dis']['numberOf_dis_relative_iteration'] > 0 else 1
        if dis_iter < numberOf_dis_relative_iteration:  # training the discriminetor multiple times for each generator update
            dis_iter += 1
            trainer.dis_update(images_a, images_b, config)
            continue
        else:
            trainer.dis_update(images_a, images_b, config)
            dis_iter = 1


        if config['council']['numberOfCouncil_dis_relative_iteration'] > 0:
            trainer.dis_council_update(images_a, images_b, config)  # the multiple iterating happens inside dis_council_update

        trainer.gen_update(images_a, images_b, config, iterations)
        torch.cuda.synchronize()
        iterations += 1




        # write training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            write_loss(iterations, trainer, train_writer)



        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            test_gen_a2b_im, test_gen_b2a_im = write_2images(test_image_outputs,
                                                             display_size * config['council']['council_size'],
                                                             image_directory, 'test_%08d' % (iterations + 1), do_a2b=config['do_a2b'], do_b2a=config['do_b2a'])
            train_gen_a2b_im, train_gen_b2a_im = write_2images(train_image_outputs,
                                                               display_size * config['council']['council_size'],
                                                               image_directory, 'train_%08d' % (iterations + 1), do_a2b=config['do_a2b'], do_b2a=config['do_b2a'])

            if config['do_a2b']:
                train_writer.add_image('a2b/train', train_gen_a2b_im, iterations)
                train_writer.add_image('a2b/test', test_gen_a2b_im, iterations)
            if config['do_b2a']:
                train_writer.add_image('b2a/train', train_gen_b2a_im, iterations)
                train_writer.add_image('b2a/test', test_gen_b2a_im, iterations)

            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images', do_a2b=config['do_a2b'], do_b2a=config['do_b2a'])

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size * config['council']['council_size'], image_directory,
                          'train_current', do_a2b=config['do_a2b'], do_b2a=config['do_b2a'])

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            try:
                trainer.save(checkpoint_directory, iterations)
            except Exception as e:
                print('================================= Faild to save check avileble memory =================================')
                print(e)
                input(" Clear space and press enter to retry ....")
                trainer.save(checkpoint_directory, iterations)







