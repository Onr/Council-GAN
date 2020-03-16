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
import sys
import torch
import os
from tqdm import tqdm
import torch.utils.data as data
import os.path
import torch
# import cv2
#
from PIL import Image
from termcolor import colored

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--input_folder', type=str, help="input image folder")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--output_path', type=str, default='.outputs', help="outputs path")

parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--b2a', action='store_true', help=" whether to run b2a defult a2b")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--output_only', action='store_true', help="whether only save the output images or also save the input images")
parser.add_argument('--num_of_images_to_test', type=int, default=10000, help="number of images to sample")

opts = parser.parse_args()

import sys
torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if not opts.b2a else config['input_dim_b']
council_size = config['council']['council_size']

# Setup model and data loader
if not 'new_size_a' in config.keys():
    config['new_size_a'] = config['new_size']
is_data_A = not opts.b2a


style_dim = config['gen']['style_dim']
if opts.b2a:
    config['do_b2a'] = True
    config['do_a2b'] = False
else:
    config['do_b2a'] = False
    config['do_a2b'] = True

trainer = Council_Trainer(config)
only_one = False
if 'gen_' in opts.checkpoint[-21:]:
    state_dict = torch.load(opts.checkpoint)
    if not opts.b2a:
        trainer.gen_a2b_s[0].load_state_dict(state_dict['a2b'])
    else:
        trainer.gen_b2a_s[0].load_state_dict(state_dict['b2a'])
    council_size = 1
    only_one = True
else:
    for i in range(council_size):
        if not opts.b2a:
            tmp_checkpoint = opts.checkpoint[:-8] + 'a2b_gen_' + str(i) + '_' + opts.checkpoint[-8:] + '.pt'
            state_dict = torch.load(tmp_checkpoint)
            trainer.gen_a2b_s[i].load_state_dict(state_dict['a2b'])
        else:
            tmp_checkpoint = opts.checkpoint[:-8] + 'b2a_gen_' + str(i) + '_' + opts.checkpoint[-8:] + '.pt'
            state_dict = torch.load(tmp_checkpoint)
            trainer.gen_b2a_s[i].load_state_dict(state_dict['b2a'])


trainer.cuda()
trainer.eval()

encode_s = []
decode_s = []
if not opts.b2a:
    for i in range(council_size):
        encode_s.append(trainer.gen_a2b_s[i].encode)  # encode function
        decode_s.append(trainer.gen_a2b_s[i].decode)  # decode function
else:
    for i in range(council_size):
        encode_s.append(trainer.gen_b2a_s[i].encode)  # encode function
        decode_s.append(trainer.gen_b2a_s[i].decode)  # decode function


def load_net(checkpoint):
    try:
        state_dict = torch.load(checkpoint)
        if 'a2b' in checkpoint:
            trainer.gen_a2b_s[0].load_state_dict(state_dict['a2b'])
            encode_s[0] = trainer.gen_a2b_s[0].encode  # encode function
            decode_s[0] = trainer.gen_a2b_s[0].decode  # decode function
        else:
            trainer.gen_b2a_s[0].load_state_dict(state_dict['b2a'])
            encode_s[0] = trainer.gen_b2a_s[0].encode  # encode function
            decode_s[0] = trainer.gen_b2a_s[0].decode  # decode function
    except Exception as e:
        print(e)
        print('FAILED to load network!')



from torchvision import transforms
from torchvision.utils import save_image
import time

telegram_res_path = './telegram_tmp'
telegram_res_path = os.path.join(telegram_res_path, time.ctime(time.time()))

if not os.path.exists(telegram_res_path):
    os.mkdir(telegram_res_path)


# from test_gui import run_net_work

def run_net_work(img_path, entropy, use_face_locations=False):
    run_net_work.counter += 1
    out_im_path = os.path.join(telegram_res_path, 'tmp_' + str(run_net_work.counter) + '.png')
    in_im_path = os.path.join(telegram_res_path, 'tmp_' + str(run_net_work.counter) + '_in.png')
    height = 128
    width = 128
    new_size = 128


    img = Image.open(img_path).convert('RGB')
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    transform_list = [transforms.ToTensor()
                      ,transforms.Normalize(mean=mean, std=std)]
    transform_list = [transforms.CenterCrop((height, width))] + transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list
    transform = transforms.Compose(transform_list)

    img = transform(img).unsqueeze(0).cuda()
    content, _ = encode_s[0](img)
    res_img = decode_s[0](content, entropy, img).detach().cpu().squeeze(0)
    res_img = transforms.Normalize(mean=(-mean/std), std=(1.0/std))(res_img)
    save_image(res_img, out_im_path)
    in_image = img.detach().cpu().squeeze(0)

    in_image = transforms.Normalize(mean=-mean/std, std=1/std)(in_image)
    save_image(in_image, in_im_path)
    return in_im_path, out_im_path

run_net_work.counter = 0



import telegram
from telegram.ext import Updater, MessageHandler, Filters
confidential_yaml_file_path = './confidential_do_not_upload_to_github.yaml'

if not os.path.exists(confidential_yaml_file_path):
    with open(confidential_yaml_file_path, 'w') as confidential_yaml_file:
        confidential_yaml_file.write('bot_token: xxxx\n')
        confidential_yaml_file.write('chat_id: xxxx\n')
        confidential_yaml_file.write('bot_token_test: xxxx\n')

    print(colored('create a telegram bot using a chat with \"BotFather\" and enter its token into bot_token_test in ' + confidential_yaml_file_path, color='red', attrs=['underline', 'bold', 'blink', 'reverse']))
    print('you can turn telegram report off from the config.yaml file')
    input('when you are done press Enter')
confidential_conf = get_config(confidential_yaml_file_path)

while confidential_conf['bot_token_test'] == 'xxxx':
    print(colored('TOKEN not defined yet'))
    print(colored('create a telegram bot using a chat with BotFather and enter its token into ' + confidential_yaml_file_path, color='red', attrs=['underline', 'bold', 'blink', 'reverse']))
    input('when you are done press Enter')
    confidential_conf = get_config(confidential_yaml_file_path)


telegram_bot = telegram.Bot(token=confidential_conf['bot_token_test'])
def telegram_bot_send_message(bot_message):
    try:
        telegram_bot.send_message(chat_id=confidential_conf['chat_id'], text=config['misc']['telegram_report_add_prefix'] + bot_message)
    except:
        print('telegram send_message Failed')
def telegram_bot_send_photo(bot_image, caption=None):
    try:
        telegram_bot.send_photo(chat_id=confidential_conf['chat_id'], photo=bot_image, caption=config['misc']['telegram_report_add_prefix'] + caption)
    except:
        print('telegram send_photo Failed')
def telegram_bot_send_document(bot_document_path, filename=None):
    try:
        telegram_bot.send_document(chat_id=confidential_conf['chat_id'], document=open(bot_document_path, 'rb'), filename=config['misc']['telegram_report_add_prefix'] + filename)
    except:
        print('telegram send_document Failed')

def telegram_command(update, context):

    context.bot.send_message(chat_id=update.message.chat_id, text='Prossesing')
    try:
        photo_id = update.message['photo'][0]['file_id']
        im_url = context.bot.getFile(photo_id)['file_path']
        import urllib.request
        telegram_image_save_path = os.path.join(telegram_res_path, "telegram_recived.jpg")
        urllib.request.urlretrieve(im_url, telegram_image_save_path)

        random_entropy = Variable(torch.randn(1, style_dim, 1, 1).cuda())
        run_net_work(img_path=telegram_image_save_path, entropy=random_entropy, use_face_locations=True)

        in_image_path = os.path.join(telegram_res_path, 'tmp_' + str(run_net_work.counter) + '_in.png')
        with open(in_image_path, 'rb') as in_file:
            context.bot.send_photo(chat_id=update.message.chat_id, photo=in_file, filename=config['misc']['telegram_report_add_prefix'], caption='input')

        out_image_path = os.path.join(telegram_res_path, 'tmp_' + str(run_net_work.counter) + '.png')
        with open(out_image_path, 'rb') as res_file:
            context.bot.send_photo(chat_id=update.message.chat_id, photo=res_file, filename=config['misc']['telegram_report_add_prefix'], caption='output')

        # context.bot.sendMessage(update.message.chat_id, text='enter chat_id in to: ' + confidential_yaml_file_path + ' as:')
    except Exception as e:
        context.bot.send_message(chat_id=update.message.chat_id, text='Failed')
        print(e)


updater = Updater(token=confidential_conf['bot_token_test'], use_context=True)
dispatcher = updater.dispatcher
dispatcher.add_handler(MessageHandler(Filters.photo, telegram_command))

updater.start_polling()

input(colored('telegram bot running - press enter to stop', color='yellow', attrs=['underline', 'bold', 'blink', 'reverse']))
print(colored('stoping...', color='red', attrs=['underline', 'bold', 'blink', 'reverse']))
updater.stop()
print(colored('stoped', color='red', attrs=['underline', 'bold', 'blink', 'reverse']))
