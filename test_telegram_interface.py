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

data_name = 'out'

opts = parser.parse_args()

import sys












torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if not opts.b2a else config['input_dim_b']
council_size = config['council']['council_size']


# Setup model and data loader
# image_names = ImageFolder(opts.input_folder, transform=None, return_paths=True)
if not 'new_size_a' in config.keys():
    config['new_size_a'] = config['new_size']
is_data_A = not opts.b2a
# data_loader = get_data_loader_folder(opts.input_folder, 1, False,\
#                                      new_size=config['new_size_a'] if 'new_size_a' in config.keys() else config['new_size'],\
#                                      crop=False, config=config, is_data_A=is_data_A)


style_dim = config['gen']['style_dim']
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


def run_net_work(img_path, entropy):
    out_im_path = './tmp.png'
    in_im_path = './tmp_in.png'
    height = 128
    width = 128
    new_size = 128
    img = Image.open(img_path).convert('RGB')
    transform_list = [transforms.ToTensor()]
                      # ,transforms.Normalize((0.5, 0.5, 0.5),
                      #                      (0.5, 0.5, 0.5))]
    transform_list = [transforms.CenterCrop((height, width))] + transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list
    transform = transforms.Compose(transform_list)

    img = transform(img).unsqueeze(0).cuda()
    content, _ = encode_s[0](img)
    res_img = decode_s[0](content, entropy, img).detach().cpu().squeeze(0)
    # res_img = transforms.ToPILImage()(res_img)
    # res_img.save(out_im_path)
    save_image(res_img, out_im_path)

    # in_image = transforms.ToPILImage()(img.detach().cpu().squeeze(0))
    in_image = img.detach().cpu().squeeze(0)
    # in_image.save(in_im_path)
    save_image(in_image, in_im_path)
    return in_im_path, out_im_path
















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
    # if str(update.message.text) == 'cofig':
    #     with open(opts.config, 'r') as tmp_conf:
    #         telegram_bot_send_message(tmp_conf.read())
    # for photo in update.message['photo']:
    #     print(photo['file_id'])

    photo_id = update.message['photo'][0]['file_id']
    im_url = context.bot.getFile(photo_id)['file_path']
    import urllib.request
    telegram_image_save_path = "./telegram_recived.jpg"
    urllib.request.urlretrieve(im_url, telegram_image_save_path)

    random_entropy = Variable(torch.randn(1, style_dim, 1, 1).cuda())
    run_net_work(telegram_image_save_path, random_entropy)
    out_image_path ='./tmp.png'
    with open(out_image_path, 'rb') as res_file:
        context.bot.send_photo(update.message.chat_id, photo=res_file, filename=config['misc']['telegram_report_add_prefix'])

    # context.bot.sendMessage(update.message.chat_id, text='enter chat_id in to: ' + confidential_yaml_file_path + ' as:')




updater = Updater(token=confidential_conf['bot_token_test'], use_context=True)
dispatcher = updater.dispatcher
dispatcher.add_handler(MessageHandler(Filters.photo, telegram_command))

updater.start_polling()


input(colored('telegram bot running - press enter to stop', color='yellow', attrs=['underline', 'bold', 'blink', 'reverse']))
updater.stop()





