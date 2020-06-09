"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04
from trainer_council import Council_Trainer
import argparse
from torch.autograd import Variable
import numpy as np
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import os.path
import torch
from PIL import Image
import warnings
from termcolor import colored
use_face_locations = True

if use_face_locations:
    from PIL import Image
    try:
        import face_recognition
    except:
        warnings.warn("Filed to import face_recognition, setting use_face_locations to FALSE")
        use_face_locations = False

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/anime2face_council_folder.yaml', help='Path to the config file.')

parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b 0 for b2a")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_style', type=int, default=10, help="number of styles to sample")
parser.add_argument('--output_only', action='store_true', help="whether only save the output images or also save the input images")
parser.add_argument('--num_of_images_to_test', type=int, default=10000, help="number of images to sample")

opts = parser.parse_args()

import sys
torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']
council_size = config['council']['council_size']

# Setup model and data loader
if not 'new_size_a' in config.keys():
    config['new_size_a'] = config['new_size']
is_data_A = opts.a2b


style_dim = config['gen']['style_dim']
if opts.a2b:
    config['do_b2a'] = False
    config['do_a2b'] = True
else:
    config['do_b2a'] = True
    config['do_a2b'] = False
trainer = Council_Trainer(config)
only_one = False
if 'gen_' in opts.checkpoint[-21:]:
    state_dict = torch.load(opts.checkpoint, map_location=torch.device('cuda:0'))
    if opts.a2b:
        trainer.gen_a2b_s[0].load_state_dict(state_dict['a2b'])
    else:
        trainer.gen_b2a_s[0].load_state_dict(state_dict['b2a'])
    council_size = 1
    only_one = True
else:
    for i in range(council_size):
        if opts.a2b:
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
if opts.a2b:
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

telegram_res_path = './telegram_tmp'
if not os.path.exists(telegram_res_path):
    os.mkdir(telegram_res_path)


from test_gui import run_net_work
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

if not 'bot_token_test' in confidential_conf.keys():
    confidential_conf['bot_token_test'] = confidential_conf['bot_token']
    print('bot_token_test not defined in confidential_do_not_upload_to_github.ymal using bot_token instead')
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
        number_of_result_to_return = 3
        move_up_by_ratio_vec = [0.21, 0.2, 0.15, 0.1, 0, -0.05] # celebe
        face_increes_by_dev_ratio_vec = [1.6, 1.7, 1.8, 2, 2.3, 2.5] # celebe
        # move_up_by_ratio_vec = [-0.1, 0, 0.1] # anime
        # face_increes_by_dev_ratio_vec = [2, 2.3, 2.5, 3, 4] # anime
        run_net_work.counter += 1
        print(f'number of image prossesed: {run_net_work.counter}')
        for i in range(number_of_result_to_return):
            random_entropy = Variable(torch.randn(1, style_dim, 1, 1).cuda())
            # change sample range
            random_entropy *= 0.9
            bais = (Variable(torch.randint(low=0, high=2, size=(1, style_dim, 1, 1))) - 0.5) * 2
            bais *= 0.2
            bais = bais.type(torch.FloatTensor).cuda()
            random_entropy = np.random.choice([random_entropy - bais, random_entropy + bais])

            move_up_by_ratio = np.random.choice(move_up_by_ratio_vec)
            face_increes_by_dev_ratio = np.random.choice(face_increes_by_dev_ratio_vec)
            in_image_path, out_image_path = run_net_work(img_path=telegram_image_save_path,
                                                         entropy=random_entropy,
                                                         use_face_locations=True,
                                                         config=config,
                                                         face_increes_by_dev_ratio=face_increes_by_dev_ratio,
                                                         move_up_by_ratio=move_up_by_ratio)

            with open(out_image_path, 'rb') as res_file:
                context.bot.send_photo(chat_id=update.message.chat_id, photo=res_file, filename=config['misc']['telegram_report_add_prefix'], caption='output')
                # context.bot.send_photo(chat_id=update.message.chat_id, photo=res_file, filename=config['misc']['telegram_report_add_prefix'], caption='output'+ 'face_increes_by_dev_ratio:'+ str(face_increes_by_dev_ratio) + ' move_up_by_ratio:' + str(move_up_by_ratio))
            os.remove(in_image_path)
            os.remove(out_image_path)
    except Exception as e:
        context.bot.send_message(chat_id=update.message.chat_id, text='Failed')
        print(e)

updater = Updater(token=confidential_conf['bot_token_test'], use_context=True)
dispatcher = updater.dispatcher
dispatcher.add_handler(MessageHandler(Filters.photo, telegram_command))

updater.start_polling()

input(colored('Telegram bot running - press enter to stop', color='yellow', attrs=['underline', 'bold', 'blink', 'reverse']))
print(colored('Stoping...', color='red', attrs=['underline', 'bold', 'blink', 'reverse']))
updater.stop()
print(colored('Stoped', color='red', attrs=['underline', 'bold', 'blink', 'reverse']))
