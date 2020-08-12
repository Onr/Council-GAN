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
import random


from trainer_council import Council_Trainer

from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer, \
    load_inception

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
import sys, traceback
import tensorboardX
import shutil
import threading
import torchvision.utils as vutils
import math
from scipy.stats import binom
from tqdm import tqdm
import time
from termcolor import colored

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/glasses_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.outputs', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--cuda_device", type=str, default='cuda:0', help="gpu to run on")
opts = parser.parse_args()


# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path
config['cuda_device'] = opts.cuda_device

# FOR REPRODUCIBILITY
def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_torch(config['random_seed'])

# Setup model and data loader
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

try:
    train_display_images_a = torch.stack([train_loader_a[0].dataset[np.random.randint(train_loader_a[0].__len__())] for _ in range(display_size)]).cuda(config['cuda_device'])
except:
    train_display_images_a = torch.stack([train_loader_a[0].dataset[np.random.randint(train_loader_a[0].__len__())] for _ in range(display_size)]).cuda(config['cuda_device'])
try:
    train_display_images_b = torch.stack([train_loader_b[0].dataset[np.random.randint(train_loader_b[0].__len__())] for _ in range(display_size)]).cuda(config['cuda_device'])
except:
    train_display_images_b = torch.stack([train_loader_b[0].dataset[np.random.randint(train_loader_b[0].__len__())] for _ in range(display_size)]).cuda(config['cuda_device'])
try:
    test_display_images_a = torch.stack([test_loader_a[0].dataset[np.random.randint(test_loader_a[0].__len__())] for _ in range(display_size)]).cuda(config['cuda_device'])
except:
    # test_display_images_a = torch.stack([test_loader_a[0].dataset[np.random.randint(test_loader_a[0].__len__())] for _ in range(display_size)]).cuda()
    test_display_images_a = None
try:
    test_display_images_b = torch.stack([test_loader_b[0].dataset[np.random.randint(test_loader_b[0].__len__())] for _ in range(display_size)]).cuda(config['cuda_device'])
except:
    test_display_images_b = torch.stack([test_loader_b[0].dataset[np.random.randint(test_loader_b[0].__len__())] for _ in range(display_size)]).cuda(config['cuda_device'])

trainer = Council_Trainer(config, config['cuda_device'])

trainer.cuda(config['cuda_device'])

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path, model_name)
checkpoint_directory, image_directory, log_directory = prepare_sub_folder(output_directory)

config_backup_folder = os.path.join(output_directory, 'config_backup')
if not os.path.exists(config_backup_folder):
    os.mkdir(config_backup_folder)
shutil.copy(opts.config, os.path.join(config_backup_folder, ('config_backup_' + str(datetime.datetime.now())[:19] + '.yaml').replace(' ', '_')))  # copy config file to output folder


m1_1_a2b, s1_1_a2b, m1_1_b2a, s1_1_b2a = None, None, None, None # save statisices for the fid calculation

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0


def launchTensorBoard(port=6006):
    import os
    os.system('tensorboard --logdir=' + log_directory + ' --port=' + str(port) + ' > /dev/null 2>/dev/null')
    return

if config['misc']['start_tensor_board']:
    port = config['misc']['start_tensor_board port']
    t_tensorBoard = threading.Thread(target=launchTensorBoard, args=([port]))  # launches TensorBoard in a diffrent thread
    t_tensorBoard.start()
    print(colored('tensorboard board launched at http://127.0.0.1:' + str(port), color='yellow', attrs=['underline', 'bold', 'blink', 'reverse']))
train_writer = tensorboardX.SummaryWriter(log_directory, purge_step=iterations)

if config['misc']['do_telegram_report']:           
    try:
        import telegram
        from telegram.ext import Updater, MessageHandler, Filters
    except:
        print(colored('Failed to load Telegram Try: \n1) conda install -c conda-forge python-telegram-bot. \n OR \n2) in \".yaml\" file change do_telegram_report to False \n in the meantime Continuing without ....', color='red', attrs=['underline', 'bold', 'blink', 'reverse']))
        config['misc']['do_telegram_report'] = False
if config['misc']['do_telegram_report']:           
    in_ = ''
    confidential_yaml_file_path = './confidential_do_not_upload_to_github.yaml'
    if not os.path.exists(confidential_yaml_file_path):
        
        in_ = input(colored('do_telegram_report is set to True If you would like to set up telegram press Enter. If you just Want to continue write \"NO\": \n'))
        if in_.upper() != 'NO':    
            with open(confidential_yaml_file_path, 'w') as confidential_yaml_file:
                confidential_yaml_file.write('bot_token: xxxx\n')
                confidential_yaml_file.write('chat_id: xxxx')
            print(colored('Create a telegram bot. this is done by: \n1) downloding and signing into telegram.  \n2) starting a chat with \"BotFather\" \n3) send \"BotFather\" the text "/newbot", then follow the "BotFather" instraction to creat the bot \n4)when you are done you will recive a the new bot token. enter the token into the file: "' + confidential_yaml_file_path + ' which was create in the currnt directory', color='red', attrs=['underline', 'bold', 'blink', 'reverse']))
            print('==== You can turn telegram report OFF from the config.yaml file ====')
            input('when you are done press ENTER.')
            

    if in_.upper() != 'NO':
        confidential_conf = get_config(confidential_yaml_file_path)
        while confidential_conf['bot_token'] == 'xxxx':
            print(colored('TOKEN not defined yet'))
            print(colored('Create a telegram bot. this is done by 1) downloding and signing into telegram.  \n2) starting a chat with \"BotFather\" \n3) send him the text "/newbot", then follow the "BotFather" instraction to creat the bot \n4)when you are done you will recive a the new bot token. enter the token into the file: "' + confidential_yaml_file_path + 'which was create in the currnt directory', color='red', attrs=['underline', 'bold', 'blink', 'reverse']))
            print('==== You can turn telegram report OFF from the config.yaml file ====')
            input(colored('when you are done press Enter'))
            confidential_conf = get_config(confidential_yaml_file_path)


        def telegram_command(update, context):

            context.bot.sendMessage(update.message.chat_id, text='enter chat_id in to: ' + confidential_yaml_file_path + ' as:')
            context.bot.sendMessage(update.message.chat_id, text='chat_id: ' + str(update.message.chat_id))


        updater = Updater(token=confidential_conf['bot_token'], use_context=True)
        dispatcher = updater.dispatcher
        dispatcher.add_handler(MessageHandler(Filters.text, telegram_command))
        updater.start_polling()


        while confidential_conf['chat_id'] == 'xxxx':
            print(colored('CHAT ID is not defined send your telegram bot a random message to get your chat id, then enter it into the file:' + confidential_yaml_file_path, color='red', attrs=['underline', 'bold', 'blink', 'reverse']))
            print('==== You can turn telegram report OFF from the config.yaml file ====')
            input('when you are done press Enter')
            confidential_conf = get_config(confidential_yaml_file_path)
        updater.stop()

        telegram_bot = telegram.Bot(token=confidential_conf['bot_token'])
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
    else:
        config['misc']['do_telegram_report'] = False
        print('You can set do_telegram_report to False to not be asked again')

def test_fid(dataset1, dataset2, iteration, train_writer, name, m1=None, s1=None, retun_m1_s1=False, batch_size=10, dims=2048, cuda=True):
    import pytorch_fid.fid_score
    fid_paths = [dataset1, dataset2]
    try:
        fid_value, m1, s1 = pytorch_fid.fid_score.calculate_fid_given_paths_save_first_domain_statistic(paths=fid_paths,
                                                                                                batch_size=batch_size,
                                                                                                cuda=cuda,
                                                                                                dims=dims,
                                                                                                m1=m1, s1=s1)

        train_writer.add_scalar('FID score/' + name, fid_value, iterations)

        print(colored('iteration: ' + str(iteration) + ' ,' + name + ' aprox FID: ' + str(fid_value), color='green', attrs=['underline', 'bold', 'blink', 'reverse']))

        if config['misc']['do_telegram_report']:
            telegram_bot_send_message('iteration: ' + str(iteration) + ' ,' + name + ' aprox FID: ' + str(fid_value))
    except Exception as e:
        print(str(e))
        fid_value, m1, s1 = 0, None, None
    if not retun_m1_s1:
        return
    return m1, s1


t = time.time()
dis_iter = 1
try:
    if config['misc']['do_telegram_report']:
        telegram_bot_send_message('Started Training!')
        send_telegram_config = config['misc']['do_telegram_send_config_file']
        if send_telegram_config:
            try:
                telegram_bot_send_document(bot_document_path=opts.config, filename='config.txt')
            except:
                print('telegram config message send failed')
    while True:
        tmp_train_loader_a, tmp_train_loader_b = (train_loader_a[0], train_loader_b[0])
        for it, (images_a, images_b) in enumerate(zip(tmp_train_loader_a, tmp_train_loader_b)):
            images_a, images_b = images_a.cuda(config['cuda_device']).detach(), images_b.cuda(config['cuda_device']).detach()

            print("Iteration: " + str(iterations + 1) + "/" + str(max_iter) + " Elapsed time " + str(time.time()-t)[:5])
            t = time.time()

            if iterations > max_iter:
                sys.exit('Finish training')

            # Main training code
            config['iteration'] = iterations
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
            torch.cuda.synchronize(device=config['cuda_device'])
            iterations += 1

            # write training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                write_loss(iterations, trainer, train_writer)
            # test FID
            if config['misc']['do_test_Fid'] and (iterations + 1) % config['misc']['test_Fid_iter'] == 0:
                if config['do_a2b']:
                    tmp_path_im_a2b = image_directory + '/tmp/a2b'
                    if not os.path.exists(tmp_path_im_a2b):
                        print("Creating directory: {}".format(tmp_path_im_a2b))
                        os.makedirs(tmp_path_im_a2b)
                    filelist2 = [f for f in os.listdir(tmp_path_im_a2b) if f.endswith(".jpg")]
                    for f2 in filelist2:
                        os.remove(os.path.join(tmp_path_im_a2b, f2))

                if config['do_b2a']:
                    tmp_path_im_b2a = image_directory + '/tmp/b2a'
                    if not os.path.exists(tmp_path_im_b2a):
                        print("Creating directory: {}".format(tmp_path_im_b2a))
                        os.makedirs(tmp_path_im_b2a)
                    filelist1 = [f for f in os.listdir(tmp_path_im_b2a) if f.endswith(".jpg")]
                    for f1 in filelist1:
                        os.remove(os.path.join(tmp_path_im_b2a, f1))

                if config['do_a2b']:
                    tmp_images_a = test_loader_a[0].dataset[0].cuda(config['cuda_device']).unsqueeze(0)
                if config['do_b2a']:
                    tmp_images_b = test_loader_b[0].dataset[0].cuda(config['cuda_device']).unsqueeze(0)
                ind_a2b = 0
                ind_b2a = 0
                for k in tqdm(range(1, config['misc']['test_Fid_num_of_im']), desc='Creating images for tests'):
                    c_ind = np.random.randint(config['council']['council_size'])
                    if config['do_a2b']:
                        tmp_images_a = test_loader_a[0].dataset[k].cuda(config['cuda_device']).unsqueeze(0)

                        styles = torch.randn(tmp_images_a.shape[0], config['gen']['style_dim'], 1, 1).cuda(config['cuda_device'])
                        tmp_res_imges_a2b = trainer.sample(x_a=tmp_images_a, x_b=None, s_a=styles, s_b=styles)
                        tmp_res_imges_a2b = tmp_res_imges_a2b[2][c_ind].unsqueeze(0)
                        for tmp_res_imges_a2b_t in tmp_res_imges_a2b:
                            vutils.save_image(tmp_res_imges_a2b_t, tmp_path_im_a2b + '/' + str(ind_a2b) + '.jpg')
                            ind_a2b += 1
                    if config['do_b2a']:
                        tmp_images_b = test_loader_b[0].dataset[k].cuda(config['cuda_device']).unsqueeze(0)
                        styles = torch.randn(tmp_images_b.shape[0], config['gen']['style_dim'], 1, 1).cuda(config['cuda_device'])
                        tmp_res_imges_b2a = trainer.sample(x_a=None, x_b=tmp_images_b, s_a=styles, s_b=styles)

                        tmp_res_imges_b2a = tmp_res_imges_b2a[6][c_ind].unsqueeze(0)
                        for tmp_res_imges_b2a_t in tmp_res_imges_b2a:
                            vutils.save_image(tmp_res_imges_b2a_t, tmp_path_im_b2a + '/' + str(ind_b2a) + '.jpg')
                            ind_b2a += 1

                if config['do_a2b']:
                    dataset_for_fid_B = os.path.join(config['data_root'], 'testB')
                    tmp_path_a2b_save_stat = dataset_for_fid_B
                    if os.path.exists(tmp_path_a2b_save_stat + '/m1'):
                        with open(tmp_path_a2b_save_stat + '/m1', 'rb') as f:
                            m1_1_a2b = pickle.load(f)
                    if os.path.exists(tmp_path_a2b_save_stat + '/s1'):
                        with open(tmp_path_a2b_save_stat + '/s1', 'rb') as f:
                            s1_1_a2b = pickle.load(f)

                    if m1_1_a2b is None or s1_1_a2b is None:
                        print('fid test initialization')
                        m1_1_a2b, s1_1_a2b = test_fid(dataset_for_fid_B, tmp_path_im_a2b, iterations, train_writer, 'B', retun_m1_s1=True, batch_size=10)
                        if m1_1_a2b is not None and s1_1_a2b is not None:
                            with open(tmp_path_a2b_save_stat + '/m1', 'wb') as f:
                                pickle.dump(m1_1_a2b, f)
                            with open(tmp_path_a2b_save_stat + '/s1', 'wb') as f:
                                pickle.dump(s1_1_a2b, f)
                    else:
                        _ = test_fid(dataset_for_fid_B, tmp_path_im_a2b, iterations, train_writer, 'B', m1_1_a2b, s1_1_a2b)

                if config['do_b2a']:
                    dataset_for_fid_A = os.path.join(config['data_root'], 'testA')
                    tmp_path_b2a_save_stat = dataset_for_fid_A
                    if os.path.exists(tmp_path_b2a_save_stat + '/m1'):
                        with open(tmp_path_b2a_save_stat + '/m1', 'rb') as f:
                            m1_1_b2a = pickle.load(f)
                    if os.path.exists(tmp_path_b2a_save_stat + '/s1'):
                        with open(tmp_path_b2a_save_stat + '/s1', 'rb') as f:
                            s1_1_b2a = pickle.load(f)

                    if m1_1_b2a is None or s1_1_b2a is None:
                        print('fid test initialization')
                        m1_1_b2a, s1_1_b2a = test_fid(dataset_for_fid_A, tmp_path_im_b2a, iterations, train_writer, 'A', retun_m1_s1=True, batch_size=10)
                        if m1_1_b2a is not None and s1_1_b2a is not None:
                            with open(tmp_path_b2a_save_stat + '/m1', 'wb') as f:
                                pickle.dump(m1_1_b2a, f)
                            with open(tmp_path_b2a_save_stat + '/s1', 'wb') as f:
                                pickle.dump(s1_1_b2a, f)
                    else:
                        _ = test_fid(dataset_for_fid_A, tmp_path_im_b2a, iterations, train_writer, 'A', m1_1_b2a, s1_1_b2a)


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
                    if config['misc']['do_telegram_report']:
                        telegram_bot_send_message('iteration: ' + str(iterations))
                        telegram_bot_send_document(os.path.join(image_directory, 'gen_a2b_' + 'train_%08d' % (iterations + 1) + '.jpg'), filename='train_gen_a2b_im-iteration: ' + str(iterations) + '.jpg')
                        telegram_bot_send_document(os.path.join(image_directory, 'gen_a2b_' + 'test_%08d' % (iterations + 1) + '.jpg'), filename='test_gen_a2b_im-iteration: ' + str(iterations) + '.jpg')

                if config['do_b2a']:
                    train_writer.add_image('b2a/train', train_gen_b2a_im, iterations)
                    train_writer.add_image('b2a/test', test_gen_b2a_im, iterations)
                    if config['misc']['do_telegram_report']:
                        telegram_bot_send_message('iteration: ' + str(iterations))
                        telegram_bot_send_document(os.path.join(image_directory, 'gen_b2a_' + 'train_%08d' % (iterations + 1) + '.jpg'), filename='train_gen_b2a_im-iteration: ' + str(iterations) + '.jpg')
                        telegram_bot_send_document(os.path.join(image_directory, 'gen_b2a_' + 'test_%08d' % (iterations + 1) + '.jpg'), filename='test_gen_b2a_im-iteration: ' + str(iterations) + '.jpg')

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
                    if config['misc']['do_telegram_report']:
                        telegram_bot_send_message('problem Occurred need attention!')
                    input("Clear space and press enter to retry ....")
                    print("retrying to save...")
                    trainer.save(checkpoint_directory, iterations)
                if config['misc']['do_telegram_report']:
                    telegram_bot_send_message('snapshot saved iter: ' + str(iterations))
            trainer.update_learning_rate()

except Exception as e:
    print('Error')
    print('-' * 60)
    traceback.print_exc(file=sys.stdout)
    print('-' * 60)
    print(e)
    print(colored('Training STOPED!', color='red', attrs=['underline', 'bold', 'blink', 'reverse']))
    if config['misc']['do_telegram_report']:
        telegram_bot_send_message('Error Training STOPED!')
