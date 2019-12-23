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
import cv2

from PIL import Image

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
    except  Exception as e:
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


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class DropLabel(QLineEdit):
    def __init__(self, *args, **kwargs):
        QLabel.__init__(self, *args, **kwargs)
        self.setAcceptDrops(True)
        self.setEnabled(True)
        self.res_im = None

class Style_Slider(QSlider):
    def __init__(self, *args, **kwargs):
        QSlider.__init__(self, *args, **kwargs)

class App(QWidget):


    def redraw_in_and_out(self):
        if self.__dict__.get('img_path') is None:
            return
        h = 256
        w = 256
        max_added_val = 250
        random_entropy_direction_mult = (self.slider.value() - self.slider.maximum() / 2) / (self.slider.maximum())
        print('random_entropy +  ' + str(random_entropy_direction_mult) + ' * random_entropy_direction')
        random_entropy = self.random_entropy + max_added_val * self.random_entropy_direction * random_entropy_direction_mult

        self.in_im_path, self.res_im_path = run_net_work(img_path=self.img_path, entropy=random_entropy)
        self.im_out = QPixmap(self.res_im_path)
        self.out_image_label.setPixmap(self.im_out.scaled(w, h))
        self.out_image_label.repaint()


        self.im_in = QPixmap(self.in_im_path)
        self.in_image_label.setPixmap(self.im_in.scaled(w, h))
        self.in_image_label.repaint()


    def sliderReleased(self):
        # self.slider.value()
        # print(self.slider.value())
        self.redraw_in_and_out()

    def dropEvent(self, event):
        self.img_path = event.mimeData().text()[7:-2]
        print('prossing image: ' + self.img_path)
        self.redraw_in_and_out()

    def dropEvent_new_net(self, event):
        self.net_path = event.mimeData().text()[7:-2]
        load_net(self.net_path)
        self.redraw_in_and_out()
        self.label_net.setText(self.net_path)

    def random_button_pressed(self):
        self.random_entropy = Variable(torch.randn(1, style_dim, 1, 1).cuda())
        self.random_entropy_direction = Variable(torch.randn(1, style_dim, 1, 1).cuda())
        self.random_entropy_direction /= torch.norm(self.random_entropy_direction)
        self.redraw_in_and_out()

    def take_pic_button_pressed(self):
        self.slider.setEnabled(False)
        self.pushbutton_random_entropy.setEnabled(False)
        self.label.setEnabled(False)
        self.pushbutton_take_pic.setEnabled(False)
        self.label_net.setEnabled(False)
        self.pushbutton_take_pic.setText('Press Esc to stop')

        print('press Esc to stop')
        self.img_path = './cap_tmp_in.png'

        cap = cv2.VideoCapture(0)
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            press_key = cv2.waitKey(1)

            if press_key & 0xFF == ord('q') or press_key == 27:
                break


            cv2.imwrite(self.img_path, frame)
            # cv2.imwrite(self.img_path, gray)
            self.redraw_in_and_out()

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()



        cv2.imwrite(self.img_path,frame)

        self.redraw_in_and_out()
        self.slider.setEnabled(True)
        self.pushbutton_random_entropy.setEnabled(True)
        self.label.setEnabled(True)
        self.pushbutton_take_pic.setEnabled(True)
        self.label_net.setEnabled(True)

        self.pushbutton_take_pic.setText('take a pic')


    def __init__(self):
        super().__init__()
        self.title = 'Council GAN example'
        self.left = 10
        self.top = 10
        self.width = 640 # 640
        self.height = 480 # 480
        self.layout = QVBoxLayout()
        self.hbox = QHBoxLayout()
        self.random_entropy = Variable(torch.randn(1, style_dim, 1, 1).cuda())
        self.random_entropy_direction = Variable(torch.randn(1, style_dim, 1, 1).cuda())
        self.random_entropy_direction /= torch.norm(self.random_entropy_direction)
        self.res_im_path = None

        self.label = DropLabel("drag & drop image into this line")
        self.label.dropEvent = self.dropEvent
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.layout.addLayout(self.hbox)
        self.in_image_label = QLabel("in")
        self.in_image_label.setUpdatesEnabled(True)
        self.hbox.addWidget(self.in_image_label)
        self.hbox.addStretch()
        self.layout.addStretch()

        self.out_image_label = QLabel("out")
        self.out_image_label.setUpdatesEnabled(True)
        self.out_image_label.resize(256, 256)
        self.hbox.addWidget(self.out_image_label)


        self.pushbutton_take_pic = QPushButton(text='take a pic')
        self.pushbutton_take_pic.pressed.connect(self.take_pic_button_pressed)
        self.layout.addWidget(self.pushbutton_take_pic)


        self.pushbutton_random_entropy = QPushButton(text='new random entropy vector & entropy vector direction')
        self.pushbutton_random_entropy.pressed.connect(self.random_button_pressed)
        self.layout.addWidget(self.pushbutton_random_entropy)


        self.slider = Style_Slider(orientation=Qt.Horizontal)
        self.slider.setValue(50)
        self.slider.sliderReleased.connect(self.sliderReleased)
        self.layout.addWidget(self.slider, Qt.AlignBottom)

        self.label_net = DropLabel("drag & drop net \".pt\" file into this line")
        self.label_net.dropEvent = self.dropEvent_new_net
        self.label_net.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label_net)

        self.setLayout(self.layout)
        self.initUI()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    # def dropEvent(self, e):
    #     print(e.mimeData().text())
    #     self.setText(e.mimeData().text())




if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)

    ex = App()
    sys.exit(app.exec_())
