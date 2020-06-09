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
import cv2
import time
import warnings

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

import torch



import torchvision.utils as vutils
import warnings

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
import time
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename
import os
import shutil
from utils import write_2images

use_face_locations = True
# use_face_locations = False

if use_face_locations:
    from PIL import Image
    try:
        import face_recognition
    except:
        warnings.warn("Filed to import face_recognition, setting use_face_locations to FALSE")
        use_face_locations = False

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--input_folder', type=str, help="input image folder")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--output_path', type=str, default='.outputs', help="outputs path")

parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b 0 for b2a")
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
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']
council_size = config['council']['council_size']

# Setup model and data loader
if not 'new_size_a' in config.keys():
    config['new_size_a'] = config['new_size']
is_data_A = opts.a2b

style_dim = config['gen']['style_dim']
trainer = Council_Trainer(config)
only_one = False
if 'gen_' in opts.checkpoint[-21:]:
    state_dict = torch.load(opts.checkpoint, map_location={'cuda:1':'cuda:0'})
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
            state_dict = torch.load(tmp_checkpoint, map_location={'cuda:1':'cuda:0'})
            trainer.gen_a2b_s[i].load_state_dict(state_dict['a2b'])
        else:
            tmp_checkpoint = opts.checkpoint[:-8] + 'b2a_gen_' + str(i) + '_' + opts.checkpoint[-8:] + '.pt'
            state_dict = torch.load(tmp_checkpoint, map_location={'cuda:1':'cuda:0'})
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
        state_dict = torch.load(checkpoint, map_location={'cuda:1':'cuda:0'})
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
        warnings.warn('FAILED to load network! the yaml config file might be wrong ')




from threading import Thread, Lock

class WebcamVideoStream :
    def __init__(self, src=0, width=640, height=480) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()





from torchvision import transforms
from torchvision.utils import save_image


def run_net_work(img_path, entropy, config=config, use_face_locations=False, face_increes_by_dev_ratio=1.7, move_up_by_ratio=0):
    out_im_path = './tmp.jpg'
    in_im_path = './tmp_in.jpg'
    net_hight = config['crop_image_height']
    net_width = config['crop_image_width']
    net_new_size = config['new_size']
    do_pad_with_zeros_if_not_squared = True

    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    transform_list = [transforms.ToTensor(), transforms.Normalize(mean=mean.tolist(), std=std.tolist())]

    transform_list = [transforms.CenterCrop((net_hight, net_width))] + transform_list
    transform_list = [transforms.Resize(net_new_size)] + transform_list
    transform = transforms.Compose(transform_list)

    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    if use_face_locations:
        img = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(img=img, number_of_times_to_upsample=0)
        img_h = img.shape[0]
        img_w = img.shape[1]

        if do_pad_with_zeros_if_not_squared:
            # padd with zeros if the image is not squared
            left_index = 0
            up_index = 0
            if img_h != img_w:
                new_size = max(img_h, img_w)
                new_im = np.zeros((new_size, new_size, img.shape[2]))
                if img_h > img_w:
                    left_index = int((new_size - img_w)/2)
                    new_im[:, left_index:left_index+img_w, :] = img
                else:
                    up_index = int((new_size - img_h)/2)
                    new_im[up_index:up_index+img_h, :, :] = img
                for cur, face_location in enumerate(face_locations):
                    # (top, right, bottom, left)
                    face_location = (face_location[0]+up_index, face_location[1]+left_index, face_location[2]+up_index, face_location[3]+left_index)
                    face_locations[cur] = face_location
                img_h = new_size
                img_w = new_size
                old_img = img
                img = new_im.astype(np.uint8)

        final_res_img = transforms.ToTensor()(img)
        in_img = transforms.ToTensor()(img)

    else:
        img = Image.open(img_path)
        img = np.array(img)
        face_locations = [[0, img.shape[1], img.shape[0], 0]]
        img_h = img.shape[0]
        img_w = img.shape[1]
        final_res_img = transforms.ToTensor()(img)
        in_img = transforms.ToTensor()(img)

    for face_location in face_locations:
        top, right, bottom, left = face_location

        hight = bottom - top
        width = right - left
        if use_face_locations:
            # making the image larger because face_recognition  cuts the faces
            increes_by = int(max(hight, width) / face_increes_by_dev_ratio)

            if hight + increes_by > img_h or width + increes_by > img_w:
                # increes_by is too big
                increes_by_max_h = int((img_h - hight) / 2)
                increes_by_max_w = int((img_w - width) / 2)
                increes_by = min(increes_by_max_h, increes_by_max_w)

            top, right, bottom, left = top - increes_by, right + increes_by, bottom + increes_by, left - increes_by
            hight = bottom - top
            width = right - left

            if top < 0:
                top = 0
                bottom = hight if hight < img_h else img_h-1
            if bottom >= img_h:
                bottom = img_h - 1
                top = bottom - hight if bottom - hight >= 0 else 0
            if left < 0:
                left = 0
                right = width if width < img_w else img_w - 1
            if right >= img_w:
                right = img_w - 1
                left = right - width if right - width >= 0 else 0

            hight = bottom - top
            width = right - left

            #make squer
            bottom = top + min(hight, width, img_h, img_w)
            right = left + min(hight, width, img_h, img_w)

            hight = bottom - top
            width = right - left

            # move the up the face square
            move_up_by_ratio_pix = int(hight * move_up_by_ratio)
            if move_up_by_ratio_pix > 0:
                move_up_by_ratio_pix = min(move_up_by_ratio_pix, top)
            else:
                move_up_by_ratio_pix = max(move_up_by_ratio_pix, bottom - img_h) + 1
            top -= move_up_by_ratio_pix
            bottom -= move_up_by_ratio_pix

            # last checks
            if top < 0:
                top = 0
            if bottom >= img_h:
                bottom = img_h - 1
            if left < 0:
                left = 0
            if right >= img_w:
                right = img_w - 1
        curr_face_image = img[top:bottom, left:right]
        curr_face_image = transform(Image.fromarray(curr_face_image)).unsqueeze(0).cuda()
        content, _ = encode_s[0](curr_face_image)
        res_img = decode_s[0](content, entropy, curr_face_image).detach().cpu().squeeze(0)
        res_img = transforms.Normalize(mean=(-1 * mean / std).tolist(), std=(1.0 / std).tolist())(res_img)
        # resize the network output to fit the original image
        transforms_size_prossesing = [transforms.ToPILImage(), transforms.Resize(size=(hight, width)), transforms.ToTensor()]
        transforms_size_prossesing = transforms.Compose(transforms_size_prossesing)
        res_img = transforms_size_prossesing(res_img)
        if bottom - top < res_img.shape[1]:
            bottom += 1
        if right - left < res_img.shape[2]:
            left += 1
        final_res_img[:, top:bottom, left:right] = transforms_size_prossesing(res_img)

        curr_face_image = curr_face_image.cpu().squeeze(0)
        curr_face_image = transforms.Normalize(mean=(-1 * mean / std).tolist(), std=(1.0 / std).tolist())(curr_face_image)
        in_img[:, top:bottom, left:right] = transforms_size_prossesing(curr_face_image.cpu().squeeze(0))
    if use_face_locations:
        if do_pad_with_zeros_if_not_squared:
            if up_index > 0:
                final_res_img = final_res_img[:, up_index:-up_index, :]
                in_img = in_img[:, up_index:-up_index, :]
            if left_index > 0:
                final_res_img = final_res_img[:, :, left_index:-left_index]
                in_img = in_img[:, :, left_index:-left_index]

    save_image(final_res_img, out_im_path)
    save_image(in_img, in_im_path)
    return in_im_path, out_im_path





if __name__ == '__main__':


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
            max_added_val = 50
            random_entropy_direction_mult = (self.slider.value() - self.slider.maximum() / 2) / (self.slider.maximum())
            random_entropy = self.random_entropy + max_added_val * self.random_entropy_direction * random_entropy_direction_mult

            self.in_im_path, self.res_im_path = run_net_work(img_path=self.img_path, entropy=random_entropy,
                                                             use_face_locations=self.use_face_locations,
                                                             face_increes_by_dev_ratio=self.face_increes_by_dev_ratio)
            self.im_out = QPixmap(self.res_im_path)
            self.out_image_label.setPixmap(self.im_out.scaled(w, h))
            self.out_image_label.repaint()
            self.im_in = QPixmap(self.in_im_path)
            self.in_image_label.setPixmap(self.im_in.scaled(w, h))
            self.in_image_label.repaint()

        def sliderReleased(self):
            self.redraw_in_and_out()

        def face_incres_sliderReleased(self):
            self.face_increes_by_dev_ratio = self.min_face_range + self.face_slider_range * self.slider_face_increse.value() / 100
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
            if self.live_view_on:
                self.live_view_on = False
                return
            self.pushbutton_take_pic.setText('Press Here to Stop')
            self.pushbutton_record.setEnabled(True)

            print('press Esc to stop')
            self.img_path = './cap_tmp_in.png'
            # cap = cv2.VideoCapture(0)
            cap = WebcamVideoStream(src=0, width=640, height=480).start()
            self.live_view_on = True
            start_time = time.time()
            while (self.live_view_on):
                frame = cap.read()
                # Display the resulting frame
                if frame is None:
                    break
                cv2.imshow('press ENTER to stop', frame)
                press_key = cv2.waitKey(1)


                if press_key & 0xFF == ord('q') or press_key == 27 or press_key == 13:
                    break

                stop_time = time.time()
                duration = stop_time - start_time
                print('FPS: ' + str(1/duration))
                start_time = time.time()

                cv2.imwrite(self.img_path, frame)
                self.redraw_in_and_out()

                if self.do_record_vid:
                    res_img = cv2.imread(self.res_im_path)
                    in_img = cv2.imread(self.in_im_path)
                    to_save_frame = np.concatenate((in_img, res_img), axis=1)
                    self.out_vid.write(to_save_frame)
            cap.stop()

            cv2.destroyAllWindows()
            if frame is not None:
                cv2.imwrite(self.img_path, frame)
                self.redraw_in_and_out()
            self.slider.setEnabled(True)
            self.pushbutton_random_entropy.setEnabled(True)
            self.label.setEnabled(True)
            self.pushbutton_take_pic.setEnabled(True)
            self.label_net.setEnabled(True)
            self.cb_ft.setEnabled(True)
            self.pushbutton_record.setEnabled(False)


            cv2.imwrite(self.img_path, frame)
            self.pushbutton_take_pic.setText('live webcam view')

        def save_image_pressed(self):
            Tk().withdraw()
            savepath = asksaveasfilename( defaultextension=".png")
            if savepath is None:
                return
            print('saving image to: ' + savepath)
            shutil.copyfile(self.in_im_path, savepath[:-4] + '_in' + savepath[-4:])
            shutil.copyfile(self.res_im_path, savepath[:-4] + '_out' + savepath[-4:])


        def cb_face_traucker_changed(self,cb):
            self.use_face_locations = self.cb_ft.isChecked()
            self.slider_face_increse.setEnabled(self.use_face_locations)

        def record_vid(self):
            if not self.do_record_vid:
                self.pushbutton_record.setText('Stop')
                self.out_vid = cv2.VideoWriter('output_vid.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (2*640, 480))
                self.do_record_vid = True

            else:
                self.do_record_vid = False
                self.pushbutton_record.setText('Record')
                self.out_vid.release()


        def __init__(self):
            super().__init__()
            self.title = 'Council GAN example'
            self.left = 10
            self.top = 10
            self.width = 640  # 640
            self.height = 480  # 480
            self.min_face_range = 0.5
            self.max_face_range = 6.5
            self.face_slider_range = self.max_face_range - self.min_face_range
            self.layout = QVBoxLayout()
            self.hbox = QHBoxLayout()
            self.hbox2 = QHBoxLayout()
            self.random_entropy = Variable(torch.randn(1, style_dim, 1, 1).cuda())
            self.random_entropy_direction = Variable(torch.randn(1, style_dim, 1, 1).cuda())
            self.random_entropy_direction /= torch.norm(self.random_entropy_direction)
            self.res_im_path = None
            self.use_face_locations = False
            self.live_view_on = False
            self.do_record_vid = False
            self.out_vid = None
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

            self.label = DropLabel("drag & drop image into this line")
            self.label.dropEvent = self.dropEvent
            self.label.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(self.label)


            self.layout.addLayout(self.hbox2)

            self.cb_ft = QCheckBox("face trucker")
            self.cb_ft.setChecked(False)
            self.cb_ft.stateChanged.connect(self.cb_face_traucker_changed)
            self.hbox2.addWidget(self.cb_ft, stretch=1)

            self.pushbutton_take_pic = QPushButton(text='live webcam view')
            self.pushbutton_take_pic.pressed.connect(self.take_pic_button_pressed)
            self.hbox2.addWidget(self.pushbutton_take_pic, stretch=4)

            self.pushbutton_record = QPushButton(text='Record')
            self.pushbutton_record.pressed.connect(self.record_vid)
            self.hbox2.addWidget(self.pushbutton_record, stretch=1)
            self.pushbutton_record.setEnabled(False)



            self.slider_face_increse = Style_Slider(orientation=Qt.Horizontal)
            self.slider_face_increse.setValue(27)
            self.slider_face_increse.sliderReleased.connect(self.face_incres_sliderReleased)
            self.slider_face_increse.valueChanged.connect(self.face_incres_sliderReleased)
            self.face_increes_by_dev_ratio = self.min_face_range + self.face_slider_range * self.slider_face_increse.value() / 100
            self.slider_face_increse.setEnabled(self.use_face_locations)
            self.layout.addWidget(self.slider_face_increse, Qt.AlignBottom)

            self.pushbutton_random_entropy = QPushButton(text='new random entropy vector & entropy vector direction')
            self.pushbutton_random_entropy.pressed.connect(self.random_button_pressed)
            self.layout.addWidget(self.pushbutton_random_entropy)


            self.slider = Style_Slider(orientation=Qt.Horizontal)
            self.slider.setValue(50)
            self.slider.sliderReleased.connect(self.sliderReleased)
            self.slider.valueChanged.connect(self.sliderReleased)
            self.layout.addWidget(self.slider, Qt.AlignBottom)

            self.label_net = DropLabel("drag & drop net \".pt\" file into this line")
            self.label_net.dropEvent = self.dropEvent_new_net
            self.label_net.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(self.label_net)

            self.save_image = QPushButton(text='save image')
            self.save_image.pressed.connect(self.save_image_pressed)
            self.layout.addWidget(self.save_image)

            self.setLayout(self.layout)
            self.initUI()

        def initUI(self):
            self.setWindowTitle(self.title)
            self.setGeometry(self.left, self.top, self.width, self.height)
            self.show()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)

    ex = App()
    sys.exit(app.exec_())
