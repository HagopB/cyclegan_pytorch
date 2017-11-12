import numpy as np
import os
import ntpath
import time
from . import util
from scipy.misc import *

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, visuals, total_steps):
        image_dir = self.img_dir
        name = '{}.png'.format(total_steps)
        
        h = self.opt.fineSize
        w = h
        nh = 6
        nw = self.opt.batchSize
        img = np.zeros((h*nh, w*nw, 3))

        n = 0
        for label, image_numpy in visuals.items():
            j = int(n/nw)
            i = int(n%nw)
            if n >= nh*nw: break
            img[j*h:j*h+h, i*w:i*w+w, :] = image_numpy
            n += 1
        imsave(os.path.join(image_dir,name), img)
        
    def save_preds(self, visuals, results_dir, step):
        image_dir = results_dir
        if os.path.exists(image_dir) == False:
            os.mkdir(image_dir)
        
        name = '{}.png'.format(step)
        
        h = self.opt.fineSize
        w = h
        nh = 2
        nw = self.opt.batchSize
        img = np.zeros((h*nh, w*nw, 3))

        n = 0
        for label, image_numpy in visuals.items():
            j = int(n/nw)
            i = int(n%nw)
            if n >= nh*nw: break
            img[j*h:j*h+h, i*w:i*w+w, :] = image_numpy
            n += 1
        imsave(os.path.join(image_dir,name), img)