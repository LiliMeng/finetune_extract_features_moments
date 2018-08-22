'''
Fine-tune the ResNet pretrained on ImageNet finetuned on Moments to reduce
the dimension of features. Previously we extract the last layer of Convolutional
features which has dimension of [15,2048,8,8] (the input image size 256x256, if the 
input size is [224, 224], the last Conv dim is [15, 2048, 7, 7])

Author: Lili Meng (menglili@cs.ubc.ca)
Date: August 22nd, 2018
'''
import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse

import torch
import torchvision
import torch.nn.parallel
import torch.optim
from torch.nn.utils import clip_grad_norm
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import shutil

from opts import parser
from test_model import load_model, load_categories, load_transform
from spatial_dataloader import *
from network import *
from moments_dataloader import *
from utils import *

parser = argparse.ArgumentParser(description='Moments spatial stream on resnet101')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_false', help='evaluate model on validation set')
parser.add_argument('--resume', default='./record/spatial/model_best.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--num_classes', default=339, type=int, metavar='N', help='number of classes in the dataset')


def main():

    global arg
    arg = parser.parse_args()
    print(arg)

    categories, train_list, val_list, root_path, prefix = return_moments()
    num_class = len(categories)
    assert(num_class == arg.num_classes)

    dataloader = spatial_dataloader(BATCH_SIZE=arg.batch_size,
                                    num_workers=8,
                                    path='/media/lili/fce9875a-a5c8-4c35-8f60-db60be29ea5d/Moments_in_Time_Raw/',
                                    train_list ='./img_list/new_moments_train_list.txt',
                                    test_list = './img_list/new_moments_validation_list.txt')

    train_loader,val_loader, test_video = dataloader.run()

    model = Spatial_CNN(nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        train_loader=train_loader,
                        test_loader=val_loader,
                        test_video=test_video
    )

    #Training
    model.run()
   
   
class Spatial_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, test_video):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.test_video=test_video

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet50(pretrained=True, channel=3, num_classes=arg.num_classes).cuda()
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)
    
    def resume_and_evaluate(self):
        weight_file = 'moments_RGB_resnet50_imagenetpretrained.pth.tar'
        if not os.access(weight_file, os.W_OK):
            weight_url = 'http://moments.csail.mit.edu/moments_models/' + weight_file
            os.system('wget ' + weight_url)
        model = models.__dict__['resnet50'](num_classes=arg.num_classes)
        checkpoint = torch.load(weight_file)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        # if self.resume:
        #     model = load_model(model_id, categories).cuda()
        #     if os.path.isfile(self.resume):
        #         print("==> loading checkpoint '{}'".format(self.resume))
        #         checkpoint = torch.load(self.resume)
        #         self.start_epoch = checkpoint['epoch']
        #         self.best_prec1 = checkpoint['best_prec1']
        #         self.model.load_state_dict(checkpoint['state_dict'])
        #         self.optimizer.load_state_dict(checkpoint['optimizer'])
        #         print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
        #           .format(self.resume, checkpoint['epoch'], self.best_prec1))
        #     else:
        #         print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        log_dir = os.path.join('./train_cnn_log', 'hmdb51_3_classes'+time.strftime("_%b_%d_%H_%M", time.localtime()))

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)

        for self.epoch in range(self.start_epoch, self.nb_epochs):
            train_prec1, train_loss=self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            writer.add_scalar('train_loss', train_loss, self.epoch)
            writer.add_scalar('train_accuracy', train_prec1, self.epoch)
            writer.add_scalar('test_loss', val_loss, self.epoch)
            writer.add_scalar('test_accuracy', prec1, self.epoch)
            is_best = prec1 > self.best_prec1
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_prec1 = prec1
                with open('record/spatial/spatial_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                f.close()
            
                save_checkpoint({
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': self.best_prec1,
                    'optimizer' : self.optimizer.state_dict()
                },is_best,'./saved_checkpoints/checkpoint.pth.tar','record/spatial/model_best.pth.tar')

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data_dict,label) in enumerate(progress):

    
            # measure data loading time
            data_time.update(time.time() - end)
            
            label = label.cuda(async=True)
            target_var = Variable(label).cuda()

            # compute output
            output = Variable(torch.zeros(len(data_dict['img1']),arg.num_classes).float()).cuda()
            for i in range(len(data_dict)):
                key = 'img'+str(i)
                data = data_dict[key]
                input_var = Variable(data).cuda()
                output += self.model(input_var)

            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            losses.update(loss.data[0], data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':round(losses.avg,5),
                'Prec@1':round(top1.avg,4),
                'Prec@5':round(top5.avg,4),
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/spatial/rgb_train.csv','train')
        
        return top1.avg, losses.avg

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        for i, (keys,data,label) in enumerate(progress):
            
            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #Calculate video level prediction
            preds = output.data.cpu().numpy()
            nb_data = preds.shape[0]
            for j in range(nb_data):
                videoName = keys[j]#.split('/',1)[0]
                if videoName not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[videoName] = preds[j,:]
                else:
                    self.dic_video_level_preds[videoName] += preds[j,:]

        video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()
            

        info = {'Epoch':[self.epoch],
                'Batch Time':[(batch_time.avg,3)],
                'Loss':round(video_loss[0],5),
                'Prec@1':round(video_top1,3),
                'Prec@5':round(video_top5,3)}
        record_info(info, 'record/spatial/rgb_test.csv','test')
        
        return video_top1, video_loss[0]

    def frame2_video_level_accuracy(self):
            
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),arg.num_classes))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
        for name in sorted(self.dic_video_level_preds.keys()):
            preds = self.dic_video_level_preds[name]
#            label = int(self.test_video[name])-1
            label = (self.test_video[name])
            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label
            ii+=1         
            if np.argmax(preds) == (label):
                correct+=1

        #top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()
            
        top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))
        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())     
                            
        top1 = float(top1.numpy())
        top5 = float(top5.numpy())
            
        #print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
        return top1,top5,loss.data.cpu().numpy()

if __name__== "__main__":
  main()
