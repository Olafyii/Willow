import os
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms, utils, models
from torch.autograd import Variable
from torch import nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from tqdm import tqdm
import time
import torch.backends.cudnn as cudnn
from utils import AverageMeter, accuracy, record_info
from dataset import Willow

# if __name__ == '__main__':  # multithread loading needs to specify this.
#     trainset = Willow(mode='train')
#     trainloader = DataLoader(trainset, batch_size=64, num_workers=0)
#     testset = Willow(mode='test')
#     testloader = DataLoader(testset, batch_size=64, num_workers=0)

#     alexnet = models.alexnet(pretrained=True).cuda()
#     criterion = nn.CrossEntropyLoss().cuda()
#     optimizer = torch.optim.SGD(alexnet.parameters(), 1e-5, momentum=0.9)
#     scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1,verbose=True)

#     for epoch in range(10):
#         progress = tqdm(trainloader)
#         t = time.time()
#         for i, (data, label) in enumerate(progress):
#             print('load time:', time.time() - t)
#             t = time.time()
#             label = label.cuda(async=True)
#             input_var = Variable(data).cuda()
#             target_var = Variable(label).cuda()
#             print('load to cuda:', time.time() - t)

#             t = time.time()
#             output = alexnet(input_var)
#             loss = criterion(output, target_var)
#             # print('loss:', loss.item())
#             print('calculate loss:', time.time() - t)

#             t = time.time()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             print('backward:', time.time() - t)
#             t = time.time()


# def train_1epoch():
#     for i, (data, label) in enumerate(progress):
#         print('load time:', time.time() - t)
#         t = time.time()
#         label = label.cuda(async=True)
#         input_var = Variable(data).cuda()
#         target_var = Variable(label).cuda()
#         print('load to cuda:', time.time() - t)

#         t = time.time()
#         output = alexnet(input_var)
#         loss = criterion(output, target_var)
#         # print('loss:', loss.item())
#         print('calculate loss:', time.time() - t)

#         t = time.time()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print('backward:', time.time() - t)
#         t = time.time()


def main():
    trainset = Willow('train')
    testset = Willow('test')
    trainloader = DataLoader(trainset, 64)
    testloader = DataLoader(testset, 64)
    m = model(500, 1e-3, trainloader, testloader)
    m.run()


class model():
    def __init__(self, nb_epochs, lr, trainloader, testloader):
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.trainloader = trainloader
        self.testloader = testloader
    
    def build_model(self):
        self.model = models.alexnet(pretrained=True).cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)
    
    def run(self):
        self.build_model()
        cudnn.benchmark = True

        for self.epoch in range(self.nb_epochs):
            self.train_1epoch()
            self.validate_1epoch()
    
    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.model.train()
        end = time.time()

        progress = tqdm(self.trainloader)
        for i, (data, label) in enumerate(progress):
            data_time.update(time.time() - end)
            
            label = label.cuda(async=True)
            input_var = Variable(data).cuda()
            target_var = Variable(label).cuda()
            # print('load to cuda:', time.time() - t)

            # t = time.time()
            output = self.model(input_var)
            loss = self.criterion(output, target_var)
            # print('loss:', loss.item())
            # print('calculate loss:', time.time() - t)
            prec1, prec5 = accuracy(output.data, label, topk=(1,5))
            losses.update(loss.item(), output.data.size(0))
            top1.update(prec1.item(), output.data.size(0))
            top5.update(prec5.item(), output.data.size(0))
            # print('test', output.data.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {
            'Epoch':[self.epoch],
            'Batch Time':[round(batch_time.avg, 3)],
            'Data Time':[round(data_time.avg, 3)],
            'Loss':[round(losses.avg, 5)],
            'Prec@1':[round(top1.avg, 4)],
            'Prec@5':[round(top5.avg, 4)],
            'lr':self.optimizer.param_groups[0]['lr']
        }

        record_info(info, './record/spatial/rgb_train.csv','train')
    
    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.model.eval()
        end = time.time()

        progress = tqdm(self.testloader)
        for i, (data, label) in enumerate(progress):
            label = label.cuda(async=True)
            input_var = Variable(data).cuda()
            target_var = Variable(label).cuda()
            # print('load to cuda:', time.time() - t)

            output = self.model(input_var)
            prec1, prec5 = accuracy(output.data, label, topk=(1,5))
            top1.update(prec1.item(), output.data.size(0))
            top5.update(prec5.item(), output.data.size(0))

        info = {
            'Epoch':[self.epoch],
            'Batch Time':[round(batch_time.avg, 3)],
            'Loss':[0],
            'Prec@1':[round(top1.avg, 4)],
            'Prec@5':[round(top5.avg, 4)],
        }

        record_info(info, 'record/spatial/rgb_test.csv','test')

if __name__ == '__main__':
    main()