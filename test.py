import os
import torch
import numpy as np
import argparse
import time
import torch.nn.functional as F

from tqdm import trange
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

from dataset_utility import dataset, ToTensor
from ncd import NCD

os.environ['CUDA_VISIBLE_DEVICES'] ='0, 1'
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='ncd')

parser.add_argument('--dataset', type=str, default='i-raven')
parser.add_argument('--root', type=str, default='../dataset/rpm')
parser.add_argument('--pretrained_model', type=str, default='pretrained_models/model_iraven.pth')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=123)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available:
    torch.cuda.manual_seed(args.seed)
tf = transforms.Compose([ToTensor()])    
    
if args.dataset == 'raven' or args.dataset == 'i-raven':   
    mode = 'r'
    args.img_size = 256
    args.batch_size = 64
elif args.dataset == 'pgm':   
    mode = 'rc'
    args.img_size = 96
    args.batch_size = 256

model = NCD(mode).to(device)  
if torch.cuda.device_count() > 0:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)    
model.load_state_dict(torch.load(args.pretrained_model))       

    
def test(test_loader):
    model.eval()
    metrics = {'correct': [], 'count': []}
            
    test_loader_iter = iter(test_loader)
    for _ in trange(len(test_loader_iter)):
        image, target = next(test_loader_iter)
        
        image = Variable(image, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)
                
        with torch.no_grad():   
            predict = model(image)    
            
        pred = torch.max(predict[:, 2:], 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        
        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))
        
        accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count']) 
            
    return metrics


if __name__ == '__main__':

    if args.dataset == 'raven' or args.dataset == 'i-raven':
        fig_types = ['center_single', 'distribute_four', 'distribute_nine', 
        'left_center_single_right_center_single', 'up_center_single_down_center_single', 
        'in_center_single_out_center_single', 'in_distribute_four_out_center_single']
    
        accuracy_list = []
        for i in range(len(fig_types)):
            test_set = dataset(os.path.join(args.root, args.dataset), 'test', fig_types[i], args.img_size, tf)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
            metrics_test = test(test_loader)
            acc_test = 100 * np.sum(metrics_test['correct']) / np.sum(metrics_test['count']) 
            accuracy_list.append(acc_test)
    
            print ('FigType: {:s}, Accuracy: {:.3f} \n'.format(fig_types[i], acc_test))
                  
        print (accuracy_list)
        print ('Average Accuracy: {:.3f} \n'.format(np.mean(accuracy_list)))
        
    elif args.dataset == 'pgm':
        test_set = dataset(args.root, 'test', 'interpolation', args.img_size, tf)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
                
        metrics_test = test(test_loader)
        acc_test = 100 * np.sum(metrics_test['correct']) / np.sum(metrics_test['count']) 

        print ('Average Accuracy: {:.3f} \n'.format(acc_test))
        
  
