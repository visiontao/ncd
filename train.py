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
parser.add_argument('--num_neg', type=int, default=4)

parser.add_argument('--fig_type', type=str, default='*') 
parser.add_argument('--dataset', type=str, default='i-raven')
parser.add_argument('--root', type=str, default='../dataset/rpm')

#parser.add_argument('--fig_type', type=str, default='neutral')   # neutral, interpolation, extrapolation
#parser.add_argument('--dataset', type=str, default='pgm')
#parser.add_argument('--root', type=str, default='../dataset/rpm/pgm')

parser.add_argument('--train_mode', type=bool, default=True)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--seed', type=int, default=123)

args = parser.parse_args()


if args.dataset == 'raven' or args.dataset == 'i-raven':   
    mode = 'r'
    args.img_size = 256
    args.batch_size = 64
elif args.dataset == 'pgm':   
    mode = 'rc'
    args.img_size = 96
    args.batch_size = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available:
    torch.cuda.manual_seed(args.seed)

tf = transforms.Compose([ToTensor()])    

train_set = dataset(os.path.join(args.root, args.dataset), 'train', args.fig_type, args.img_size, tf, args.train_mode)
test_set = dataset(os.path.join(args.root, args.dataset), 'test', args.fig_type, args.img_size, tf)

print ('test length', len(test_set), args.fig_type)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

save_name = args.model_name + '_' + args.fig_type + '_' + str(args.num_neg) + '_' + str(args.img_size) + '_' + str(args.batch_size)

save_path_model = os.path.join(args.dataset, 'models', save_name)    
if not os.path.exists(save_path_model):
    os.makedirs(save_path_model)    
    
save_path_log = os.path.join(args.dataset, 'logs')    
if not os.path.exists(save_path_log):
    os.makedirs(save_path_log)   
    
model = NCD(mode)
if torch.cuda.device_count() > 0:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)    
model.to(device)    
#model.load_state_dict(torch.load(save_path_model+'/model_06.pth'))    

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

time_now = datetime.now().strftime('%D-%H:%M:%S')      
save_log_name = os.path.join(save_path_log, 'log_{:s}.txt'.format(save_name)) 
with open(save_log_name, 'a') as f:
    f.write('\n------ lr: {:f}, batch_size: {:d}, img_size: {:d}, time: {:s} ------\n'.format(
        args.lr, args.batch_size, args.img_size, time_now))
f.close() 


def replace_ans(x, k=8):    
    if k > 0:
        x1 = x[:, 0:16-k]
        x2 = x[:, 16-k:]
    
        indices = np.random.permutation(x.size(0))
        x = torch.cat((x1, x2[indices,:]), dim=1)
    
    return x

def compute_loss(predict):    
    pseudo_target = torch.zeros(predict.shape)
    pseudo_target = Variable(pseudo_target, requires_grad=False).to(device)
    pseudo_target[:, 0:2] = 1    
         
    return F.binary_cross_entropy_with_logits(predict, pseudo_target)


def train(epoch):
    model.train()    
    metrics = {'loss': []}
    
    train_loader_iter = iter(train_loader)
    for batch_idx in trange(len(train_loader_iter)):
        image, _ = next(train_loader_iter)
        
        image = Variable(image, requires_grad=True).to(device)
        
        image = replace_ans(image, args.num_neg)    
        predict = model(image)             
        loss = compute_loss(predict)        
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        metrics['loss'].append(loss.item())
               
        if batch_idx > 1 and batch_idx % 12000 == 0:
            print ('Epoch: {:d}/{:d},  Loss: {:.3f}'.format(epoch, args.epochs, np.mean(metrics['loss'])))
        
    
    print ('Epoch: {:d}/{:d},  Loss: {:.3f}'.format(epoch, args.epochs, np.mean(metrics['loss'])))
            
    return metrics


def test(epoch):
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

    print ('Testing Epoch: {:d}/{:d}, Accuracy: {:.3f} \n'.format(epoch, args.epochs, accuracy))
            
    return metrics


if __name__ == '__main__':
    for epoch in range(1, args.epochs+1):      

        #metrics_test = test(epoch)
        #break

        metrics_train = train(epoch)

        # Save model
        if epoch > 0:
            save_name = os.path.join(save_path_model, 'model_{:02d}.pth'.format(epoch))
            torch.save(model.state_dict(), save_name)

        metrics_test = test(epoch)

        loss_train = np.mean(metrics_train['loss'])
        acc_test = 100 * np.sum(metrics_test['correct']) / np.sum(metrics_test['count']) 
                
        time_now = datetime.now().strftime('%H:%M:%S')    
        
        with open(save_log_name, 'a') as f:
            f.write('Epoch {:02d}: Accuracy: {:.3f}, Loss: {:.3f}, Time: {:s}\n'.format(
                epoch, acc_test, loss_train, time_now))
        f.close() 
        

        
   
