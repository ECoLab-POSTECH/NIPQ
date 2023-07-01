from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import datetime
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import argparse
from my_lib.train_test import (
    resume_checkpoint,
    create_checkpoint,
    test,
    train_aux,
    train_aux_target_avgbit,
    train_aux_target_bops,
    CosineWithWarmup,
    bops_cal
)

import pickle
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument("--data", help='dataset directory', default='/SSD/ILSVRC2012')
parser.add_argument("--ckpt", help="checkpoint directory", default='./checkpoint')

parser.add_argument("--model", 
                    choices=["mobilenetv2"])
parser.add_argument("--dataset", default="cifar100", 
                    choices=['imagenet'])

parser.add_argument("--lr", default=0.4, type=float)
parser.add_argument("--decay", default=4e-5, type=float)
parser.add_argument("--batch", default=256, type=int)
parser.add_argument("--epoch", default=120, type=int)

parser.add_argument("--warmuplen", default=3, type=int, help='scheduler warm up epoch')
parser.add_argument("--ft_epoch", default=3, type=int, help='tuning epoch')

parser.add_argument('--mode', default='avgbit', choices=['avgbit', 'bops'], help='average bit mode')

parser.add_argument("--a_scale", default=0, type=float)
parser.add_argument("--w_scale", default=0, type=float)
parser.add_argument("--bops_scale", default=0, type=float, help='using teacher')

parser.add_argument("--target", default=4, type=float, help='target bitops or avgbit')

parser.add_argument("--ts", action='store_true', help='using teacher')
args = parser.parse_args()

ckpt_root = args.ckpt  
data_root = args.data  # your path
use_cuda = torch.cuda.is_available()

print("==> Prepare data..")
if args.dataset == 'imagenet':
    from my_lib.datasets.imagenet_basic import get_loader
    valid_size = 0 
    test_loader, train_loader, _ = get_loader(data_root, test_batch=100, train_batch=args.batch, valid_size=valid_size , num_workers=16, random_seed=7777)
    class_num = 1000
else:
    raise NotImplementedError()

print("==> Prepare model..")    
if args.model == "mobilenetv2":
    if args.dataset == 'imagenet':
        from models.MobileNetV2_nq import mobilenet_v2
        model = mobilenet_v2()
        ckpt = './checkpoint/mobilenetv2_baseline/mobilenetv2_imagenet_best.pth'
        # kd boost
        if args.ts : 
            ckpt = './checkpoint/mobilenetv2_kdboost/mobilenetv2_imagenet_best.pth'
else:
        raise NotImplementedError()

model_ema=None

ckpt = torch.load(ckpt) #NOTE
model.load_state_dict(ckpt, False) #NOTE
# criterion = nn.CrossEntropyLoss()
# acc_base, _ = test(test_loader, model.cuda(), criterion, 0, False)
# print(f'** Pre-train model acc : {acc_base}')


if args.ts :
    print('==> Using teacher model')
    import torchvision
    from torchvision.models.efficientnet import efficientnet_b0
    model_t = efficientnet_b0(pretrained=True)

    for params in model_t.parameters():
        params.requires_grad = False

else :
    model_t=None

if args.mode == 'avgbit':
    print(f'** act_q : {args.a_scale} / weight_q : {args.w_scale}')
elif args.mode == 'bops':
    print(f'** bops scale : {args.bops_scale}')
else :
    raise NotImplementedError()

from my_lib.nipq import QuantOps as Q
Q.initialize(model, act=args.a_scale > 0, weight=args.w_scale > 0)

img = torch.Tensor(1, 3, 224, 224) if args.dataset =='imagenet' else torch.Tensor(1, 3, 32, 32)  

def forward_hook(module, inputs, outputs):
    module.out_shape = outputs.shape
    
hooks = []
for name, module in model.named_modules():
    if isinstance(module, (Q.ReLU, Q.Sym, Q.HSwish)):
        hooks.append(module.register_forward_hook(forward_hook))
    
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        hooks.append(module.register_forward_hook(forward_hook))

model.eval()
model.cuda()
model(img.cuda())

for hook in hooks:
    hook.remove()

if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    if model_ema is not None:
        model_ema.cuda()
        model_ema = torch.nn.DataParallel(model_ema, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
prefix = "%s_%s" % (args.model, args.dataset)

def bit_cal(model):
    numel_a = 0
    numel_w = 0
    loss_bit_a = 0
    loss_bit_au = 0
    loss_bit_w = 0
    loss_bit_wu = 0

    w_bit=-1
    a_bit=-1
    au_bit=-1
    wu_bit=-1
    for name, module in model.named_modules():
        if hasattr(module, "bit") and hasattr(module, "weight") and module.quant:
            bit = 2 + torch.sigmoid(module.bit)*12
            loss_bit_w += bit * module.weight.numel()
            loss_bit_wu += torch.round(bit) * module.weight.numel()
            numel_w += module.weight.numel()
            
        if hasattr(module, "bit") and hasattr(module, "out_shape") and module.quant:
            bit = 2 + torch.sigmoid(module.bit)*12
            loss_bit_a += bit * np.prod(module.out_shape)
            loss_bit_au += torch.round(bit) * np.prod(module.out_shape)
            numel_a += np.prod(module.out_shape)
        
    if numel_a > 0:
        a_bit = (loss_bit_a / numel_a).item()
        au_bit = (loss_bit_au / numel_a).item()

    if numel_w > 0:
        w_bit = (loss_bit_w / numel_w).item()
        wu_bit = (loss_bit_wu / numel_w).item()
    
    return a_bit, au_bit, w_bit, wu_bit

def categorize_param(model):    
    weight = []
    bnbias = []   

    for name, param in model.named_parameters():         
        if not param.requires_grad:
            continue  
        elif len(param.shape) == 1 or name.endswith(".bias"): # bnbias and quant parameters
            bnbias.append(param)
        else:
            weight.append(param)
    return (weight, bnbias)

def get_optimizer(weight, bnbias, train_weight):    
    optimizer = optim.SGD([        
        {'params': bnbias, 'weight_decay': 0., 'lr': args.lr},
        {'params': weight, 'weight_decay': args.decay if train_weight else 0, 'lr': args.lr if train_weight else 0},
    ], momentum=0.9, nesterov=True)
    return optimizer

weight, bnbias = categorize_param(model)
optimizer = get_optimizer(weight, bnbias, True)
scheduler = CosineWithWarmup(optimizer, warmup_len=args.warmuplen, warmup_start_multiplier=0.1,
                max_epochs=args.epoch+args.ft_epoch, eta_min=1e-3, last_epoch=-1)

best_acc = 0
best_acc_ema = 0 

if args.mode == 'bops' :
    print(f'mode : {args.mode}')
    train_aux_target_avgbit = train_aux_target_bops

for epoch in range(args.epoch):
    train_acc, train_loss, _ = train_aux_target_avgbit(train_loader, model, model_ema, model_t, criterion, optimizer, epoch, 
                                                        ema_rate=0.99, bit_scale_a = args.a_scale, bit_scale_w = args.w_scale,
                                                        target_ours = args.target, scale=args.bops_scale)
    # train_regular(train_loader, model, model_ema, None, 0, 0, criterion, optimizer, epoch, [1.,], ema_rate=0.9997)
    acc_base, test_loss = test(test_loader, model, criterion, epoch, False)

    acc_ema = 0        
    if model_ema is not None:
        acc_ema, loss_ema = test(test_loader, model_ema, criterion, epoch)

    a_bit, au_bit, w_bit, wu_bit = bit_cal(model)
    bops_total = bops_cal(model)
    print(f'Epoch : [{epoch}] / a_bit : {au_bit}bit / w_bit : {wu_bit}bit / bops : {bops_total.item()}GBops')

    is_best = acc_base > best_acc
    best_acc = max(acc_base, best_acc)
    
    is_best_ema = acc_ema > best_acc_ema
    best_acc_ema = max(acc_ema, best_acc_ema)    
    
    print('==> Save the model')  
    if args.ckpt is not None:
        create_checkpoint(
            optimizer, epoch, ckpt_root,
            model, is_best, best_acc, 
            model_ema, is_best_ema, best_acc_ema, prefix=prefix)
    scheduler.step()                       


# BN tuning phase
Q.initialize(model, act=args.a_scale > 0, weight=args.w_scale > 0, noise=False)

for name, module in model.named_modules():
    if isinstance(module, (Q.ReLU, Q.Sym, Q.HSwish, Q.Conv2d, Q.Linear)):
        module.bit.requires_grad = False

best_acc = 0
best_acc_ema = 0 


for epoch in range(args.epoch, args.epoch+args.ft_epoch):
    train_acc, train_loss, _  = train_aux(train_loader, model, model_ema, model_t, criterion, optimizer, epoch, 
                                          ema_rate=0.99, bit_scale_a = 0, bit_scale_w = 0)
    acc_base, test_loss = test(test_loader, model, criterion, epoch, False)

    acc_ema = 0        
    if model_ema is not None:
        acc_ema, _ = test(test_loader, model_ema, criterion, epoch)

    is_best = acc_base > best_acc
    best_acc = max(acc_base, best_acc)
    
    is_best_ema = acc_ema > best_acc_ema
    best_acc_ema = max(acc_ema, best_acc_ema)    

    if args.ckpt is not None:
        create_checkpoint(
            optimizer, epoch, ckpt_root,
            model, is_best, best_acc, 
            model_ema, is_best_ema, best_acc_ema, prefix=prefix)
    scheduler.step()                 
