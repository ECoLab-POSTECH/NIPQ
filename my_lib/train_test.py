from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import copy
import glob
import time
import torch
import shutil
import tempfile
import collections
import numpy as np 
import pathlib
import math
import torch.nn.functional as F
import torch
import torch.nn as nn

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

_print_freq = 10
_temp_dir = tempfile.mkdtemp()


def set_print_freq(freq):
    global _print_freq
    _print_freq = freq


def get_tmp_dir():
    return _temp_dir


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def create_checkpoint(
        optimizer, epoch, root, 
        model, is_best, best_acc,
        model_ema=None, is_best_ema=None, best_acc_ema=None, 
        save_freq=1, prefix='train'):        
    pathlib.Path(root).mkdir(parents=True, exist_ok=True) 
    filename = os.path.join(root, '{}_{}.ckpt'.format(prefix, epoch))
    bestname = os.path.join(root, '{}_best.pth'.format(prefix))
    bestemaname = os.path.join(root, '{}_best_ema.pth'.format(prefix))
    tempname = os.path.join(_temp_dir, '{}_tmp.pth'.format(prefix))

    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    if model_ema is not None: 
        if isinstance(model_ema, torch.nn.DataParallel):
            model_ema_state = model_ema.module.state_dict()
        else:
            model_ema_state = model_ema.state_dict()
    else:
        model_ema_state = None

    if is_best:
        torch.save(model_state, bestname)

    if is_best_ema:        
        torch.save(model_ema_state, bestemaname)
        
    if (epoch+1) > 0 and ((epoch+1) % save_freq) == 0:
        state = {
            'model': model_state,
            'model_ema': model_ema_state,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'best_acc_ema': best_acc_ema
        }
        torch.save(state, filename)


def resume_checkpoint(optimizer, root, model, model_ema=None, prefix='train', target_idx=-1):
    files = glob.glob(os.path.join(root, "{}_*.ckpt".format(prefix)))

    if target_idx == -1:
        max_idx = -1
        for file in files:
            num = re.search("{}_(\d+).ckpt".format(prefix), file)
            if num is not None:
                num = num.group(1)
                max_idx = max(max_idx, int(num))
    else:
        max_idx = target_idx

    if max_idx != -1:
        checkpoint = torch.load(
            os.path.join(root, "{}_{}.ckpt".format(prefix, max_idx)))
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])

        if model_ema is not None:
            if isinstance(model_ema, torch.nn.DataParallel):
                model_ema.module.load_state_dict(checkpoint["model_ema"])
            else:
                model_ema.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        best_acc_ema = checkpoint["best_acc_ema"]
        return (epoch, best_acc, best_acc_ema)
    else:
        print("==> Can't find checkpoint...training from initial stage")
        return (-1, 0, 0)


def test(val_loader, model, criterion, epoch, train=False, verbose=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.train(train)

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():        
            if not isinstance(model, torch.nn.DataParallel):
                input = input.cuda()
            # target = target.cuda()
            target = target.cuda(non_blocking=True)        
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            # input_var = input_var.cpu()
            # target=target.cpu()
            output = model(input_var)      

        if isinstance(output, tuple):
            loss = criterion(output[0], target_var)
            prec1, prec5 = accuracy(output[0].data, target, topk=(1, 5))
        else:
            loss = criterion(output, target_var)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        # record loss and accuracy
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i+1) % _print_freq) == 0 and verbose:
            print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i+1, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    if verbose:
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg, losses.avg

class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):    
        return input.round()
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
def bit_loss(model, epoch, bit_scale_a, bit_scale_w, target_bit, is_linear):
    loss_bit_a = torch.Tensor([0]).cuda()
    loss_bit_w = torch.Tensor([0]).cuda()


    target_bit_a = torch.Tensor([0]).cuda()
    target_bit_w = torch.Tensor([0]).cuda()
    target_bit = target_bit

    for name, module in model.named_modules():
        if hasattr(module, "bit") and hasattr(module, "weight") and module.quant:

            bit = 2 + torch.sigmoid(module.bit)*12
            bit = bit.round() + (bit - bit.detach())
            size = bit * module.weight.numel()
            loss_bit_w += size
            
            target_size = target_bit*module.weight.numel()
            target_bit_w += target_size
            
        if hasattr(module, "bit") and hasattr(module, "out_shape") and module.quant:
            bit = 2 + torch.sigmoid(module.bit)*12
            bit = bit.round() + (bit - bit.detach())
            size = bit * np.prod(module.out_shape)
            loss_bit_a += size

            target_size = target_bit*np.prod(module.out_shape)
            target_bit_a += target_size
    
    loss_a_bit = F.smooth_l1_loss(loss_bit_a, target_bit_a)
    loss_w_bit = F.smooth_l1_loss(loss_bit_w, target_bit_w)

    if is_linear :
        bit_loss =   (bit_scale_a * (epoch+1)/30) * loss_a_bit / 2 ** 23 \
                   + (bit_scale_w * (epoch+1)/30) * loss_w_bit / 2 ** 23  
    else :  
        bit_loss = bit_scale_a * loss_a_bit / 2 ** 23 + bit_scale_w * loss_w_bit / 2 ** 23
    return bit_loss

def compute_bops(
    kernel_size, in_channels, filter_per_channel, h, w, bits_w=32, bits_a=32
):
    conv_per_position_flops = (
        kernel_size * kernel_size * in_channels * filter_per_channel
    )
    active_elements_count = h * w
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bops = overall_conv_flops * bits_w * bits_a
    return bops


def bops_loss(model, epoch, target_bops):
    loss_bops_total = torch.Tensor([0]).cuda()

    bops = torch.Tensor(1).cuda()
    bops.data.fill_(target_bops)

    for name, module in model.named_modules():
        if hasattr(module, "bit") and hasattr(module, "weight") and module.quant:
            bits_weight = 2 + torch.sigmoid(module.bit)*12
            bits_weight = bits_weight.round() + (bits_weight - bits_weight.detach())

            if module.act_func is not None :
                bits_actv = 2 + torch.sigmoid(module.act_func.bit)*12
                bits_actv = bits_actv.round() + (bits_actv - bits_actv.detach())
            else :
                bits_act = 32
            
            if isinstance(module, nn.Conv2d):
                _, _, h, w = module.out_shape
                bop = compute_bops(
                    module.kernel_size[0],
                    module.in_channels,
                    module.out_channels // module.groups, h, w,
                    bits_weight,
                    bits_act,
                    )
            else :
                bop = compute_bops(
                    1,
                    module.in_features,
                    module.out_features, 1, 1,
                    bits_weight,
                    bits_act,
                    )

            loss_bops_total += bop
    
    loss = F.smooth_l1_loss( loss_bops_total/(10**9), bops )
    return loss

def bops_cal(model):
    bops_total = torch.Tensor([0]).cuda()

    for name, module in model.named_modules():
        if hasattr(module, "bit") and hasattr(module, "weight") and module.quant:

            bits_weight = (2 + torch.sigmoid(module.bit)*12).round()
            
            if module.act_func is not None :
                bits_act = (2 + torch.sigmoid(module.act_func.bit)*12).round()
            else :
                bits_act = 32
            
            if isinstance(module, nn.Conv2d):
                _, _, h, w = module.out_shape
                bop = compute_bops(
                    module.kernel_size[0],
                    module.in_channels,
                    module.out_channels // module.groups, h, w,
                    bits_weight,
                    bits_act,
                    )
            else :
                bop = compute_bops(
                    1,
                    module.in_features,
                    module.out_features, 1, 1,
                    bits_weight,
                    bits_act,
                    )

            bops_total += bop
               
    return bops_total

def train_aux_target_bops(train_loader, model, model_ema, model_t, criterion, optimizer, epoch, 
                          ema_rate=0.9999, bit_scale_a = 0, bit_scale_w = 0, target_ours=0, scale=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    if model_t is not None:
        model_t.eval()
        model_t.cuda()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if not isinstance(model, torch.nn.DataParallel):
            input = input.cuda()

        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target)

            if model_t is not None:
                output_t = model_t(input_var)      
        output = model(input_var)

        loss = None        
        loss_class = criterion(output, target_var)
        
        loss_bit = bops_loss(model, epoch, target_ours)
        loss_class = loss_class + scale * loss_bit 
        
        if model_t is not None:
            loss_kd = -1 * torch.mean(
                torch.sum(torch.nn.functional.softmax(output_t, dim=1) 
                        * torch.nn.functional.log_softmax(output, dim=1), dim=1))
            loss = loss_class + loss_kd 
        else:
            loss = loss_class
        losses.update(loss_class.data.item(), input.size(0))

        # measure accuracy and record loss
        if isinstance(output, tuple):
            prec1, prec5 = accuracy(output[0].data, target, topk=(1, 5))
        else:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if model_ema is not None:
            for module, module_ema in zip(model.module.modules(), model_ema.module.modules()):
                target = []

                if hasattr(module, "c"):    # QIL
                    target.append("c")
                    target.append("d")
                
                if hasattr(module, "p"):    # PACT
                    target.append("p")

                if hasattr(module, "s"):    # lsq
                    target.append("s")

                if hasattr(module, "e"):    # proposed
                    target.append("e")    
                    target.append("f")  


                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    target.extend(["weight", "bias"])
                    
                    if hasattr(module, "scale"):
                        target.extend(["scale", "shift"])

                if isinstance(module, (torch.nn.BatchNorm2d)):
                    target.extend(["weight", "bias", "running_mean", "running_var"])

                    if module.num_batches_tracked is not None:
                        module_ema.num_batches_tracked.data = module.num_batches_tracked.data

                for t in target:
                    base = getattr(module, t, None)    
                    ema = getattr(module_ema, t, None)    

                    if base is not None and hasattr(base, "data"):                        
                        ema.data += (1 - ema_rate) * (base.data - ema.data)   

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bops = bops_cal(model) / 10**9

        if ((i+1) % _print_freq) == 0:
            numel_a = 0
            numel_w = 0
            loss_bit_a = 0
            loss_bit_au = 0
            loss_bit_w = 0
            loss_bit_wu = 0
            
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
            else:
                a_bit = -1
                au_bit = -1
            
            if numel_w > 0:
                w_bit = (loss_bit_w / numel_w).item()
                wu_bit = (loss_bit_wu / numel_w).item()
            else:
                w_bit = -1
                wu_bit = -1
                
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                'a bit {a_bit:.2f}[{au_bit:.2f}]\t'
                'w bit {w_bit:.2f}[{wu_bit:.2f}]\t'
                'bops : {bops:.3f} \t'.format(
                    epoch, i+1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, 
                    a_bit = a_bit, au_bit = au_bit, 
                    w_bit = w_bit, wu_bit = wu_bit, 
                    bops = bops.item()))
    return top1.avg, losses.avg, None

def train_aux_target_avgbit(train_loader, model, model_ema, model_t, criterion, optimizer, epoch, 
                     ema_rate=0.9999, bit_scale_a = 0, bit_scale_w = 0, target_ours=None, scale=0):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    if model_t is not None:
        model_t.eval()
        model_t.cuda()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if not isinstance(model, torch.nn.DataParallel):
            input = input.cuda()

        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target)

            if model_t is not None:
                output_t = model_t(input_var)      
        output = model(input_var)

        loss = None        
        loss_class = criterion(output, target_var)
        
        loss_bit = bit_loss(model, epoch, bit_scale_a, bit_scale_w, target_ours, False)
        loss_class = loss_class + loss_bit
        
        if model_t is not None:
            loss_kd = -1 * torch.mean(
                torch.sum(torch.nn.functional.softmax(output_t, dim=1) 
                        * torch.nn.functional.log_softmax(output, dim=1), dim=1))
            loss = loss_class + loss_kd 
        else:
            loss = loss_class
        losses.update(loss_class.data.item(), input.size(0))

        # measure accuracy and record loss
        if isinstance(output, tuple):
            prec1, prec5 = accuracy(output[0].data, target, topk=(1, 5))
        else:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if model_ema is not None:
            for module, module_ema in zip(model.module.modules(), model_ema.module.modules()):
                target = []

                if hasattr(module, "c"):    # QIL
                    target.append("c")
                    target.append("d")
                
                if hasattr(module, "p"):    # PACT
                    target.append("p")

                if hasattr(module, "s"):    # lsq
                    target.append("s")

                if hasattr(module, "e"):    # proposed
                    target.append("e")    
                    target.append("f")  


                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    target.extend(["weight", "bias"])
                    
                    if hasattr(module, "scale"):
                        target.extend(["scale", "shift"])

                if isinstance(module, (torch.nn.BatchNorm2d)):
                    target.extend(["weight", "bias", "running_mean", "running_var"])

                    if module.num_batches_tracked is not None:
                        module_ema.num_batches_tracked.data = module.num_batches_tracked.data

                for t in target:
                    base = getattr(module, t, None)    
                    ema = getattr(module_ema, t, None)    

                    if base is not None and hasattr(base, "data"):                        
                        ema.data += (1 - ema_rate) * (base.data - ema.data)   

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i+1) % _print_freq) == 0:
            numel_a = 0
            numel_w = 0
            loss_bit_a = 0
            loss_bit_au = 0
            loss_bit_w = 0
            loss_bit_wu = 0
            
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
            else:
                a_bit = -1
                au_bit = -1
            
            if numel_w > 0:
                w_bit = (loss_bit_w / numel_w).item()
                wu_bit = (loss_bit_wu / numel_w).item()
            else:
                w_bit = -1
                wu_bit = -1
                
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                'a bit {a_bit:.2f}[{au_bit:.2f}]\t'
                'w bit {w_bit:.2f}[{wu_bit:.2f}]\t'.format(
                    epoch, i+1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, 
                    a_bit = a_bit, au_bit = au_bit, 
                    w_bit = w_bit, wu_bit = wu_bit))          
    return top1.avg, losses.avg, None

def train_aux(train_loader, model, model_ema, model_t, criterion, optimizer, epoch, ema_rate=0.9999, bit_scale_a = 0, bit_scale_w = 0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    if model_t is not None:
        # model_t.train()
        model_t.eval()
        model_t.cuda()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if not isinstance(model, torch.nn.DataParallel):
            input = input.cuda()

        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target)

            if model_t is not None:
                output_t = model_t(input_var)      
        output = model(input_var)

        loss = None        
        loss_class = criterion(output, target_var)
        loss_bit_a = 0
        loss_bit_w = 0


        for name, module in model.named_modules():
            if hasattr(module, "bit") and hasattr(module, "weight") and module.quant:
                bit = 2 + torch.sigmoid(module.bit)*12
                size = bit * module.weight.numel()
                loss_bit_w += size
                
            if hasattr(module, "bit") and hasattr(module, "out_shape") and module.quant:
                bit = 2 + torch.sigmoid(module.bit)*12
                size = bit * np.prod(module.out_shape)
                loss_bit_a += size
                
        loss_class = loss_class + bit_scale_a * loss_bit_a / 2 ** 23 + bit_scale_w * loss_bit_w / 2 ** 23
        
        if model_t is not None:
            loss_kd = -1 * torch.mean(
                torch.sum(torch.nn.functional.softmax(output_t, dim=1) 
                        * torch.nn.functional.log_softmax(output, dim=1), dim=1))
            loss = loss_class + loss_kd 
        else:
            loss = loss_class
        losses.update(loss_class.data.item(), input.size(0))

        # measure accuracy and record loss
        if isinstance(output, tuple):
            prec1, prec5 = accuracy(output[0].data, target, topk=(1, 5))
        else:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if model_ema is not None:
            for module, module_ema in zip(model.module.modules(), model_ema.module.modules()):
                target = []

                if hasattr(module, "c"):    # QIL
                    target.append("c")
                    target.append("d")
                
                if hasattr(module, "p"):    # PACT
                    target.append("p")

                if hasattr(module, "s"):    # lsq
                    target.append("s")

                if hasattr(module, "e"):    # proposed
                    target.append("e")    
                    target.append("f")  


                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    target.extend(["weight", "bias"])
                    
                    if hasattr(module, "scale"):
                        target.extend(["scale", "shift"])

                if isinstance(module, (torch.nn.BatchNorm2d)):
                    target.extend(["weight", "bias", "running_mean", "running_var"])

                    if module.num_batches_tracked is not None:
                        module_ema.num_batches_tracked.data = module.num_batches_tracked.data

                for t in target:
                    base = getattr(module, t, None)    
                    ema = getattr(module_ema, t, None)    

                    if base is not None and hasattr(base, "data"):                        
                        ema.data += (1 - ema_rate) * (base.data - ema.data)   

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i+1) % _print_freq) == 0:
            numel_a = 0
            numel_w = 0
            loss_bit_a = 0
            loss_bit_au = 0
            loss_bit_w = 0
            loss_bit_wu = 0
            
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
            else:
                a_bit = -1
                au_bit = -1
            
            if numel_w > 0:
                w_bit = (loss_bit_w / numel_w).item()
                wu_bit = (loss_bit_wu / numel_w).item()
            else:
                w_bit = -1
                wu_bit = -1
                
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                'a bit {a_bit:.2f}[{au_bit:.2f}]\t'
                'w bit {w_bit:.2f}[{wu_bit:.2f}]\t'.format(
                    epoch, i+1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, 
                    a_bit = a_bit, au_bit = au_bit, 
                    w_bit = w_bit, wu_bit = wu_bit))
            
            
    return top1.avg, losses.avg, None

def train_regular(train_loader, model, model_ema, model_t, criterion, optimizer, epoch, aux_rate=[1.,], ema_rate=0.9999):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_aux = AverageMeter()
    losses_reg = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    if model_t is not None:
        model_t.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if not isinstance(model, torch.nn.DataParallel):
            input = input.cuda()

        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            if model_t is not None:
                output_t = model_t(input_var)
      
        output = model(input_var)
        
        loss = 0

        if isinstance(output, tuple):
            for aux, out in zip(aux_rate, output):
                if loss is None:
                    loss_out = criterion(out, target_var) 
                    losses.update(loss_out.data.item(), input.size(0))
                    loss = aux * loss_out
                else:
                    loss_out = criterion(out, target_var) 
                    losses_aux.update(loss_out.data.item(), input.size(0))
                    loss += aux * loss_out
        else:
            loss += criterion(output, target_var)
            losses.update(loss.data.item(), input.size(0))

        # measure accuracy and record loss
        if isinstance(output, tuple):
            prec1, prec5 = accuracy(output[0].data, target, topk=(1, 5))
        else:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if model_ema is not None:
            for module, module_ema in zip(model.module.modules(), model_ema.module.modules()):
                target = []

                if hasattr(module, "c"):    # QIL
                    target.append("c")
                    target.append("d")
                
                if hasattr(module, "p"):    # PACT
                    target.append("p")

                if hasattr(module, "s"):    # lsq
                    target.append("s")

                if hasattr(module, "e"):    # proposed
                    target.append("e")    
                    target.append("f")  


                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    target.extend(["weight", "bias"])
                    
                    if hasattr(module, "scale"):
                        target.extend(["scale", "shift"])
                        
                    if hasattr(module, "weight_center"):
                        target.extend(["weight_center"])

                if isinstance(module, (torch.nn.BatchNorm2d)):
                    target.extend(["weight", "bias", "running_mean", "running_var"])

                    if module.num_batches_tracked is not None:
                        module_ema.num_batches_tracked.data = module.num_batches_tracked.data

                for t in target:
                    base = getattr(module, t, None)    
                    ema = getattr(module_ema, t, None)    

                    if base is not None and hasattr(base, "data"):                        
                        ema.data += (1 - ema_rate) * (base.data - ema.data)   

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i+1) % _print_freq) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Loss reg {loss_reg.val:.4f} ({loss_reg.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i+1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, loss_reg=losses_reg, top1=top1, top5=top5))
    return losses.avg, losses_aux.avg


class LabelSmoothingCrossEntropy(torch.nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """ Implements a schedule where the first few epochs are linear warmup, and
    then there's cosine annealing after that."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_len: int,
                 warmup_start_multiplier: float, max_epochs: int, 
                 eta_min: float = 0.0, last_epoch: int = -1):
        if warmup_len < 0:
            raise ValueError("Warmup can't be less than 0.")
        self.warmup_len = warmup_len
        if not (0.0 <= warmup_start_multiplier <= 1.0):
            raise ValueError(
                "Warmup start multiplier must be within [0.0, 1.0].")
        self.warmup_start_multiplier = warmup_start_multiplier
        if max_epochs < 1 or max_epochs < warmup_len:
            raise ValueError("Max epochs must be longer than warm-up.")
        self.max_epochs = max_epochs
        self.cosine_len = self.max_epochs - self.warmup_len
        self.eta_min = eta_min  # Final LR multiplier of cosine annealing
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.max_epochs:
            raise ValueError(
                "Epoch may not be greater than max_epochs={}.".format(
                    self.max_epochs))
        if self.last_epoch < self.warmup_len or self.cosine_len == 0:
            # We're in warm-up, increase LR linearly. End multiplier is implicit 1.0.
            slope = (1.0 - self.warmup_start_multiplier) / self.warmup_len
            lr_multiplier = self.warmup_start_multiplier + slope * self.last_epoch
        else:
            # We're in the cosine annealing part. Note that the implementation
            # is different from the paper in that there's no additive part and
            # the "low" LR is not limited by eta_min. Instead, eta_min is
            # treated as a multiplier as well. The paper implementation is
            # designed for SGDR.
            cosine_epoch = self.last_epoch - self.warmup_len
            lr_multiplier = self.eta_min + (1.0 - self.eta_min) * (
                1 + math.cos(math.pi * cosine_epoch / self.cosine_len)) / 2
        assert lr_multiplier >= 0.0
        return [base_lr * lr_multiplier for base_lr in self.base_lrs]
