import torch.nn as nn
import torch


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def free_params(module: nn.Module):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = False


def clear_grads(things):
    for thing in things:
        if thing is None:
            continue
        thing.zero_grad()


def weights_init_dcgan(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            # m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)


def weights_init_generator(net):
    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)


def weights_init_adversary(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
