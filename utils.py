"""
Ref: https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/utils.py
"""
import sys
import os.path as osp
import time
from PIL import Image

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as D
from torchvision import transforms

from timm.data.auto_augment import auto_augment_transform, rand_augment_transform

sys.path.append('../../..')
import tllib.vision.datasets as datasets
import models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.vision.datasets.imagelist import MultipleDomainsDataset

import os
import numpy as np
import sklearn.metrics
from sklearn.metrics import accuracy_score, f1_score
from data.MNIST.digits import MNISTWithConcepts, MNIST_M, label2concept
from data.Skincon.fitz import SkinConDataset
from torch.utils.data import Subset
import random
import pandas as pd



def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True, input_channels=3): 
    if model_name in models.__dict__:
        # load models from models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
            
    if hasattr(backbone, 'conv1'):
        original_conv1 = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            input_channels, 
            original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=original_conv1.bias is not None
        )

        if input_channels == 1:
            backbone.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        elif input_channels == 3:
            backbone.conv1.weight.data = original_conv1.weight.data
    
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + ['Digits']


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name == "Digits":
        train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), download=True,
                                                            transform=train_source_transform)
        train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                            transform=train_target_transform)
        val_dataset = test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                                  download=True, transform=val_transform)
        class_names = datasets.MNIST.get_classes()
        num_classes = len(class_names)
        
    ##########################################
    # Waterbirds dataset
    ##########################################
    
    elif dataset_name == "waterbirds" or dataset_name == "waterbirds-cub" or dataset_name == "cub":

        class_names = ['waterbird', 'landbird'] # not use
        num_classes = len(class_names) # not use
        train_source_dataset = os.path.join(root, "train_source.pkl")
        train_target_dataset = os.path.join(root, "train_target.pkl")
        val_dataset = test_dataset = os.path.join(root, "test.pkl")
    
    ##########################################
    # MNIST dataset
    ##########################################
        
    elif dataset_name == "m2mm":
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        num_classes = len(class_names)
        transform_m = transforms.Compose([transforms.Resize([32, 32]),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])
        transform_mm = transforms.Compose([transforms.Resize([32, 32]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])
        img_dir = "./data/MNIST"
        img_dir_m = "./data/MNIST/mnist_m"
        train_source_dataset = D.MNIST(img_dir, train=True, download=True, transform=transform_m)
        train_source_dataset = MNISTWithConcepts(train_source_dataset, label2concept)
        train_target_dataset = MNIST_M(img_dir_m, train=True, transform=transform_mm)
        train_source_dataset = MNISTWithConcepts(train_target_dataset, label2concept)
        test_dataset = MNIST_M(img_dir_m, train=False, transform=transform_mm)
        val_dataset = test_dataset = MNISTWithConcepts(test_dataset, label2concept)
        
    elif dataset_name == "s2m":
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        num_classes = len(class_names)
        transform_s = transforms.Compose([
                                transforms.Resize([32, 32]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])
        transform_m = transforms.Compose([transforms.Resize([32, 32]),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])
        img_dir = "./data/MNIST"
        train_source_dataset = D.SVHN(img_dir, split='train', download=True, transform=transform_s)
        train_source_dataset = MNISTWithConcepts(train_source_dataset, label2concept)
        train_target_dataset = D.MNIST(img_dir, train=True, download=True, transform=transform_m)
        train_target_dataset = MNISTWithConcepts(train_target_dataset, label2concept)
        test_dataset = D.MNIST(img_dir, train=False, download=True, transform=transform_m)
        val_dataset = test_dataset = MNISTWithConcepts(test_dataset, label2concept)
        
    elif dataset_name == "m2u":
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        num_classes = len(class_names)
        transform = transforms.Compose([transforms.Resize([32, 32]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])
        img_dir = "./data/MNIST"
        train_source_dataset = D.MNIST(img_dir, train=True, download=True, transform=transform)
        train_source_dataset = MNISTWithConcepts(train_source_dataset, label2concept)
        train_target_dataset = D.USPS(img_dir, train=True, download=True, transform=transform)
        train_target_dataset = MNISTWithConcepts(train_target_dataset, label2concept)
        test_dataset = D.USPS(img_dir, train=False, download=True, transform=transform)
        val_dataset = test_dataset = MNISTWithConcepts(test_dataset, label2concept)
        
        
    ##########################################
    # Skin Lesion dataset
    ##########################################
    elif dataset_name == 'skincon-wl' or  dataset_name == "skincon-lw"  or dataset_name == 'skincon-ld':
        img_dir = "./data/Skincon/fitz/finalfitz17k"
        datasets_dir = "./data/Skincon/datasets"
        class_names = ['benign', 'malignant', 'non-neoplastic']
        num_classes = len(class_names)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
        ])

        test_transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
        ])
        if dataset_name == 'skincon-wl' :
            source_file = "white_dataset.csv"
            target_file = "light_dataset.csv"
        elif dataset_name == 'skincon-lw':
            source_file = "light_dataset.csv"
            target_file = "white_dataset.csv"
        elif dataset_name == 'skincon-ld':
            source_file = "light_dataset.csv"
            target_file = "dark_dataset.csv"
        source_csv = os.path.join(datasets_dir, source_file)
        target_csv = os.path.join(datasets_dir, target_file)
        train_source_dataset = SkinConDataset(csv_path=source_csv, root_dir=img_dir, transform=train_transform)
        train_target_dataset = SkinConDataset(csv_path=target_csv, root_dir=img_dir, transform=test_transform)
        val_dataset = test_dataset = SkinConDataset(csv_path=target_csv, root_dir=img_dir, transform=test_transform)
        
        
    ##########################################
    elif dataset_name in datasets.__dict__:
        # load datasets from tllib.vision.datasets
        dataset = datasets.__dict__[dataset_name]

        def concat_dataset(tasks, start_idx, **kwargs):
            # return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])
            return MultipleDomainsDataset([dataset(task=task, **kwargs) for task in tasks], tasks,
                                          domain_ids=list(range(start_idx, start_idx + len(tasks))))

        train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform,
                                              start_idx=0)
        train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform,
                                              start_idx=len(source))
        val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform,
                                     start_idx=len(source))
        if dataset_name == 'DomainNet':
            test_dataset = concat_dataset(root=root, tasks=target, split='test', download=True, transform=val_transform,
                                          start_idx=len(source))
        else:
            test_dataset = val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)
    else:
        raise NotImplementedError(dataset_name)
    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    c_acc = AverageMeter('C Acc', ':6.2f')
    c_f1 = AverageMeter('C F1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, c_acc, c_f1],
        prefix='Test: ')
    
    criterion = torch.nn.BCELoss()

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target, concepts = data[:3]
            images = images.to(device)
            target = target.to(device)
            concepts = concepts.to(device)

            # compute output
            output, c_pred = model(images)
            concept_loss = criterion(c_pred, concepts.float())
            loss = F.cross_entropy(output, target) + concept_loss
            
            concept_acc = CAccuracy(c_pred, concepts)
            concept_f1 = C_f1_score(c_pred, concepts)
            
            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            c_acc.update(concept_acc, images.size(0))
            c_f1.update(concept_f1, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        print(' * C Acc {c_acc.avg:.3f}'.format(c_acc=c_acc))
        print(' * C F1 {c_f1.avg:.3f}'.format(c_f1=c_f1))
        if confmat:
            print(confmat.format(args.class_names))

    return top1.avg, c_acc.avg, c_f1.avg

def save_batch_with_global_idx(c_pred, concepts, start_idx):
    
    c_pred = (c_pred >= 0.5).int()
    c_pred = c_pred.cpu().numpy()
    concepts = concepts.cpu().numpy()

    batch_results = []
    for sample_idx in range(c_pred.shape[0]):
        batch_results.append({
            "idx": start_idx + sample_idx,  
            "c_pred": c_pred[sample_idx].tolist(), 
            "concepts": concepts[sample_idx].tolist()  
        })

    return batch_results


def save_results_to_csv(results, save_dir):
    
    df = pd.DataFrame(results)

    save_path = os.path.join(save_dir, "c_pred_and_concepts.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved all predictions and concepts to {save_path}")


def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    transformed_img_size = 224
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224, scale=scale, ratio=ratio)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    elif random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])


def empirical_risk_minimization(train_source_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    c_accs = AverageMeter('C Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            
            
def CAccuracy(pred, target):
    pred = (pred >= 0.5).int()
    target = (target > 0.5).int()
    # print("Predict concepts:", pred)
    
    # Calculate accuracy per concept
    correct = (pred == target).float()
    concept_acc = correct.mean(dim=0).mean().item() * 100
    return concept_acc


def C_f1_score(pred, target):
    pred = (pred >= 0.5).int()
    target = (target > 0.5).int()

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    c_f1 = 0.0
    f1_per_concept = []

    for i in range(target_np.shape[-1]): 
        true_vars = target_np[:, i]
        pred_vars = pred_np[:, i]

        f1 = sklearn.metrics.f1_score(true_vars, pred_vars,  average='macro')
        f1_per_concept.append(f1)

    c_f1 = np.mean(f1_per_concept) * 100  

    # print(f"Macro F1 for concepts: {c_f1}")
    return c_f1


def validateo(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

    return top1.avg


