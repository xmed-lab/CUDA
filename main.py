import os
import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
from models.domain_adversarial_network import DomainDiscriminator
from alignemnt.dann import DomainAdversarialLoss, ImageClassifier

from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger

from analysis import collect_feature, tsne, a_distance, pca, collect_concept_feature
from torchvision import transforms
from data.Waterbirds.cub import load_cub_data
from utils import CAccuracy, C_f1_score
import numpy as np
from typing import TextIO
import wandb


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)
    
    wandb.init(
        project=args.project, 
        name=args.log,
        config=args
    )

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    # print("train_transform: ", train_transform)
    # print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
 
    if args.data == 'waterbirds' or args.data == 'waterbirds-cub':
        normalizer = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        train_source_loader = load_cub_data([train_source_dataset], use_attr=True, no_img=False, 
                batch_size=args.batch_size, uncertain_label=False, image_dir=args.root, resol=299, normalizer=normalizer,
                n_classes=args.num_classes, resampling=False)
        train_target_loader = load_cub_data([train_target_dataset], use_attr=True, no_img=False, 
                batch_size=args.batch_size, uncertain_label=False, image_dir=args.root, resol=299, normalizer=normalizer,
                n_classes=args.num_classes, resampling=False)
        val_loader = load_cub_data([test_dataset], use_attr=True, no_img=False, 
                    batch_size=args.batch_size, uncertain_label=False, image_dir=args.root, resol=299, normalizer=normalizer,
                    n_classes=args.num_classes, resampling=False)
        test_loader = load_cub_data([test_dataset], use_attr=True, no_img=False, 
                    batch_size=args.batch_size, uncertain_label=False, image_dir=args.root, resol=299, normalizer=normalizer,
                    n_classes=args.num_classes, resampling=False)
    else:
        train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.workers, drop_last=True)
        train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.workers, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch, input_channels=args.input_channels)
    # indentify the pool layer
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, args.num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch, concept_emb_dim=args.concept_emb_dim).to(device)
    domain_discri = DomainDiscriminator(in_feature=args.bottleneck_dim * args.concept_emb_dim, n_domains=args.n_domains, hidden_size=512).to(device)
    wandb.watch(classifier, log="all", log_freq=args.print_freq)
    wandb.config.update({"model_parameters": count_parameters(classifier)})
    wandb.log({"Classifier Parameters": count_parameters(classifier)})
    wandb.log({"Domain Discriminator Parameters": count_parameters(domain_discri)})


    # define optimizer and lr scheduler
    optimizer = Adam(classifier.get_parameters(),
                  lr=args.lr, 
                  betas=(0.9, 0.999),  
                  weight_decay=args.weight_decay)
    
    domain_optimizer = Adam(domain_discri.get_parameters(),
                  lr=args.lr, 
                  betas=(0.9, 0.999),  
                  weight_decay=args.weight_decay)
    
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    domain_lr_scheduler = LambdaLR(domain_optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))


    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        # tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        pca_filename = osp.join(logger.visualize_directory, 'PCA.png')
        # tsne.visualize(source_feature, target_feature, tSNE_filename)
        pca.visualize(source_feature, target_feature, pca_filename)
        # print("Saving t-SNE to", tSNE_filename)
        # print("Saving PCA to", pca_filename)
        # # calculate A-distance, which is a measure for distribution discrepancy
        # A_distance = a_distance.calculate(source_feature, target_feature, device)
        # print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1, c_acc, c_f1 = utils.validate(test_loader, classifier, args, device)
        print(acc1, c_acc, c_f1)
        return
    
    if args.phase == 'concept-analysis':
        concept_extractor = classifier
        source_feature, source_concept = collect_concept_feature(train_source_loader, concept_extractor, device)
        target_feature, target_concept = collect_concept_feature(train_target_loader, concept_extractor, device)
        print("source_feature:", source_feature.shape)
        print("target_feature:", target_feature.shape)
        # save the concept features
        source_feature_path = osp.join(logger.visualize_directory, 'source_feature.npy')
        target_feature_path = osp.join(logger.visualize_directory, 'target_feature.npy')
        source_concept_path = osp.join(logger.visualize_directory, 'source_concept.npy')
        target_concept_path = osp.join(logger.visualize_directory, 'target_concept.npy')
        np.save(source_feature_path, source_feature)
        np.save(target_feature_path, target_feature)
        np.save(source_concept_path, source_concept)
        np.save(target_concept_path, target_concept)
        return
        

    # start training
    best_acc1 = 0.
    # write loss into logger folder
    lossfile_path = osp.join(logger.root, 'realtime_transfer_loss_relaxed.csv')
    lossfile = open(lossfile_path, 'w')
    lossfile.write('epoch,iteration,loss\n')  
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr()[0])
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, domain_adv, optimizer,
              lr_scheduler, domain_optimizer, domain_lr_scheduler, epoch, args, lossfile)

        # evaluate on validation set
        acc1, c_acc, c_f1 = utils.validate(val_loader, classifier, args, device)
        print("epoch: {:d}, val_acc1 = {:3.1f}".format(epoch, acc1, c_acc, c_f1))
        wandb.log({
            "val_accuracy": acc1,
            "val_concept_accuracy": c_acc,
            "val_concept_f1_score": c_f1,
            "epoch": epoch
        })

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
        
    lossfile.close()
    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1, c_acc, c_f1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1, c_acc, c_f1))
    wandb.log({
        "test_accuracy": acc1,
        "test_concept_accuracy": c_acc,
        "test_concept_f1_score": c_f1
    })


    logger.close()
    wandb.finish()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, domain_adv: DomainAdversarialLoss, optimizer: Adam,
          lr_scheduler: LambdaLR, domain_optimizer: Adam,
          domain_lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace, lossfile: TextIO):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    concept_accs = AverageMeter('Concept Acc', ':3.1f')
    concept_f1s = AverageMeter('Concept F1', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, domain_accs, concept_accs, concept_f1s],
        prefix="Epoch: [{}]".format(epoch))
    
    criterion = torch.nn.BCELoss()


    # switch to train mode
    model.train()
    domain_adv.train()
    model_size = count_parameters(model)
    print("model_size:", model_size)
    domain_adv_size = count_parameters(domain_adv)
    print("domain_adv_size:", domain_adv_size)

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s, c_s = next(train_source_iter)[:3]
        x_t, = next(train_target_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        c_s = c_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        # using image classifier to get the output and features
        y, f, c_pred = model(x)
        
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)
        c_pred_s, c_pred_t = c_pred.chunk(2, dim=0)
    
        
        concept_loss = criterion(c_pred_s, c_s.float())
        
        cls_loss = F.cross_entropy(y_s, labels_s)
        
        # 1. Train domain discriminator independently
        domain_optimizer.zero_grad()
        transfer_loss = domain_adv(f_s.detach(), f_t.detach())  # Detach to train discriminator only
        transfer_loss.backward()
        domain_optimizer.step()
        
        # 2. Joint training of classifier and domain discriminator with relaxed loss
        optimizer.zero_grad()

        if args.tau > 0:
            transfer_loss_relaxed = torch.clamp(domain_adv(f_s, f_t), max=args.tau)
        else:
            transfer_loss_relaxed = domain_adv(f_s, f_t)
        
        loss = cls_loss + args.lambda_c * concept_loss - args.lambda_t * transfer_loss_relaxed
        loss.backward()
        optimizer.step()


        # Update metrics
        concept_acc = CAccuracy(c_pred_s, c_s)
        concept_f1 = C_f1_score(c_pred_s, c_s)
        cls_acc = accuracy(y_s, labels_s)[0]
        domain_acc = domain_adv.domain_discriminator_accuracy

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))
        concept_accs.update(concept_acc, x_s.size(0))
        concept_f1s.update(concept_f1, x_s.size(0))
        
        # Write relaxed transfer loss to file
        lossfile.write(f'{epoch},{i},{transfer_loss_relaxed.item()}\n')

        # Log all losses and metrics to wandb
        wandb.log({
            "epoch": epoch,
            "iteration": i,
            "overall_loss": loss.item(),
            "concept_loss": concept_loss.item(),
            "classification_loss": cls_loss.item(),
            "transfer_loss": transfer_loss.item(),
            "transfer_loss_relaxed": transfer_loss_relaxed.item(),
            "concept_accuracy": concept_acc,
            "concept_f1_score": concept_f1,
            "classification_accuracy": cls_acc.item(),
            "domain_accuracy": domain_acc.item(),
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        lr_scheduler.step()
        domain_lr_scheduler.step()
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('-root', metavar='DIR', default="./data/Waterbirds/waterbirds-dann-2",
                        help='root path of dataset')
    parser.add_argument('--project', default='DACBM-wb', type=str, help='project name')
    parser.add_argument('--device', default=0, type=int, 
                        help='device for training')
    parser.add_argument('-d', '--data', metavar='DATA', default='waterbirds',
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: waterbirds, mnist)')
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis', 'concept-analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels of the network')
    parser.add_argument('--lambda_c', default=5, type=float,
                        help='the trade-off hyper-parameter for concept loss')
    parser.add_argument('--lambda_t', default=0.3, type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--n_domains', default=1, type=int,
                        help='number of domains')
    parser.add_argument('--tau', default=0.7, type=float,
                        help='the threshold hyper-parameter for transfer loss')
    parser.add_argument('--num_classes', default=200, type=int,
                        help='number of classes')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=299,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
    parser.add_argument('--bottleneck_dim', default=112, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--concept_emb_dim', default=64, type=int,)
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=0.01, type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    args = parser.parse_args()
    
    args.log = f"{args.data}-seed{args.seed}-lambda_c{args.lambda_c}-lambda_t{args.lambda_t}-tau{args.tau}-class{args.num_classes}-b{args.batch_size}"
    print("Log name:", args.log)
    
    # print(args)
    if args.device == -1:
        device = torch.device("cpu")  
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    main(args)