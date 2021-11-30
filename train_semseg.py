'''
Descripttion: 
version: 
Author: Axelrod
Date: 2021-11-28 23:06:45
LastEditors: Axelrod
LastEditTime: 2021-11-29 20:25:04
'''

import omegaconf
import hydra
import argparse
import os
from dataset import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


@hydra.main(config_path='config', config_name='semseg')
def main(args):

    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    # root = 'data/Stanford3dDataset_v1.2_Aligned_Version/'
    root = hydra.utils.to_absolute_path(
        '/home/dell/Axelrod/Point-Transformers-master/data/s3dis/trainval_fullarea')
    # root = hydra.utils.to_absolute_path(
    #     '/home/dell/Axelrod/Point-Transformers-master/data/s3dis/test')
    args.input_dim = 9  # (xyzrgbx'y'z')
    args.num_class = 13
    NUM_CLASSES = 13
    NUM_POINT = args.num_point
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT,
                                 test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    print("start loading test data ...")
    TEST_DATASET = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT,
                                test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    logger.info("The number of training data is: %d" % len(TRAIN_DATASET))
    logger.info("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    shutil.copy(hydra.utils.to_absolute_path(
        'models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(
        args.model.name)), 'PointTransformerSeg')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        logger.info('**** Epoch %d (%d/%s) ****' %
                    (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay **
                 (epoch // args.step_size)), LEARNING_RATE_CLIP)
        logger.info('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * \
            (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(
            lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            # print("points.size() = {}".format(points.size()))

            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            # points = points.transpose(2, 1)
            seg_pred = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        logger.info('Training mean loss: %f' % (loss_sum / num_batches))
        logger.info('Training accuracy: %f' %
                    (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = 'best_model.pth'
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            logger.info('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            logger.info('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                # points = points.transpose(2, 1)

                seg_pred = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum(
                        (pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(
                        ((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(
                np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) /
                           (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            logger.info('eval mean loss: %f' % (loss_sum / float(num_batches)))
            logger.info('eval point avg class IoU: %f' % (mIoU))
            logger.info('eval point accuracy: %f' %
                        (total_correct / float(total_seen)))
            logger.info('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' *
                    (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            logger.info(iou_per_class_str)
            logger.info('Eval mean loss: %f' % (loss_sum / num_batches))
            logger.info('Eval accuracy: %f' %
                        (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = 'best_model.pth'
                logger.info('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                logger.info('Saving model....')
            logger.info('Best mIoU: %f' % best_iou)
        global_epoch += 1


if __name__ == '__main__':
    main()
