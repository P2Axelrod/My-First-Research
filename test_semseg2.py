"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from dataset import ScannetDatasetWholeScene
import torch
import hydra
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
from models.Hengshuang.model2 import PointTransformerSeg
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='Hengshuang',
                        help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=500, type=int,
                        help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001,
                        type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str,
                        default='Hengshuang', help='Log path [default: None]')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight_decay [default: 1e-4]')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='Point Number [default: 1024]')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7,
                        help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5,
                        help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3,
                        help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--nneighbor', type=int, default=16
                        )
    parser.add_argument('--nblocks', type=int, default=4
                        )
    parser.add_argument('--transformer_dim', type=int, default=512
                        )
    parser.add_argument('--name', type=str, default='Hengshuang'
                        )
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    # print(vote_label_pool.shape)
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]),
                                int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    args.input_dim = 9
    NUM_CLASSES = 13
    args.num_class = NUM_CLASSES
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    root = hydra.utils.to_absolute_path(
        './data/s3dis/trainval_fullarea/')

    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(
        root, split='test', test_area=args.test_area, block_points=NUM_POINT)
    log_string("The number of test data is: %d" %
               len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    classifier = PointTransformerSeg(args).cuda()
    checkpoint = torch.load(
        "/home/dell/Axelrod/Point-Transformers-master/log/semseg/Hengshuang/best_model.pth")
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)
        print("num_batches={}".format(num_batches))

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." %
                  (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros(
                (whole_scene_label.shape[0], NUM_CLASSES))
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[
                    batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size,
                                      ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_data[:, :, 3:6] /= 1.0

                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    seg_pred = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[
                        1].numpy()
                    # print(batch_smpw[0:real_batch_size, ...])
                    # batch_smpw 表示点的权重
                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])
            print(vote_label_pool)
            pred_label = np.argmax(vote_label_pool, 1)

            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                total_correct_class_tmp[l] += np.sum(
                    (pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum(
                    ((pred_label == l) | (whole_scene_label == l)))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / \
                (np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            print('----------------------------')

        IoU = np.array(total_correct_class) / \
            (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
            np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
