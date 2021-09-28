from argparse import ArgumentParser
import numpy as np
import csv
import glob
import os
import sys
import dataset_wave as dataset
from model.model import Pointnet2Backbone
from sklearn.preprocessing import normalize
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import glob

# disable tensorflow debug information:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

def main(in_files, density, kNN, out_folder, thinFactor, save_txt, ground, others, arch):
    spacing = np.sqrt(kNN*thinFactor/(np.pi*density)) * np.sqrt(2)/2 * 0.95  # 5% MARGIN
    print("Using a spacing of %.2f m" % spacing)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    NUM_POINTS=kNN
    NUM_CLASSES = args.NUM_CLASSES
    point = np.ones((1, NUM_POINTS, 163))

    recall = np.zeros((6))
    prec = np.zeros((6))
    f1 = np.zeros((6))
    modelname = ""



    '''MODEL LOADING'''
    if(arch == 1):
        model = Pointnet2Backbone(input_feature_dim=160-3).cuda()
    elif(arch == 2):
        model = model2(input_feature_dim=160-3).cuda()
    elif(arch == 3):
        model = model3(input_feature_dim=160-3).cuda()

    elif(arch == 4):
        model = model4(input_feature_dim=160-3).cuda()
    elif(arch == 5):
        model = model5(input_feature_dim=160-3).cuda()   
    files = glob.glob(args.model + "/*.pth")
    print(args.model + "/*.pth")
    for i in range(len(files)):
        model_path =files[i]
        print(model_path)

        if model_path is not None:
            
            try:
                checkpoint = torch.load(model_path)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['model_state_dict'])
                print('loaded model')
            except:
                print('No existing model, starting training from scratch...')
                continue

        model = model.eval()

    

        for file_pattern in in_files:
            for file in glob.glob(file_pattern):
                print("Loading file %s" % file)
                d = dataset.kNNBatchDataset(file=file, k=int(kNN*thinFactor), spacing=spacing)
                pred = np.zeros((len(d), NUM_CLASSES))
                count = np.zeros(len(d))

                out_name = d.filename.replace('.la', '_test.la')  # laz or las
                out_path = os.path.join(out_folder, out_name)
                gt_name = d.filename.replace('.la', '_gt.la')  # laz or las
                gt_path = os.path.join(out_folder, gt_name)
                while True:
                    print("Processing batch %d/%d" % (d.currIdx, d.num_batches))
                    points_and_features, _, idx = d.getBatchsWithIdx(batch_size=1)
                    idx_to_use = np.sort(np.random.choice(range(int(thinFactor*kNN)), kNN))
                    # print(idx_to_use)
                    # print(idx)

                    # print(points_and_features[0][idx_to_use].shape)
                    if points_and_features is not None:
                        # print(points_and_features[0][idx_to_use].shape)
                        point[0, :, :] = points_and_features[0][:][:,:]
                        points = torch.from_numpy(point).cuda()
                        points = points.float()
                        # print(points.shape)
                        if(arch == 1):
                            pred_batch = model(points)
                        elif(arch==2):
                            pred_batch,_,_ = model(points)
                        elif(arch==3 or arch==4 or arch==5):
                            pred_batch , _ , _ , _ , _ = model(points)
                            # print(pred_batch.size())
                            # print(wave_.size())
                        # pred_batch = pred_batch.view(-1, args.NUM_CLASSES)
                        pred_batch = pred_batch.cpu().data.numpy()

                        # print(pred_batch.shape)
                        # print (pred[idx[:, idx_to_use], :].shape)
                        pred[idx[:, :], :] += pred_batch
                        count[idx[0, :]] += 1
                        check = np.argmax(pred_batch, axis=2)
                        # print(check)
                        # print(np.sum(check, axis=1))

                    else:  # no more data
                        break

                new_classes = np.argmax(pred, axis=1) + 1
                

                # new_classes = np.where(new_classes >= 1, others , new_classes)
                # new_classes = np.where(new_classes == 0, ground, new_classes)
                new_index = count.nonzero()[0]
                d.labels[new_index] = d.labels[new_index] + 1
                dataset.Dataset.Save(out_path, d.points_and_features[new_index], d.names,
                                    labels=new_classes[new_index])
                dataset.Dataset.Save(gt_path, d.points_and_features[new_index], d.names,
                                    labels=d.labels [new_index])
                print("Save to %s" % out_path)
                if(len(new_classes[new_index]) == len(new_classes)):
                    print("same dim")
                else:
                    print("dim is changed")

                # evel
                gt = d.labels[new_index] 
                temp_prec = (precision_score(gt, new_classes[new_index], average=None))
                temp_recall = (recall_score(gt, new_classes[new_index], average=None))
                temp_f1 = (f1_score(gt, new_classes[new_index], average=None))
                print(temp_prec)
                print(temp_recall)
                print(temp_f1)

                if(np.mean(temp_prec) >  np.mean(prec) and np.mean(temp_recall) >  np.mean(recall) and np.mean(temp_f1) >  np.mean(f1)):
                     prec = temp_prec
                     recall = temp_recall
                     f1 = temp_f1
                     modelname = model_path

    print(modelname)
    print(prec)
    print(recall)
    print(f1)
    




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--inFiles',
                        default=[],
                        required=True,
                        help='input files (wildcard supported)',
                        action='append')
    parser.add_argument('--density', default=15, type=float, help='average point density')
    parser.add_argument('--kNN', default=200000, type=int, help='how many points per batch [default: 200000]')
    parser.add_argument('--outFolder', required=True, help='where to write output files and statistics to')
    parser.add_argument('--model', required=True, help='tensorflow model ckpt file')
    parser.add_argument('--NUM_CLASSES', default=3, type=int,help='python architecture file')
    parser.add_argument('--thinFactor', default=1., type=float,
                        help='factor to thin out points by (2=use half of the points)')
    parser.add_argument('--normalize', default=1, type=int,
                        help='normalize fields and coordinates [default: 1][1/0]')
    parser.add_argument('--Ground', default=2, type=int,
                        help='ground class flag')
    parser.add_argument('--Others', default=6, type=int,
                        help='not ground class flag')
    parser.add_argument('--saveTxt', action="store_true",
                        help='save txt format file')
    parser.add_argument('--arch', default=1, type=int,
                        help='choese model')
    args = parser.parse_args()

    main(args.inFiles, args.density, args.kNN, args.outFolder, args.thinFactor, args.saveTxt, args.Ground, args.Others, args.arch)
