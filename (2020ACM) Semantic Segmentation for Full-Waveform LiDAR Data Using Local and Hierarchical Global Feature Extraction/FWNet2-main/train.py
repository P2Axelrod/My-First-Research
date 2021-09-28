import os, sys
import numpy as np
import argparse
from model.model import Pointnet2Backbone
from dataset_wave import Dataset
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, os, sys, signal
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import glob

#Disable TF debug messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
resolution=1
def f_op(pcloud_np, resolution = 1):
    # Current (inefficient) code to quantize into 5x5 XY 'bins' and take mean Z values in each bin
    pcloud_np[:, 0] =( pcloud_np[:, 0]  - np.min(pcloud_np[:, 0]) )* 1000000
    pcloud_np[:, 1] = (pcloud_np[:, 1] - np.min(pcloud_np[:, 1]) ) * 1000000
    print(np.max(pcloud_np[:, 0]))

    print(np.min(pcloud_np[:, 0]))
    
    print(np.max(pcloud_np[:, 1]))
    print(np.min(pcloud_np[:, 1]))    
    pcloud_np[:, 0:2] = np.round(pcloud_np[:, 0:2]/float(resolution))*float(resolution) # Round XY values to nearest 5

    num_x = int(np.max(pcloud_np[:, 0])/resolution) + 1
    num_y = int(np.max(pcloud_np[:, 1])/resolution) + 1

    mean_height = []#np.zeros((num_x * num_y), 3)
    mean_label = []#np.zeros((num_x * num_y), 1)
    

    # Loop over each x-y bin and calculate mean z value 
    x_val = 0
    for x in range(num_x):
        y_val = 0
        for y in range(num_y):
            indx = np.where((pcloud_np[:,0] == float(x_val)) & (pcloud_np[:,1] == float(y_val)))
            height_vals = pcloud_np[indx, 2]
            label_vals = pcloud_np[indx, 3]
            if height_vals.size != 0:
                indx_h = np.argmin(height_vals, 1)
#                print(height_vals[0])
#                print(height_vals.shape)
                mean_height.append([x, y, height_vals[:,indx_h]])
                mean_label.append(label_vals[:,indx_h])
            y_val += resolution
        x_val += resolution

    return np.array(mean_height), np.array(mean_label)


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss


def main(args):
    inlist = args.inList
    threshold = args.threshold
    train_size = args.trainSize
    lr = args.learningRate
    normalize_vals = args.normalize == 1

    with open(inlist, "rb") as f:
       _ = f.readline()  # remove header
       rest = f.readlines()
    
    #list dir
    # rest = glob.glob(inlist+ "/*.las")

    datasets = []
    all_ds = []
    for line in rest:
        line = line.decode('utf-8')
        linespl = line.split(",")
        dataset_path = os.path.join(os.path.dirname(inlist), linespl[0])
        print(linespl)
        # if( 'training_4' in linespl[0]  or 'training_5' in linespl[0]  or 'training_6' in linespl[0] or 'training_11' in linespl[0]  or 'training_12' in linespl[0]):
        # if float(linespl[4]) > 0 and float(linespl[5]) > 0 and float(linespl[6]) > 0 and float(linespl[7]) > 0:
        datasets.append(dataset_path)
        all_ds.append(dataset_path)
    print(len(datasets))
    np.random.shuffle(datasets)
    datasets_th = []
    for idx, dataset in enumerate(datasets):
        print("Loading dataset %s of %s (%s)" % (idx+1, len(datasets), os.path.basename(dataset)))
        #print(dataset)
        ds = Dataset(dataset, load=False, normalize=normalize_vals, istrain = True)
        datasets_th.append(ds)
    print("%s datasets loaded." % len(datasets_th))
    sys.stdout.flush()

    '''MODEL LOADING'''
    if(args.archFile == "1"):
        model = model1(input_feature_dim=160-3, NUM_CLASSES = args.classes).cuda()
    if(args.archFile == "2"):
        model = model2(input_feature_dim=160-3, NUM_CLASSES = args.classes).cuda()
    if(args.archFile == "3"):
        model = model3(input_feature_dim=160-3, NUM_CLASSES = args.classes).cuda()
    if(args.archFile == "4"):
        model = model4(input_feature_dim=160-3, NUM_CLASSES = args.classes).cuda()
    if(args.archFile == "5"):
        model = model5(input_feature_dim=160-3, NUM_CLASSES = args.classes).cuda()
    if(args.archFile == "6"):
        model = model6(input_feature_dim=160-3, NUM_CLASSES = args.classes).cuda()

    weights = torch.ones(args.classes).cuda()
    if(args.classes==6):
        weights[0] = 2
        weights[1] = 2
        weights[2] = 2
        weights[3] = 2
        weights[4] = 2
        weights[5] = 2

        # weights[0] = 1
        # weights[1] = 1
        # weights[2] = 1
        # weights[3] = 20
        # weights[4] = 20
        # weights[5] = 1
    # if(args.classes==3):
    #     weights[0] = 2
    #     weights[1] = 1
    #     weights[2] = 1

    criterion = get_loss().cuda()

    if args.continueModel is not None:
        try:
            checkpoint = torch.load(args.continueModel)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Use pretrain model')
        except:
            print('No existing model, starting training from scratch...')
            start_epoch = 0



    '''SET OPTIMIZER'''
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learningRate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
        scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') 
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learningRate, momentum=0.9)

    print(datasets_th[0].load_data())

    global_epoch = 0
    best_iou = 0

    start = time.time()
    NUM_CLASSES = args.classes
    NUM_POINT = args.points
    BATCH_SIZE = args.batchsize

    # labels = torch.ones(1, 50000, 1).long().cuda()
    # points = torch.rand(1, 50000, 6).cuda()
    label = np.ones((BATCH_SIZE, NUM_POINT, 1))
    point = np.ones((BATCH_SIZE, NUM_POINT, 163))
#    input_arr = np.ones((NUM_POINT, 4))

    for j in range(args.multiTrain):
        minibatch =0
        total_loss = 0.0 
        for i in range(len(datasets_th)):

            file_in = datasets_th[i]
            # print(file_in)
            if isinstance(file_in, Dataset):
                ds = file_in
            else:
                ds = Dataset(file_in)
            label[0,:,0] = ds.labels
            point[0,:,:] = ds.points_and_features[:,:]
#            point[0,:,:2] = normalize( point[0,:,:2])
#            input_arr[:,:3] = point[0,:,:3]
#            input_arr[:,3] = label[0,:,0]
#            input_xyz, input_label = f_op(input_arr)
#            plt.imshow(input_xyz)
#            plt.show()
#            
            # print(ds.labels.shape)
            # print(point.shape)
            # set pytorch tensor
            labels = torch.from_numpy(label).cuda()
            points = torch.from_numpy(point).cuda()
            labels = labels.long()
            points = points.float()
            # labels = torch.ones(16, 10000, 1).long().cuda()
            # points = torch.rand(16, 10000, 6).cuda()
            # set variable tensor
            labels = torch.autograd.Variable(labels)
            points = torch.autograd.Variable(points)

            batch_label = labels.view(-1, 1)[:, 0].cpu().data.numpy()
            labels = labels.view(-1, 1)[:, 0]
            # optimizer
            optimizer.zero_grad()
            model = model.train()
            if(args.archFile == "1"):
                seg_pred = model(points)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                # seg_pred = seg_pred.view(-1, NUM_CLASSES)
                loss = criterion(seg_pred, labels, weights)
                total_loss += loss
                loss.backward()
            elif(args.archFile == "2" or args.archFile == "6"):
                seg_both, seg_pred_local, seg_pred_global= model(points)
                seg_pred_local = seg_pred_local.view(-1, NUM_CLASSES)
                seg_pred_global = seg_pred_global.view(-1, NUM_CLASSES)
                seg_pred = seg_both.contiguous().view(-1, NUM_CLASSES)
                # seg_pred = seg_both.view(-1, NUM_CLASSES)
                loss = criterion(seg_pred, labels, weights) + criterion(seg_pred_local, labels, weights) + criterion(seg_pred_global, labels, weights)
                total_loss += loss
                loss.backward()
            elif(args.archFile == "3" or args.archFile == "4" or args.archFile == "5"):
                seg_both, seg_pred_local, seg_pred_global, seg_point, seg_pointwave= model(points)
                seg_pred_local = seg_pred_local.view(-1, NUM_CLASSES)
                seg_pred_global = seg_pred_global.view(-1, NUM_CLASSES)
                seg_pred = seg_both.contiguous().view(-1, NUM_CLASSES)
                seg_point = seg_point.view(-1, NUM_CLASSES)
                seg_pointwave = seg_pointwave.contiguous().view(-1, NUM_CLASSES)

                # seg_pred = seg_both.view(-1, NUM_CLASSES)
                loss = criterion(seg_pred, labels, weights) + criterion(seg_pred_local, labels, weights) + criterion(seg_pred_global, labels, weights) + criterion(seg_point, labels, weights) + criterion(seg_pointwave, labels, weights)
                total_loss += loss
                loss.backward()
            # seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            # print(seg_pred.size())
            # print(labels.size())
            
            
            optimizer.step()
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)/len(pred_choice)
            # total_correct += correct
            # total_seen += (BATCH_SIZE * NUM_POINT)
            # loss_sum += loss
            print(precision_score(batch_label, pred_choice, average=None))
            print(recall_score(batch_label, pred_choice, average=None))
            print(f1_score(batch_label, pred_choice, average=None))

            print("Training datasets loss is %s %s to %s (%s total) at %s epoch" % (loss.item(), i*train_size,
                                                             min((i+1)*train_size, len(datasets_th)),
                                                             len(datasets_th), j))
        total_loss = total_loss/len(datasets_th)
        scheduler.step(total_loss)


            # print("SAVE MODEL")

        elapsed_time = time.time() - start

        savepath = args.outDir + '/model_' + str(j) + '.pth'
        print('Saving at %f' % elapsed_time)
        state = {
            'epoch': j,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
        #mysaver = tf.train.Saver()
        #mysaver.save(inst._session, os.path.join(args.outDir, 'models', 'model_%d' % (j), 'alsNet.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inList', help='input text file, must be csv with filename;stddev;...')
    parser.add_argument('--threshold', type=float, help='upper threshold for class stddev')
    parser.add_argument('--minBuild', type=float, help='lower threshold for buildings class [0-1]')
    parser.add_argument('--outDir', required=True, help='directory to write html log to')
    # parser.add_argument('--multiclass', default=True, type=bool, help='label into multiple classes ' +
    #                                                                  '(not only ground/nonground) [default: True]')
    parser.add_argument('--multiTrain', default=200, type=int,
                       help='how often to feed the whole training dataset [default: 1]')
    parser.add_argument('--trainSize', default=1, type=int,
                       help='how many plots to train at once [default: 1]')
    parser.add_argument('--learningRate', default=0.0005, type=float,
                       help='learning rate [default: 0.001]')
    parser.add_argument('--archFile', default="1", type=str,
                       help='architecture file to import [default: default architecture]')
    parser.add_argument('--continueModel', default=None, type=str,
                        help='continue training an existing model [default: start new model]')
    parser.add_argument('--lossFn', default='fp_high', type=str,
                        help='loss function to use [default: fp_high][simple/fp_high]')
    parser.add_argument('--normalize', default=1, type=int,
                        help='normalize fields and coordinates [default: 1][1/0]')
    parser.add_argument('--points', default = 100000, type=int,
                        help='normalize fields and coordinates [default: 1][1/0]')
    parser.add_argument('--classes', default=6, type=int,
                        help='normalize fields and coordinates [default: 1][1/0]')
    parser.add_argument('--batchsize', default=2, type=int,
                        help='normalize fields and coordinates [default: 1][1/0]')
    # parser.add_argument('--testList', help='list with files to test on')
    parser.add_argument('--gpuID', default=0, help='which GPU to run on (default: CPU only)')
    parser.add_argument('--optimizer', default="Adam", help='which Optimizer (default: Adam)')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    args = parser.parse_args()
    main(args)
