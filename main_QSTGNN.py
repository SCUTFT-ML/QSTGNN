"code by Chaoxiong Lin"
import csv
import itertools
import torch
import numpy as np
import scipy.sparse as sp
import argparse
import time
from net import qstnet
from util import *
from trainer import Trainer
import copy
import numpy as np
from layer import *
import  earlystopping

# 构造四元数时间序列方法
def expand(trainx, hm):
    expanded_x = torch.zeros(trainx.size(0),4*trainx.size(1),trainx.size(2),trainx.size(3)).to(args.device) 
    if args.in_dim == 4:
        expanded_x[:, 0:1, :, :] = trainx
    else:
        expanded_x[:, 0:2, :, :] = trainx.squeeze(1)
    for i in range(1, 4):
        h = hm * i
        time_slice = trainx[:, :, :, h:].to(args.device)   # 获取对应时间维度的切片
        zero_slice = torch.zeros(trainx.size(0), trainx.size(1), trainx.size(2), h).to(args.device)   # 创建全零张量
        if args.in_dim == 4:
            # 将切片和全零张量进行合并
            expanded_x[:, i:i+1, :, :] = torch.cat((time_slice, zero_slice), dim=3).to(args.device)
        else:
            expanded_x[:, 2*i:2*(i+1), :, :] = torch.cat((time_slice, zero_slice), dim=3).squeeze(1).to(args.device)
    return expanded_x

def main():
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)
    #load data
    device = torch.device(args.device)
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    # 加载两个距离邻接矩阵和语义邻接矩阵
    predefined_A = load_adj(args.adj_data)#这个是加载pkl文件的代码，在这里，有几个数据集都是pkl文件的，自己构建的07,08数据集才不是pkl
    predefined_A = torch.tensor(predefined_A)-torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device).to(torch.float32)
    predefined_A = normalize(predefined_A, 'sym')

    # predefined_A = np.load(args.adj_data)
    # predefined_A = torch.tensor(predefined_A)-torch.eye(args.num_nodes)
    # predefined_A = predefined_A.to(device).to(torch.float32)

    predefined_dtw = np.load(args.adj_dtw_data)
    predefined_dtw = torch.tensor(predefined_dtw)# -torch.eye(args.num_nodes) #两个交通数据集要注释掉这里
    predefined_dtw = predefined_dtw.to(device).to(torch.float32)
    predefined_dtw = normalize(predefined_dtw, 'sym')

    model = qstnet(args.gcn_depth, args.num_nodes, args.kernel_size,
                  device, predefined_A=predefined_A,predefined_dtw = predefined_dtw,
                  dropout=args.dropout, subgraph_size=args.subgraph,
                  node_dim=args.node_dim,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)

    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.lr, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler, device)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    minl = 1e5
    for i in range(1,args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            # 将原数据拓展为四元数
            trainx = expand(trainx, args.hm)

            tx = trainx[:, :, :, :]
            ty = trainy[:, :, :, :]
            metrics = engine.train(tx, ty[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

            if iter % 500 == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} GPU occupy: {:.6f} MiB'.format(i, gpu_mem_alloc))
        
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            testx = expand(testx, args.hm)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)

        if mvalid_loss<minl:
            torch.save(engine.model.state_dict(), args.save + "experiment" +".pth")
            minl = mvalid_loss
        if es.step(mvalid_loss):
            print('Early stopping.')
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save + "experiment" +".pth"))
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))

    #valid data
    outputs = []
    realy = torch.Tensor(dataloader['y_val']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        testx = expand(testx,args.hm)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]
    pred = scaler.inverse_transform(yhat)
    vmae, vmape, vrmse = metric(pred,realy)

    #test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testx = expand(testx,args.hm)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    mae = []
    mape = []
    rmse = []
    for i in range(args.seq_out_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
    
    return vmae, vmape, vrmse, mae, mape, rmse


if __name__ == '__main__':
    # 定义参数网格
    param_grid = {
        'gcn_depth': [2],
        'dropout': [0.3],
        'conv_channels': [64],
        'residual_channels': [64],
        'skip_channels': [64],
        'end_channels':[128],
        'lr': [0.0008],
        'weight_decay': [0.0001],
        'propalpha': [0.05],
        'tanhalpha': [3],
        'batch_size':[32],
        'layers': [2],
        'subgraph': [10],
        'hm':[1,2,3,4],
        'kernel_size':[3,5,7]
    }

    # 将默认参数放在这里,注意在加载数据时，注释了两行。'in_dim':4或8。要看具体数据集。
    default_args = {
        'epochs':500,
        'device':'cuda:0',
        'data':'data/PEMSD7(M)',
        'adj_data':'data/sensor_graph/PEMSD7(M)/adj_mx.pkl',
        'adj_dtw_data':'data/sensor_graph/PEMSD7(M)/adj_dtw.npy',
        'node_dim':40,
        'in_dim':8,
        'seq_in_len':12,
        'seq_out_len':12,
        'num_nodes': 325,
        'seed':1000,
        'patience':30,
        'clip':5,
        'step_size1':2500,
        'save':'./save/',
    }

    results = []
    # 遍历参数网格
    for combination in itertools.product(*param_grid.values()):
        args_dict = copy.deepcopy(default_args)
        for key, value in zip(param_grid.keys(), combination):
            args_dict[key] = value
        args = argparse.Namespace(**args_dict)
        print(f"Running experiment with args: {args}")

        vmae, vmape, vrmse, mae, mape, rmse = main()

        print(mae[2], mape[2], rmse[2], mae[5], mape[5], rmse[5], mae[11], mape[11], rmse[11])
        
    
    print("All finished")
