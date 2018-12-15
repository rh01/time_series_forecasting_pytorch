from model import *
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import shuffle
import pickle as p
import time
import numpy as np


def train(trainX, trainY, epoch, lr, batchSize, modelPath, lookBack, lookAhead, method, checkPoint=10):

    lossFilePath = "./model/loss_ResRNN-4.pkl"
    output = open(lossFilePath, 'wb')
    lossList = []

    hidden_num = 128

    n = trainX.shape[0]
    print("trainx num is:", n)
    batchNum = n // batchSize - 1
    print("batch num is:", batchNum)

    if method == "RNN":
        net = RNNModel(inputDim=1, hiddenNum=hidden_num, outputDim=lookAhead, layerNum=1, cell="RNN")
    if method == "RNN_ALL":
        net = Multi_Hidden_RNN_Model(inputDim=1, hiddenNum=hidden_num, outputDim=lookAhead, layerNum=1, cell="RNN", seq_len=lookBack, merge="concate")
    if method == "LSTM":
        net = LSTMModel(inputDim=1, hiddenNum=hidden_num, outputDim=lookAhead, layerNum=1, cell="LSTM")
    if method == "GRU":
        net = GRUModel(inputDim=1, hiddenNum=hidden_num, outputDim=lookAhead, layerNum=1, cell="GRU")
    if method == "GRU_ALL":
        net = Multi_Hidden_GRU_Model(inputDim=1, hiddenNum=hidden_num, outputDim=lookAhead, layerNum=1, cell="GRU", seq_len=lookBack, merge="mean")
    if method == "ANN":
        net = ANNModel(inputDim=lookBack, hiddenNum=hidden_num, outputDim=1)
    if method == "ResRNN":
        #net = ResidualRNNModel(inputDim=1, hiddenNum=100, outputDim=1, layerNum=1, cell="RNNCell")
        net = ResRNNModel(inputDim=1, hiddenNum=hidden_num, outputDim=lookAhead, resDepth=-1)

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=0.9)

    t1 = time.time()
    for epoch_idx in range(epoch):

        trainX, trainY = shuffle(trainX, trainY, random_state=epoch)

        batchStart = 0
        lossSum = 0

        for batch_idx in range(batchNum):

            x = trainX[batchStart:batchStart+batchSize, :, :]
            y = trainY[batchStart:batchStart+batchSize]

            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            x, y = Variable(x), Variable(y)

            optimizer.zero_grad()

            pred = net.forward(x, batchSize=batchSize)
            criterion = nn.MSELoss()
            loss = criterion(pred, y)

            lossSum += loss.data.numpy()[0]

            if (batch_idx + 1) % checkPoint == 0 or (batch_idx + 1) == batchNum:
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.
                      format(epoch_idx, batch_idx + 1,  lossSum / (batch_idx + 1)))

            loss.backward()
            optimizer.step()

            batchStart += batchSize

    t2 = time.time()
    print("train time:", t2-t1)

    p.dump(lossList, output, -1)

    torch.save(net, modelPath)


def predict(testX, modelFileName):

    net = torch.load(modelFileName)
    testBatchSize = testX.shape[0]
    testX = torch.from_numpy(testX)
    testX = Variable(testX)
    pred = net(testX, testBatchSize)

    return pred.data.numpy()


def predict_iteration(testX, lookAhead, modelFileName):

    net = torch.load(modelFileName)
    testBatchSize = testX.shape[0]
    ans = []
    for i in range(lookAhead):

        testX_torch = torch.from_numpy(testX)
        testX_torch = Variable(testX_torch)
        pred = net(testX_torch, testBatchSize)
        pred = pred.data.numpy()
        pred = np.squeeze(pred)
        ans.append(pred)

        testX = testX[:, 1:]
        pred = pred.reshape((testBatchSize, 1, 1))
        testX = np.append(testX, pred, axis=1)

    ans = np.array(ans)
    ans = ans.transpose([1, 0])
    return ans


