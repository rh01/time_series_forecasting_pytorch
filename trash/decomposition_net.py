from model import RNNModel, GRUModel, ANNModel, AttentionRNNModel, DecompositionNetModel
import torch.optim as optim
from sklearn.utils import shuffle
from code.util import createSamples, divideTrainTest, align
from sklearn.preprocessing import MinMaxScaler
from code.season_decompose import seasonDecompose


def decomopsitionNet(data, lookBack, batchSize):


    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)

    # 分割序列为样本,并整理成RNN的输入形式
    trainData, testData = divideTrainTest(dataset)

    trainX, trainY = createSamples(trainData, lookBack, RNN=False)
    testX, testY = createSamples(testData, lookBack, RNN=False)
    print("testX shape:", testX.shape)
    print("testy shape:", testY.shape)
    print("trainX shape:", trainX.shape)
    print("trainy shape:", trainY.shape)

    net1 = DecompositionNetModel(inputDim=24, hiddenNum=100, outputDim=24)
    net2 = RNNModel(inputDim=1, hiddenNum=100, outputDim=1, layerNum=1, cell="RNN")

    optimizer1 = optim.RMSprop(net1.parameters(), lr=1e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=1e-3)

    prime = net1.forward()
