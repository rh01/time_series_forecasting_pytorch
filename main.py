from train_NN import train, predict_iteration, predict
from util import load_data, createSamples, divideTrainTest
from sklearn.preprocessing import MinMaxScaler
import eval


def run(data, lookBack, train_lookAhead, test_lookAhead, epoch, lr, batchSize, method, modelPath):

    # 归一化数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)

    # 分割训练测试样本
    trainData, testData = divideTrainTest(dataset)

    # 分割序列为样本, 支持RNN或者普通样本形式
    flag = True
    trainX, trainY = createSamples(trainData, lookBack=lookBack, lookAhead=train_lookAhead, RNN=flag)
    testX, testY = createSamples(testData, lookBack=lookBack, lookAhead=test_lookAhead, RNN=flag)
    print("testX shape:", testX.shape)
    print("testy shape:", testY.shape)
    print("trainX shape:", trainX.shape)
    print("trainy shape:", trainY.shape)

    train(trainX, trainY,  epoch=epoch, lr=lr, batchSize=batchSize, modelPath=modelPath,
          lookBack=lookBack, lookAhead=train_lookAhead, method=method)

    # testPred = predict(testX, MODEL_PATH)
    testPred = predict_iteration(testX,  test_lookAhead, MODEL_PATH)
    print("testPred shape:", testPred.shape)
    # trainPred = predict(trainX, MODEL_PATH)
    # print("trainPred shape:", trainPred.shape)

    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    return testPred, MAE, MRSE, SMAPE


if __name__ == "__main__":

    lookBack = 24
    train_lookAhead = 1
    test_lookAhead = 1
    batchSize = 64
    epoch = 20
    MODEL_PATH = "./model/RNN_model.pkl"
    METHOD = "GRU"
    lr = 1e-4

    print("look back:", lookBack)
    print("look ahead:", test_lookAhead)
    print("batchSize", batchSize)
    print("epoch:", epoch)
    print("METHOD:", METHOD)
    print("MODEL_PATH:", MODEL_PATH)
    print("lr:", lr)

    ts, data = load_data("./data/NSW2013.csv",  columnName="TOTALDEMAND")
    # ts, data = load_data("./data/bike_hour.csv", indexName="dteday", columnName="cnt")
    # ts, data = load_data("./data/traffic_data_in_bits.csv", indexName="datetime", columnName="value")
    # ts, data = load_data("./data/TAS2016.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("./data/AEMO/TT30GEN.csv", indexName="TRADING_INTERVAL", columnName="VALUE")

    testPred, MAE, MRSE, SMAPE = run(data=data, lookBack=lookBack, train_lookAhead=train_lookAhead, test_lookAhead=test_lookAhead,
                                epoch=epoch,  lr=lr, batchSize=batchSize, method=METHOD, modelPath=MODEL_PATH)

    # trainPred, testPred, MAE, MRSE, SMAPE = FCD_Train(ts=ts, dataset=data, freq=freq, lookBack=lookBack,
    #                                                   batchSize=batchSize,
    #                                                   epoch=epoch, lr=lr, method=METHOD)

    # test(data, lookBack, epoch, 1e-4, batchSize,  method=METHOD, modelPath=MODEL_PATH)