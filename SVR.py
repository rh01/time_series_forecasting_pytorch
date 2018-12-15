from sklearn.svm import SVR
import eval
from util import *
from sklearn.preprocessing import MinMaxScaler


def trainSVM(trainX, trainY):

    n = trainX.shape[0]
    print("trainx num is:", n)
    svrModel = SVR(C=0.001, epsilon=0.01, kernel="rbf")
    svrModel.fit(trainX, trainY)

    return svrModel


def predict_SVM(testX, svrModel):

    n = testX.shape[0]
    print("testx num is:", n)
    testy = svrModel.predict(testX)

    return testy


def predict_SVM_iteration(testX, lookAhead, svrModel):

    testBatchSize = testX.shape[0]
    ans = []

    for i in range(lookAhead):

        pred = svrModel.predict(testX)  # (test_num, )
        ans.append(pred)

        testX = testX[:, 1:]
        pred = pred.reshape((testBatchSize, 1))
        testX = np.append(testX, pred, axis=1)

    ans = np.array(ans)
    ans = ans.transpose([1, 0])
    return ans


def run_SVM(data, lookBack, train_lookAhead, test_lookAhead):

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)

    # 分割序列为样本
    trainData, testData = divideTrainTest(dataset)

    flag = False
    trainX, trainY = createSamples(trainData, lookBack, lookAhead=train_lookAhead, RNN=flag)
    testX, testY = createSamples(testData, lookBack, lookAhead=test_lookAhead, RNN=flag)
    print("testX shape:", testX.shape)
    print("testy shape:", testY.shape)
    print("trainX shape:", trainX.shape)
    print("trainy shape:", trainY.shape)

    model = trainSVM(trainX, trainY)

    testPred = predict_SVM_iteration(testX, test_lookAhead, model)
    print("testPred shape:", testPred.shape)

    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    return testPred, MAE, MRSE, SMAPE


if __name__ == '__main__':

    lookBack = 24
    train_lookAhead = 1
    test_lookAhead = 7

    print("looback:", lookBack)
    print("train look ahead:", train_lookAhead)
    print("test look ahead:", test_lookAhead)

    ts, data = load_data("./data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = load_data("./data/bike_hour.csv", indexName="dteday", columnName="cnt")
    # ts, data = load_data("./data/traffic_data_in_bits.csv", indexName="datetime", columnName="value")
    # ts, data = load_data("./data/TAS2016.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("./data/AEMO/TT30GEN.csv", indexName="TRADING_INTERVAL", columnName="VALUE")

    testPred, MAE, MRSE, SMAPE = run_SVM(data, lookBack, train_lookAhead=train_lookAhead, test_lookAhead=test_lookAhead)

