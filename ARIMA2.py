from util import *
import eval
import numpy as np
import pyflux as pf


def run_ARIMA(data, p, q, lookBack, train_lookAhead, test_lookAhead):

    # 分割序列为样本
    trainData, testData = divideTrainTest(data)

    flag = False
    trainX, trainY = createSamples(trainData, lookBack, lookAhead=train_lookAhead, RNN=flag)
    testX, testY = createSamples(testData, lookBack, lookAhead=test_lookAhead, RNN=flag)
    print("testX shape:", testX.shape)
    print("testy shape:", testY.shape)
    print("trainX shape:", trainX.shape)
    print("trainy shape:", trainY.shape)

    testPred = predict_ARIMA(trainData, testX, test_lookAhead, p, q)
    print("testPred shape:", testPred.shape)

    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    return testPred, MAE, MRSE, SMAPE


def predict_ARIMA(trainData, testX, lookAhead, p, q):

    total_train = np.concatenate(trainData, testX)
    model = pf.ARIMA(data=total_train, ar=p, ma=q, family=pf.Normal())
    model.fit(method="MLE")

    pred = model.predict(lookAhead, intervals=False)

    return pred


def predict_ARIMA_iteration(trainData, testX, lookAhead, p, q):

    testBatchSize = testX.shape[0]
    ans = []

    for i in range(lookAhead):

        total_train = np.concatenate(trainData, testX)
        model = pf.ARIMA(data=total_train, ar=p, ma=q, family=pf.Normal())
        model.fit(method="MLE")

        pred = model.predict(1, intervals=False)
        ans.append(pred)

        testX = testX[:, 1:]
        pred = pred.reshape((testBatchSize, 1))
        testX = np.append(testX, pred, axis=1)

    ans = np.array(ans)
    ans = ans.transpose([1, 0])
    return ans


if __name__ == '__main__':

    ts, data = load_data("./data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = load_data("../data/bike_hour.csv", indexName="dteday", columnName="cnt")
    # ts, data = load_data("../data/traffic_data_in_bits.csv",  columnName="value")
    #  ts, data = load_data("../data/TAS2016.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
    # ts, data = util.load_data("../data/AEMO/TT30GEN.csv", indexName="TRADING_INTERVAL", columnName="VALUE")

    dataset = ts.values[:]
    X = np.array(dataset,dtype="float64")
    # size = int(len(X) * 0.9)
    # train, test = X[0:size], X[size:len(X)]
    train, test = divideTrainTest(dataset)
    history = [x for x in train]
    predictions = []
    realTestY = []

    for t in range(len(test)):
        # order = st.arma_order_select_ic(history, max_ar=5, max_ma=5, ic=['aic', 'bic', 'hqic'])
        # print(order.bic_min_order)
            #model = ARIMA(history, order=(3, 2, 1))
        model = pf.ARIMA(data=np.array(history), ar=4, ma=4, family=pf.Normal())
        model.fit(method="MLE")

        output = model.predict(1, intervals=False)

        yhat = output.values[0][0]

        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        realTestY.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))

    realTestY = np.array(test)
    predictions = np.array(predictions).reshape(-1)
    MAE = eval.calcMAE(realTestY, predictions)
    RMSE = eval.calcRMSE(realTestY, predictions)
    MAPE = eval.calcSMAPE(realTestY, predictions)
    print('Test MAE: %.8f' % MAE)
    print('Test RMSE: %.8f' % RMSE)
    print('Test MAPE: %.8f' % MAPE)

    # plot
    # pyplot.plot(test)
    # pyplot.plot(predictions, color='red')
    # pyplot.show()