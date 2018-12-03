#encoding=utf-8
import pandas as pd
import numpy as np
import statsmodels.stats.diagnostic
import statsmodels.api as sm


# 加载csv数据，返回时间序列series和array形式的数据,以行形式返回
def load_data(filename, columnName):

    '''
    :param filename: csv file name
    :param columnName: target ts name
    :return: ts, pandas ts; data, np array with shape (sea_len, )
    '''

    df = pd.read_csv(filename)
    # df.index = pd.to_datetime(df.index)
    ts = df[columnName]
    data = ts.values.reshape(-1).astype("float32")
    return ts, data


# 分割时间序列作为样本
def createSamples(ts, lookBack, lookAhead=1, RNN=True):

    '''
    :param ts: input ts np array
    :param lookBack: history window size
    :param lookAhead: forecasting window size
    :param RNN: if use RNN input format
    :return: trainx with shape (sample_num, look_back, 1) or and trainy with shape (sample_num, look_ahead)
    '''

    dataX, dataY = [], []
    for i in range(len(ts) - lookBack - lookAhead):
        history_seq = ts[i: i + lookBack]
        future_seq = ts[i + lookBack: i + lookBack + lookAhead]
        dataX.append(history_seq)
        dataY.append(future_seq)
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    if RNN:
        dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))
        # dataY = np.reshape(dataY, (dataY.shape[0], dataY.shape[1], 1))
    return dataX, dataY


# 划分训练集和测试集
def divideTrainTest(dataset, rate=0.75):

    '''
    :param dataset: ts np array
    :param rate: train and test ratio
    :return: train and test np array with shape (sea_len, )
    '''

    train_size = int(len(dataset) * rate)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:]
    return train, test


if __name__ == "__main__":

    # load data test
    # ts, data = load_data("./data/AEMO/NSW/nsw.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
    # ts, data = load_data("../data/TAS2016.csv", columnName="TOTALDEMAND")
    ts, data = load_data("../data/TT30GEN.csv", columnName="VALUE")
    # ts, data = load_data("../data/bike_hour.csv", columnName="cnt")
    # ts,data = load_data_xls("./data/air_quality/AirQuality.xlsx", indexName="Date", columnName="PT08.S2(NMHC)")
    # ts,data = load_data_txt("./data/house_data/house_power.csv", indexName="Date", columnName="Global_active_power")
    # ts, data = load_data_txt("./data/house_data/house_power.csv", indexName="Date", columnName="Global_active_power")#
    # data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    print(ts)
    print(data.shape)

    # train/test divide test
    train, test = divideTrainTest(data, 0.75)
    print(train.shape)
    print(test.shape)

    # create samples test
    trainX, trainY = createSamples(train, lookBack=21, lookAhead=7, RNN=True)
    testX, testY = createSamples(test, lookBack=21, lookAhead=7, RNN=True)
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)



    # vtrainX,vtrainY = createVariableDataset(train, 10, 20,step=5)
    # vtestX, vtestY = createVariableDataset(test, 10, 20, step=5)
    # vtestY = transformGroundTruth(vtestY,10, 20, 5)
    # ptrainX, ptrainY = createPaddedDataset(train, 30, 40)

    # print (vtrainX.shape)
    # print (vtrainY.shape)
    # print(vtestX.shape)
    # print(vtestY.shape)
    # print (ptrainX.shape)
    # print (ptrainY.shape)






