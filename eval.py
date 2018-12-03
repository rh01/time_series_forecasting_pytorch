from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


# 计算RMSE
def calcRMSE(true, pred):
    return np.sqrt(mean_squared_error(true, pred, multioutput='uniform_average'))


# 计算MAE
def calcMAE(true, pred):
    #pred = pred[:, 0]
    return mean_absolute_error(true, pred, multioutput='uniform_average')


# 计算MAPE
def calcMAPE(true, pred, epsion = 0.0000000):
    #pred = pred[:,0] # 列转行，便于广播计算误差指标
    # print (true-pred).shape
    # print true.shape
    # print pred.shape
    true += epsion
    return np.sum(np.abs((true-pred)/true))/len(true)*100
    #return mean_absolute_percentage_error(true, pred)


# 计算SMAPE
def calcSMAPE(true, pred):

    # pred = pred.reshape((-1, ))  # 统一shape为 (sample_num, )
    # true = true.reshape((-1, ))
    delim = (true+pred)/2.0
    numerator = np.abs((true-pred))
    tmp = numerator/delim
    sum_error = np.mean(tmp)
    return sum_error * 100


if __name__ == '__main__':

    pred = np.array([1, 2, 3, 4])
    true = np.array([2, 3, 4, 5])
    print(pred.shape)
    print(true.shape)
    mae = calcMAE(pred, true)
    rmse = calcRMSE(pred, true)
    smape = calcSMAPE(pred, true)
    print(mae, rmse, smape)