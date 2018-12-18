
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(1)


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


# 模型基类，主要是用于指定参数和cell类型
class BaseModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):

        super(BaseModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = layerNum
        if cell == "RNN":
            self.cell = nn.RNN(input_size=self.inputDim, hidden_size=self.hiddenNum,
                        num_layers=self.layerNum, dropout=0.0,
                         nonlinearity="tanh", batch_first=True,)
        if cell == "LSTM":
            self.cell = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenNum,
                               num_layers=self.layerNum, dropout=0.0,
                               batch_first=True, )
        if cell == "GRU":
            self.cell = nn.GRU(input_size=self.inputDim, hidden_size=self.hiddenNum,
                                num_layers=self.layerNum, dropout=0.0,
                                 batch_first=True, )
        print(self.cell)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)


# 标准RNN模型
class RNNModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):

        super(RNNModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)

    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, h0)
        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput


class Multi_Hidden_RNN_Model(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, seq_len, merge="mean"):

        super(Multi_Hidden_RNN_Model, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)
        if merge == "mean":
            self.dense = nn.Linear(hiddenNum, outputDim)
        if merge == "concate":
            self.dense = nn.Linear(hiddenNum * seq_len, outputDim)
        self.hiddenNum = hiddenNum
        self.merge = merge
        self.seq_len = seq_len

    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, h0)  # shape (batch_size, seq_len, hidden_num)
        if self.merge == "mean":
            sum_hidden = torch.mean(rnnOutput, 1)
            x = sum_hidden.view(-1, self.hiddenNum)
        if self.merge == "concate":
            rnnOutput = rnnOutput.contiguous()
            x = rnnOutput.view(-1, self.hiddenNum * self.seq_len)
        # x = nn.Dropout(0.5)(x)
        fcOutput = self.dense(x)

        return fcOutput


class RNN_Attention(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, seq_len, cell="RNN", merge="concate"):

        super(RNN_Attention, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)
        self.att_fc = nn.Linear(hiddenNum, 1)
        self.time_distribut_layer = TimeDistributed(self.att_fc)
        if merge == "mean":
            self.dense = nn.Linear(hiddenNum, outputDim)
        if merge == "concate":
            self.dense = nn.Linear(hiddenNum * seq_len, outputDim)
        self.hiddenNum = hiddenNum
        self.merge = merge
        self.seq_len = seq_len

    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, h0)

        attention_out = self.time_distribut_layer(rnnOutput)
        attention_out = attention_out.view((batchSize, -1))
        attention_out = F.softmax(attention_out)
        attention_out = attention_out.view(batchSize, -1, 1)

        rnnOutput = rnnOutput * attention_out

        if self.merge == "mean":
            sum_hidden = torch.mean(rnnOutput, 1)
            x = sum_hidden.view(-1, self.hiddenNum)
        if self.merge == "concate":
            rnnOutput = rnnOutput.contiguous()
            x = rnnOutput.view(-1, self.hiddenNum * self.seq_len)

        fcOutput = self.dense(x)

        return fcOutput


# LSTM模型
class LSTMModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):
        super(LSTMModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)

    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        c0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, (h0, c0))  # rnnOutput 12,20,50 hn 1,20,50
        hn = hn[0].view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput


class Multi_Hidden_LSTM_Model(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, seq_len, merge="mean"):

        super(Multi_Hidden_LSTM_Model, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)
        if merge == "mean":
            self.dense = nn.Linear(hiddenNum, outputDim)
        if merge == "concate":
            self.dense = nn.Linear(hiddenNum * seq_len, outputDim)
        self.hiddenNum = hiddenNum
        self.merge = merge
        self.seq_len = seq_len

    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        c0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, (h0, c0))

        if self.merge == "mean":
            sum_hidden = torch.mean(rnnOutput, 1)
            x = sum_hidden.view(-1, self.hiddenNum)
        if self.merge == "concate":
            rnnOutput = rnnOutput.contiguous()
            x = rnnOutput.view(-1, self.hiddenNum * self.seq_len)
        # x = nn.Dropout(0.5)(x)
        fcOutput = self.dense(x)

        return fcOutput


# GRU模型
class GRUModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):
        super(GRUModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)

    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, h0)  # rnnOutput 12,20,50 hn 1,20,50
        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput


class Multi_Hidden_GRU_Model(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, seq_len, merge="mean"):

        super(Multi_Hidden_GRU_Model, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)
        if merge == "mean":
            self.dense = nn.Linear(hiddenNum, outputDim)
        if merge == "concate":
            self.dense = nn.Linear(hiddenNum * seq_len, outputDim)
        self.hiddenNum = hiddenNum
        self.merge = merge
        self.seq_len = seq_len

    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, h0)  # shape (batch_size, seq_len, hidden_num)
        if self.merge == "mean":
            sum_hidden = torch.mean(rnnOutput, 1)
            x = sum_hidden.view(-1, self.hiddenNum)
        if self.merge == "concate":
            rnnOutput = rnnOutput.contiguous()
            x = rnnOutput.view(-1, self.hiddenNum * self.seq_len)
        # x = nn.Dropout(0.5)(x)
        fcOutput = self.dense(x)

        return fcOutput


# 标准ANN模型
class ANNModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim):

        super(ANNModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.fc1 = nn.Linear(self.inputDim, self.hiddenNum)
        self.fc2 = nn.Linear(self.hiddenNum, self.outputDim)

    def forward(self,x, batchSize):

        output = self.fc1(x)
        output = self.fc2(output)

        return output

# ResRNN模型
class ResRNNModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, resDepth):

        super(ResRNNModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = 1
        self.resDepth = resDepth
        self.i2h = nn.Linear(self.inputDim, self.hiddenNum, bias=True)
        self.h2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        self.h2o = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.ht2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        #self.tanh = nn.Tanh()

    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        # output = []
        inputLen = x.data.size()[1]
        ht = h0
        for i in range(inputLen):
            hn = self.i2h(x[:, i, :]) + self.h2h(h0)
            if self.resDepth == 0:
                h0 = nn.Tanh()(hn)
            if self.resDepth == 1:
                # res depth = 1
                h0 = nn.Tanh()(hn + h0)
            if self.resDepth >= 2:
                # res depth = N
                if i % self.resDepth == 0 and i != 0:
                    h0 = nn.Tanh()(hn + ht)
                    ht = hn
                else:
                    h0 = nn.Tanh()(hn)


            # 首尾加入res
            if self.resDepth == -1:
                if i == 0:
                    hstart = hn
                if i == inputLen-2:
                    h0 = nn.Tanh()(hn+hstart)
                else:
                    if i % 4 == 0 and i != 0:
                        h0 = nn.Tanh()(hn + ht)
                        ht = hn
                    else:
                        h0 = nn.Tanh()(hn)

        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput


# 加入注意机制的RNN模型
class AttentionRNNModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, seqLen):
        super(AttentionRNNModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = 1
        self.i2h = nn.Linear(self.inputDim, self.hiddenNum, bias=True)
        self.h2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        self.h2o = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.fc = nn.Linear(self.hiddenNum*seqLen, self.outputDim, bias=True)
        # self.tanh = nn.Tanh()

    def forward(self, x, batchSize):
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        hiddenList = []
        inputLen = x.data.size()[1]
        for i in range(inputLen):
            hn = self.i2h(x[:, i, :]) + self.h2h(h0)
            h0 = nn.Tanh()(hn)
            ht = h0.view(batchSize, self.hiddenNum)
            hiddenList.append(ht)
        flanten = torch.cat(hiddenList, dim=1)

        fcOutput = self.fc(flanten)

        return fcOutput


class IndRNNCell(nn.Module):

    def __init__(self, inpdim, recdim, act=F.tanh):
        super().__init__()
        self.inpdim = inpdim
        self.recdim = recdim
        self.act = F.relu if act is None else act
        self.w = nn.Parameter(torch.randn(inpdim, recdim))
        self.u = nn.Parameter(torch.randn(recdim))
        self.b = nn.Parameter(torch.randn(recdim))
        self.F = nn.Linear(recdim, 1)

    def forward(self, x_t, h_tm1):
        return self.act(h_tm1 * self.u + x_t @ self.w + self.b)


class IndRNN(nn.Module):
    def __init__(self, inpdim, recdim, depth=1):
        """
        inpdim      : dimension D in (Batch, Time, D)
        recdim      : recurrent dimension/ Units/
        depth       : stack depth
        """
        super().__init__()
        self.inpdim = inpdim
        self.recdim = recdim
        self.cells = [IndRNNCell(inpdim, recdim) for _ in range(depth)]
        self.depth = depth

    def forward(self, x, h_0):
        # h_tm1 = Variable(torch.ones(self.recdim))
        seq = []
        for i in range(x.size()[1]):
            x_t = x[:, i, :]
            for cell in self.cells:
                h_0 = cell.forward(x_t, h_0)
            seq.append(h_0)
        return torch.stack(seq, dim=1), h_0


class IndRNNModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim):
        super(IndRNNModel, self).__init__()
        self.inputDim = inputDim
        self.hiddenNum = hiddenNum
        self.outputDim = outputDim
        self.cell = IndRNN(inputDim, hiddenNum)
        self.fc = nn.Linear(hiddenNum, outputDim)

    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, h0)
        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput


