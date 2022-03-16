"""
Author:
    Ghaith Arar (ghaith01@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ujson as json

def get_tf_scores(path='C:/Users/ghait/PycharmProjects/R-Net_plus/data/tf_scores.json'):
    with open(path) as tf_file:
        tf_scores = json.load(tf_file)

    del tf_scores['total']

    score_len = max([int(i) for i in tf_scores.keys()]) + 1

    tf_score_tensor = torch.zeros((score_len, 1))

    for k, v in tf_scores.items():
        tf_score_tensor[int(k), 0] = v

    return tf_score_tensor


class BRNNs(nn.Module):
    def __init__(self, in_size, h_size, n_layers=1, type='LSTM'):
        super(BRNNs, self).__init__()
        self.type = {'LSTM': nn.LSTM, 'GRU': nn.GRU}
        assert (type in self.type.keys()), 'Type must be LSTM or GRU'


        self.number_layers = n_layers
        self.brnns = nn.ModuleList()
        for n in range(self.number_layers):
            in_size = h_size * 2 if n > 0 else in_size
            self.brnns.append(self.type[type](in_size, h_size, batch_first=True, num_layers=1, bidirectional=True))


    def forward(self, input, input_mask):
        assert input is not None
        assert input_mask is not None
        assert isinstance(input, torch.Tensor)
        assert isinstance(input_mask, torch.Tensor)

        lengths = input_mask.data.eq(0).long().sum(1).squeeze() #Input mask 1 for true input, 0 for padding
        dim_soreted , sorted_idx = torch.sort(lengths, dim=0, descending=True)
        _dim_soreted, idx_unsort = torch.sort(sorted_idx, dim=0)



        input_rnn = input.index_select(0, sorted_idx)

        input_rnn = nn.utils.rnn.pack_padded_sequence(input_rnn, lengths.cpu(), batch_first=True, enforce_sorted=False)

        outputs = [input_rnn]

        for i in range(self.number_layers):
            rnn_input = outputs[-1]
            outputs.append(self.brnns[i](rnn_input)[0])

        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        output = torch.cat(outputs[1:], 2)
        output = output.transpose(0,1)
        #print(output.shape)
        return output

class BRNNs_char(nn.Module):
    def __init__(self, in_size, h_size, n_layers=1, type='LSTM'):
        super(BRNNs_char, self).__init__()


        self.type = {'LSTM': nn.LSTM, 'GRU': nn.GRU}
        assert (type in self.type.keys()), 'Type must be LSTM or GRU'

        self.number_layers = n_layers
        self.rnn = self.type[type]
        self.h_size = h_size
        self.brnns = nn.ModuleList()
        for n in range(self.number_layers):
            in_size = h_size * 2 if n > 0 else in_size
            self.brnns.append(self.rnn(in_size, h_size, batch_first=False, num_layers=1, bidirectional=True))


    def forward(self, input):
        assert input is not None
        assert isinstance(input, torch.Tensor)
        B, W, C, C_embed = input.size()
        #print(input.size())
        input = input.view(B * W, C, C_embed)
        #print(input.size())
        outputs = [input]

        for layer in range(self.number_layers):
            rnn_input = outputs[-1]
            outputs.append(self.brnns[layer](rnn_input)[0])

        output = outputs[-1]
        #print('outpit shape', output.shape)
        output = output[:, C-1, :].squeeze()
        #print('outpit shape', output.shape)
        output = output.view(B, W, self.h_size*2)
        #print('outpit shape', output.shape)

        return output



class FusionCell(nn.Module):
    def __init__(self, input_size, f_size):
        super(FusionCell, self).__init__()
        self.r = nn.Linear(f_size + input_size, input_size)
        self.g = nn.Linear(f_size + input_size, input_size)

    def forward(self, x, f):
        r = torch.tanh(self.r(torch.cat([x, f], 2)))
        g = torch.sigmoid(self.g(torch.cat([x, f], 2)))
        o = g * r + (1 - g) * x
        return o

class Attntion_Unit(nn.Module):
    def __init__(self, in_size):
        super(Attntion_Unit, self).__init__()
        self.linear = nn.Linear(in_size, in_size)

    def forward(self, C, Q, Q_mask):
        C = self.linear(C)
        C= F.relu(C)

        Q = self.linear(Q)
        Q = F.relu(Q)

        scores = C.bmm(Q.transpose(2, 1))

        scores.data.masked_fill_(Q_mask.unsqueeze(1).expand(scores.size()).data, -float('inf'))

        a = F.softmax(scores, dim=2)

        return a.bmm(Q)

class Self_Attntion_Unit(nn.Module):
    def __init__(self, input_size):
        super(Self_Attntion_Unit, self).__init__()

    def forward(self, x, x_mask):
        scores = x.bmm(x.transpose(2, 1))
        x_len = x.size(1)
        for i in range(x_len):
            scores[:, i, i] = 0

        x_mask = x_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        a = torch.softmax(scores, dim=2)

        return a.bmm(x)

class Output_Pointer(nn.Module):
    def __init__(self, x_size, y_size, h_size, unit_num = 1):
        super(Output_Pointer, self).__init__()
        self.unit_num = unit_num

        self.FF_begin = nn.ModuleList()
        self.Fusion_begin = nn.ModuleList()


        self.FF_end = nn.ModuleList()
        self.Fusion_end = nn.ModuleList()

        self.dropout = nn.Dropout(p=0.2)

        for i in range(unit_num):
            self.FF_begin.append(FFN(x_size + y_size + 2 * h_size, h_size, 1))
            self.Fusion_begin.append(FusionCell(y_size, 2 * h_size))

            self.FF_end.append(FFN(x_size + y_size + 2 * h_size, h_size, 1))
            self.Fusion_end.append(FusionCell(y_size, 2 * h_size))

    def forward(self, C, Q, C_mask):
        z_s = Q[:, -1, :].unsqueeze(1)
        z_e, s, E, P1, P2 = None, None, None, None, None


        for i in range(self.unit_num):
            z_s_ = z_s.repeat(1, C.size(1), 1)
            s = self.FF_begin[i](torch.cat([C, z_s_, C * z_s_], 2)).squeeze(2)
            s.data.masked_fill_(C_mask.data, -float('inf'))

            p1 = torch.softmax(s, dim=1)
            u_s = p1.unsqueeze(1).bmm(C)
            z_e = self.Fusion_begin[i](z_s, u_s)
            z_e_ = z_e.repeat(1, C.size(1), 1)

            e = self.FF_end[i](torch.cat([C, z_e_, C * z_e_], 2)).squeeze(2)
            e.data.masked_fill_(C_mask.data, -float('inf'))

            p2 = torch.softmax(e, dim=1)
            u_e = p2.unsqueeze(1).bmm(C)
            z_s = self.Fusion_end[i](z_e, u_e)

        p1 = torch.log_softmax(s, dim=1)
        p2 = torch.log_softmax(e, dim=1)

        return p1, p2

class FFN(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(FFN, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(in_size, h_size)
        self.linear2 = nn.Linear(h_size, out_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        return self.linear2(x)