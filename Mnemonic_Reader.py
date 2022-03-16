"""
Author:
    Ghaith Arar (ghaith01@stanford.edu)
    CS224N Final Project
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import MnemonicReader_layers
import util

class Mnemonic_Reader(nn.Module):
    def __init__(self, word_vec, char_vec, char_h_size=40, h_size=100, tf_dim=1, exact_match_dim=1, word_dim=300, dropout=0.15, units=3):
        super(Mnemonic_Reader, self).__init__()
        self.units_number = units
        self.dropout_rate = dropout

        # Word Embedding
        self.word_embedding = nn.Embedding.from_pretrained(word_vec)

        # Char Embedding
        self.char_embedding = nn.Embedding.from_pretrained(char_vec)

        # Normalized Term Frequency
        self.TF_embedding = nn.Embedding.from_pretrained(MnemonicReader_layers.get_tf_scores())

        # RNN Char encoder
        self.char_rnn = MnemonicReader_layers.BRNNs_char(64, char_h_size, n_layers=1)

        # RNN encoder
        self.rnn_encoder = nn.LSTM(word_dim+(char_h_size*2)+tf_dim+exact_match_dim, h_size, num_layers=1, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)




        self.reader_aligners = nn.ModuleList()
        self.reader_fusion = nn.ModuleList()
        self.self_atten = nn.ModuleList()
        self.self_fusion = nn.ModuleList()
        self.rnns = nn.ModuleList()

        for i in range(self.units_number):
            self.reader_aligners.append(MnemonicReader_layers.Attntion_Unit(h_size * 2))
            self.reader_fusion.append(MnemonicReader_layers.FusionCell(h_size * 2, h_size * 2 * 3))
            self.self_atten.append(MnemonicReader_layers.Self_Attntion_Unit(h_size * 2))
            self.self_fusion.append(MnemonicReader_layers.FusionCell(h_size * 2, h_size * 2 * 3))
            self.rnns.append(MnemonicReader_layers.BRNNs(h_size * 2, h_size, n_layers=1, type='LSTM'))

        self.output_pointer = MnemonicReader_layers.Output_Pointer(h_size * 2, h_size * 2, h_size, unit_num=self.units_number)

    def forward(self, C_words, Q_words, C_char, Q_char):
        C_mask = torch.zeros_like(C_words) == C_words
        Q_mask = torch.zeros_like(Q_words) == Q_words

        CW_emb = self.word_embedding(C_words)
        C_tf = self.TF_embedding(C_words)

        QW_emb = self.word_embedding(Q_words)
        Q_tf = self.TF_embedding(Q_words)

        C_char_emb = self.char_embedding(C_char)
        Q_char_emb = self.char_embedding(Q_char)

        if self.dropout:
            CW_emb = self.dropout(CW_emb)
            QW_emb = self.dropout(QW_emb)
            C_char_emb = self.dropout(C_char_emb)
            Q_char_emb = self.dropout(Q_char_emb)

        C_char_encoding = self.char_rnn(C_char_emb)

        Q_char_encoding = self.char_rnn(Q_char_emb)

        # Get exact match for C and Q
        C_em_mask = torch.zeros_like(C_words) != C_words
        C_em = torch.bitwise_and(torch.isin(C_words, Q_words), C_em_mask)
        C_em[:, 0] = 0
        C_em = C_em.unsqueeze(2)

        Q_em_mask = torch.zeros_like(Q_words) != Q_words
        Q_em = torch.bitwise_and(torch.isin(Q_words, C_words), Q_em_mask)
        Q_em[:, 0] = 0
        Q_em = Q_em.unsqueeze(2)

        C_rnn_input = [CW_emb, C_tf, C_char_encoding, C_em]
        C_rnn_input = torch.cat(C_rnn_input, 2)
        C_mask_len = C_em_mask.eq(1).sum(1).cpu()
        #print(C_mask_len.shape, C_mask_len.dtype)
        C_rnn_input_packed = torch.nn.utils.rnn.pack_padded_sequence(C_rnn_input, C_mask_len, batch_first=True, enforce_sorted=False)


        Q_rnn_input = [QW_emb, Q_tf, Q_char_encoding, Q_em]
        Q_rnn_input = torch.cat(Q_rnn_input, 2)
        Q_mask_len = Q_em_mask.eq(1).sum(1).cpu()
        #print(Q_mask_len.shape, Q_mask_len.dtype)
        Q_rnn_input_packed = torch.nn.utils.rnn.pack_padded_sequence(Q_rnn_input, Q_mask_len, batch_first=True, enforce_sorted=False)

        # Encode document
        C, (ch_n, cc_n) = self.rnn_encoder(C_rnn_input_packed)
        C_padded, len_C = torch.nn.utils.rnn.pad_packed_sequence(C, batch_first=True)

        # Encode question
        Q, (qh_n, qc_n) = self.rnn_encoder(Q_rnn_input_packed )
        Q_padded, lec_Q = torch.nn.utils.rnn.pad_packed_sequence(Q, batch_first=True)


        Current_Input = C_padded

        for unit in range(self.units_number):
            q_t = self.reader_aligners[unit](Current_Input, Q_padded, Q_mask)
            #print("1 unit: ", unit, q_t.shape)

            c_bar = self.reader_fusion[unit](Current_Input,
                                                     torch.cat([q_t, Current_Input * q_t, Current_Input - q_t], 2))
            #print("2 unit: ", unit, c_bar.shape)

            c_t = self.self_atten[unit](c_bar, C_mask)
            #print("3 unit: ", unit, c_t.shape)

            c_hat = self.reader_fusion[unit](c_bar, torch.cat([c_t, c_bar * c_t, c_bar - c_t], 2))
            #print("4 unit: ", unit, c_hat.shape)

            Current_Input = self.rnns[unit](c_hat, C_mask)
            #print("5 unit: ", unit, Current_Input.shape)

            
            
            
            

        p1, p2 = self.output_pointer(Current_Input, Q_padded, C_mask)

        return (p1, p2)
