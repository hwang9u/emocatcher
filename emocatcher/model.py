# EmoCatcher: Convoltuional Attention based Bi-directional GRU
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask_ind(x, input_lengths):    
    mask_ind = torch.zeros_like(x)
    for b in range(x.shape[0]):
        x_size = input_lengths[b]
        mask_ind[b, x_size: ,:] = 1
    return mask_ind

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_z = nn.Linear(hidden_dim, 1)
        
    def forward(self, query, key, value, lengths):
        q = self.linear_q(query)
        k = self.linear_k(key)
        att_score = self.linear_z(torch.tanh(q+k)) 
        mask = get_mask_ind(att_score, lengths)
        att_score[mask.type(torch.bool)] = -1e9 # masking
        att_weights = F.softmax(att_score, dim = 1) 
        att_value = torch.bmm(att_weights.transpose(1,2).contiguous(), value) 
        
        return att_value, att_weights


class ConvLN(nn.Module):
    def __init__(self, n_feats):
        super(ConvLN, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.layer_norm(x)
        return x.transpose(1, 2).contiguous()
    

class Conv1dLNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p = 0.2, **conv_kwargs):
        super(Conv1dLNBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels = out_channels, kernel_size= kernel_size, **conv_kwargs),
            ConvLN(out_channels),
            nn.GELU(),
             nn.Dropout(p)
            )
        
    def forward(self, x):
        out = self.conv_block(x)
        return out

class EmoCatcher(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_classes = 8):
        super(EmoCatcher, self).__init__()
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_block1 = Conv1dLNBlock(input_dim, hidden_dim//2, kernel_size=kernel_size, padding = 'same')
        self.conv_block2 = Conv1dLNBlock(hidden_dim//2, hidden_dim//2, kernel_size=kernel_size, padding = 'same')
        self.conv_block3 = Conv1dLNBlock(hidden_dim//2, hidden_dim, kernel_size=kernel_size, padding = 'same')
  
        self.maxpool1 = nn.MaxPool1d(2)
        
        self.gru_layernorm = nn.LayerNorm( hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim//2,
                                batch_first = True,
                                num_layers = 1,
                                bidirectional =True,
                                )
        self.attention= BahdanauAttention(hidden_dim)
        
        self.linear1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.BatchNorm1d(hidden_dim//2),
                                    nn.GELU(),
                                    nn.Dropout(0.1),
                                    )
                                    
        self.linear2 = nn.Linear(hidden_dim//2, self.num_classes)
        
        
        self.initialize_weights() # init weights
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU): 
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        
            if isinstance(m, BahdanauAttention):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        if 'z' in name:
                            torch.nn.init.kaiming_normal_(param.data)
                        else:
                            torch.nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

        
    def forward(self,x, L, return_attention_weights = False):
        x = x.squeeze(1)
        # conv
        z = self.conv_block1(x) ## (N, h//2, L)
        z = self.conv_block2(z)
        z = self.conv_block3(z)
        
        # maxpool
        z_mp = self.maxpool1(z)
        L_mp = torch.div(L - 2, 2, rounding_mode='trunc') + 1 ## length list after MaxPool1d 
        z_mp = z_mp.transpose(1,2).contiguous() ## (N, L, h//2) # transpose to feed gru layer
        
        # gru
        z_ln= self.gru_layernorm(z_mp)
        z_pp = nn.utils.rnn.pack_padded_sequence(input=z_ln, lengths = L_mp, batch_first=True,enforce_sorted=False)
        o, h = self.gru(z_pp)
        o_pad, L_pad = nn.utils.rnn.pad_packed_sequence(o, batch_first = True,padding_value=0.)
        
        # attention 
        h_c = torch.concat( (h[0], h[1]), axis = -1).unsqueeze(1)
        o_att, att_weights = self.attention(query = h_c, key = o_pad, value = o_pad, lengths= L_pad)
        o_att = o_att.squeeze(1)
        
        # fc
        z2 = self.linear1(o_att)
        y = self.linear2(z2)
        
        if return_attention_weights:
            return y, att_weights
        else:
            return y
