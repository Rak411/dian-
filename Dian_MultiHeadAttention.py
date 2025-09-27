import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
       
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
       
        self.d_k = embed_dim // num_heads  
        
        
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        
        
        self.o = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)


    def forward(self, query, key, value, mask=None):
        
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        Q = self.q(query)  
        K = self.k(key)    
        V = self.v(value)  

        print(f"Q 矩阵形状: {Q.shape}")  
        
        
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        print(f"重塑后 Q 矩阵形状: {Q.shape}")  
        
       
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        
        if mask is not None:
            mask = mask.unsqueeze(1)  
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
       
        output = torch.matmul(attention_weights, V)
    
        
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.embed_dim)
        print(f"output 矩阵形状: {output.shape}")  

    
        output = self.o(output)
        
        return output, attention_weights
    
def test_multihead_attention():
    
    batch_size, seq_len, num_heads, embed_dim = 2, 5, 4, 16
    
    
    Q = torch.randn(batch_size, seq_len, embed_dim)  

    K = torch.randn(batch_size, seq_len, embed_dim)  

    V = torch.randn(batch_size, seq_len, embed_dim)  
    
    print("Q 形状:", Q.shape)
    print("K 形状:", K.shape)
    print("V 形状:", V.shape)

    
    model = MultiHeadAttention(embed_dim, num_heads)
    model.forward(Q,K,V,None)

test_multihead_attention()

def test_linear():
    
    linear_layer = nn.Linear(in_features=3, out_features=2, bias=True)

    print("权重矩阵 W 的形状:", linear_layer.weight.shape)  
    print("偏置向量 b 的形状:", linear_layer.bias.shape)    
    print("\n初始化的参数:")
    print("权重 W:\n", linear_layer.weight)
    print("偏置 b:", linear_layer.bias)

def demonstrate_linear_in_attention():
    
    embed_dim = 16
    num_heads = 4
    d_k = embed_dim // num_heads  
    
    
    w_q = nn.Linear(embed_dim, embed_dim)  
    
    batch_size, seq_len = 2, 5
    input_sequence = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"输入序列形状: {input_sequence.shape}")  
    
    
    Q = w_q(input_sequence)
    print(f"Q 矩阵形状: {Q.shape}")  
    
    
    print(f"\n线性层权重形状: {w_q.weight.shape}")  
    print(f"线性层偏置形状: {w_q.bias.shape}")     

