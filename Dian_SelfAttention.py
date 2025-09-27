import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#Self Attention层实现
class SelfAttention(nn.Module):
    
    def __init__(self):
        
        super(SelfAttention, self).__init__()
        
    def forward(self, q, k, v, mask):
       
        # 1. 计算 Q * K^T
        k_transposed = k.transpose(0, 1)  # [3, 2] -> [2, 3]
        print("K转置后形状:", k_transposed.shape)

        d_k = k.size(1)
        scale = d_k
        attention_scores = torch.matmul(q, k_transposed)
        # attention_scores: [seq_len, seq_len]
        
        print("\n注意力分数矩阵:")
        print(attention_scores)
        print("分数矩阵形状:", attention_scores.shape)

        # 2. 注意力分数缩放
        d_k = k.size(1)
        scale = d_k
        attention_scores = attention_scores / scale

        # 3. 应用注意力掩码
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 4. 应用 softmax 得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        print("\n注意力权重:")
        print(attention_weights)
        
        # 5. 应用注意力权重到 V
        output = torch.matmul(attention_weights, v)
        print("\n最终输出:")
        print(output)
        print("输出形状:", output.shape)
        
        return output, attention_weights

def test_self_attention():
    # 参数：序列长度=3, 维度=2
    seq_len, d_k = 3, 2
    
    # 创建 Q, K, V
    Q = torch.tensor([[1.0, 2.0], 
                        [3.0, 4.0], 
                        [5.0, 6.0]])  # [3, 2]
    
    K = torch.tensor([[0.5, 1.5], 
                        [1.0, 2.0], 
                        [1.5, 2.5]])  # [3, 2]
    
    V = torch.tensor([[0.1, 0.2], 
                        [0.3, 0.4], 
                        [0.5, 0.6]])  # [3, 2]
    print("Q 形状:", Q.shape)
    print("K 形状:", K.shape)
    print("V 形状:", V.shape)

    model = SelfAttention()
    model.forward(Q,K,V,None)

test_self_attention()