# author: jingxiaolun
# date: 2023.08.31
# description: offical code in X-Pool (modules/transformer.py)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_mha_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    
    def forward(self, text_embeds, video_embeds, video_mask):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        num_texts, _ = text_embeds.shape
        q, k, v = text_embeds, video_embeds, video_embeds
        
        # num_texts x embed_dim
        #q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1,2,0)

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        #k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)

        # num_vids x num_frames x embed_dim
        #v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim) 

        # num_vids x num_frames -> num_vids x num_heads x num_frames x num_texts
        #if video_mask is not None:
        #    video_mask_ = video_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, num_texts)
        #    fill_value = torch.tensor(-1e9)
        #    attention_logits = attention_logits.masked_fill(video_mask_==0, fill_value)

        attention_weights = F.softmax(attention_logits, dim=2)
        #print(f'attn_weights: {attention_weights[0, 0, :, 0]}')

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        #o = self.out_proj(attention)
        o = attention
        return o

    def get_conditional_visual_output_plus(self, text_embeds, video_embeds, video_mask):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_texts x num_frames x num_vids x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        num_texts, _ = text_embeds.shape
        q, k, v = text_embeds, video_embeds, video_embeds

        # num_texts x num_heads x head_dim
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_texts x num_heads x head_dim x 1
        q = q.unsqueeze(-1)

        num_frames, num_vids, _ = video_embeds.shape[1:]
        # num_texts x num_frames x num_vids x num_heads x head_dim
        k = k.reshape(num_texts, num_frames, num_vids, self.num_heads, self.head_dim)
        # num_vids x num_texts x num_heads x num_frames x head_dim
        k = k.permute(2,0,3,1,4)

        # num_texts x num_frames x num_vids x num_heads x head_dim
        v = v.reshape(num_texts, num_frames, num_vids, self.num_heads, self.head_dim)
        # num_vids x num_texts x num_heads x head_dim x num_frames
        v = v.permute(2,0,3,4,1)

        # num_vids x num_texts x num_heads x num_frames x 1
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=3)

        # num_vids x num_texts x num_heads x head_dim x 1
        attention = v @ attention_weights
        # num_vids x num_texts x num_heads x head_dim 
        attention = attention.squeeze(-1)
        # num_vids x num_texts x embed_dim
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)
        o = attention
        return o

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_mha_heads, transformer_dropout):
        super().__init__()
        self.cross_attn = MultiHeadedAttention(embed_dim, num_mha_heads)

        self.linear_proj = nn.Linear(embed_dim, embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(transformer_dropout)

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, text_embeds, video_embeds, video_mask):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_texts x num_vids x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        if video_embeds.dim()==3:
            attn_out = self.cross_attn(text_embeds, video_embeds, video_mask)
        elif video_embeds.dim()==4:
            attn_out = self.cross_attn.get_conditional_visual_output_plus(text_embeds, video_embeds, video_mask)

        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        #out = self.layer_norm3(linear_out)
        
        # num_texts x num_vids x embed_dim
        out = out.permute(1, 0, 2).contiguous()
        return out

class Conditional_Transformer(nn.Module):
    def __init__(self, embed_dim, num_mha_heads, transformer_dropout):
        super().__init__()
        self.pool_frames = Transformer(embed_dim, num_mha_heads, transformer_dropout)

    def forward(self, text_embeds, video_embeds, video_mask, return_all_frames=False):
        video_pool_embeds = self.pool_frames(text_embeds, video_embeds, video_mask)
        if return_all_frames:
            return text_embeds, video_embeds, video_pool_embeds
        else:
            return text_embeds, video_pool_embeds
