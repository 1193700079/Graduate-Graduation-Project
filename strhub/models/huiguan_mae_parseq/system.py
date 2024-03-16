# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from itertools import permutations
from typing import Sequence, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from strhub.models.huiguan_mae_parseq.modules import DecoderLayer, Decoder, TokenEmbedding


from transformers import GPT2Model, GPT2Config


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
from einops import rearrange, repeat


import torch
import numpy as np
import random

# 定义一个种子值
seed = 42

# 设置随机种子
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 对于 CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了进一步确保模型的确定性，你可以设置如下选项
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from torch.optim.lr_scheduler import OneCycleLR

    
from einops import rearrange
from strhub.models.huiguan_mae_parseq import clip
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
class MAE_PARSeq(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool,
                 decode_ar: bool, refine_iters: int, dropout: float, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()

        self.coef_lr = kwargs["coef_lr"] if "coef_lr" in kwargs.keys() else 1.0
        self.coef_wd = kwargs["coef_wd"] if "coef_wd" in kwargs.keys() else 1.0
        self.image_freeze_nlayer = kwargs["image_freeze_nlayer"] if "image_freeze_nlayer" in kwargs.keys() else -1
        self.text_freeze_nlayer = kwargs["text_freeze_nlayer"] if "text_freeze_nlayer" in kwargs.keys() else -1

        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters

        import strhub.models.models_mae as models_mae
        
        mae_model = getattr(models_mae, 'mae_vit_base_patch4')()

        self.encoder = mae_model
        embed_dim = 768
        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

        

        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored

        # # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        
        # encoder_embed_dim = 768 # 768
        
        self.adpater = Adapter(512,4)
        # self.adpater_clip_textual = Adapter(768,4)
        # self.adpater_clip_visual = Adapter(768,4)
        self.adpater2 = Adapter(768,4)
        # self.adpater_decoder = Adapter(1024,32)

       
        
        self.clip_model, _ = clip.load("ViT-B/16", device='cpu', jit=False)
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.clip_model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
        
        del self.clip_model.token_embedding   
        # if self.clip_model is not None:
        #     for param in self.clip_model.parameters():
        #         param.requires_grad = False # fix teacher_model model

        #     self.clip_model.eval()  
       
        self.w_att = nn.Linear(2 * 768, 768)   
        # scale3 = 77 ** -0.5
        # self.proj_to_77 =   nn.Parameter(scale3 * torch.randn(197, 77))


        # scale = 512 ** -0.5
        scale2 = 768 ** -0.5
        scale3 = 77 ** -0.5
        scale4 = 197 ** -0.5
        # self.proj_to_512  = nn.Parameter(scale * torch.randn(768, 512))
        self.proj_to_768  = nn.Parameter(scale2 * torch.randn(512, 768))
        self.proj_to_77 =   nn.Parameter(scale3 * torch.randn(197, 77))
        # self.proj_clip_visual_features =   nn.Parameter(scale3 * torch.randn(512, 768))
        # self.proj_clip_features =   nn.Parameter(scale3 * torch.randn(512, 768))
        
        if img_size[0] == 32 and img_size[1] == 128:
            # scale4 = 257 ** -0.5
            self.proj_to_257 =   nn.Parameter(scale4 * torch.randn(77, 257))
        elif   img_size[0] == 224 and img_size[1] == 224:
            # scale4 = 197 ** -0.5
            self.proj_to_memory =   nn.Parameter(scale4 * torch.randn(77, 197))


        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std=.02)
        

    # @torch.jit.ignore
    # def no_weight_decay(self):
        # param_names = {'text_embed.embedding.weight', 'pos_queries'}
        # enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        # return param_names.union(enc_param_names)
        # return param_names  

    def encode(self, mae_img,clip_img,mask_ratio=0):
        mae_memory,cls_tokens,encoder_features = self.encoder(mae_img, mask_ratio) #B * 257 *768 
        
        target = self.clip_model.encode_image(clip_img, return_all_tokens=True)  #B*197*768  //  B*197*512
        clip_visual_features  = target
        # clip_visual_features = target @ self.proj_clip_visual_features
        # clip_visual_features = self.adpater_clip_visual(clip_visual_features)

        target = target.permute(0,2,1)
        target = target @ self.proj_to_77
        target = target.permute(0,2,1)
        target = self.adpater(target)
        
        text_features = self.clip_model.encode_text_feature(target)   #B*77*512

        text_features = text_features.permute(0,2,1)
        text_features = text_features @ self.proj_to_257
        text_features = text_features.permute(0,2,1)
        text_features = text_features @ self.proj_to_768
        text_features = self.adpater2(text_features)

        # clip_feature = clip_visual_features + text_features
        # clip_feature = clip_feature @ self.proj_clip_features
        # clip_feature = self.adpater_clip(clip_feature)

        
        f = torch.cat((mae_memory, text_features), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        memory = f_att * mae_memory + (1 - f_att) * text_features# B*197*768
 
        # # memory = memory+target
        # for blk in self.blocks: 
        #     memory = blk(memory)
        # memory = self.norm(memory) #B*197*768
        
        # memory = torch.cat((memory, target), dim=1)
        # memory = self.adpater(memory)
        
        return memory
    


    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        N, L = tgt.shape #B,tgt_len
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1)) #B*T*768
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1) #B*T*768
        tgt_query = self.dropout(tgt_query)#B*T*768
        return self.decoder(query=tgt_query, content=tgt_emb, memory=memory, query_mask=tgt_query_mask, content_mask=tgt_mask, content_key_padding_mask=tgt_padding_mask)
    
    


    def decodeAR(self,bs, num_steps,memory,tgt_mask,pos_queries,query_mask,testing):
        tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
        tgt_in[:, 0] = self.bos_id

        logits = []
        for i in range(num_steps):
            j = i + 1  # next token index
            # Efficient decoding:
            # Input the context up to the ith token. We use only one query (at position = i) at a time.
            # This works because of the lookahead masking effect of the canonical (forward) AR context.
            # Past tokens have no access to future tokens, hence are fixed once computed.
            tgt_out = self.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                                    tgt_query_mask=query_mask[i:j, :j])
            # the next token probability is in the output's ith token position
            p_i = self.head(tgt_out)
            logits.append(p_i)
            if j < num_steps:
                # greedy decode. add the next token index to the target input
                tgt_in[:, j] = p_i.squeeze().argmax(-1)
                # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                    break

        logits = torch.cat(logits, dim=1)
        return logits

    def decodeNAR(self,bs,memory,pos_queries):
        tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
        tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
        logits = self.head(tgt_out)
        return logits
    
    def decodeRI(self,logits,bs,query_mask,num_steps,memory,tgt_mask,pos_queries):
        query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
        bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
        for i in range(self.refine_iters):
            # Prior context is the previous output.
            tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
            tgt_padding_mask = ((tgt_in == self.eos_id).int().cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
            tgt_out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask,
                                    tgt_query=pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]])
            logits = self.head(tgt_out)
        return logits
        
    def forward(self, mae_images,clip_images, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length) #25
        bs = mae_images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1 #26
        memory = self.encode(mae_images,clip_images) #1*257*768
        
        # Query positions up to `num_steps` #self.pos_queries 1*26*768
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

        if self.decode_ar:
            logits = self.decodeAR(bs, num_steps,memory,tgt_mask,pos_queries,query_mask,testing)
        
        else:
        # No prior context, so input is just <bos>. We query all positions.
            logits = self.decodeNAR(bs,memory,pos_queries)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            logits = self.decodeRI(logits,bs,query_mask,num_steps,memory,tgt_mask,pos_queries)

        
        # codebook_cluster_size = self.quantize.cluster_size
        # zero_cnt = (codebook_cluster_size == 0).sum().item()
        # print(f"Unused code in codebook: {zero_cnt}")
        
        return logits
        # return [logits_1,logits_2,logits_1_3,logits_2_3]

    def gen_tgt_perms(self, tgt):
        """Generate shared permutations for the whole batch.
           This works because the same attention mask can be used for the shorter sequences
           because of the padding mask.
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        # For 4-char sequences and shorter, we generate all permutations and sample from the pool to avoid collisions
        # Note that this code path might NEVER get executed since the labels in a mini-batch typically exceed 4 chars.
        if max_num_chars < 5:
            # Pool of permutations to sample from. We only need the first half (if complementary option is selected)
            # Special handling for max_num_chars == 4 which correctly divides the pool into the flipped halves
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)), device=self._device)[selector]
            # If the forward permutation is always selected, no need to add it to the pool for sampling
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend([torch.randperm(max_num_chars, device=self._device) for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)
        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            # Stack in such a way that the pairs are next to each other.
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        # NOTE:
        # The only meaningful way of permuting the EOS position is by moving it one character position at a time.
        # However, since the number of permutations = T! and number of EOS positions = T + 1, the number of possible EOS
        # positions will always be much less than the number of permutations (unless a low perm_num is set).
        # Thus, it would be simpler to just train EOS using the full and null contexts rather than trying to evenly
        # distribute it across the chosen number of permutations.
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for the reverse direction. This does two things:
        # 1. Reverse context for the characters
        # 2. Null context for [EOS] (required for learning to predict [EOS] in NAR mode)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
        return perms

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float('-inf')  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device) # B * (T+1)
        
        # len_target, char_target =self.mgp_covert.char_encode(labels,self._device)
        # bpe_target = self.mgp_covert.bpe_encode(labels,self._device)
        # wp_target = self.mgp_covert.wp_encode(labels,self._device)

        # Encode the source sequence (i.e. the image codes)
        memory = self.encode(images) # B*197*768
        
        # text = clip.tokenize(labels).to(self._device)
        # text_features,x = self.clip_model.encode_text(text) # B*77*512
        # text_features = text_features.to(self._device) @  self.proj  # B*77*768
        
        
        # print(target.shape)
        # memory = torch.cat((memory, target), dim=1)
        # print("f.shape",f.shape)
        # f_att = torch.sigmoid(self.w_att(f))
        # memory = f_att * memory + (1 - f_att) * text_features# B*197*768
            
        # char_attn, x_char = self.char_tokenLearner(memory)
        # x_char = self.char_head(x_char)
        # char_preds = x_char

        # # bpe
        # bpe_attn, x_bpe = self.bpe_tokenLearner(memory)
        # bpe_preds = self.bpe_head(x_bpe)

        # # wp
        # wp_attn, x_wp = self.wp_tokenLearner(memory)
        # wp_preds = self.wp_head(x_wp)
        
        # char_loss =  F.cross_entropy(char_preds.view(-1, char_preds.shape[-1]), char_target.contiguous().view(-1))
        # bpe_pred_cost =  F.cross_entropy(bpe_preds.view(-1, bpe_preds.shape[-1]), bpe_target.contiguous().view(-1)) 
        # wp_pred_cost =  F.cross_entropy(wp_preds.view(-1, wp_preds.shape[-1]), wp_target.contiguous().view(-1)) 
        
        
        # Prepare the target sequences (input and output)
        tgt_perms = self.gen_tgt_perms(tgt) # k*(T+1)
        tgt_in = tgt[:, :-1] #B*32
        tgt_out = tgt[:, 1:] #B*32
        # # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)#B*32

        loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm) #32*32  ,  32*32 
            out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask) # B * T * 768
            # out = self
            logits = self.head(out).flatten(end_dim=1)  #(B * len(label+1),NUM_CLASS)
            loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        
        # out =  self.gpt2_decoder(memory) 
        logits = self.head(out).flatten(end_dim=1)
        loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
        loss_numel += n        
        loss /= loss_numel
        
        # loss+=char_loss+bpe_pred_cost+wp_pred_cost

        self.log('loss', loss)
        return loss
    def configure_optimizers(self):
        agb = self.trainer.accumulate_grad_batches
        # Linear scaling so that the effective learning rate is constant regardless of the number of GPUs used with DDP.
        # lr_scale = agb * math.sqrt(self.trainer.num_devices) * self.batch_size / 256.
        lr_scale = agb * (self.trainer.num_devices * self.batch_size) / 512  #5*2*16/512
        lr = lr_scale * self.lr

        # https://github.com/mlfoundations/open_clip/blob/b4cf9269b0b11c0eea47cb16039369a46bd67449/src/training/main.py#L171
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n \
                            or "pos_queries" in n or "text_embed" in n
        include = lambda n, p: not exclude(n, p)

        # encoder parameters
        encoder_params = list(self.clip_model.named_parameters())
        enc_gain_or_bias_params = [p for n, p in encoder_params if exclude(n, p) and p.requires_grad]
        enc_rest_params = [p for n, p in encoder_params if include(n, p) and p.requires_grad]

        # decoder parameters
        decoder_params = [(n, p) for n, p in list(self.named_parameters()) if "clip_model" not in n]
        dec_gain_or_bias_params = [p for n, p in decoder_params if exclude(n, p) and p.requires_grad]
        dec_rest_params = [p for n, p in decoder_params if include(n, p) and p.requires_grad]

        # rank_zero_info("[VL4STR] The length of encoder params with and without weight decay is {} and {}, respectively.".format(
        #     len(enc_rest_params), len(enc_gain_or_bias_params)
        # ))
        # rank_zero_info("[VL4STR] The length of decoder params with and without weight decay is {} and {}, respectively.".format(
        #     len(dec_rest_params), len(dec_gain_or_bias_params)
        # ))

        optimizer = torch.optim.AdamW(
            [
                {"params": enc_gain_or_bias_params, "weight_decay": 0., 'lr': lr},
                {"params": enc_rest_params, "weight_decay": self.weight_decay, 'lr': lr},
                {"params": dec_gain_or_bias_params, "weight_decay": 0., 'lr': lr * self.coef_lr},
                {"params": dec_rest_params, "weight_decay": self.weight_decay * self.coef_wd, 'lr': lr * self.coef_lr},
            ],
            lr=lr, betas=(0.9, 0.98), eps=1.0e-6,
            )
        sched = OneCycleLR(optimizer, [lr, lr, lr * self.coef_lr, lr * self.coef_lr],
                            self.trainer.estimated_stepping_batches, pct_start=self.warmup_pct,
                            cycle_momentum=False)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

    # def _freeze_backbones(self):
    #     """set frozen backbones to eval mode"""
    #     for name, mod in self.clip_model.named_modules():
    #         if name.startswith("visual.transformer.resblocks."):
    #             layer_num = int(name.split(".resblocks.")[1].split(".")[0])
    #             if layer_num < self.image_freeze_nlayer:
    #                 mod.eval()

    #             # if self.image_freeze_layer_divisor > 0 and (layer_num + 1) % self.image_freeze_layer_divisor == 0:
    #             #     mod.eval()
    #         elif name.startswith("transformer.resblocks."):
    #             layer_num = int(name.split(".resblocks.")[1].split(".")[0])
    #             if layer_num < self.text_freeze_nlayer:
    #                 mod.eval()

    #     # if self.freeze_language_backbone:
    #     #     self.clip_model.transformer.eval()
    #     #     self.clip_model.ln_final.eval()

    # def freeze_cip_layers(self, image_freeze_nlayer, text_freeze_nlayer, image_freeze_layer_divisor=-1, image_only_fc=False):
    #     """
    #     freeze the parameters of layers with No.layer < image_freeze_nlayer or text_freeze_nlayer,
    #     """
    #     assert image_freeze_nlayer <= 12 and text_freeze_nlayer <=12 and image_freeze_layer_divisor <= 12
    #     if hasattr(self, "clip_model"):
    #         if image_freeze_nlayer > -1:
    #             for name, param in self.clip_model.visual.named_parameters():
    #                 # top layers always need to train
    #                 if name.startswith("ln_post.") or name.startswith("proj") or name.startswith("conv1") or name.startswith("ln_pre"):
    #                     continue
    #                 elif name.startswith("transformer.resblocks."):
    #                     layer_num = int(name.split(".resblocks.")[1].split(".")[0])
    #                     if layer_num >= image_freeze_nlayer:
    #                         continue
    #                 param.requires_grad = False

    #         #### freeze the layers which the index can be divided by image_freeze_layer_divisor
    #         # if image_freeze_layer_divisor > 0:
    #         #     for name, param in self.clip_model.visual.named_parameters():
    #         #         if name.startswith("transformer.resblocks."):
    #         #             layer_num = int(name.split(".resblocks.")[1].split(".")[0])
    #         #             if (layer_num + 1) % image_freeze_layer_divisor == 0:
    #         #                 param.requires_grad = False

    #         #### only train the top fc layer
    #         # if image_only_fc:
    #         #     for name, param in self.clip_model.visual.named_parameters():
    #         #         if "out_proj" in name or "conv1" in name or name.startswith("ln_post.") or name.startswith("proj"):
    #         #             continue
    #         #         param.requires_grad = False

    #         if text_freeze_nlayer > -1:
    #             for name, param in self.clip_model.named_parameters():
    #                 # top layers always need to train
    #                 if name.startswith("ln_final.") or name.startswith("text_projection") or name.startswith("visual"):
    #                     continue
    #                 elif name.startswith("transformer.resblocks."):
    #                     layer_num = int(name.split(".resblocks.")[1].split(".")[0])
    #                     if layer_num >= text_freeze_nlayer:
    #                         continue
    #                 param.requires_grad = False

    #         # freeze the whole backbones and related parameters
    #         if text_freeze_nlayer >= 12:
    #             for n, p in self.clip_model.named_parameters():
    #                 # exclude visual parameters
    #                 if "visual" not in n:                
    #                     if "transformer" in n or "token_embedding" in n or "ln_final" in n or "text_projection" in n:
    #                         p.requires_grad = False
            
    #         if image_freeze_nlayer >= 12:
    #             for n, p in self.clip_model.visual.named_parameters():
    #                 p.requires_grad = False               

    # def train(self, mode=True):
    #     """
    #     Override the default train() to freeze the BN parameters
    #     """
    #     super().train(mode)
    #     if mode:
    #         self._freeze_backbones()

    #     return self

