import json
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate_fn_map, collate
from torchvision import transforms
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from util.loss import FocalLoss
from net.image_encoding import ImageEncoderEfficientNet
from net.question_encoding import QuestionEncoderBERT
from einops import rearrange
from hflayers import Hopfield,HopfieldPooling
from math import sqrt
from functools import partial
from net.initialization import init_weights
from net.stm import STM
from net.mca import MCA_ED,make_mask
from net.ep_memory import MACUnit 

import torch
import torch.nn as nn
import torch.nn.functional as fn
class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, embed_dim:int =768,k=30):
        super().__init__()
        self.W_b = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.randn(k, embed_dim))
        self.W_q = nn.Parameter(torch.randn(k, embed_dim))
        self.w_hv = nn.Parameter(torch.randn(k, k))
        self.w_hq = nn.Parameter(torch.randn(k, k))
        self.tanh = nn.Tanh()
        #self.W_w = nn.Parameter(torch.randn(embed_dim, embed_dim))
        #self.W_p = nn.Parameter(torch.randn(embed_dim*2, embed_dim))
        #self.W_s = nn.Parameter(torch.randn(embed_dim*2, embed_dim))

        # self.W_w = nn.Linear(embed_dim, embed_dim)
        # self.W_p = nn.Linear(embed_dim*2, embed_dim)
        # self.W_s = nn.Linear(embed_dim*2, embed_dim)
    def forward(self, V, Q):  # V : B x 512 x 196, Q : B x L x 512
        V = V.permute(0,2,1)
        C = torch.matmul(Q, torch.matmul(self.W_b, V)) # B x L x 196

        H_v = self.tanh(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))                            # B x k x 196
        H_q = self.tanh(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))           # B x k x L

        #a_v = torch.squeeze(fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)) # B x 196
        #a_q = torch.squeeze(fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)) # B x L

        a_v = fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2) # B x 1 x 196
        a_q = fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2) # B x 1 x L

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1))) # B x 512
        q = torch.squeeze(torch.matmul(a_q, Q))                  # B x 512

        return v, q
def max_neg_value(t):
    return -torch.finfo(t.dtype).max
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
class LayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6
        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x):
        return self.fn(x) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn, sandwich = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim) if sandwich else nn.Identity()
        self.fn = fn

    def forward(self, x):
        x = self.norm(x)
        x = self.fn(x)
        return self.norm_out(x)
# implementation of 2D positional encoding from https://github.com/gaopengcuhk/Stable-Pix2Seq
def create_pos_encoding(args):
    temperature = 10000
    hidden_size = args.hidden_size
    max_position_embeddings = args.max_position_embeddings  # args.num_image_tokens + 3 + args.num_question_tokens
    num_pos_feats = hidden_size // 4
    img_mask = torch.ones((1, args.img_feat_size, args.img_feat_size))  # bs, img_tokens
    y_embed = img_mask.cumsum(1, dtype=torch.float32)
    x_embed = img_mask.cumsum(2, dtype=torch.float32)

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t

    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_img = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    pos_img = pos_img.flatten(2)

    # extend pos_img with zeros to match the size of pos_seq
    pos_img = torch.cat(
        (pos_img, torch.zeros((pos_img.shape[0], pos_img.shape[1], max_position_embeddings - pos_img.shape[2]), device=pos_img.device)), dim=2)
    # switch last two dimensions
    pos_img = pos_img.permute(0, 2, 1)

    return pos_img


class MyBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size // 2)  # other half is reserved for spatial image embeddings
        if args.progressive:
            self.token_type_embeddings = nn.Embedding(4, config.hidden_size)  # image, history Q, history A, current question
        else:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.img_pos_embeddings = create_pos_encoding(self.args)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values_length: int = 0,

    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            position_embeddings = torch.cat((self.img_pos_embeddings.to(position_embeddings.device), position_embeddings),
                                            dim=-1)  # add image position embeddings and sequence position embeddings
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class HopfieldLayer(nn.Module):
    def __init__(self,dim,n_prototype=1000,dropout=0.1):
        super().__init__()
        self.beta = 1./sqrt(dim)
        self.lookup_matrix = nn.Linear(dim, n_prototype, bias = False)
        self.content_matrix = nn.Linear(n_prototype,dim,bias = False)
        self.softmax = torch.softmax
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        lookup = self.softmax(self.lookup_matrix(x) * self.beta, dim=-1)
        content = self.content_matrix(lookup)
        return self.dropout(content)

class SelfAttention(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        self.beta = 1./sqrt(dim_head)
        self.heads = heads
        self.to_qk = nn.Linear(dim, inner_dim*2, bias = False)
        self.to_out = nn.Sequential(
                        nn.Linear(inner_dim, dim),
                        nn.Dropout(dropout),
                        )
        self.softmax = torch.softmax
        
    def forward(self, x):
        h, device = self.heads, x.device
        qk = self.to_qk(x).chunk(2, dim = -1)       
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qk)        

        bqk = torch.einsum('b h i d, b h j d -> b h i j', q* self.beta, k)
        mask_value = max_neg_value(bqk)

        # causality
        i, j = bqk.shape[-2:]
        mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()    
        bqk = self.softmax(bqk.masked_fill_(mask, mask_value), dim=-1)

        bqkk = torch.einsum('b h i j, b h j d -> b h i d', bqk, k)
        out = rearrange(bqkk, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class SelfAttention_qkv(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        self.beta = 1./sqrt(dim_head)
        self.heads = heads
        self.q = nn.Linear(dim, inner_dim, bias = False)
        self.k = nn.Linear(dim, inner_dim, bias = False)
        self.v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Sequential(
                        nn.Linear(inner_dim, dim),
                        nn.Dropout(dropout),
                        )
        self.softmax = torch.softmax
        
    def forward(self, x):
        h, device = self.heads, x.device
        q = rearrange(self.q(x), 'b n (h d) -> b h n d', h = h)
        k = rearrange(self.k(x), 'b n (h d) -> b h n d', h = h)
        v = rearrange(self.v(x), 'b n (h d) -> b h n d', h = h)         

        bqk = torch.einsum('b h i d, b h j d -> b h i j', q* self.beta, k)
        mask_value = max_neg_value(bqk)

        # causality
        i, j = bqk.shape[-2:]
        mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()    
        bqk = self.softmax(bqk.masked_fill_(mask, mask_value), dim=-1)

        bqkv = torch.einsum('b h i j, b h j d -> b h i d', bqk, v)
        out = rearrange(bqkv, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class PrototypeBlock(nn.Module):
    def __init__(self,dim,n_block,heads=8,dim_head=64,num_prototype=1000):    
        super().__init__()    
        self.layers = nn.ModuleList([])    
        for i in range(n_block):                 
            self.layers.append(nn.ModuleList([
                LayerScale(dim,i+1,PreNorm(dim,HopfieldLayer(dim,num_prototype))),
                LayerScale(dim,i+1,PreNorm(dim,SelfAttention(dim,heads=heads,dim_head=dim_head))),
            ]))
        pos_emb = None      
        self.register_buffer('pos_emb', pos_emb)
    def forward(self, x):
        for (f, g) in self.layers:
            x = x + f(x)
            x = x + g(x)
        return x
class MyBertModel(BertModel):
    """

    Overwrite BERTModel in order to adapt the positional embeddings for images
    """

    def __init__(self, config, add_pooling_layer=True, args=None):
        super().__init__(config)
        self.config = config

        self.embeddings = MyBertEmbeddings(config, args=args)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        self.image_encoder = ImageEncoderEfficientNet(args)
        self.question_encoder = QuestionEncoderBERT(args)
        self.gui_attn = MCA_ED(args)
        # self.associate_question_memory = Hopfield(input_size=args.hidden_size,
        #                                 hidden_size= args.hidden_size,
        #                                 num_heads=8, 
        #                                 #normalize_hopfield_space = True,                          
        #                                 #stored_pattern_as_static=True,
        #                                 scaling=args.scaling,
        #                                 dropout=args.classifier_hopfield,
        #                             )
        # self.associate_image_memory = Hopfield(input_size=args.hidden_size,
        #                                 hidden_size= args.hidden_size,
        #                                 num_heads=args.heads, 
        #                                 #normalize_hopfield_space = True,                          
        #                                 #stored_pattern_as_static=True,
        #                                 scaling=args.scaling,
        #                                 dropout=args.classifier_hopfield,
        #                             )
        self.co_attn = CoattentionNet()
        self.memory = STM(args.hidden_size,args.hidden_size)
        #self.visio_linguistic = SelfAttention(dim=args.hidden_size,dim_head=128,dropout=0.4)
        #self.associate_question_memory = STM(input_size=args.hidden_size,output_size=args.hidden_size,out_att_size=args.hidden_size)
        #self.associate_image_memory = STM(input_size=args.hidden_size,output_size=args.hidden_size,out_att_size=args.hidden_size)
        #self.associate_memory = STM(input_size=args.hidden_size,output_size=args.hidden_size,slot_size=args.slot_size*2,mlp_size=args.mlp_size*2,rel_size=args.rel_size*2)
        self.fusion = PrototypeBlock(dim=args.hidden_size,n_block=args.n_block,num_prototype=1024)
        # self.pool = HopfieldPooling(input_size=args.hidden_size,
        #                             hidden_size=16,
        #                             scaling=args.scaling,
        #                             quantity=args.quantity) 
        self.classifier = nn.Sequential(
                nn.Dropout(args.classifier_dropout),
                nn.Linear(args.hidden_size, 512),
                nn.ReLU(),
                nn.Linear(512, args.num_classes))

    def forward(self, img, input_ids, q_attn_mask):

        image_features = self.image_encoder(img)
        text_features = self.question_encoder(input_ids, q_attn_mask)
        t_mask = make_mask(text_features)
        i_mask = make_mask(image_features)
        t,i = self.gui_attn(text_features,image_features,t_mask,i_mask)
        v,q = self.co_attn(image_features,text_features)
        m = v+q
        #h = torch.cat((image_features, text_features), dim=1)
        m,(_,_,_) = self.memory(m.permute(1,0,2))
        #att_r_t,(_,_,_) = self.associate_question_memory(text_features)
        #att_r_f,(_,_,_) = self.associate_memory(h)
        # i = i + self.associate_image_memory((image_features,i,image_features))
        # t = t + self.associate_question_memory((text_features,t,text_features))
        # image_features = image_features + i
        # text_features = text_features + t
        #c_vl = self.visio_linguistic(h)
        image_features = image_features + i
        text_features = text_features + t
        enriched_c = torch.cat((image_features,m.unsqueeze(1), text_features), dim=1)
        # h= h + enriched_c
        
        #out = torch.cat((att_r_i, att_r_t,att_r_f),dim=1)
        out = self.fusion(enriched_c)
        #out = self.pool(out)
        logits = self.classifier(out.mean(1))

        return logits


class ModelWrapper(pl.LightningModule):
    def __init__(self, args, train_df=None, val_df=None):
        super(ModelWrapper, self).__init__()
        self.args = args
        self.model = Model(args)
        self.train_df = train_df
        self.val_df = val_df

        self.loss_fn = FocalLoss(sqrt(3))

        self.train_preds = []
        self.val_preds = []
        self.val_infos = []
        self.val_soft_scores = []
        self.val_gen_preds = []
        self.train_gen_preds = []
        self.train_targets = []
        self.train_soft_scores = []
        self.train_infos = []
        self.train_gen_labels = []
        self.val_targets = []
        self.val_gen_labels = []
        self.train_answer_types = []
        self.val_answer_types = []

    def forward(self, img, input_ids, q_attn_mask):
        return self.model(img, input_ids, q_attn_mask)

    def training_step(self, batch, batch_idx):
        img, question_token, q_attention_mask, target,answer_type  = batch
        question_token = question_token.squeeze(1)
        q_attention_mask = q_attention_mask.squeeze(1)

        out = self(img, question_token, q_attention_mask)

        logits = out
        pred = logits.softmax(1).argmax(1).detach()

        self.train_preds.append(pred)
        self.train_targets.append(target)

        self.train_answer_types.append(answer_type)

        loss = self.loss_fn(logits.softmax(1), target)

        self.log('Loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, question_token, q_attention_mask, target,answer_type  = batch

        question_token = question_token.squeeze(1)
        q_attention_mask = q_attention_mask.squeeze(1)

        out= self(img, question_token, q_attention_mask)

        logits = out

        pred = logits.softmax(1).argmax(1).detach()
        self.val_soft_scores.append(logits.softmax(1).detach())


        self.val_preds.append(pred)
        self.val_targets.append(target)
        self.val_answer_types.append(answer_type)


        loss = self.loss_fn(logits[target != -1].softmax(1), target[target != -1])

        self.log('Loss/val', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=14, gamma=0.3)
        return [optimizer],[scheduler]

    def on_training_epoch_end(self, outputs) -> None:
        preds = torch.cat(self.train_preds).cpu().numpy()
        targets = torch.cat(self.train_targets).cpu().numpy()


        answer_types = torch.cat(self.train_answer_types).cpu().numpy()
        total_acc = (preds == targets).mean() * 100.
        closed_acc = (preds[answer_types == 0] == targets[answer_types == 0]).mean() * 100.
        open_acc = (preds[answer_types == 1] == targets[answer_types == 1]).mean() * 100.

        self.logger.experiment.add_scalar('Acc/train', total_acc, self.current_epoch)
        self.logger.experiment.add_scalar('ClosedAcc/train', closed_acc, self.current_epoch)
        self.logger.experiment.add_scalar('OpenAcc/train', open_acc, self.current_epoch)

        self.train_preds = []
        self.train_targets = []
        self.train_soft_scores = []
        self.train_infos = []
        self.train_answer_types = []
        self.train_gen_labels = []
        self.train_gen_preds = []

    def on_validation_epoch_end(self) -> None:
        preds = torch.cat(self.val_preds).cpu().numpy()
        targets = torch.cat(self.val_targets).cpu().numpy()

        answer_types = torch.cat(self.val_answer_types).cpu().numpy()
        total_acc = (preds == targets).mean() * 100.
        closed_acc = (preds[answer_types == 0] == targets[answer_types == 0]).mean() * 100.
        open_acc = (preds[answer_types == 1] == targets[answer_types == 1]).mean() * 100.

        self.logger.experiment.add_scalar('Acc/val', total_acc, self.current_epoch)
        self.logger.experiment.add_scalar('ClosedAcc/val', closed_acc, self.current_epoch)
        self.logger.experiment.add_scalar('OpenAcc/val', open_acc, self.current_epoch)

        # clean accuracies without samples not occuring in the training set
        total_acc1 = (preds[targets != -1] == targets[targets != -1]).mean() * 100.

        closed_acc1 = (preds[targets != -1][answer_types[targets != -1] == 0] ==
                        targets[targets != -1][answer_types[targets != -1] == 0]).mean() * 100.
        open_acc1 = (preds[targets != -1][answer_types[targets != -1] == 1] ==
                        targets[targets != -1][answer_types[targets != -1] == 1]).mean() * 100.
        # log
        self.log('Acc/val_clean', total_acc1, on_step=False, on_epoch=True, prog_bar=True, logger=True)  # for saving

        self.logger.experiment.add_scalar('Acc/val_clean', total_acc1, self.current_epoch)
        self.logger.experiment.add_scalar('ClosedAcc/val_clean', closed_acc1, self.current_epoch)
        self.logger.experiment.add_scalar('OpenAcc/val_clean', open_acc1, self.current_epoch)

        self.val_preds = []
        self.val_targets = []
        self.val_soft_scores = []
        self.val_answer_types = []
        self.val_gen_labels = []
        self.val_gen_preds = []
        self.val_infos = []
