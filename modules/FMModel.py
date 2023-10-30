import torch
from torch import nn
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils.KLLoss import KLLoss
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def trunc_normal_(x, mean=0., std=1.):
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class TAggregate(nn.Module):
    def __init__(self, clip_length=None, embed_dim=2048, n_layers=6):
        super(TAggregate, self).__init__()
        self.clip_length = clip_length
        drop_rate = 0.
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(
            embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        nvids = x.shape[0]

        cls_tokens = self.cls_token.expand(nvids, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)

        return o[0]


class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))


class FMCLIP(nn.Module):
    def __init__(self, sim_head, clip_state_dict, T):
        super().__init__()
        self.sim_header = sim_head
        self.T = T
        assert sim_head in ["meanP", "LSTM", "Transf", "Conv_1D", "Transf_cls"]

        if self.sim_header == "LSTM" or self.sim_header == "Transf" or self.sim_header == "Transf_cls" or self.sim_header == "Conv_1D" :
            embed_dim = clip_state_dict["text_projection"].shape[1]

            context_length = clip_state_dict["positional_embedding"].shape[0]
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64

            transformer_layers = len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
        if self.sim_header == "Transf" :
            self.transformer = TemporalTransformer(width=embed_dim, layers=6, heads=transformer_heads)
        if self.sim_header == "LSTM":
            self.lstm_visual = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim,
                                       batch_first=True, bidirectional=False, num_layers=1)

        self.apply(self.init_weights)

        if self.sim_header == "Transf_cls":
            self.transformer = TAggregate(clip_length=self.T, embed_dim=embed_dim, n_layers=6)

        if self.sim_header == 'Conv_1D' :
            self.shift = nn.Conv1d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False)
            weight = torch.zeros(embed_dim, 1, 3)
            weight[:embed_dim // 4, 0, 0] = 1.0
            weight[embed_dim // 4:embed_dim // 4 + embed_dim // 2, 0, 1] = 1.0
            weight[-embed_dim // 4:, 0, 2] = 1.0
            self.shift.weight = nn.Parameter(weight)

        self.loss_item = KLLoss()
        self.loss_score = KLLoss()
        # self.loss_fct = nn.CrossEntropyLoss()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def video_encoder(self, x, return_hidden=False):
        b, t, c = x.size()
        x = x.contiguous()
        if self.sim_header == "meanP":
            pass
        elif self.sim_header == 'Conv_1D':
            x_original = x
            x = x.view(-1, c, t)
            x = self.shift(x.float())
            x = x.permute(0, 2, 1)
            x = x.type(x_original.dtype) + x_original

        elif self.sim_header == "Transf":
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) + x_original

        elif self.sim_header == "LSTM":
            x_original = x
            x, _ = self.lstm_visual(x.float())
            self.lstm_visual.flatten_parameters()
            x = torch.cat((x, x_original[:, x.size(1):, ...].contiguous()), dim=1)
            x = x.type(x_original.dtype) + x_original
        elif self.sim_header == "Transf_cls":
            x_original = x
            return self.transformer(x).type(x_original.dtype)

        else:
            raise ValueError('Unknown optimizer: {}'.format(self.sim_header))
        if return_hidden:
            return x, x.mean(dim=1, keepdim=False)
        else:
            return x.mean(dim=1, keepdim=False)

    def cross_modal_sim(self, text, video, logit_scale):
        cross_sim = logit_scale * torch.matmul(video, text.t())
        # cross_sim = logit_scale * torch.sum(s * torch.softmax(s / 1e-2, dim=1), dim=1)
        logpt = F.log_softmax(cross_sim, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

    def coarse_grained_sim(self, video, text, item, score, logit_scale, item_truth, score_truth):
        bs, dim = video.shape
        nb_item, dim = item.shape

        video_item = video @ item.t()  # (bs, 4)
        text_item = text @ item.t()
        fuse_item = (video_item + text_item) / 2
        # fuse_item = text_item

        score = score.view(-1, dim)  # (4 * 3, d)
        video_score = torch.matmul(video, score.t())
        text_score = torch.matmul(text, score.t())
        fuse_score = (video_score + text_score) / 2
        # fuse_score = text_score
        fuse_score = fuse_score.view(bs, nb_item, -1)
        return fuse_item, fuse_score

    def hierachy_sim(self, fuse_item, fuse_score, video, item, score, logit_scale, item_truth, score_truth):
        vs2vi = torch.softmax(fuse_score, dim=-1).max(-1, keepdim=False)[0] # (bs, 4)
        vs2vi_sim = torch.matmul(vs2vi.unsqueeze(-1) * item.unsqueeze(0), video.unsqueeze(1).permute(0,2,1)).squeeze()  # (bs, 4)
        vi = (torch.softmax(vs2vi_sim, dim=-1) + torch.softmax(fuse_item, dim=-1)) / 2   # (bs, 4)

        bs, nb_item, nb_score = fuse_score.shape
        vi2vs = fuse_item.unsqueeze(-1).repeat(1, 1, 3).unsqueeze(-1) * score.view(nb_item, nb_score, -1).unsqueeze(0)   # (bs, 4, 3, d)
        vi2vs_sim = torch.matmul(vi2vs.view(bs, nb_item*nb_score, -1), video.unsqueeze(-1)).squeeze() # (bs, 4 * 3)
        vs = (torch.softmax(vi2vs_sim, dim=-1) + torch.softmax(fuse_score, dim=-1).view(bs, -1)) / 2
        filter = (vi == vi.max(dim=1, keepdim=True)[0]) * 1.0
        vs = (filter.unsqueeze(-1) * vs.view(bs, nb_item, nb_score)).view(bs, -1).type(vi.dtype)

        vs2vi_loss = self.loss_item(logit_scale * vi, item_truth)
        vi2vs_loss = self.loss_score(logit_scale * vs, score_truth)

        # vs2vi_loss = self.loss_fct(vi, item_truth)
        # vi2vs_loss = self.loss_fct(vs, score_truth)
        loss = vs2vi_loss + vi2vs_loss

        return vi, vs, loss

    def forward(self, frame, motion, item_des_emb, score_des_emb, logit_scale, item_truth, score_truth):
        '''
        INPUT:
            frame: (bs, nb_frame, d)
            motion: (bs, d) body movements
            item_des_emb: (nb_item, d)
            score_des_emb: (nb_score, d)
            item_truth: (bs, 4)
            score_truth: (bs, 12)
        OUTPUT:
            video_emb: (bs, d)
        '''
        video_emb = self.video_encoder(frame)

        motion = motion / motion.norm(dim=-1, keepdim=True)
        video_emb = video_emb / video_emb.norm(dim=-1, keepdim=True)
        item_des_emb = item_des_emb / item_des_emb.norm(dim=-1, keepdim=True)
        score_des_emb = score_des_emb / score_des_emb.norm(dim=-1, keepdim=True)

        cross_loss = self.cross_modal_sim(motion, video_emb, logit_scale)
        fuse_item, fuse_score = self.coarse_grained_sim(video_emb, motion, item_des_emb, score_des_emb, logit_scale, item_truth, score_truth)
        vs2vi, vi2vs, hierachy_loss = self.hierachy_sim(fuse_item, fuse_score, video_emb, item_des_emb, score_des_emb, logit_scale, item_truth, score_truth)
        print(vs2vi, vi2vs)
        # bs, nb_item, nb_score = fuse_score.shape
        # filter = (fuse_item == fuse_item.max(dim=1, keepdim=True)[0]) * 1.0
        # fuse_score = (filter.unsqueeze(-1) * fuse_score.view(bs, nb_item, nb_score)).view(bs, -1).type(fuse_item.dtype)
        # hierachy_loss = self.loss_item(logit_scale * fuse_item, item_truth) + self.loss_score(logit_scale * fuse_score, score_truth)

        loss = cross_loss + hierachy_loss
        return vs2vi, vi2vs, loss
