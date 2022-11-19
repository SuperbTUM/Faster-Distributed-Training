import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Transformer(nn.Module):
    '''
    Creating a Transformer model for a classification task
    Arguments:
    n_class     - number of possible classes in final prediction
    vocab       - vocabulary size of tokenizer, used by embedding layer
    n_layers    - number of encoder layers used
    h           - number of heads in MHA module
    d_model     - dimensions of features per token throughout whole model
    d_ff        - dimensions of features in middle layer of FFN module
    d_hidden    - dimensions of features in classifier hidden layer
    maxlen      - max possible token length for input sentence
    dropout_... - dropout rate for respective layer
    '''
    def __init__(self,
                 n_class,
                 vocab,
                 n_layers=6,
                 h=8,
                 d_model=512,
                 d_ff=1024,
                 d_hidden=1024,
                 maxlen=512,
                 dropout_encodings=0.1,
                 dropout_connection_attention=0.1,
                 dropout_connection_ffn=0.1,
                 dropout_attention=0.1,
                 dropout_ffn=0.1):
        super(Transformer, self).__init__()
        self.input_embeddings = Embeddings(d_model, vocab, maxlen)
        self.input_encodings = PositionalEncoding(d_model, dropout_encodings, maxlen)
        # self.layernorm = LayerNorm(d_model)
        self.sublayer_attention = nn.ModuleList()
        self.sublayer_ffn = nn.ModuleList()
        for _ in range(n_layers):
            self.sublayer_attention.append(sublayerConnectionAttention(
                h, d_model, dropout_attention, dropout_connection_attention))
            self.sublayer_ffn.append(sublayerConnectionFFN(
                d_model, d_ff, dropout_ffn, dropout_connection_ffn))
        self.classifier = Classifier(d_model, d_hidden, n_class)
        self.n_layers = n_layers

        self.init_params()

    def forward(self, x, mask=None):
        embeddings = self.input_embeddings(x)
        encodings = self.input_encodings(embeddings)
        x = embeddings + encodings
        for i in range(self.n_layers):
            x = self.sublayer_attention[i](x, mask)
            x = self.sublayer_ffn[i](x)
            # x = self.layernorm(x)
        cls_repre = x[:, 0, :]
        outputs = self.classifier(cls_repre)
        return outputs

    def init_params(self, default_initialization=False):
        # Not mentioned in the paper, but other implementations used xavier.
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

class PositionalEncoding(nn.Module):
    '''
    Generating positional encoding to preserve spatial information in input tokens
    Arguments:
    d_model - dimensions of features per token throughout whole model
    dropout - dropout rate for input data
    max_len - max possible token length for input sentence
    '''
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros((max_len, d_model), device=device)
        position = torch.arange(0, max_len, device=device).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * scale)
        pe[:, 1::2] = torch.cos(position * scale)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class Embeddings(nn.Module):
    '''
    Transforming token id into d_model features
    Arguments:
    d_model - dimensions of features per token throughout whole model
    vocab   - vocabulary size of tokenizer, input dim of embedding layer
    maxlen  - max possible token length for input sentence
    '''
    def __init__(self, d_model, vocab, maxlen):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(vocab, d_model)
        self.pos_embedding = nn.Embedding(maxlen, d_model)
        self.d_model = d_model
        self.maxlen = maxlen

    def forward(self, x):
        positions = self.pos_embedding(torch.arange(start=0, end=self.maxlen, device=device))[:x.size(1), :]
        tokens = self.token_embedding(x)
        return (positions + tokens) * math.sqrt(self.d_model)

class PositionalWiseFFN(nn.Module):
    '''
    FFN module, contains two linear layer and one dropout layer
    Arguments:
    d_model - dimensions of features per token throughout whole model
    d_ffn   - dimensions of features in middle layer of FFN module
    dropout - dropout rate for output
    '''
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionalWiseFFN, self).__init__()
        # self.w_1 = nn.Linear(d_model, d_ff)
        # self.w_2 = nn.Linear(d_ff, d_model)
        self.fusedmlp = FusedMLP(d_model, d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # return self.w_2(self.dropout(F.relu(self.w_1(x))))
        return self.dropout(self.fusedmlp(x))


def ScaledDotProduct(query, key, values, dropout=None, mask=None):
    '''
    Realizing scaled dot product between query, key and values
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.squeeze(1)
        scores = scores.masked_fill_(mask == 0, -1e-9)
    p_atten = F.softmax(scores, dim=-1)
    if dropout:
        p_atten = dropout(p_atten)
    return torch.matmul(p_atten, values), p_atten


class MultiheadAttention(nn.Module):
    '''
    Class of multi-head attention, break input features into h heads,
    do attention, then concatenated together.
    Arguments:
    h       - number of heads
    d_model - dimensions of features per token throughout whole model
    dropout - dropout rate for output
    '''
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.heads = nn.ModuleList()
        self.attn = None
        for _ in range(3):
            self.heads.append(nn.Linear(d_model, d_model, device=device))
        self.output = nn.Linear(d_model, d_model, device=device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.heads, (query, key, value))]
        x, self.attn = ScaledDotProduct(query, key, value, self.dropout, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = self.output(x)
        return x


class LayerNorm(nn.Module):
    '''Construct a layernorm module'''
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features, device=device))
        self.b_2 = nn.Parameter(torch.zeros(features, device=device))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class sublayerConnectionAttention(nn.Module):
    '''Construct a sublayer with MHA, layernorm, dropout and shorcut'''
    def __init__(self, h, d_model, dropout_head=0.1, dropout_connection=0.1):
        super(sublayerConnectionAttention, self).__init__()
        self.multiheads = MultiheadAttention(h, d_model, dropout_head)
        self.layernorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_connection)

    def forward(self, x, mask=None):
        original = x
        x = self.layernorm(x)
        x = self.multiheads(x, x, x, mask)
        x = self.dropout(x)
        return x + original


class sublayerConnectionFFN(nn.Module):
    '''Construct a sublayer with FFN, layernorm, dropout and shorcut'''
    def __init__(self, d_model, d_ff, dropout_ffn=0.1, dropout_connection=0.1):
        super(sublayerConnectionFFN, self).__init__()
        self.ffn = PositionalWiseFFN(d_model, d_ff, dropout_ffn).to(device)
        self.layernorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_connection)

    def forward(self, x):
        original = x
        x = self.layernorm(x)
        x = self.dropout(self.ffn(x))
        return x + original


class Classifier(nn.Module):
    '''Final classifier with one linear layer'''
    def __init__(self, d_model, d_hidden, n_class):
        super(Classifier, self).__init__()
        # self.hidden = nn.Linear(d_model, d_hidden)
        # self.classifier = nn.Linear(d_hidden, n_class)
        self.classifier = FusedMLP(d_model, d_hidden, n_class)

    def forward(self, x):
        # return self.classifier(F.relu(self.hidden(x)))
        return self.classifier(x)


class MLPScratch(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, X, W1, b1, W2, b2):
        ctx.save_for_backward(X, W1, b1, W2)
        linear = F.linear(X, W1, b1)
        activated = F.relu(linear)
        output = F.linear(activated, W2, b2)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_out):
        X, W1, b1, W2 = ctx.saved_tensors
        linear = F.linear(X, W1, b1)
        activated = F.relu(linear)
        # output = F.linear(activated, W2, b2)

        grad_b2 = torch.mean(grad_out, dim=0, keepdim=True)
        if grad_out.ndim == 2:
            grad_out_transpose = grad_out.T
        else:
            grad_out_transpose = grad_out.permute(*(i for i in range(grad_out.ndim - 2)),
                                                  grad_out.ndim - 1, grad_out.ndim - 2).contiguous()
        grad_W2 = grad_out_transpose @ activated
        grad_activated = grad_out @ W2
        grad_activated_shape = grad_activated.size()
        grad_activated_flattened = grad_activated.view(-1)
        grad_before_activated = torch.empty_like(grad_activated_flattened)
        for i in range(grad_before_activated.size(0)):
            grad_before_activated[i] = grad_activated_flattened[i] if grad_activated_flattened[i] > 0 else 0
        grad_before_activated = grad_before_activated.reshape(grad_activated_shape)
        grad_b1 = torch.mean(grad_before_activated, dim=0, keepdim=True)

        if grad_out.ndim == 2:
            grad_before_activated_transpose = grad_before_activated.T
        else:
            grad_before_activated_transpose = grad_before_activated.permute(
                *(i for i in range(grad_before_activated.ndim - 2)),
                grad_before_activated.ndim - 1, grad_before_activated.ndim - 2).contiguous()
        grad_W1 = grad_before_activated_transpose @ X

        grad_X = grad_before_activated @ W1
        return grad_X, grad_W1, grad_b1, grad_W2, grad_b2


class FusedMLP(nn.Module):
    def __init__(self, input_channel, hidden_channel, output_channel, device=None, dtype=None):
        super(FusedMLP, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        hidden_shape_weight = (hidden_channel, input_channel)
        hidden_shape_bias = (1, hidden_channel)
        output_shape_weight = (output_channel, hidden_channel)
        output_shape_bias = (1, output_channel)
        self.W1 = nn.Parameter(torch.empty(*hidden_shape_weight, **factory_kwargs))
        self.b1 = nn.Parameter(torch.empty(*hidden_shape_bias, **factory_kwargs))
        self.W2 = nn.Parameter(torch.empty(*output_shape_weight, **factory_kwargs))
        self.b2 = nn.Parameter(torch.empty(*output_shape_bias, **factory_kwargs))

        self.reset_parameters()

    def forward(self, X):
        return MLPScratch.apply(X, self.W1, self.b1, self.W2, self.b2)

    def reset_parameters(self):
        self.W1 = torch.nn.init.xavier_uniform_(self.W1)
        self.W2 = torch.nn.init.xavier_uniform_(self.W2)
