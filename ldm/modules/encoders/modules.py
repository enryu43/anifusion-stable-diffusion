import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia
import numpy as np

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


import numpy as np
import transformers
import hashlib
import zlib
from base64 import urlsafe_b64decode as b64d
import six

class TagTokenizer(object):
    def __init__(self, ar_model_path, max_length, device='cuda'):
        self.device = device
        self.max_length = max_length

        # TODO: this is hacky, can we do smth nicer? OmegaConf doesn't seem to allow specifying relative paths.
        import os
        our_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(our_dir, 'tags.config'), 'rb') as f:
          raw_tags = f.read()
        id2tag = zlib.decompress(b64d(raw_tags))
        id2tag = six.ensure_str(id2tag).split(',')
        tag2id = {t: i for i, t in enumerate(id2tag)}
        id2tag = {i: t for t, i in tag2id.items()}
        
        self.vocab_size = len(id2tag)
        self.tag2id = tag2id
        self.id2tag = id2tag
        self.cache = {}
        self.bos = 2999
        self.eos = 2999
        self.pad = 2999
        self.pad_token_id = self.pad

        vocab_size = len(id2tag)
        self.BOS = vocab_size
        self.EOS = vocab_size + 1
        self.PAD = vocab_size + 2
       
        self.model = transformers.AutoModelForCausalLM.from_pretrained(ar_model_path).to(self.device)
        
        BLACKLIST = 'censored,mosaic_censoring,photoshop_(medium),comic,monochrome,translated,greyscale,jpeg_artifacts,hard_translated,bar_censor,lowres,bad_pixiv_id,bad_id,translation_request,transparent_background,realistic,photo_(medium)'.split(',')
        META = 'rating_e,rating_s,rating_q,score_perc_10,score_perc_20,score_perc_40,score_perc_60,score_perc_80,score_perc_90,score_perc_100,adjusted_score_perc_10,adjusted_score_perc_20,adjusted_score_perc_40,adjusted_score_perc_60,adjusted_score_perc_80,adjusted_score_perc_90,adjusted_score_perc_100'.split(',')
        self.BLACKLIST = [[tag2id[t]] for t in BLACKLIST]
        self.META = [tag2id[t] for t in META]


    def token_id(self, token):
        if token in self.tag2id:
          return self.tag2id[token]
        h = self.pad + 1 + (int(hashlib.sha1(token.encode('utf-8')).hexdigest(), 16) % (10 ** 9))
        self.cache[h] = token
        return h

    def token(self, token_id):
        if token_id in self.id2tag:
          return self.id2tag[token_id]
        assert token_id in self.cache
        return self.cache[token_id]

    def generate(self, prompt, max_augment=50, seed=None):
      result = []
      ids = [self.BOS]
      index_mapping = []
      for t in prompt:
         result.append(self.token_id(t))
         if t in self.tag2id:
           ids.append(self.tag2id[t])
           index_mapping.append(len(result) - 1)
      if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
      gens = self.model.generate(
          input_ids=torch.tensor(np.array([ids], dtype=np.int32)).to(self.device),
          temperature=1.0,  # HERE
          top_p=0.9,
          do_sample=True,
          min_length=max_augment + 2, max_length=max_augment + 2,
          pad_token_id=self.PAD, eos_token=self.EOS, bos_token_id=self.BOS,
          no_repeat_ngram_size=1,
          bad_words_ids=self.BLACKLIST + [[x] for x in self.META]).cpu().numpy()[0, 1:-1]
      assert len(gens) >= len(index_mapping)
      for i in range(len(index_mapping)):
          assert result[index_mapping[i]] == gens[i]
      for i in range(len(index_mapping), len(gens)):
          result.append(gens[i])

      return [self.token(x) for x in result]


    def encode_one(self, text, max_length=50, add_special_tokens=True, augment=True, max_augment=50, seed=None, **kwargs):
        text = text.replace(',', ' ').split(' ')
        text = [s for s in text if s]
        if '__NO_AUGMENT__' in text:
          augment = False
          text = [s for s in text if s != '__NO_AUGMENT__']
        print('Called for:', text)
        if len(text) > 0 and augment:
          text = self.generate(text, max_augment=max_augment, seed=seed)
          print('Generated:', text)
        text = [self.token_id(s) for s in text]
        text = text[:max_length]
        if add_special_tokens:
          text.extend([self.pad] * (max_length - len(text)))
        return np.array(text)

    def __call__(self, text, max_length=50, add_special_tokens=True, augment=True, max_augment=50, **kwargs):
        print('Final call with:', augment, max_augment)
        if isinstance(text, list):
            res = np.array([self.encode_one(s, add_special_tokens=add_special_tokens, augment=augment, max_augment=max_augment, **kwargs) for s in text])
        else:
          res = self.encode_one(text, add_special_tokens=add_special_tokens, augment=augment, max_augment=max_augment, **kwargs)
        res = torch.tensor(res)
        return {'input_ids': res}



class WrappedTransformerEmbedder(TransformerEmbedder):
    def __init__(self, device='cuda', max_seq_len=77, ar_model_path='', **kwargs):
        super().__init__(device=device, max_seq_len=max_seq_len, **kwargs)
        self.tokenizer = TagTokenizer(ar_model_path, max_seq_len, device=device)
        self.device = device
        self.max_length = max_seq_len
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, augment=True, **kwargs):
        batch_encoding = self.tokenizer(text, max_length=self.max_length, augment=augment, **kwargs)
        if isinstance(batch_encoding, dict):
            batch_encoding = batch_encoding['input_ids']
        tokens = torch.clamp(batch_encoding, 0, self.tokenizer.pad)
        tokens = batch_encoding.to(self.device)
        z = super().forward(tokens)
        return z

    def encode(self, text):
        return self(text)





if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)
