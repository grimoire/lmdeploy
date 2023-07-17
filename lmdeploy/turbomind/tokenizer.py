# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Sequence, Union

import torch
from torch.nn.utils.rnn import pad_sequence


class SentencePieceTokenizer:
    """Tokenizer of sentencepiece.

    Args:
        model_file (str): the path of the tokenizer model
    """

    def __init__(self, model_file: str):
        from sentencepiece import SentencePieceProcessor
        self.model = SentencePieceProcessor(model_file=model_file)

    @property
    def vocab_size(self):
        """vocabulary size."""
        return self.model.vocab_size()

    @property
    def bos_token_id(self):
        """begine of the sentence token id."""
        return self.model.bos_id()

    @property
    def eos_token_id(self):
        """end of the sentence token id."""
        return self.model.eos_id()

    def encode(self, s: str):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
        Returns:
            list[int]: token ids
        """
        add_bos = False
        add_eos = False
        if s.find('<BOS>') != -1:
            s = s.replace('<BOS>', '')
            add_bos = True
        if s == '<EOS>':
            s = ''
            add_eos = True
        return self.model.Encode(s, add_bos=add_bos, add_eos=add_eos)

    def decode(self, t: Sequence[int]):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
        Returns:
            str: text of decoding tokens
        """
        if isinstance(t, torch.Tensor):
            t = t.tolist()
        return self.model.Decode(t)


class HuggingFaceTokenizer:
    """Tokenizer of sentencepiece.

    Args:
        model_dir (str): the directory of the tokenizer model
    """

    def __init__(self, model_dir: str):
        from transformers import AutoTokenizer
        model_file = osp.join(model_dir, 'tokenizer.model')
        backend_tokenizer_file = osp.join(model_dir, 'tokenizer.json')
        model_file_exists = osp.exists(model_file)
        if not osp.exists(backend_tokenizer_file) and model_file_exists:
            print('WARNING: Can not find tokenizer.json. '
                  'It may take long time to initialize the tokenizer.')
        self.model = AutoTokenizer.from_pretrained(model_dir,
                                                   trust_remote_code=True)
        # save tokenizer.json to reuse
        if not osp.exists(backend_tokenizer_file) and model_file_exists:
            self.model.backend_tokenizer.save(backend_tokenizer_file)

    @property
    def vocab_size(self):
        """vocabulary size."""
        return self.model.vocab_size

    @property
    def bos_token_id(self):
        """begine of the sentence token id."""
        return self.model.bos_token_id

    @property
    def eos_token_id(self):
        """end of the sentence token id."""
        return self.model.eos_token_id

    def encode(self, s: str):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
        Returns:
            list[int]: token ids
        """
        add_special_tokens = False
        if s.find('<BOS>') != -1:
            s = s.replace('<BOS>', '<s>')
        if s == '<EOS>':
            s = '</s>'
        if len(s) == 0:
            add_special_tokens = True
        return self.model.encode(s, add_special_tokens=add_special_tokens)

    def decode(self, t: Sequence[int]):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
        Returns:
            str: text of decoding tokens
        """
        skip_special_tokens = True
        return self.model.decode(t, skip_special_tokens=skip_special_tokens)


class Tokenizer:
    """Tokenize prompts or de-tokenize tokens into texts.

    Args:
        model_file (str): the path of the tokenizer model
    """

    def __init__(self, model_file: str):
        if model_file.endswith('.model'):
            model_folder = osp.split(model_file)[0]
        else:
            model_folder = model_file
            model_file = osp.join(model_folder, 'tokenizer.model')
        tokenizer_config_file = osp.join(model_folder, 'tokenizer_config.json')

        model_file_exists = osp.exists(model_file)
        config_exists = osp.exists(tokenizer_config_file)
        use_hf_model = config_exists or not model_file_exists

        if not use_hf_model:
            self.model = SentencePieceTokenizer(model_file)
        else:
            self.model = HuggingFaceTokenizer(model_folder)

    @property
    def vocab_size(self):
        """vocabulary size."""
        return self.model.vocab_size

    @property
    def bos_token_id(self):
        """begine of the sentence token id."""
        return self.model.bos_token_id

    @property
    def eos_token_id(self):
        """end of the sentence token id."""
        return self.model.eos_token_id

    def encode(self, s: str):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
        Returns:
            list[int]: token ids
        """
        return self.model.encode(s)

    def decode(self, t: Sequence[int]):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
        Returns:
            str: text of decoding tokens
        """
        return self.model.decode(t)


class Preprocessor:
    """Tokenize prompts.

    Args:
        tokenizer (Tokenizer): an instance of tokenizer
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def infer(self, prompts: Union[str, Sequence[str]]) -> tuple:
        """Tokenize the input prompts.

        Args:
            prompts(str | Sequence[str]): user's prompt, or a batch prompts

        Returns:
            Tuple(torch.Tensor, torch.Tensor): prompt's token
            ids, ids' length and requested output length
        """
        if isinstance(prompts, str):
            _ = [[prompts]]
        elif isinstance(prompts, Sequence):
            _ = [[prompt] for prompt in prompts]
        else:
            assert 0, f'str or Sequence[str] prompts are expected but got ' \
                      f'{type(prompts)}'

        start_ids = [
            torch.IntTensor(self.tokenizer.encode(prompt))
            for prompt in prompts
        ]
        start_lengths = torch.IntTensor([[len(ids)] for ids in start_ids])
        start_ids = pad_sequence(start_ids,
                                 batch_first=True,
                                 padding_value=self.eos_token_id)
        return start_ids, start_lengths


class Postprocessor:
    """De-tokenize token ids.

    Args:
        tokenizer (Tokenizer): an instance of tokenizer
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def infer(self, output_ids: torch.Tensor, seqlen: torch.Tensor):
        """De-tokenize tokens for text.

        Args:
            output_ids(torch.Tensor): tokens' id
            seqlen(torch.Tensor): sequence length

        Returns:
            str: decoded tokens
        """
        outputs = []
        for tokens, _len in zip(output_ids, seqlen):
            output = self.tokenizer.decode(tokens[:_len])
            outputs.append(output)
        return outputs
