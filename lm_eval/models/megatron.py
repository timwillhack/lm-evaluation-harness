import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from lm_eval.base import LM, TokenizedLM
from lm_eval import utils
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod
from typing import Iterable

import gpt_neox
from gpt_neox.megatron.text_generation_utils import generate_samples_from_prompt

class MegatronLM(TorchLM):

    def __init__(self, neox_args, device='cuda', batch_size=1):
      
        # This needs to be fixed but every other function should work.
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(batch_size, int)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained(pretrained, revision=revision +("/" + subfolder if subfolder is not None else "")).to(self.device)
        self.gpt2.eval()

        self.tokenizer = neox_args.tokenizer

        self.vocab_size = self.tokenizer.vocab_size
        self.eot_token_id = self.tokenizer.eos_token_id # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        self.max_gen_toks = 256

        # multithreading and batching
        gpus = torch.cuda.device_count()
        batch_size_per_gpu = batch_size # todo: adaptive batch size

        # TODO: fix multi-gpu
        self.batch_size = batch_size_per_gpu# * gpus

        # TODO: fix multi-gpu
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)
    
    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits retuned from the model
        """

        data_wrapped = iter([{'text': F.pad(inps, pad=(0, 1))}])
        if self.neox_args.is_pipe_parallel:
            # need these flags to stop deepspeed from hanging
            self.model.first_output_send = True
            self.model.pipe_recv_buf = None
        _, logits = self._forward_step_fn(model=self.model, data_iterator=data_wrapped)
        return logits
    
    def _model_generate(self, context, max_length, eos_token_id):
        return generate_samples_from_prompt(
            neox_args=neox_args, 
            model = self.model,
            text=context,
            eos_token_id = eos_token_id,
            maximum_tokens = max_lenth
            recompute = neox_args.recompute, 
            temperature = neox_args.temperature,
            top_k = neox_args.top_k, 
            top_p = neox_args.top_p,
            stop_tokens = neox_args.stop_tokens
        )
