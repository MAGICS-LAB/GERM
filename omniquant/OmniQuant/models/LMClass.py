import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pdb
from transformers_language.models.bert_attention import (
    AttentionGateType,
    BertUnpadSelfAttentionWithExtras,
)
from transformers_language.models.softmax import SOFTMAX_MAPPING

class LMClass(BaseLM):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model_config = args.model
        config = AutoConfig.from_pretrained(
            args.model, attn_implementation=args.attn_implementation, trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False)
        # self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16,trust_remote_code=True)
        # for layer_idx in range(len(model.bert.encoder.layer)):
        #     old_self = model.bert.encoder.layer[layer_idx].attention.self
        #     print("----------------------------------------------------------")
        #     print("Inside BERT custom attention")
        #     print("----------------------------------------------------------")
        #     new_self = BertUnpadSelfAttentionWithExtras(
        #         config,
        #         position_embedding_type=None,
        #         softmax_fn=SOFTMAX_MAPPING[args.attn_softmax],
        #         ssm_eps=None,
        #         tau=None,
        #         max_seq_length=args.model_max_length,
        #         skip_attn=False,
        #         fine_tuning=False,
        #     )

        #     # copy loaded weights
        #     if args.model is not None:
        #         new_self.load_state_dict(old_self.state_dict(), strict=False)
        #     model.bert.encoder.layer[layer_idx].attention.self = new_self

        # # Gating -> load the model again to load missing alpha
        # if args.model is not None and AttentionGateType.none.name != "none":
        #     state_dict = torch.load(str(Path(args.model_name_or_path) / "pytorch_model.bin"))
        #     new_state_dict = {}
        #     for name, val in state_dict.items():
        #         if "alpha" in name:
        #             new_state_dict[name] = val
        #     model.load_state_dict(new_state_dict, strict=False)
        # # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # # on a small vocab and want a smaller embedding size, remove this test.
        # embedding_size = model.get_input_embeddings().weight.shape[0]
        # if len(self.tokenizer) > embedding_size:
        #     print("Resizing token embeddings to fit tokenizer vocab size")
        #     model.resize_token_embeddings(len(tokenizer))

        self.model = model
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab size: ", self.vocab_size)

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():

            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
