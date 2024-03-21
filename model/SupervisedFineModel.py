import math
import torch
import torch.nn as nn
from algorithm.llm.model.Model import Model
from algorithm.llm.model.Model import SavePeftModelTrainer
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv
from typing import Optional, Tuple
from types import MethodType
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_pt_utils import LabelSmoother

is_flash_attn_2_available = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    is_flash_attn_2_available = True
except ImportError:
    is_flash_attn_2_available = False


# Copied from: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llmtuner/extras/patches/llama_patch.py
class LlamaShiftShortAttention(LlamaAttention):

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:  # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if getattr(self, "num_key_value_groups"):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        if getattr(self.config, "group_size_ratio", None) and self.training:  # shift
            groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
            assert q_len % groupsz == 0, "q_len {} should be divisible by group size {}.".format(q_len, groupsz)
            num_groups = q_len // groupsz

            def shift(state: torch.Tensor) -> torch.Tensor:
                state = state.transpose(1, 2)  # output: (bsz, seq_len, n_heads, head_dim)
                state = torch.cat((
                    state[:, :, :self.num_heads // 2], state[:, :, self.num_heads // 2:].roll(-groupsz // 2, dims=1)
                ), dim=2)
                return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim).transpose(1, 2)

            query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, :groupsz, :groupsz].repeat(num_groups, 1, 1, 1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)  # (bsz, :, seq_len, :) or (bsz*n_group, :, groupsz, :)
        attn_output = attn_output.transpose(1, 2).contiguous()

        if getattr(self.config, "group_size_ratio", None) and self.training:  # shift back
            groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
            attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
            attn_output = torch.cat((
                attn_output[:, :, :self.num_heads // 2],
                attn_output[:, :, self.num_heads // 2:].roll(groupsz // 2, dims=1)
            ))

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # FlashAttention requires the input to have the shape (bsz, seq_len, n_heads, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:  # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # cast to half precision
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            self.logger.warning("The input hidden states seems to be silently casted in float32.")
            query_states = query_states.to(self.config.torch_dtype)
            key_states = key_states.to(self.config.torch_dtype)
            value_states = value_states.to(self.config.torch_dtype)

        if getattr(self, "num_key_value_groups", None):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = query_states.transpose(1, 2)  # (bsz, seq_len, n_heads, head_dim)
        key_states = key_states.transpose(1, 2)  # (bsz, seq_len, n_heads, head_dim)
        value_states = value_states.transpose(1, 2)  # (bsz, seq_len, n_heads, head_dim)

        if getattr(self.config, "group_size_ratio", None) and self.training:  # shift
            groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
            assert q_len % groupsz == 0, "q_len {} should be divisible by group size {}.".format(q_len, groupsz)
            num_groups = q_len // groupsz

            def shift(state: torch.Tensor) -> torch.Tensor:
                state = torch.cat((
                    state[:, :, :self.num_heads // 2], state[:, :, self.num_heads // 2:].roll(-groupsz // 2, dims=1)
                ), dim=2)
                return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim)

            query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(bsz * num_groups, groupsz)

        if attention_mask is not None:
            self.logger.warning("Padded sequences are less efficient in FlashAttention.")
            # -q_len: assumes left padding when q_len != kv_len
            unpadded_q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(query_states, attention_mask[:, -q_len:])
            unpadded_k, _, cu_seqlens_k, max_seqlen_k = unpad_input(key_states, attention_mask)
            unpadded_v, _, _, _ = unpad_input(value_states, attention_mask)
            attn_output_unpad = flash_attn_varlen_func(
                unpadded_q,
                unpadded_k,
                unpadded_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=True,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, bsz, q_len)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, 0.0, softmax_scale=None, causal=True
            )

        if getattr(self.config, "group_size_ratio", None) and self.training:  # shift back
            groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
            attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
            attn_output = torch.cat((
                attn_output[:, :, :self.num_heads // 2],
                attn_output[:, :, self.num_heads // 2:].roll(groupsz // 2, dims=1)
            ))

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def _prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: torch.Tensor,
        inputs_embeds: torch.Tensor,
        past_key_values_length: int
) -> torch.Tensor:
    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask


class SupervisedFineModel(Model):

    def __init__(self, **kwargs):
        # self.model_args = kwargs["model_args"]
        # self.script_args = kwargs["script_args"]
        # self.training_args = kwargs["training_args"]
        # self.logger = kwargs["logger"]
        # self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[self.model_args.model_type]
        super(SupervisedFineModel, self).__init__(**kwargs)
        # self.trainer = None

    def before_load_model(self):
        config, torch_dtype = super().before_load_model()
        # Set RoPE scaling
        if self.model_args.rope_scaling is not None:
            if hasattr(config, "use_dynamic_ntk"):  # for Qwen models
                self.logger.warning("Qwen model does not support RoPE scaling in training.")
            elif hasattr(config, "rope_scaling"):  # for LLaMA and Falcon models
                if self.model_args.rope_scaling == "dynamic":
                    self.logger.warning(
                        "Dynamic NTK may not work well with fine-tuning. "
                        "See: https://github.com/huggingface/transformers/pull/24653"
                    )
                current_max_length = getattr(config, "max_position_embeddings", None)
                if current_max_length and self.script_args.model_max_length > current_max_length:
                    scaling_factor = float(math.ceil(self.script_args.model_max_length / current_max_length))
                else:
                    self.logger.warning(
                        f"The model_max_length({self.script_args.model_max_length}) is smaller than max "
                        f"length({current_max_length}). Consider increase model_max_length.")
                    scaling_factor = 1.0

                setattr(config, "rope_scaling", {"type": self.model_args.rope_scaling, "factor": scaling_factor})
                self.logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                    self.model_args.rope_scaling, scaling_factor
                ))
            else:
                self.logger.warning("Current model does not support RoPE scaling.")

        # Set FlashAttention-2
        if self.model_args.flash_attn:
            if getattr(config, "model_type", None) == "llama":
                if is_flash_attn_2_available:
                    modeling_llama.LlamaAttention = LlamaFlashAttention2
                    modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
                    self.logger.info("Using FlashAttention-2 for faster training and inference.")
                else:
                    self.logger.warning("FlashAttention-2 is not installed.")
            elif getattr(config, "model_type", None) == "qwen":
                self.logger.info("Qwen models automatically enable FlashAttention if installed.")
            else:
                self.logger.warning("Current model does not support FlashAttention-2.")
        elif self.model_args.shift_attn and getattr(config, "model_type", None) == "llama":
            modeling_llama.LlamaAttention = LlamaShiftShortAttention
            self.logger.warning("Using `--flash_attn` for faster training in large context length, enable if your GPU"
                                " is RTX4090, A100 or H100.")

        # Set shift short attention (S^2-Attn)
        if self.model_args.shift_attn:
            if getattr(config, "model_type", None) == "llama":
                setattr(config, "group_size_ratio", 0.25)
                self.logger.info("Using shift short attention with group_size_ratio=1/4.")
            else:
                self.logger.warning("Current model does not support shift short attention.")
        return config, torch_dtype
    def after_load_model(self, model, config):
        # Fix ChatGLM2 and ChatGLM3 LM head
        if getattr(config, "model_type", None) == "chatglm":
            setattr(model, "lm_head", model.transformer.output_layer)
            setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

        # Set NEFTune trick for fine-tuning
        if self.model_args.neft_alpha > 0:
            input_embed = model.get_input_embeddings()
            if isinstance(input_embed, torch.nn.Embedding):
                def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
                    embeddings = input_embed.__class__.forward(self, x)
                    dims = self.num_embeddings * self.embedding_dim
                    mag_norm = self.model_args.neft_alpha / (dims ** 0.5)
                    embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
                    return embeddings

                input_embed.forward = MethodType(noisy_forward, input_embed)
                self.logger.info("Using noisy embedding with alpha={:.2f}".format(self.model_args.neft_alpha))
            else:
                self.logger.warning(
                    "Input embeddings are not normal nn.Embedding, cannot transform into noisy embedding.")
        return model

    def before_fine_tuning_model(self, model):
        # Set fp32 forward hook for lm_head
        output_layer = getattr(model, "lm_head")
        if isinstance(output_layer, torch.nn.Linear):
            def fp32_forward_pre_hook(module: torch.nn.Module, args: Tuple[torch.Tensor]):
                return args[0].to(output_layer.weight.dtype)

            def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                return output.to(torch.float32)

            output_layer.register_forward_pre_hook(fp32_forward_pre_hook)
            output_layer.register_forward_hook(fp32_forward_post_hook)
        return model

    def initial_trainer(self, model, tokenizer, train_dataset, eval_dataset):
        IGNORE_INDEX = LabelSmoother.ignore_index if self.data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        # Initialize our Trainer
        if self.trainer:
            return
        self.before_initial_trainer(model)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=IGNORE_INDEX,
            pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,  # for shift short attention
        )
        # Initialize our Trainer
        self.trainer = SavePeftModelTrainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset if self.training_args.do_train else None,
            eval_dataset=eval_dataset if self.training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    def before_train(self, tokenizer):
        IGNORE_INDEX = LabelSmoother.ignore_index if self.data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        if self.trainer.is_world_process_zero():
            sample = next(iter(self.trainer.get_train_dataloader()))
            self.logger.debug(f"Train dataloader example: {sample}")
            self.logger.debug(
                f"Detail input_ids: {list(sample['input_ids'])[:3]}, \nlabels: {list(sample['labels'])[:3]}")
            self.logger.debug(f"Decode input_ids[0]: {tokenizer.decode(sample['input_ids'][0])}")
            replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id for label in
                               sample['labels'][0]]
            self.logger.debug(f"Decode labels[0]: {tokenizer.decode(replaced_labels)}")
