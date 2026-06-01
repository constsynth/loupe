import gc
import torch
import warnings
import typing as tp
from dataclasses import dataclass
from interpretability.sae.sae import SAE
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


@dataclass
class SAEInterventionConfig:
    """
    Configuration for an SAE intervention hook.

    `feature_indices` selects latent SAE features. `token_positions` optionally
    localizes the intervention for activations shaped [batch, sequence, hidden].
    """

    feature_indices: int | tp.List[int] | torch.Tensor | None = None
    intervention_value: float = 1.0
    token_positions: int | tp.List[int] | torch.Tensor | None = None
    enabled: bool = True


class LLM:

    def __init__(
        self,
        model_name_or_path: str,
        device: str = 'cuda',
    ) -> None:
        self.device = self._resolve_device(device)
        self.model, self.tokenizer = self.create_model(
            model_name_or_path
        )
        self.model.to(self.device)
        self.model.eval()
        self.sae_hooks: dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.sae_interventions: dict[str, SAEInterventionConfig] = {}
        self.sae_modules: dict[str, SAE] = {}

    @staticmethod
    def cleanup_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device

    @staticmethod
    def create_model(
        model_name_or_path: str,
    ) -> tp.Tuple[AutoModelForCausalLM, AutoTokenizer]:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )
        return model, tokenizer
    
    def get_hidden_state(
        self,
        input_text: str | tp.List[str],
        layer_name: str | None = None,
        return_tokens: bool = False,
        valid_tokens_only: bool = False,
        batch_size: int | None = None,
        detach: bool = True,
        move_to_cpu: bool = True,
        **forward_kwargs,
    ) -> torch.Tensor:
        """
        Return activations for a selected layer or the model's final hidden state.

        Args:
            input_text: Input string or batch of strings.
            layer_name: Exact module name from `self.model.named_modules()`.
            return_tokens: If True, return token activations with shape
                [batch, sequence, hidden]. If False, return attention-mask mean
                pooled activations with shape [batch, hidden].
            valid_tokens_only: If True with `return_tokens=True`, return only
                non-padding token activations with shape [valid_tokens, hidden].
                This is the preferred mode for token-level SAE training.
            batch_size: Optional number of input texts to process per forward
                pass. Use this when extracting activations for large datasets to
                avoid CUDA out-of-memory errors.
            detach: Detach activations from autograd before returning.
            move_to_cpu: Move activations to CPU before returning.

        Returns:
            Tensor with shape [batch, hidden], [batch, sequence, hidden], or
            [valid_tokens, hidden] when `valid_tokens_only=True`.
        """
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive when provided")

        input_batches = self._input_text_batches(input_text, batch_size)
        hidden_state_batches: list[torch.Tensor] = []

        def prepare_hidden_state(
            hidden_state: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            mask = attention_mask.to(device=hidden_state.device)
            if return_tokens and valid_tokens_only:
                hidden_state = hidden_state[mask].reshape(-1, hidden_state.shape[-1])
            elif not return_tokens:
                hidden_state = self._masked_mean_pool(hidden_state, mask)
            if detach:
                hidden_state = hidden_state.detach()
            if move_to_cpu:
                hidden_state = hidden_state.cpu()
            return hidden_state

        module = self._get_module(layer_name) if layer_name else None
        if layer_name is None:
            warnings.warn("`layer_name` is None, `last_hidden_state` used")

        model_forward_kwargs = dict(forward_kwargs)
        model_forward_kwargs.setdefault("output_hidden_states", layer_name is None)
        model_forward_kwargs.setdefault("use_cache", False)

        for input_batch in input_batches:
            inputs = self.tokenizer(
                input_batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to(self.device)
            attention_mask = inputs["attention_mask"].bool()
            batch_hidden_states: list[torch.Tensor] = []
            hook_handle = None

            def hook_fn(module, input, output):
                hidden_state = self._extract_hidden_state_from_module_output(output)
                batch_hidden_states.append(prepare_hidden_state(hidden_state, attention_mask))

            if module is not None:
                hook_handle = module.register_forward_hook(hook_fn)

            try:
                with torch.no_grad():
                    output = self.model(**inputs, **model_forward_kwargs)
            finally:
                if hook_handle is not None:
                    hook_handle.remove()

            if layer_name:
                if not batch_hidden_states:
                    raise ValueError(f"Layer did not produce hidden states: {layer_name}")
                hidden_state_batches.append(batch_hidden_states[-1])
            else:
                hidden_state_batches.append(
                    prepare_hidden_state(output.hidden_states[-1], attention_mask)
                )
            del inputs
            self.cleanup_memory()

        self.cleanup_memory()
        return self._concat_hidden_state_batches(
            hidden_state_batches,
            return_tokens=return_tokens,
            valid_tokens_only=valid_tokens_only,
        )

    def add_sae(
        self,
        sae: SAE,
        layer_num: int = -1,
        layer_name: str | None = None,
        feature_indices: int | tp.List[int] | torch.Tensor | None = None,
        intervention_value: float = 1.0,
        token_positions: int | tp.List[int] | torch.Tensor | None = None,
        enabled: bool = True,
    ) -> str:
        """
        Attach SAE to a transformer layer with an optional latent intervention.

        During the hooked forward pass, selected SAE features are converted to
        decoder directions and added to the layer activation before subsequent
        model layers run.

        Args:
            sae: Trained SAE with `in_hidden_state_size` matching the hooked layer.
            layer_num: Transformer block index. Negative values count from the end.
            layer_name: Exact module name from `self.model.named_modules()`.
            feature_indices: SAE latent feature index or indices to intervene on.
                If omitted, SAE reconstructs the activation without changing `z`.
            intervention_value: Scalar value for the intervention.
            token_positions: Optional token positions for token-level activations.
            enabled: Whether the intervention hook is active immediately.

        Returns:
            Resolved layer name.
        """
        resolved_layer_name = self._resolve_layer_name(layer_name, layer_num)
        module = self._get_module(resolved_layer_name)

        self.remove_sae(resolved_layer_name)

        model_device, model_dtype = self._model_device_dtype()
        sae = sae.to(device=model_device, dtype=model_dtype)
        sae.eval()

        config = SAEInterventionConfig(
            feature_indices=feature_indices,
            intervention_value=intervention_value,
            token_positions=token_positions,
            enabled=enabled,
        )

        def hook_fn(module, inputs, output):
            if not self.sae_interventions[resolved_layer_name].enabled:
                return output

            hidden_state = self._extract_hidden_state_from_module_output(output)
            active_config = self.sae_interventions[resolved_layer_name]

            if active_config.feature_indices is None:
                modified_hidden_state = sae(hidden_state)
            else:
                token_positions = self._valid_token_positions(
                    active_config.token_positions,
                    hidden_state,
                )
                if active_config.token_positions is not None and token_positions is None:
                    return output

                intervention_delta = self._additive_decoder_direction_delta(
                    sae=sae,
                    hidden_state=hidden_state,
                    feature_indices=active_config.feature_indices,
                    intervention_value=active_config.intervention_value,
                    token_positions=token_positions,
                )
                modified_hidden_state = hidden_state + intervention_delta

            return self._replace_hidden_state_in_module_output(output, modified_hidden_state)

        self.sae_modules[resolved_layer_name] = sae
        self.sae_interventions[resolved_layer_name] = config
        self.sae_hooks[resolved_layer_name] = module.register_forward_hook(hook_fn)
        return resolved_layer_name

    def set_sae_intervention(
        self,
        layer_name: str,
        feature_indices: int | tp.List[int] | torch.Tensor | None = None,
        intervention_value: float | None = None,
        token_positions: int | tp.List[int] | torch.Tensor | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Update intervention settings for an already attached SAE hook."""
        if layer_name not in self.sae_interventions:
            raise ValueError(f"No SAE hook registered for layer: {layer_name}")

        config = self.sae_interventions[layer_name]
        if feature_indices is not None:
            config.feature_indices = feature_indices
        if intervention_value is not None:
            config.intervention_value = intervention_value
        if token_positions is not None:
            config.token_positions = token_positions
        if enabled is not None:
            config.enabled = enabled

    def clear_sae_intervention_features(self, layer_name: str) -> None:
        """Keep the SAE hook attached but switch it to reconstruction-only mode."""
        if layer_name not in self.sae_interventions:
            raise ValueError(f"No SAE hook registered for layer: {layer_name}")
        self.sae_interventions[layer_name].feature_indices = None

    def enable_sae_intervention(self, layer_name: str) -> None:
        """Enable an attached SAE hook."""
        self.set_sae_intervention(layer_name, enabled=True)

    def disable_sae_intervention(self, layer_name: str) -> None:
        """Disable an attached SAE hook without removing it."""
        self.set_sae_intervention(layer_name, enabled=False)

    def remove_sae(self, layer_name: str) -> None:
        """Remove SAE hook and associated configuration for one layer."""
        handle = self.sae_hooks.pop(layer_name, None)
        if handle is not None:
            handle.remove()
        self.sae_interventions.pop(layer_name, None)
        self.sae_modules.pop(layer_name, None)

    def clear_sae_hooks(self) -> None:
        """Remove all SAE hooks from the model."""
        for layer_name in list(self.sae_hooks):
            self.remove_sae(layer_name)

    def _resolve_layer_name(self, layer_name: str | None, layer_num: int) -> str:
        if layer_name is not None:
            self._get_module(layer_name)
            return layer_name

        named_modules = dict(self.model.named_modules())
        candidate_names = self._candidate_layer_names(layer_num)
        for candidate_name in candidate_names:
            if candidate_name in named_modules:
                return candidate_name

        raise ValueError(
            f"Could not resolve layer_num={layer_num}. Pass exact layer_name from model.named_modules()."
        )

    def _candidate_layer_names(self, layer_num: int) -> list[str]:
        candidates: list[str] = []

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            n_layers = len(self.model.model.layers)
            layer_idx = layer_num if layer_num >= 0 else n_layers + layer_num
            candidates.append(f"model.layers.{layer_idx}")

        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            n_layers = len(self.model.transformer.h)
            layer_idx = layer_num if layer_num >= 0 else n_layers + layer_num
            candidates.append(f"transformer.h.{layer_idx}")

        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            n_layers = len(self.model.gpt_neox.layers)
            layer_idx = layer_num if layer_num >= 0 else n_layers + layer_num
            candidates.append(f"gpt_neox.layers.{layer_idx}")

        candidates.extend(
            [
                f"layers.{layer_num}",
                f"model.layers.{layer_num}",
                f"transformer.h.{layer_num}",
                f"gpt_neox.layers.{layer_num}",
            ]
        )
        return candidates

    def _get_module(self, layer_name: str) -> torch.nn.Module:
        named_modules = dict(self.model.named_modules())
        if layer_name not in named_modules:
            raise ValueError(f"Layer not found: {layer_name}")
        return named_modules[layer_name]

    def _model_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.model.parameters())
        return parameter.device, parameter.dtype

    @staticmethod
    def _extract_hidden_state_from_module_output(output) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, tuple):
            return output[0]
        if isinstance(output, list):
            return output[0]
        raise TypeError(f"Unsupported module output type for SAE intervention: {type(output)}")

    @staticmethod
    def _masked_mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool token activations over non-padding positions."""
        weights = attention_mask.to(device=hidden_state.device, dtype=hidden_state.dtype).unsqueeze(-1)
        return (hidden_state * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _input_text_batches(
        input_text: str | tp.List[str],
        batch_size: int | None,
    ) -> list[str | tp.List[str]]:
        if isinstance(input_text, str):
            return [input_text]
        if batch_size is None:
            return [input_text]
        return [
            input_text[start:start + batch_size]
            for start in range(0, len(input_text), batch_size)
        ]

    @staticmethod
    def _concat_hidden_state_batches(
        hidden_state_batches: list[torch.Tensor],
        return_tokens: bool,
        valid_tokens_only: bool,
    ) -> torch.Tensor:
        if not hidden_state_batches:
            raise ValueError("No hidden states were produced")

        if return_tokens and not valid_tokens_only:
            max_sequence_length = max(batch.shape[1] for batch in hidden_state_batches)
            padded_batches = []
            for batch in hidden_state_batches:
                if batch.shape[1] == max_sequence_length:
                    padded_batches.append(batch)
                    continue
                padding = batch.new_zeros(
                    batch.shape[0],
                    max_sequence_length - batch.shape[1],
                    batch.shape[2],
                )
                padded_batches.append(torch.cat([batch, padding], dim=1))
            hidden_state_batches = padded_batches

        return torch.cat(hidden_state_batches, dim=0)

    @staticmethod
    def _replace_hidden_state_in_module_output(output, hidden_state: torch.Tensor):
        if torch.is_tensor(output):
            return hidden_state
        if isinstance(output, tuple):
            return (hidden_state, *output[1:])
        if isinstance(output, list):
            return [hidden_state, *output[1:]]
        raise TypeError(f"Unsupported module output type for SAE intervention: {type(output)}")

    @staticmethod
    def _valid_token_positions(
        token_positions: int | tp.List[int] | torch.Tensor | None,
        hidden_state: torch.Tensor,
    ) -> int | list[int] | torch.Tensor | None:
        if token_positions is None or hidden_state.ndim < 3:
            return token_positions

        sequence_length = hidden_state.shape[-2]
        positions = torch.as_tensor(token_positions, dtype=torch.long).flatten()
        positions = torch.where(positions < 0, positions + sequence_length, positions)
        positions = positions[(positions >= 0) & (positions < sequence_length)]
        if positions.numel() == 0:
            return None
        return positions.to(device=hidden_state.device)

    @staticmethod
    def _additive_decoder_direction_delta(
        sae: SAE,
        hidden_state: torch.Tensor,
        feature_indices: int | tp.List[int] | torch.Tensor,
        intervention_value: float,
        token_positions: int | tp.List[int] | torch.Tensor | None,
    ) -> torch.Tensor:
        """Return `alpha * W_dec[:, feature]` broadcast to hooked activations."""
        direction = sae.decoder_direction_for_feature(
            feature_indices=feature_indices,
            intervention_value=intervention_value,
        ).to(device=hidden_state.device, dtype=hidden_state.dtype)

        if token_positions is None:
            view_shape = (1,) * (hidden_state.ndim - 1) + (direction.shape[-1],)
            return direction.reshape(view_shape).expand_as(hidden_state)

        if hidden_state.ndim < 3:
            raise ValueError("token_positions require hidden state shape [batch, sequence, hidden]")

        delta = torch.zeros_like(hidden_state)
        delta[:, token_positions, :] = direction
        return delta

    def generate(
        self,
        input_text: str,
        return_full_text: bool = False,
        strip: bool = True,
        input_max_length: int | None = None,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        use_chat_template: bool = True,
        **generate_kwargs
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            input_text: Prompt text.
            return_full_text: If True, return prompt plus completion. If False,
                return only newly generated completion tokens.
            strip: Strip leading/trailing whitespace from decoded text.
            input_max_length: Optional tokenizer truncation length for the prompt.
            system_prompt: Optional system message used with tokenizer chat template.
            use_chat_template: If True, format the user prompt with the
                tokenizer's chat template before generation.
            **generate_kwargs: Arguments forwarded to `transformers.generate`.

        Returns:
            Decoded generated text.
        """
        model_input_text = self._format_generation_prompt(
            input_text=input_text,
            system_prompt=system_prompt,
            use_chat_template=use_chat_template,
        )
        tokenizer_kwargs: dict[str, tp.Any] = {"return_tensors": "pt"}
        if input_max_length is not None:
            tokenizer_kwargs.update({"truncation": True, "max_length": input_max_length})
        inputs = self.tokenizer(model_input_text, **tokenizer_kwargs).to(self.device)

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_token_id is not None and "pad_token_id" not in generate_kwargs:
            generate_kwargs["pad_token_id"] = pad_token_id

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)
        output_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
        ids_to_decode = output_ids[0]
        if not return_full_text:
            prompt_length = inputs["input_ids"].shape[-1]
            ids_to_decode = ids_to_decode[prompt_length:]

        text = self.tokenizer.decode(ids_to_decode, skip_special_tokens=True)
        self.cleanup_memory()
        return text.strip() if strip else text

    def _format_generation_prompt(
        self,
        input_text: str,
        system_prompt: str | None,
        use_chat_template: bool,
    ) -> str:
        if not use_chat_template:
            return input_text

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": input_text})

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, ValueError):
            return input_text
