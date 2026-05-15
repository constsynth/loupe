import gc
import torch
import warnings
import typing as tp
from dataclasses import dataclass
from interpretability.sae.sae import SAE
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class SAEInterventionConfig:
    """
    Configuration for an SAE intervention hook.

    `feature_indices` selects latent SAE features. `token_positions` optionally
    localizes the intervention for activations shaped [batch, sequence, hidden].
    """

    feature_indices: int | tp.List[int] | torch.Tensor | None = None
    intervention_value: float = 1.0
    mode: str = "add"
    token_positions: int | tp.List[int] | torch.Tensor | None = None
    enabled: bool = True


class LLM:

    def __init__(
    self,
    model_name_or_path: str,
    device: str = 'cuda'
    ) -> None:
        self.device = device
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
    def create_model(
        model_name_or_path: str,
    ) -> tp.Tuple[AutoModelForCausalLM, AutoTokenizer]:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )
        return model, tokenizer
    
    def get_hidden_state(
        self,
        input_text: str | tp.List[str],
        layer_name: str = None,
        **generate_kwargs
    ) -> torch.Tensor:
        """
        Returns torch.Tensor with a certain layer activations for input batch.
        """
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        # forward pass saving activations using `hook_fn`
        hidden_states = []
        hook_handle = None
        def hook_fn(module, input, output):
            hidden_state = self._extract_hidden_state_from_module_output(output)
            hidden_states.append(hidden_state.mean(dim=1).detach().cpu()) # Mean pooling for all the tokens
        if layer_name:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    hook_handle = module.register_forward_hook(hook_fn)
                    break
        else:
            warnings.warn("`layer_name` is None, `last_hidden_state` used")
        with torch.no_grad():
            _ = self.model(**inputs, output_hidden_states=True, **generate_kwargs)
        if hook_handle is not None:
            hook_handle.remove()
        self.cleanup_memory()
        if layer_name and not hidden_states:
            raise ValueError(f"Layer not found or did not produce hidden states: {layer_name}")
        return hidden_states[-1] if layer_name else _.hidden_states[0].mean(dim=1).detach().cpu()

    def add_sae(
        self,
        sae: SAE,
        layer_num: int = -1,
        layer_name: str | None = None,
        feature_indices: int | tp.List[int] | torch.Tensor | None = None,
        intervention_value: float = 1.0,
        mode: str = "add",
        token_positions: int | tp.List[int] | torch.Tensor | None = None,
        enabled: bool = True,
    ) -> str:
        """
        Attach SAE to a transformer layer with an optional latent intervention.

        During the hooked forward pass, the layer activation `h` is encoded into
        SAE latents `z`, selected features are modified as `z_prime`, and the
        decoded activation `h_prime` is passed to subsequent model layers.

        Args:
            sae: Trained SAE with `in_hidden_state_size` matching the hooked layer.
            layer_num: Transformer block index. Negative values count from the end.
            layer_name: Exact module name from `self.model.named_modules()`.
            feature_indices: SAE latent feature index or indices to intervene on.
                If omitted, SAE reconstructs the activation without changing `z`.
            intervention_value: Scalar value for the intervention.
            mode: `add`, `set`, or `multiply`.
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
            mode=mode,
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

                modified_hidden_state = sae.intervene(
                    hidden_state=hidden_state,
                    feature_indices=active_config.feature_indices,
                    intervention_value=active_config.intervention_value,
                    mode=active_config.mode,
                    token_positions=token_positions,
                )

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
        mode: str | None = None,
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
        if mode is not None:
            config.mode = mode
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

    def generate(
        self,
        input_text: str,
        **generate_kwargs
    ) -> str:
        """
        Basic output generation method.
        """
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        # Default generation settings, may be reinitiated via kwargs (max_length, temperature, num_beams etc.)
        outputs = self.model.generate(**inputs, **generate_kwargs)
        text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        self.cleanup_memory()
        return text
