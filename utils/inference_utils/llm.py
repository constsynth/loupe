import gc
import torch
import warnings
import typing as tp
from interpretability.sae.sae import SAE
from transformers import AutoTokenizer, AutoModelForCausalLM


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

    @staticmethod
    def cleanup_memory():
        gc.collect()
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
        def hook_fn(module, input, output):
            hidden_states.append(output[0].mean(dim=1).detach().cpu()) # Mean pooling for all the tokens
        if layer_name:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    module.register_forward_hook(hook_fn)
                    break
        else:
            warnings.warn("`layer_name` is None, `last_hidden_state` used")
        with torch.no_grad():
            _ = self.model(**inputs, output_hidden_states=True, **generate_kwargs)
        self.cleanup_memory()
        return hidden_states[-1] if layer_name else _.hidden_states[0].mean(dim=1).detach().cpu()

    def add_sae(
        self,
        sae: SAE,
        layer_num: int = -1
    ):
        pass

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
