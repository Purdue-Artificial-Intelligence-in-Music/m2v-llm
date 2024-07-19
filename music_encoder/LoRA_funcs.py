import torch
import transformers
from peft import LoraConfig, get_peft_model

def get_LoRA_trainable_modules(model: torch.nn.Module, return_type_tuples: bool = False, separate_out_adapters: bool = True):
    '''
    This method returns a list of LoRA trainable parameters given a PyTorch model.

    Input:
    - model: a torch.nn.Module (a model).
    Output: 
    - trainable_modules: a list of trainable modules if return_type_tuples is false or a list of (name, module_type) tuples if return_type_tuples is true.
    - adapter_modules: a list of trainable adapter modules (removed from trainable_modules) if return_type_tuples is false or a list of (name, module_type) tuples if return_type_tuples is true.

    Parameters:
    - return_type_tuples: whether to return (name, module_type) tuples or just the names of the trainable modules.
    - separate_out_adapters: whether to separate out the adapter modules from the trainable modules.

    '''
    modules = [(n, type(m)) for n, m in model.named_modules()]
    trainable_modules = []
    adapter_modules = []
    for elem in modules:
        name, module_type = elem
        if module_type == torch.nn.Linear or module_type == torch.nn.Embedding or module_type == torch.nn.Conv2d or module_type == transformers.pytorch_utils.Conv1D:
            if separate_out_adapters and name.startswith('adapter'):
                adapter_modules.append((name, module_type) if return_type_tuples else name)
            else:
                trainable_modules.append((name, module_type) if return_type_tuples else name)
    if separate_out_adapters:
        return trainable_modules, adapter_modules
    return trainable_modules

def get_LoRA_model(model: torch.nn.Module, r: int = 8, lora_alpha: int = 8):
    '''
    This method returns a LoRA model given a PyTorch model.

    Input:
    - model: a torch.nn.Module (a model).
    Output: a PEFT-based LoRA model.

    Parameters:
    - r: LoRA attention dimension.
    - lora_alpha: LoRA alpha (scaling) parameter.

    '''
    trainable_modules = get_LoRA_trainable_modules(model)
    config = LoraConfig(
        target_modules=trainable_modules,
        r = r,
        lora_alpha=lora_alpha,
        bias='lora_only',
        use_rslora=True,
    )
    return get_peft_model(model, config)