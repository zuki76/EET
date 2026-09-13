import torch
from torch import nn


def is_eet_parameter(name):
    return any(part.startswith("DMPW_LoRAGenerater_") for part in name.split("."))


def is_prompt_parameter(name):
    return name.startswith("prompt_learner.") or any(
        part.startswith("VPT") for part in name.split(".")
    )


class FrozenExpertModel(nn.Module):

    def train(self, mode=True):
        super().train(mode)
        self.domain_model.eval()
        return self


def configure_trainable_parameters(model):
    for name, param in model.named_parameters():
        trainable = is_eet_parameter(name) or is_prompt_parameter(name)
        param.requires_grad_(trainable and "no_grad" not in name)
    model.domain_model.eval()


def build_eet_optimizer(model, optim_cfg):
    configs = list(optim_cfg.PARAM_GROUPS)
    names = [group["NAME"] for group in configs]
    if len(names) != 2 or set(names) != {"EET", "prompt"}:
        raise ValueError("OPTIM.PARAM_GROUPS must contain EET and prompt exactly once")

    groups = {"EET": [], "prompt": []}
    if any(p.requires_grad for p in model.domain_model.parameters()):
        raise RuntimeError("The domain expert must be frozen")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_eet_parameter(name):
            group = "EET"
        elif is_prompt_parameter(name):
            group = "prompt"
        else:
            raise RuntimeError(f"Unclassified trainable parameter: {name}")
        groups[group].append(param)
        print(f"Trainable [{group}]: {name} {tuple(param.shape)}")

    param_groups = []
    for config in configs:
        name = config["NAME"]
        if not groups[name]:
            raise RuntimeError(f"No trainable {name} parameters")
        group = {"params": groups[name], "lr": config["LR"], "name": name}
        for key in ("WEIGHT_DECAY", "MOMENTUM"):
            if key in config:
                group[key.lower()] = config[key]
        param_groups.append(group)
        print(f"Param group {name}: {len(groups[name])} tensors, LR={config['LR']}")

    optimizer = optim_cfg.NAME.lower()
    weight_decay = optim_cfg.get("WEIGHT_DECAY", 5e-4)
    if optimizer == "sgd":
        return torch.optim.SGD(
            param_groups, momentum=optim_cfg.get("MOMENTUM", 0.9),
            weight_decay=weight_decay,
        )
    if optimizer == "adamw":
        if any("MOMENTUM" in config for config in configs):
            raise ValueError("MOMENTUM is an SGD option; remove it for AdamW")
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    raise ValueError(f"Unsupported EET optimizer: {optimizer}")


def load_eet_weights(model, state_dict):
    state_dict = dict(state_dict)
    for name in ("prompt_learner.token_prefix", "prompt_learner.token_suffix"):
        state_dict.pop(name, None)
    required = {
        name: param for name, param in model.named_parameters()
        if is_eet_parameter(name) or is_prompt_parameter(name)
    }
    missing = [name for name in required if name not in state_dict]
    mismatched = [
        f"{name}: checkpoint {tuple(state_dict[name].shape)}, model {tuple(param.shape)}"
        for name, param in required.items()
        if name in state_dict and state_dict[name].shape != param.shape
    ]
    if missing or mismatched:
        raise RuntimeError(
            "Incompatible EET checkpoint. This release requires retraining and "
            "reevaluation; 50-token position tables cannot be loaded into full-patch "
            f"generators. Missing learned keys: {missing}. Shape mismatches: {mismatched}"
        )
    return model.load_state_dict(state_dict, strict=False)
