from torch import nn

NAME = "convnext"
MODEL_ID = "convnext_base.fb_in22k_ft_in1k"
INPUT_SIZE = 224
NUM_STAGES = 4


def build_backbone(checkpoint: str = "") -> nn.Module:
    import timm
    import torch

    model = timm.create_model(MODEL_ID, pretrained=True, num_classes=0)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model


def unfreeze_parameters(model: nn.Module, stages: int = 1) -> list[nn.Parameter]:
    if not 0 <= stages <= NUM_STAGES:
        raise ValueError(f"stages must be between 0 and {NUM_STAGES}")
    modules = [*model.stages[NUM_STAGES - stages :], model.head]
    if stages == NUM_STAGES:
        modules.append(model.stem)
    return [p for m in modules for p in m.parameters()]
