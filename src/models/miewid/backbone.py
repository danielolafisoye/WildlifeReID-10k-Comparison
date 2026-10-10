import warnings

from torch import nn

NAME = "miewid"
MODEL_ID = "conservationxlabs/miewid-msv3"
REVISION = "4f1d7f2b521149e5fe34bb85f377248ce9971a7d"
INPUT_SIZE = 440
NUM_STAGES = 6


def build_backbone(checkpoint: str = "") -> nn.Module:
    import torch
    from transformers import AutoModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model


def unfreeze_parameters(model: nn.Module, stages: int = 1) -> list[nn.Parameter]:
    if not 0 <= stages <= NUM_STAGES:
        raise ValueError(f"stages must be between 0 and {NUM_STAGES}")
    bb = model.backbone
    modules = [*bb.blocks[NUM_STAGES - stages :], bb.conv_head, bb.bn2, bb.global_pool, model.bn]
    if stages == NUM_STAGES:
        modules += [bb.conv_stem, bb.bn1]
    return [p for m in modules for p in m.parameters()]
