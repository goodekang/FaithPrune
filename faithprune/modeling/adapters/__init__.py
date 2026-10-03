from .base import GEN, INS, SYS, VIS, ModelAdapter, PreparedInputs, grid_centers


def build_adapter(model_cfg) -> ModelAdapter:
    kind = model_cfg.adapter
    kwargs = dict(path=model_cfg.path, dtype=model_cfg.dtype, device=model_cfg.device)
    if kind == "llava":
        from .llava import LlavaAdapter
        return LlavaAdapter(**kwargs)
    if kind == "llava_next":
        from .llava_next import LlavaNextAdapter
        return LlavaNextAdapter(**kwargs)
    if kind == "qwen2_5_vl":
        from .qwen2_5_vl import Qwen25VLAdapter
        return Qwen25VLAdapter(image_size=model_cfg.image_size, **kwargs)
    raise ValueError(f"Unknown adapter '{kind}'")
