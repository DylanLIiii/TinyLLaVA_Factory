from .clip import CLIPVisionTower
from .dinov2 import Dinov2VisionTower
from .siglip import SiglipVisionTower
from .mof import MoFVisionTower
from .pe_vision_tower import PEVisionTower # Added this line

VISION_TOWER_REGISTRY = {
    'clip': CLIPVisionTower,
    'dinov2': Dinov2VisionTower,
    'siglip': SiglipVisionTower,
    'mof': MoFVisionTower,
    'pe': PEVisionTower, # Added this line
}

def build_vision_tower(cfg, **kwargs):
    vision_tower_identifier = cfg.vision_tower_name # This will be like "pe:PE-Core-G14-448" or "clip"
    
    # Handle cases where it might be a direct name (e.g., "clip") or a prefixed name (e.g., "pe:PE-Core-G14-448")
    registry_key = vision_tower_identifier.split(':')[0] # Get "pe" or "clip"

    if registry_key not in VISION_TOWER_REGISTRY:
        raise ValueError(f"Unknown vision tower type: {registry_key} (from {vision_tower_identifier})")
    
    # The vision tower class (e.g., PEVisionTower) will receive the full config `cfg`.
    # PEVisionTower's __init__ takes `cfg`, and its `_load_model` will use `cfg.vision_tower_name`
    # (which is the PE-specific model name like "PE-Core-G14-448" due to train.py's processing)
    # or more directly, the `vision_tower_name` argument passed to `_load_model`.
    # The `TinyLlavaConfig` object (cfg) will have `cfg.vision_model_name_or_path` as the full identifier "pe:PE-Core-G14-448".
    # The `PEVisionTower`'s `_load_model` expects the PE model name like "PE-Core-G14-448", which it gets from
    # `model_args['vision_tower']['model_name_or_path']` in `train.py`'s `model.load_vision_tower` call.
    
    # We might need to pass the specific PE model name to PEVisionTower if it can't reliably get it.
    # For now, let's assume PEVisionTower's _load_model gets the correct name.
    # The `cfg` object passed to the vision tower constructor is `TinyLlavaConfig.vision_config`,
    # which might be problematic if AutoConfig failed for "PE-Core-G14-448".
    # However, `PEVisionTower` uses `VisionTransformer.from_config` which handles its own config.

    # Let's ensure the vision tower class gets the main `TinyLlavaConfig` (`cfg_model` in this context,
    # which is `model_config` in train.py) not just `cfg.vision_config`.
    # The current factory signature passes `cfg` which is `config.vision_config` in `TinyLlavaForConditionalGeneration.__init__`.
    # This needs to be compatible with how other vision towers expect their config.
    # Let's assume the existing `cfg` (which is `vision_config` from `TinyLlavaConfig`) is what towers expect or can handle.
    # `PEVisionTower` uses `self.config` (passed as `cfg` to `__init__`) for `self.config.vision_tower` in `num_patches`,
    # so `cfg` should be the main `TinyLlavaConfig` object.

    # Revisiting TinyLlavaForConditionalGeneration.__init__():
    # self.vision_tower = VisionTowerFactory(config.vision_model_name_or_path)(config)
    # Here, `config` is the main `TinyLlavaConfig` object. So PEVisionTower will receive the main config.
    # This is good. `PEVisionTower`'s `_load_model` is called later by `model.load_vision_tower` with the
    # PE-specific name.

    return VISION_TOWER_REGISTRY[registry_key](cfg, **kwargs) # cfg here is TinyLlavaConfig
