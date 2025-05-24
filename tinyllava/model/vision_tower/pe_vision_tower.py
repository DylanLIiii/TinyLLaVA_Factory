import torch
from tinyllava.model.vision_tower.base import VisionTower
from tinyllava.model.vision_tower.pe.pe import VisionTransformer
from tinyllava.model.vision_tower.pe.config import fetch_pe_checkpoint, PE_VISION_CONFIG

class PEVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        # The actual model will be loaded in _load_model
        self._vision_tower = None
        # PE models typically don't have a separate image processor like CLIP,
        # transforms are often handled by the model itself or in preprocessing.
        # We might need to adjust this based on how PE expects image inputs.
        self._image_processor = None # Placeholder, adjust if PE has specific processor

    def _load_model(self, vision_tower_name, **kwargs):
        # vision_tower_name here is expected to be a key in PE_VISION_CONFIG
        # e.g., "PE-Core-G14-448"
        if vision_tower_name not in PE_VISION_CONFIG:
            raise ValueError(f"Unknown PE vision tower name: {vision_tower_name}. Available: {list(PE_VISION_CONFIG.keys())}")

        # Allow overriding checkpoint_path via kwargs, otherwise use default from fetch_pe_checkpoint
        checkpoint_path = kwargs.pop('pe_checkpoint_path', None)
        
        # TODO: The VisionTransformer.from_config expects `pretrained` and `checkpoint_path`
        # We need to ensure `fetch_pe_checkpoint` is correctly used.
        # The `pretrained` flag in `from_config` will trigger loading via `load_ckpt` which calls `fetch_pe_checkpoint`
        # if checkpoint_path is not provided to `load_ckpt`.
        # If a custom checkpoint_path is given, `fetch_pe_checkpoint` helps resolve it (e.g. hf:// paths)
        # then `from_config` uses it.

        # The `cfg` passed to __init__ might contain general vision tower config,
        # while `PE_VISION_CONFIG[vision_tower_name]` contains PE-specifics.
        # We should rely on PE_VISION_CONFIG for PE model structure.
        
        # `pretrained=True` will make `from_config` call `model.load_ckpt`
        # `load_ckpt` internally calls `fetch_pe_checkpoint` if no path is given to `load_ckpt`
        # If `checkpoint_path` is provided to `from_config`, it's passed to `load_ckpt`.
        try:
            self._vision_tower = VisionTransformer.from_config(
                name=vision_tower_name,
                pretrained=True, # This will trigger loading checkpoints
                checkpoint_path=checkpoint_path, # Pass custom path if any
                **kwargs # Pass any other overrides for PEConfig
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load PE VisionTransformer '{vision_tower_name}': {e}")

        self._vision_tower.requires_grad_(False)
        
        # For PE, image processing might be simpler or integrated.
        # If PE requires specific HuggingFace-style image processor, load it here.
        # For now, assuming preprocessing is handled externally or by VisionTransformer.
        # self._image_processor = ... 

        print(f"Loaded PE vision tower: {vision_tower_name}")

    def forward(self, x: torch.Tensor, **kwargs):
        # PE's VisionTransformer.forward takes image tensor and returns features.
        # It might have its own way of selecting layers or features.
        # The base VisionTower has a specific way of handling features.
        # We need to adapt this to PE's output.

        # Default behavior from base.py:
        # image_features = self._vision_tower(x, output_hidden_states=True)
        # image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]
        # if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
        #   image_features = image_features[:, 1:]
        # ...

        # PE VisionTransformer forward: `def forward(self, x: torch.Tensor, **kwargs)`
        # kwargs can include `layer_idx` for intermediate features.
        # It also has `pool_type` which determines output.
        # If pool_type is 'none', it returns patch features.
        # If pool_type is 'attn', 'avg', 'tok', it returns pooled features.

        # Let's assume we want patch features by default for LMMs,
        # similar to how CLIP features are handled (dropping CLS token).
        # The PE VisionTransformer.forward already handles pooling or returning all tokens.
        # If its `pool_type` is 'none', it returns (B, N, D) features.
        # If its `pool_type` is 'tok' and `use_cls_token` is true, first token is CLS.
        
        # We need to ensure the output is compatible with what the LMM expects.
        # Typically, (Batch, NumPatches, HiddenDim)
        
        # The `VisionTransformer.forward_features` with `norm=True` and `strip_cls_token=True` (if applicable)
        # seems more aligned with what LMMs might expect (patch features).
        # Or, if the main `forward` is used with `pool_type='none'`.

        layer_idx = kwargs.get('vision_feature_layer', -1) # PE uses -1 for last layer
        
        # If PE's VisionTransformer is configured with pool_type='none', 
        # its forward() method will return patch features (B, N, D)
        # If it's configured with pooling, forward() returns (B, D)
        # We need to ensure it returns patch features for typical LMM usage.
        # The PE VisionTransformer has `forward_features` which returns (B, N, D)
        # and `forward` which applies pooling based on its config.

        # Let's use forward_features to get patch features before any pooling.
        # We assume `norm=True` is desired for final features.
        # `strip_cls_token` should be true if `use_cls_token` is true for the model
        # and we only want patch tokens.
        
        # Check if the loaded PE model uses a CLS token
        strip_cls = False
        if self._vision_tower.use_cls_token and kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
            strip_cls = True

        # Use forward_features to get patch features (B, NumTokens, Dim)
        # NumTokens might include CLS token if use_cls_token is True.
        image_features = self._vision_tower.forward_features(
            x,
            norm=True, # Apply final layer norm
            layer_idx=layer_idx,
            strip_cls_token=strip_cls # remove CLS token if present and strategy is 'patch'
        )
        
        # If strategy is 'cls_patch', and CLS token was stripped by `forward_features`
        # we might need to reconsider. However, `strip_cls_token` in `forward_features`
        # is conditional on `self.use_cls_token`.
        # If 'cls_patch' is desired, `strip_cls_token` should be False.
        if kwargs.get('vision_feature_select_strategy') == 'cls_patch' and strip_cls:
            # This case means 'cls_patch' was requested, but 'patch' (which strips cls) was default for strip_cls.
            # We need to re-evaluate how to get CLS + patch or ensure strip_cls is correctly set.
            # For now, let's assume if 'cls_patch' is specified, strip_cls_token was false.
             image_features = self._vision_tower.forward_features(
                x,
                norm=True,
                layer_idx=layer_idx,
                strip_cls_token=False # Keep CLS token for 'cls_patch'
            )


        return image_features

    @property
    def image_processor(self):
        # PE vision tower might not use a HF image processor object in the same way.
        # Preprocessing logic might be in `tinyllava.model.vision_tower.pe.transforms.py`
        # For now, returning None or a placeholder.
        # This needs to be integrated with the data loading pipeline.
        if self._image_processor is None:
            # Placeholder: Load or define transforms based on PE requirements
            # from tinyllava.model.vision_tower.pe.transforms import ...
            # For example, if PE_VISION_CONFIG[self.vision_tower_name].image_size is needed
            # image_size = PE_VISION_CONFIG[self.config.vision_tower].image_size
            # self._image_processor = SomePETransform(image_size)
            print("Warning: PEVisionTower._image_processor is not fully implemented yet.")
        return self._image_processor

    # Required properties from the base class if not handled by _vision_tower directly
    @property
    def hidden_size(self): # Or feature_dim, embed_dim etc.
        if self._vision_tower:
            return self._vision_tower.width # PE VisionTransformer uses 'width' for its main dimension
        return None

    @property
    def num_patches(self):
        # This depends on patch_size and image_size
        # PE VisionTransformer calculates this internally, e.g., grid_h * grid_w
        # It's not directly stored as a property post-init in PE's VisionTransformer.
        # This might need to be calculated based on config or a dummy forward pass, or set during load.
        # For now, returning a placeholder or raising NotImplementedError
        if self._vision_tower and hasattr(self._vision_tower, 'patch_size') and hasattr(self._vision_tower, 'image_size'):
             # This calculation is for square patches and images.
             # PE's VisionTransformer dynamically handles image size.
             # This property is often used to configure the connector.
             # Let's use the default image_size from config for this.
            cfg_image_size = PE_VISION_CONFIG[self.config.vision_tower].image_size
            patch_size = self._vision_tower.patch_size
            return (cfg_image_size // patch_size) ** 2
        # raise NotImplementedError("num_patches property needs to be implemented for PEVisionTower")
        return 256 # Fallback, common for 224px/16px or 336px/14px -> (14*14=196 or 16*16=256 or 24*24=576)
                    # PE G14-448 -> 448/14 = 32 -> 32*32 = 1024 patches
                    # PE L14-336 -> 336/14 = 24 -> 24*24 = 576 patches
                    # This should be derived from actual model config at load time.


# To make this runnable for quick checks (optional)
if __name__ == '__main__':
    # Example usage (requires tinyllava and its dependencies to be in PYTHONPATH)
    # This is a mock config object that base.VisionTower and this class might expect
    class MockConfig:
        def __init__(self, vision_tower_name):
            self.vision_tower = vision_tower_name # Expected by PEVisionTower's num_patches
            self.vision_tower_name = vision_tower_name # Expected by _load_model
            # Add other fields if base class or this class needs them during init
            # self.vision_feature_layer = -1
            # self.vision_feature_select_strategy = 'patch'


    # Test with a PE config name
    # Ensure you have credentials for Hugging Face if downloading for the first time.
    # You might need to be logged in via `huggingface-cli login`
    try:
        pe_model_name = "PE-Core-L14-336" # A smaller one to test
        mock_cfg = MockConfig(vision_tower_name=pe_model_name)
        
        print(f"Attempting to initialize PEVisionTower with {pe_model_name}...")
        pe_tower = PEVisionTower(cfg=mock_cfg)
        
        print(f"Attempting to load model for {pe_model_name}...")
        # Kwargs for _load_model can be passed here if needed, e.g. pe_checkpoint_path
        pe_tower._load_model(vision_tower_name=pe_model_name) 
        print(f"Model {pe_model_name} loaded successfully into PEVisionTower.")
        
        if pe_tower._vision_tower:
            print(f"Vision tower width (hidden_size): {pe_tower.hidden_size}")
            print(f"Vision tower num_patches (calculated for default image size): {pe_tower.num_patches}")

            # Create a dummy image tensor
            # PE-Core-L14-336 expects image_size 336
            image_size = PE_VISION_CONFIG[pe_model_name].image_size
            dummy_image = torch.randn(1, 3, image_size, image_size)
            print(f"Performing a forward pass with dummy image ({image_size}x{image_size})...")
            
            # Mimic kwargs that might come from the main model
            forward_kwargs = {
                'vision_feature_layer': -1, # PE uses -1 for last layer
                'vision_feature_select_strategy': 'patch' # 'patch' or 'cls_patch'
            }
            features = pe_tower.forward(dummy_image, **forward_kwargs)
            print(f"Output features shape: {features.shape}")
            # Expected: (1, num_patches, hidden_size)
            # For L14-336, num_patches = (336/14)^2 = 24^2 = 576
            # Hidden_size (width) = 1024
            # Expected shape: (1, 576, 1024)
            
            if features.shape == (1, (image_size // pe_tower._vision_tower.patch_size)**2, pe_tower.hidden_size):
                print("Forward pass output shape is as expected.")
            else:
                print(f"Warning: Forward pass output shape {features.shape} differs from expected.")

        else:
            print("Vision tower model was not loaded.")

    except Exception as e:
        print(f"An error occurred during PEVisionTower test: {e}")
        import traceback
        traceback.print_exc()
