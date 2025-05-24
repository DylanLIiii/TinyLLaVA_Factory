import unittest
import torch

# Adjust imports based on the actual project structure if TinyLLaVA is a package
# Assuming TinyLLaVA is in the PYTHONPATH
from tinyllava.model.vision_tower.pe_vision_tower import PEVisionTower
from tinyllava.model.vision_tower.pe.pe import VisionTransformer
from tinyllava.model.vision_tower.pe.config import PE_VISION_CONFIG, fetch_pe_checkpoint
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig # For mock config

class TestPEVisionTower(unittest.TestCase):

    def test_load_and_forward_pe_vision_tower(self):
        # Use a specific, smaller PE model for testing
        # Ensure this model's checkpoint is accessible (e.g., public on Hugging Face Hub)
        pe_model_name = "PE-Core-L14-336" 
        
        if pe_model_name not in PE_VISION_CONFIG:
            self.skipTest(f"PE model config {pe_model_name} not found.")

        # Create a mock TinyLlavaConfig object, similar to how it's done in pe_vision_tower.py's main block
        # The PEVisionTower constructor expects a config object that has a `vision_tower` attribute
        # which holds the full vision tower identifier (e.g., "pe:PE-Core-L14-336")
        # and other attributes like `vision_feature_layer`, `vision_feature_select_strategy`.
        mock_main_config = TinyLlavaConfig()
        mock_main_config.vision_tower = f"pe:{pe_model_name}" # Correctly set for num_patches
        mock_main_config.vision_model_name_or_path = f"pe:{pe_model_name}" # What TinyLlavaConfig.load_from_config would set
        mock_main_config.vision_feature_layer = -1
        mock_main_config.vision_feature_select_strategy = 'patch'

        print(f"Attempting to initialize PEVisionTower with main config for {pe_model_name}...")
        try:
            # The PEVisionTower is initialized with the main TinyLlavaConfig
            pe_tower = PEVisionTower(cfg=mock_main_config)
        except Exception as e:
            self.fail(f"PEVisionTower initialization failed for {pe_model_name}: {e}")

        print(f"Attempting to load model for {pe_model_name}...")
        try:
            # _load_model is called with the specific PE model name (e.g., "PE-Core-L14-336")
            # and potentially other kwargs.
            pe_tower._load_model(vision_tower_name=pe_model_name) 
        except Exception as e:
            # Check if it's a known download issue (e.g. no internet, hf down)
            if "ConnectionError" in str(e) or "HTTPError" in str(e):
                 self.skipTest(f"Skipping test due to network error trying to download {pe_model_name}: {e}")
            self.fail(f"PEVisionTower _load_model failed for {pe_model_name}: {e}")
            
        self.assertIsNotNone(pe_tower._vision_tower, "Vision tower model was not loaded.")
        self.assertIsInstance(pe_tower._vision_tower, VisionTransformer, "Loaded model is not a VisionTransformer instance.")

        print(f"Model {pe_model_name} loaded successfully into PEVisionTower.")

        # Get image_size and patch_size from the loaded PE model's config
        pe_model_actual_config = PE_VISION_CONFIG[pe_model_name]
        image_size = pe_model_actual_config.image_size
        patch_size = pe_model_actual_config.patch_size
        
        # Create a dummy image tensor
        # Shape: (batch_size, num_channels, height, width)
        dummy_image = torch.randn(1, 3, image_size, image_size)
        
        print(f"Performing a forward pass with dummy image ({image_size}x{image_size})...")
        
        # Mimic kwargs that might come from the main model's `encode_images` or `prepare_inputs_labels_for_multimodal`
        forward_kwargs = {
            'vision_feature_layer': mock_main_config.vision_feature_layer,
            'vision_feature_select_strategy': mock_main_config.vision_feature_select_strategy
        }
        
        try:
            features = pe_tower.forward(dummy_image, **forward_kwargs)
        except Exception as e:
            self.fail(f"PEVisionTower forward pass failed: {e}")

        print(f"Output features shape: {features.shape}")
        
        # Calculate expected shape
        # NumPatches = (image_size / patch_size) ^ 2
        # HiddenDim = loaded vision_tower's width
        expected_num_patches = (image_size // patch_size) ** 2
        expected_hidden_size = pe_tower.hidden_size # Accesses _vision_tower.width
        
        self.assertEqual(features.ndim, 3, "Output features should be 3-dimensional (B, N, D).")
        self.assertEqual(features.shape[0], 1, "Batch size of output features should be 1.")
        self.assertEqual(features.shape[1], expected_num_patches, f"Number of patches in output features should be {expected_num_patches}.")
        self.assertEqual(features.shape[2], expected_hidden_size, f"Hidden size of output features should be {expected_hidden_size}.")
        print("Forward pass output shape is as expected.")

if __name__ == '__main__':
    unittest.main()
