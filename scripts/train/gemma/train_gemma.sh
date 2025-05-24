DATA_PATH=/home/ai/data/llava/dataset/text_files/blip_laion_cc_sbu_558k.json
FINETUNE_DATA_PATH=/home/ai/data/llava/dataset/text_files/llava_v1_5_mix665k.json
IMAGE_PATH=/home/ai/data/llava/dataset/llava/llava_pretrain/images
FINETUNE_IMAGE_PATH=/home/ai/data/llava/dataset

LLM_VERSION=google/gemma-2b-it
# Vision tower options:
# - Hugging Face path like 'google/siglip-so400m-patch14-384' for CLIP, SigLIP, etc.
# - For PE (Positional Encoding) vision tower, use "pe:PE_MODEL_NAME",
#   e.g., "pe:PE-Core-G14-448", "pe:PE-Core-L14-336".
#   Ensure the corresponding PE model config exists in `tinyllava.model.vision_tower.pe.config.PE_VISION_CONFIG`.
VT_VERSION=google/siglip-so400m-patch14-384 # vision tower path in huggingface, or "pe:PE_MODEL_NAME"
# Example for PE:
# VT_VERSION="pe:PE-Core-L14-336" 

VT_VERSION2=""
CN_VERSION=mlp2x_gelu
CONV_VERSION=gemma
VERSION=base
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=2048


bash scripts/train/gemma/pretrain_gemma.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train/gemma/finetune_gemma.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
