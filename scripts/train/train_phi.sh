DATA_PATH=/home/ai/data/llava/dataset/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=/home/ai/data/llava/dataset/text_files/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=/home/ai/data/llava/dataset/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=/home/ai/data/llava/dataset #finetune image dir

LLM_VERSION=microsoft/phi-2 # llm path in huggingface
# Vision tower options:
# - Hugging Face path like 'google/siglip-so400m-patch14-384' for CLIP, SigLIP, etc.
# - For PE (Positional Encoding) vision tower, use "pe:PE_MODEL_NAME",
#   e.g., "pe:PE-Core-G14-448", "pe:PE-Core-L14-336".
#   Ensure the corresponding PE model config exists in `tinyllava.model.vision_tower.pe.config.PE_VISION_CONFIG`.
VT_VERSION=google/siglip-so400m-patch14-384 # vision tower path in huggingface, or "pe:PE_MODEL_NAME"
# Example for PE:
# VT_VERSION="pe:PE-Core-L14-336" 

VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=phi #chat template, other options are: phi, llama, gemmma, etc
VERSION=base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm


bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
