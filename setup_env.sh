#!/bin/bash

# Set Hugging Face cache to use persistent volume
export HF_HOME=/workspace/huggingface
export TRANSFORMERS_CACHE=/workspace/huggingface/hub
export HF_DATASETS_CACHE=/workspace/huggingface/datasets

echo "✓ HF_HOME set to: $HF_HOME"
echo "✓ TRANSFORMERS_CACHE set to: $TRANSFORMERS_CACHE"
echo "✓ Ready to use cached models from /workspace"