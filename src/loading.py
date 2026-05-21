#!/usr/bin/env python3
# loading.py - shared helpers for run.py + run_initial_inference.py

import importlib
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def load_dataset_module(dataset_name: str):
    """Dynamically load the dataset module and handle potential errors."""
    try:
        return importlib.import_module(f"datasets.{dataset_name}")
    except ImportError:
        logging.error(
            f"Failed to import dataset module for '{dataset_name}'. "
            f"Make sure 'datasets/{dataset_name}.py' exists."
        )
        raise


def load_model_and_tokenizer(
    model_path: str,
    device_map: str,
    use_flash_attention: bool = False,
    compile_model: bool = False,
):
    """Load the model and tokenizer with optimizations for H100 (same code shared across runners)."""
    logger.info(f"Loading model: {model_path}")
    logger.info(f"Optimizations: FlashAttention={use_flash_attention}, Compile={compile_model}")

    # Don't trust remote code for Microsoft Phi models (security concern)
    trust_remote = "microsoft/Phi" not in model_path
    if not trust_remote:
        logger.warning(f"Disabling trust_remote_code for Microsoft Phi model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote)

    # Set pad_token if not set (required for batched inference)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Set left padding for decoder-only models (required for correct batched generation)
    tokenizer.padding_side = "left"
    logger.info("Set padding_side to 'left' for correct batched generation")

    # Use bfloat16 on H100+ for better performance
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    logger.info(f"Using dtype: {dtype}")

    common_kwargs = {
        "pretrained_model_name_or_path": model_path,
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote,
    }

    # Enable FlashAttention 2 if available (2-4x faster)
    if use_flash_attention:
        try:
            common_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Attempting to use FlashAttention 2")
        except Exception as e:
            logger.warning(f"FlashAttention 2 not available: {e}")
            common_kwargs["attn_implementation"] = "sdpa"  # fallback
            logger.info("Using SDPA (scaled dot product attention)")

    if device_map == "auto":
        model = AutoModelForCausalLM.from_pretrained(**common_kwargs, device_map="auto")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        logger.info(f"Using single device: {device}")
        model = AutoModelForCausalLM.from_pretrained(**common_kwargs).to(device)

    model.eval()

    # Optional: Compile model for additional speedup (requires PyTorch 2.0+)
    if compile_model and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile (first run will be slow)...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled successfully")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")

    # Log GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    return model, tokenizer
