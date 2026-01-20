---
title: VibeSpace
emoji: 🚀
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 5.24.0
app_file: app.py
pinned: false
---


step1: pip install -r ./requirements.txt

step2: create openai api key: https://platform.openai.com/api-keys

step3: run `export OPENAI_API_KEY="actual_api_key"`

step4: run `python app.py` or demo notebooks


# LLM-Driven Alpha Adjustment
The LLM planner now dynamically adjusts alpha_start and alpha_end based on the actual content of the input images and the user’s creative intent.
Previously, vibe blending used a fixed alpha range (default 0 → 1), which produced a uniform, linear transition regardless of how similar or different the images were.

By describing each image with a vision model and analyzing their shared attributes, the LLM selects an alpha range that better fits the semantics of the pair:
* Similar images → narrower alpha range for smooth, subtle transitions
* Very different images → wider alpha range (e.g. alpha_end > 1) for stronger or more expressive blending
* Creative requests → LLM may push into extrapolation (>1) to increase variation
* Subtle requests → LLM keeps alpha_end low (<1) for conservative blending

This makes transitions more meaningful, reduces visual/blending distortions and anomalies, and produces blends that better match both the images and the creative goal.

Currently, image semantics are always considered; creative intent is optional for the user.
