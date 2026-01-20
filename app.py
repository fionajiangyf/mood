# Import device_utils first to set MPS fallback environment variable before torch loads
import device_utils

import logging
import os
from typing import List, Union

import gradio as gr
from PIL import Image
import numpy as np

from vibe_blending import run_vibe_blend_safe, run_vibe_blend_not_safe
from ipadapter_model import create_image_grid
from llm_planner import analyze_pair_with_llm, judge_best_blend

USE_HUGGINGFACE_ZEROGPU = os.getenv("USE_HUGGINGFACE_ZEROGPU", "false").lower() == "false" #"true"
DEFAULT_CONFIG_PATH = "./config.yaml"

if USE_HUGGINGFACE_ZEROGPU:
    try:
        import spaces
    except ImportError:
        USE_HUGGINGFACE_ZEROGPU = False
        logging.warning("HuggingFace Spaces not available, running without GPU acceleration")

if USE_HUGGINGFACE_ZEROGPU:
    run_vibe_blend_safe = spaces.GPU(duration=60)(run_vibe_blend_safe)
    run_vibe_blend_not_safe = spaces.GPU(duration=60)(run_vibe_blend_not_safe)

    try:
        from download_models import download_ipadapter
        download_ipadapter()
    except ImportError:
        logging.warning("Could not import download_models")


def load_gradio_images_helper(pil_images: Union[List, Image.Image, str]) -> List[Image.Image]:
    """
    Convert various image input formats to a list of PIL Images.
    """
    if pil_images is None:
        return []
    
    # Handle single image
    if isinstance(pil_images, np.ndarray):
        return Image.fromarray(pil_images).convert("RGB")
    if isinstance(pil_images, Image.Image):
        return pil_images.convert("RGB")
    if isinstance(pil_images, str):
        return Image.open(pil_images).convert("RGB")
    
    # Handle list of images
    processed_images = []
    for image in pil_images:
        if isinstance(image, tuple):  # Gradio gallery format
            image = image[0]
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, Image.Image):
            pass  # Already PIL Image
        else:
            continue
        processed_images.append(image.convert("RGB"))
    
    return processed_images

def create_gradio_interface():
    theme = gr.themes.Soft(
        primary_hue="orange",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="*neutral_50",
        block_background_fill="white",
        block_border_width="1px",
        block_shadow="*shadow_sm",
    )

    demo = gr.Blocks(
        theme=theme,
        css="""
        .container { max-width: 1100px; margin: 0 auto; }
        #run_btn { height: 48px; font-weight: 600; }
        """,
    )

    with demo:
        gr.Markdown(
            """
<div class="container">
  <h1 style="margin-bottom: 0.2rem;">Vibe Blending</h1>
  <p style="margin-top: 0; opacity: 0.85;">
    Creatively connect two images using vibe-space blending + LLM planning/judging.
  </p>
</div>
            """
        )

        with gr.Row(equal_height=True):
            # -------- Left: Inputs / Controls --------
            with gr.Column(scale=5, min_width=420):
                with gr.Group():
                    gr.Markdown("### Inputs")

                    with gr.Row():
                        input1 = gr.Image(label="Input A", type="pil", height=240)
                        input2 = gr.Image(label="Input B", type="pil", height=240)

                    creative_prompt = gr.Textbox(
                        label="Creative Intent",
                        placeholder="e.g. poster-like composition, keep identities recognizable, push surreal mid-transition",
                        lines=2,
                    )

                with gr.Accordion("Poster / Film Controls", open=True):
                    poster_format = gr.Dropdown(
                        ["2:3 Poster", "1:1 Social", "16:9 Trailer Frame"],
                        value="2:3 Poster",
                        label="Output Format",
                    )
                    genre = gr.Dropdown(
                        ["Sci-Fi", "Horror", "Romance", "Action", "Luxury", "Indie"],
                        value="Sci-Fi",
                        label="Genre",
                    )
                    tagline = gr.Textbox(
                        label="Tagline (optional)",
                        placeholder="e.g. 'A love story written in neon.'",
                        lines=1,
                    )

                with gr.Accordion("Blending Controls", open=False):
                    with gr.Row():
                        alpha_start = gr.Slider(0, 2, step=0.1, value=0.0, label="α start")
                        alpha_end = gr.Slider(0, 2, step=0.1, value=1.0, label="α end")
                    n_steps = gr.Slider(4, 24, step=1, value=12, label="# outputs")

                    with gr.Row():
                        extra_images = gr.Gallery(
                            label="Extra images (optional)",
                            columns=3,
                            rows=1,
                            height=140,
                        )
                        negative_images = gr.Gallery(
                            label="Negative images (optional)",
                            columns=3,
                            rows=1,
                            height=140,
                        )

                with gr.Accordion("LLM Judge", open=False):
                    judge_toggle = gr.Checkbox(
                        label="Auto-pick best blend",
                        value=True,
                    )
                    judge_criteria = gr.Textbox(
                        label="Judge criteria",
                        placeholder="e.g. most creative but still coherent; poster-like composition",
                        value="Most creative while still coherent and aligned with the creative intent.",
                        lines=2,
                    )

                blend_button = gr.Button("Run Vibe Blending", variant="primary", elem_id="run_btn")

            # -------- Right: Output --------
            with gr.Column(scale=6, min_width=480):
                with gr.Group():
                    gr.Markdown("### Result")
                    blending_results = gr.Image(
                        label="Output",
                        show_label=False,
                        height=520,
                    )
                    llm_explanation = gr.Markdown()

                with gr.Accordion("Examples", open=False):
                    example_cases = [
                        [Image.open("./images/playviolin_hr.png"), Image.open("./images/playguitar_hr.png")],
                        [Image.open("./images/input_cat.png"), Image.open("./images/input_bread.png")],
                        [Image.open("./images/02140_left.jpg"), Image.open("./images/02140_right.jpg")],
                        [Image.open("./images/03969_l.jpg"), Image.open("./images/03969_r.jpg")],
                        [Image.open("./images/04963_l.jpg"), Image.open("./images/04963_r.jpg")],
                        [Image.open("./images/00436_l.jpg"), Image.open("./images/00436_r.jpg")],
                        [Image.open("./images/archi/input_A.jpg"), Image.open("./images/archi/input_B.jpg")],
                    ]
                    gr.Examples(examples=example_cases, inputs=[input1, input2])

        # --- Backend wrapper: keep your existing logic, just add poster brief to prompt ---
        def blend_button_click(
            input1, input2, extra_images, negative_images,
            alpha_start, alpha_end, n_steps,
            creative_prompt, judge_toggle, judge_criteria,
            poster_format, genre, tagline
        ):
            input1 = load_gradio_images_helper(input1)
            input2 = load_gradio_images_helper(input2)
            extra_images = load_gradio_images_helper(extra_images)
            negative_images = load_gradio_images_helper(negative_images)

            if extra_images is None:
                extra_images = []
            elif isinstance(extra_images, Image.Image):
                extra_images = [extra_images]

            if negative_images is None:
                negative_images = []
            elif isinstance(negative_images, Image.Image):
                negative_images = [negative_images]

            # Add poster brief (frontend feature, minimal backend effect)
            brief = f"Poster brief: format={poster_format}, genre={genre}, tagline='{tagline}'."
            full_prompt = (creative_prompt + "\n" + brief).strip()

            llm_suggestion = analyze_pair_with_llm(input1, input2, full_prompt)

            alpha_start_eff = llm_suggestion.get("alpha_start", alpha_start)
            alpha_end_eff = llm_suggestion.get("alpha_end", alpha_end)
            n_steps_eff = llm_suggestion.get("n_steps", None) or int(n_steps)

            alpha_weights = np.linspace(alpha_start_eff, alpha_end_eff, n_steps_eff + 2)[1:-1].tolist()

            blended_images = run_vibe_blend_not_safe(
                input1, input2, extra_images, negative_images, DEFAULT_CONFIG_PATH, alpha_weights
            )

            judge_reason = ""
            if judge_toggle:
                judged = judge_best_blend(input1, input2, blended_images, full_prompt, judge_criteria)
                best_i = judged["best_index"]
                judge_reason = judged["reason"]
                blended_images = [blended_images[best_i]]

            grid = create_image_grid(blended_images, rows=np.ceil(len(blended_images)/4).astype(int), cols=4)

            explanation_md = f"**LLM focus:** {', '.join(llm_suggestion.get('focus_attributes', []))}\n\n"
            explanation_md += llm_suggestion.get("explanation", "")
            if judge_toggle:
                explanation_md += f"\n\n**LLM judge:** {judge_reason}"

            return grid, explanation_md

        blend_button.click(
            blend_button_click,
            inputs=[
                input1, input2, extra_images, negative_images,
                alpha_start, alpha_end, n_steps,
                creative_prompt, judge_toggle, judge_criteria,
                poster_format, genre, tagline
            ],
            outputs=[blending_results, llm_explanation],
        )

    return demo



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    demo = create_gradio_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0" if USE_HUGGINGFACE_ZEROGPU else None,
        show_error=True
    )
