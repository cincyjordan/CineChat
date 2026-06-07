from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

from generate import generate_response, load_for_inference
from model import CHECKPOINTS_DIR, DEFAULT_MODEL_NAME


def build_interface(model, tokenizer, device) -> gr.Blocks:
    # Construct the Gradio UI. The model, tokenizer, and device are captured
    # in the closure of the respond() callback so the app holds exactly one
    # loaded model for the lifetime of the process.

    def respond(
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
    ) -> str:
        # Called on every button click or Enter keypress. Returns an empty
        # string for blank input so the output box clears rather than showing
        # a stale response.
        if not prompt.strip():
            return ""
        return generate_response(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=int(max_new_tokens),
            temperature=temperature,
            top_p=top_p,
        )

    with gr.Blocks(title="CineChat") as demo:
        gr.Markdown(
            "# CineChat\n"
            "Movie dialogue-style responses from a GPT-2 model fine-tuned on the Cornell Movie Dialogs Corpus."
        )

        with gr.Row():
            with gr.Column():
                prompt_box = gr.Textbox(
                    label="Enter a line of dialogue...",
                    placeholder="You can't handle the truth.",
                    lines=3,
                )

                # Generation controls, ranges and defaults match the PRD spec.
                temperature_slider = gr.Slider(
                    minimum=0.5,
                    maximum=1.5,
                    value=0.9,
                    step=0.05,
                    label="Temperature",
                )
                top_p_slider = gr.Slider(
                    minimum=0.8,
                    maximum=1.0,
                    value=0.95,
                    step=0.01,
                    label="Top-p",
                )
                max_tokens_slider = gr.Slider(
                    minimum=20,
                    maximum=150,
                    value=60,
                    step=5,
                    label="Max new tokens",
                )

                submit_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                output_box = gr.Textbox(
                    label="Generated response",
                    lines=6,
                    interactive=False,
                )

        # Wire both the button click and Enter key to the same callback so
        # the user can submit with either.
        _inputs = [prompt_box, temperature_slider, max_tokens_slider, top_p_slider]
        submit_btn.click(fn=respond, inputs=_inputs, outputs=output_box)
        prompt_box.submit(fn=respond, inputs=_inputs, outputs=output_box)

    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the CineChat Gradio chatbot interface."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint directory (defaults to checkpoints/best).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Generate a public Gradio share link.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("Loading model...")
    try:
        model, tokenizer, device = load_for_inference(
            checkpoint_path=args.checkpoint,
            model_name=args.model_name,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Model loaded on {device}. Launching app...")
    demo = build_interface(model, tokenizer, device)
    demo.launch(share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
