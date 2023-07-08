#!/usr/bin/env python

import pathlib

import gradio as gr

from model import run_model

DESCRIPTION = '# [CutLER](https://github.com/facebookresearch/CutLER)'

paths = sorted(pathlib.Path('CutLER/cutler/demo/imgs').glob('*.jpg'))

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            image = gr.Image(label='Input image', type='filepath')
            score_threshold = gr.Slider(label='Score threshold',
                                        minimum=0,
                                        maximum=1,
                                        value=0.5,
                                        step=0.05)
            run_button = gr.Button('Run')
        with gr.Column():
            result = gr.Image(label='Result', type='numpy')
    with gr.Row():
        gr.Examples(examples=[[path.as_posix()] for path in paths],
                    inputs=[image])

    run_button.click(fn=run_model,
                     inputs=[
                         image,
                         score_threshold,
                     ],
                     outputs=result)
demo.queue().launch()
