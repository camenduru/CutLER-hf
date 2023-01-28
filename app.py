#!/usr/bin/env python

import pathlib

import gradio as gr

from model import FULLY_SUPERVISED_MODELS, SEMI_SUPERVISED_MODELS, Model

DESCRIPTION = '''# CutLER

This is an unofficial demo for [https://github.com/facebookresearch/CutLER](https://github.com/facebookresearch/CutLER).
'''

model = Model()
paths = sorted(pathlib.Path('CutLER/cutler/demo/imgs').glob('*.jpg'))


def create_unsupervised_demo():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image = gr.Image(label='Input image', type='filepath')
                model_name = gr.Text(label='Model',
                                     value='Unsupervised',
                                     visible=False)
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

        run_button.click(fn=model,
                         inputs=[
                             image,
                             model_name,
                             score_threshold,
                         ],
                         outputs=result)

    return demo


def create_supervised_demo():
    model_names = list(SEMI_SUPERVISED_MODELS.keys()) + list(
        FULLY_SUPERVISED_MODELS.keys())
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image = gr.Image(label='Input image', type='filepath')
                model_name = gr.Dropdown(label='Model',
                                         choices=model_names,
                                         value=model_names[-1])
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

        run_button.click(fn=model,
                         inputs=[
                             image,
                             model_name,
                             score_threshold,
                         ],
                         outputs=result)

    return demo


with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('Zero-shot unsupervised'):
            create_unsupervised_demo()
        with gr.TabItem('Semi/Fully-supervised'):
            create_supervised_demo()
demo.queue().launch()
