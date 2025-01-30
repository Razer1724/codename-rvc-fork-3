import gradio as gr
import torch
from pathlib import Path
import faiss
import os

# default output directory
DEFAULT_OUTPUT_DIR = os.path.join("logs")

def create_uvcp(pth_path, index_path=None, output_path=None):
    """Create .uvcp file combining PTH and optional FAISS index"""
    pth_path = Path(pth_path)
    if not pth_path.exists():
        raise FileNotFoundError(f"PTH file not found: {pth_path}")

    uvcp_data = {"model_state": torch.load(pth_path, map_location="cpu", weights_only=True)}

    if index_path:
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        index = faiss.read_index(str(index_path))
        uvcp_data["index_data"] = faiss.serialize_index(index)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    if output_path:
        output_path = Path(output_path)
    else:
        output_path = Path(DEFAULT_OUTPUT_DIR) / pth_path.with_suffix(".uvcp").name

    torch.save(uvcp_data, output_path)
    return str(output_path)

def uvcp_interface(pth_file, index_file, output_path):
    try:
        generated_path = create_uvcp(
            pth_file,
            index_file if index_file else None,
            output_path if output_path else None
        )
        return f"UVCP file created successfully!\nSaved to: {generated_path}", generated_path
    except Exception as e:
        return f"Error: {str(e)}", None

def uvcp_creator_tab():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                pth_input = gr.File(label="Input PTH File", file_types=[".pth"])
                index_input = gr.File(label="FAISS Index File (optional)", file_types=[".index"])
                output_path = gr.Textbox(label="Custom Output Path (optional)")
                create_btn = gr.Button("Create UVCP File", variant="primary")
                
            with gr.Column():
                result_output = gr.Textbox(label="Status", interactive=False)
                file_output = gr.File(label="Download UVCP File", interactive=False)

        create_btn.click(
            uvcp_interface,
            inputs=[pth_input, index_input, output_path],
            outputs=[result_output, file_output]
        )

if __name__ == "__main__":
    with gr.Blocks(title="UVCP Creator") as demo:
        uvcp_creator_tab()
    demo.launch()