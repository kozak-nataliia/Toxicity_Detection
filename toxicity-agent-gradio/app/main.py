import gradio as gr
from model import ToxicModel

model = ToxicModel()

def classify(text: str):
    p, y = model.predict(text or "")
    return {"toxic_prob": round(p, 4), "label": int(y)}

with gr.Blocks(title="Toxicity Detector") as demo:
    gr.Markdown("# 🧪 Toxicity Detector\nEnter any text and get a toxicity probability and label.")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(lines=6, label="Your text")
            btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            out_json = gr.JSON(label="Prediction")
    btn.click(classify, inputs=inp, outputs=out_json)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
