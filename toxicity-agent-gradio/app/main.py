import os
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr

from agent import make_pipeline
from rag import load_retriever

load_dotenv()

pipe = make_pipeline()  # llm (maybe None), ToxicModel, agent (maybe None)

### ——— Plain toxicity tab
def classify_only(text: str):
    p, y = pipe.toxic.predict(text or "")
    return {"toxic_prob": round(float(p), 4), "label": int(y)}


### ——— Agent tab (exactly like terminal agent logic)
def agent_ask(user_input: str):
    if not user_input.strip():
        return "Please enter a question."
    if pipe.agent is None:
        # Fallback if no OPENAI_API_KEY: run a naive heuristic
        # If user asks about student/profile -> return retrieved context, else run toxicity
        q = user_input.lower()
        about_student = any(k in q for k in ["when was", "who is", "about student", "born", "profile", "nataliia", "student"])
        if about_student:
            return "LLM not configured. Retrieved context:\n\n" + retrieve_only(user_input)
        else:
            return f"LLM not configured. Toxicity:\n{classify_only(user_input)}"
    # With LLM: delegate to the same agent you used in terminal
    try:
        res = pipe.agent.invoke({"input": user_input})
        # LangChain returns dict with 'output' key typically
        return res.get("output", str(res))
    except Exception as e:
        return f"Agent error: {e}"

def build_ui():
    with gr.Blocks(title="🧪 Toxicity + RAG Agent") as demo:
        gr.Markdown("# 🧪 Toxicity Detector + 🎓 Student RAG Agent")

        with gr.Tab("Toxicity"):
            inp = gr.Textbox(lines=6, label="Your text")
            btn = gr.Button("Classify", variant="primary")
            out_json = gr.JSON(label="Prediction")
            btn.click(classify_only, inputs=inp, outputs=out_json)

        with gr.Tab("Agent"):
            info = (
                "This runs the agent, which has two tools — "
                "`toxicity_classifier` and `student_profile_rag`. It will decide which tool to call."
            )
            gr.Markdown(info)
            q2 = gr.Textbox(lines=2, label="Ask anything")
            btn_a = gr.Button("Ask Agent", variant="primary")
            out_a = gr.Textbox(lines=12, label="Agent reply", show_copy_button=True)
            btn_a.click(agent_ask, inputs=q2, outputs=out_a)

        footer = "OpenAI key detected ✅" if os.getenv("OPENAI_API_KEY") else "OpenAI key not set — agent uses fallback ❗"
        gr.Markdown(f"**Status:** {footer}")

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
