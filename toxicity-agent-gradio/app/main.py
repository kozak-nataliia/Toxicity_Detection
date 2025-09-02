import os
from dotenv import load_dotenv
import gradio as gr
from agent import make_pipeline

load_dotenv()
from discord_log import log_ask_to_discord, get_last_asks

pipe = make_pipeline()  # llm (maybe None), ToxicModel, agent (maybe None)

### ——— Plain toxicity tab
def classify_only(text: str):
    p, y = pipe.toxic.predict(text or "")
    return {"toxic_prob": round(float(p), 4), "label": int(y)}


### ——— Agent tab (exactly like terminal agent logic)
def agent_ask(user_input: str):
    text = (user_input or "").strip()
    if not text:
        return "Please enter a question."

    # 1) Mirror to Discord log channel (non-blocking best-effort)
    try:
        log_ask_to_discord(text, origin="gradio")
    except Exception:
        pass  # don't break the UI if Discord is unavailable

    # 2) Pull the last 20 asks (oldest→newest) from the Discord log channel
    try:
        history = get_last_asks(k=20)
    except Exception:
        history = []

    # 3) No LLM: keep your existing fallback behavior
    if pipe.agent is None:
        return f"LLM not configured. Toxicity:\n{classify_only(text)}"

    # 4) With LLM: try passing explicit 'history' first (if your agent supports it)
    ctx_blob = "\n".join(f"- {h}" for h in history)
    composed = (
        "Use the following recent Discord asks as context (oldest first). "
        "If asked, summarize and reference them. If not relevant, proceed normally.\n\n"
        "[Recent Discord Asks]\n"
        f"{ctx_blob}\n\n"
        "[User]\n"
        f"{text}"
    )

    try:
        res = pipe.agent.invoke({"input": composed, "history": history})
        return res.get("output", str(res))
    except Exception as e2:
        return f"Agent error: {e2}"


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
