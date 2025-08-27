# Toxicity Agent — Gradio (Beginner-Friendly)

Minimal UI for your toxicity classifier using **Gradio**. Runs in **Docker**.

## 1) Prepare
- If you have local weights, put the folder `toxic_roberta_model/` next to this project.
- Copy `.env.example` → `.env` and set:
```
MODEL_ID=/models/toxic_roberta_model   # or your-hf-username/toxic-roberta-model
THRESHOLD=0.5
```

## 2) Run in Docker
```bash
# from inside the project folder
cp .env.example .env
docker build -t toxicity-gradio .
docker run --rm -p 7860:7860 --env-file .env   -v "$(pwd)/toxic_roberta_model:/models/toxic_roberta_model:ro"   toxicity-gradio
# If using a HF Hub model, you can omit the -v volume flag.
```

Open http://localhost:7860 and try some text.
