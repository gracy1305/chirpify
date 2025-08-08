# app.py — Chirpify (HF InferenceClient chat_completion – works on "conversational" task)
import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN") # .env must contain: HF_TOKEN=hf_...

MODELS = [
    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "google/gemma-2-2b-it",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

PROMPT_TMPL = """You are Chirpify 🐤 — a friendly grammar coach.
Reply in EXACTLY three lines:

1) ✅ Correction: <corrected sentence> + one friendly emoji
2) 📚 Why: <one-sentence grammar reason>
3) 🐤 Motivation: <short upbeat encouragement with a bird vibe>

User sentence: "{user_sentence}"
"""

def hf_generate_chat(model: str, prompt: str, temperature=0.6, max_tokens=200) -> str:
    client = InferenceClient(model=model.strip(), token=HF_TOKEN, timeout=120)
    try:
        resp = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        return resp.choices[0].message.content
    except HfHubHTTPError as e:
        raise RuntimeError(f"HF chat_completion error: {e.response.status_code} — {e.response.text}") from e

# ---------- UI ----------
st.set_page_config(page_title="Chirpify", page_icon="🐤", layout="centered")
st.title("🐤 Chirpify")
st.subheader("Sharpen your grammar wings.")

# ---- Chirpify mascot styles ----
st.markdown("""
<style>
.bird {
  font-size: 64px;
  display: inline-block;
  animation: hop 0.6s ease-in-out infinite alternate;
}
@keyframes hop {
  0%   { transform: translateY(0) rotate(0deg); }
  100% { transform: translateY(-10px) rotate(-6deg); }
}
.chirp-dots { margin-top: 6px; }
.chirp-dot {
  display:inline-block; width:8px; height:8px; border-radius:50%;
  background:#58CC02; margin:0 4px; opacity:.2;
  animation: blink 1.2s infinite;
}
.chirp-dot:nth-child(2){ animation-delay:.2s }
.chirp-dot:nth-child(3){ animation-delay:.4s }
@keyframes blink { 0%,80%,100%{opacity:.2} 40%{opacity:1} }
</style>
""", unsafe_allow_html=True)

model = st.selectbox("Model (try Zephyr or Qwen first):", MODELS, index=0)
user_input = st.text_area("Type your sentence below:")

if st.button("Chirp It!"):
    if not HF_TOKEN:
        st.error("Missing HF token. Add HF_TOKEN=hf_xxx to your .env and restart.")
    elif not user_input.strip():
        st.warning("Please enter a sentence!")
    else:
        # Show animated bird while generating
        loading = st.empty()
        loading.markdown("""
        <div style="text-align:center; margin: 6px 0 14px 0;">
          <span class="bird">🐤</span>
          <div class="chirp-dots">
            <span class="chirp-dot"></span>
            <span class="chirp-dot"></span>
            <span class="chirp-dot"></span>
          </div>
          <div style="color:#9aa0a6; font-size:13px; margin-top:6px;">Chirping…</div>
        </div>
        """, unsafe_allow_html=True)

        try:
            prompt = PROMPT_TMPL.format(user_sentence=user_input.strip())
            out = hf_generate_chat(model, prompt)
            loading.empty()  # remove animation
            st.success(f"Here’s your Chirp (model: {model}):")
            st.markdown(out.replace("\n", "  \n"))
            st.balloons()
        except Exception as e:
            loading.empty()
            st.error(f"Inference error: {e}")
