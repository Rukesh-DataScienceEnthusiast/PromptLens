import streamlit as st
from groq import Groq
from typing import Dict, Tuple
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PromptLens – AI Prompt Refinement Studio",
    page_icon="🧠",
    layout="wide"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- PROMPT PATTERNS ----------------
PROMPT_PATTERNS = {
    "Zero-Shot": {"temperature": 0.3},
    "Few-Shot": {"temperature": 0.2},
    "Chain-of-Thought": {"temperature": 0.2},
    "Role-Play": {"temperature": 0.4},
    "Structured Output": {"temperature": 0.1},
    "Meta-Prompting": {"temperature": 0.5},
}

# ---------------- DOMAIN PATTERNS ----------------
DOMAIN_PATTERNS = {
    "coding": {
        "keywords": ["code", "function", "debug", "python", "program"],
        "temperature": 0.1
    },
    "data_analysis": {
        "keywords": ["data", "analyze", "statistics", "trend"],
        "temperature": 0.2
    },
    "creative": {
        "keywords": ["story", "write", "creative", "poem"],
        "temperature": 0.8
    },
    "research": {
        "keywords": ["research", "compare", "study", "analysis"],
        "temperature": 0.3
    }
}

# ---------------- HELPERS ----------------
def detect_domain(prompt: str) -> Tuple[str, float]:
    prompt = prompt.lower()
    scores = {}
    for domain, cfg in DOMAIN_PATTERNS.items():
        score = sum(1 for k in cfg["keywords"] if k in prompt)
        if score:
            scores[domain] = score
    if not scores:
        return "general", 0.5
    best = max(scores, key=scores.get)
    return best, min(scores[best] / 3, 1.0)

def analyze_prompt_quality(prompt: str) -> Dict:
    issues = []
    words = len(prompt.split())
    if words < 5:
        issues.append("Too short – add more context")
    if "format" not in prompt.lower():
        issues.append("No output format specified")
    return {
        "issues": issues,
        "score": max(1, 10 - len(issues) * 2)
    }

def recommend_pattern(domain: str, prompt: str) -> str:
    if "step by step" in prompt.lower():
        return "Chain-of-Thought"
    if "example" in prompt.lower():
        return "Few-Shot"
    return "Zero-Shot"

def model_config(domain: str, pattern: str) -> Dict:
    base = DOMAIN_PATTERNS.get(domain, {}).get("temperature", 0.5)
    pat = PROMPT_PATTERNS.get(pattern, {}).get("temperature", 0.5)
    return {
        "temperature": round((base + pat) / 2, 2),
        "max_tokens": 3000
    }

def build_analysis_prompt(user_prompt, domain, pattern, cfg):
    return f"""
You are an expert Prompt Engineer.

Analyze and improve the prompt using this structure:

### 🔍 Understanding
- Intent
- Ambiguities
- Missing details

### ⚠️ Issues
- Problems found

### ✨ Optimized Prompt
Provide a refined, production-ready prompt using **{pattern}** style.

### ✅ Checklist
- Role
- Context
- Task
- Constraints
- Output format

Domain: {domain}
Recommended Temperature: {cfg['temperature']}
"""

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔑 Configuration")

api_key = st.sidebar.text_input(
    "Enter your Groq API Key",
    type="password",
    help="Get your free key from https://console.groq.com"
)

model_choice = st.sidebar.selectbox(
    "Model",
    [
        "llama-3.1-8b-instant"
    ]
)


# ---------------- UI ----------------
st.markdown("""
<div class="main-header">
<h1>🧠 PromptLens – AI Prompt Refinement Studio</h1>
<p>Learn how AI understands and improves prompts</p>
</div>
""", unsafe_allow_html=True)

user_prompt = st.text_area(
    "✍️ Enter your raw prompt",
    height=180,
    placeholder="Explain anomaly detection in machine learning"
)

refine_btn = st.button("✨ Analyze & Refine Prompt", use_container_width=True)

# ---------------- PROCESS ----------------
if refine_btn:
    if not api_key:
        st.error("Please enter your Groq API key.")
    elif not user_prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("🔍 Analyzing prompt..."):
            domain, confidence = detect_domain(user_prompt)
            quality = analyze_prompt_quality(user_prompt)
            pattern = recommend_pattern(domain, user_prompt)
            cfg = model_config(domain, pattern)

            col1, col2, col3 = st.columns(3)
            col1.metric("Domain", domain)
            col2.metric("Pattern", pattern)
            col3.metric("Quality", f"{quality['score']}/10")

            if quality["issues"]:
                st.warning("⚠️ Issues detected:")
                for i in quality["issues"]:
                    st.write(f"- {i}")

            time.sleep(0.5)

        with st.spinner("🤖 Refining with Groq LLaMA-3..."):
            try:
                client = Groq(api_key=api_key)

                analysis_prompt = build_analysis_prompt(
                    user_prompt, domain, pattern, cfg
                )

                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": "You are an expert Prompt Engineer."},
                        {"role": "user", "content": analysis_prompt + "\n\nUSER PROMPT:\n" + user_prompt}
                    ],
                    temperature=cfg["temperature"],
                    max_tokens=cfg["max_tokens"]
                )

                output = response.choices[0].message.content

                st.markdown("---")
                st.markdown("### 🎯 AI Analysis & Optimized Prompt")
                st.markdown(output)

                st.markdown(f"""
                <div class="success-box">
                ✔ Completed using <strong>{model_choice}</strong><br>
                Pattern: <strong>{pattern}</strong> | Domain: <strong>{domain}</strong>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Groq API Error: {str(e)}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Groq (LLaMA-3) | Prompt Engineering Tool")
