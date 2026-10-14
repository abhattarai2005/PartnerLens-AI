import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="PartnerLens AI",
    page_icon="💞",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "PartnerLens AI"
DEFAULT_MODEL = "gpt-6-luna"
MAX_AI_MESSAGES = 50

FEATURES = {
    "communication": "Communication",
    "emotional_maturity": "Emotional Maturity",
    "independence": "Independence",
    "ambition": "Ambition",
    "family_orientation": "Family Orientation",
    "affection": "Affection",
    "social_lifestyle": "Social Lifestyle",
    "fitness": "Fitness",
    "financial_planning": "Financial Planning",
    "adventure": "Adventure / Spontaneity",
}

FEATURE_ICONS = {
    "communication": "💬",
    "emotional_maturity": "🧠",
    "independence": "🕊️",
    "ambition": "🚀",
    "family_orientation": "🏡",
    "affection": "💗",
    "social_lifestyle": "🎉",
    "fitness": "💪",
    "financial_planning": "💰",
    "adventure": "🌍",
}

QUESTIONS = {
    "communication": "How important is open and consistent communication?",
    "emotional_maturity": "How important is calm, emotionally mature conflict handling?",
    "independence": "How important is personal space and independent goals?",
    "ambition": "How important is career or personal ambition?",
    "family_orientation": "How important is being family-oriented?",
    "affection": "How important is affection and reassurance?",
    "social_lifestyle": "How important is having an active social lifestyle?",
    "fitness": "How important is an active and healthy lifestyle?",
    "financial_planning": "How important is responsible financial planning?",
    "adventure": "How important are travel, novelty, and spontaneity?",
}

SCENARIOS = {
    "Canceled plans because of work": "Your partner canceled plans because work became unexpectedly busy.",
    "Different texting needs": "One person wants frequent texts while the other prefers fewer check-ins.",
    "Need for personal space": "Your partner asks for some time alone after a stressful day.",
    "Career move to another state": "A strong career opportunity would require one partner to move.",
    "Money and spending disagreement": "You have different ideas about saving, spending, and shared expenses.",
    "Friendship and jealousy boundary": "A friendship is creating insecurity and a need for a boundary conversation.",
}

PARTNER_STYLES = {
    "Calm Communicator": "Warm, reflective, asks questions, and tries to resolve conflict collaboratively.",
    "Independent / Needs Space": "Values autonomy, personal time, and calm conversations without constant contact.",
    "Affectionate Reassurer": "Values emotional closeness, warmth, reassurance, and frequent connection.",
    "Career-Driven / Busy": "Cares about the relationship while strongly prioritizing goals and responsibilities.",
    "Conflict-Avoidant": "Finds tension uncomfortable and may need extra time before fully engaging.",
    "Direct Communicator": "Prefers clear, straightforward discussion over hints or indirect wording.",
}


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "saved_profile": None,
    "saved_profile_name": "My Profile",
    "guide_messages": [],
    "sim_messages": [],
    "sim_signature": None,
    "ai_calls": 0,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@700;800&display=swap');

:root {
    --pink: #ff4f91;
    --purple: #7c5cff;
    --blue: #45a3ff;
    --cyan: #34d7d2;
    --ink: #20243a;
    --muted: #6d7287;
    --border: rgba(105, 87, 165, .14);
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 5%, rgba(255,79,145,.14), transparent 25%),
        radial-gradient(circle at 92% 8%, rgba(124,92,255,.16), transparent 28%),
        radial-gradient(circle at 78% 90%, rgba(52,215,210,.14), transparent 25%),
        linear-gradient(135deg, #fffafe 0%, #f7f5ff 50%, #f5fcff 100%);
    color: var(--ink);
}

.block-container {
    max-width: 1380px;
    padding-top: 1.35rem;
    padding-bottom: 4rem;
}

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,.82);
    border-right: 1px solid var(--border);
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 2.25rem 2.4rem;
    margin-bottom: 1rem;
    border-radius: 30px;
    border: 1px solid var(--border);
    background: linear-gradient(120deg, rgba(255,255,255,.90), rgba(244,236,255,.85), rgba(230,248,255,.86));
    box-shadow: 0 20px 60px rgba(74,55,120,.10);
}

.hero::after {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    right: -70px;
    top: -95px;
    border-radius: 50%;
    background: linear-gradient(135deg, rgba(255,79,145,.33), rgba(124,92,255,.33));
}

.hero-kicker {
    position: relative;
    z-index: 2;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: #7557d1;
}

.hero-title {
    position: relative;
    z-index: 2;
    margin: .45rem 0 .75rem 0;
    font-family: "Manrope", sans-serif;
    font-size: clamp(2.4rem, 4vw, 4.5rem);
    line-height: 1;
    font-weight: 800;
    background: linear-gradient(90deg, #ff4f91, #7c5cff, #2f9fe8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-copy {
    position: relative;
    z-index: 2;
    max-width: 880px;
    margin: 0;
    color: #565c72;
    line-height: 1.72;
    font-size: 1.04rem;
}

.card {
    padding: 1.1rem 1.2rem;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,.78);
    box-shadow: 0 12px 35px rgba(63,48,105,.06);
    margin-bottom: .8rem;
}

.status-online {
    padding: .75rem 1rem;
    border-radius: 15px;
    background: rgba(52,215,210,.11);
    border: 1px solid rgba(52,180,190,.18);
}

.status-offline {
    padding: .75rem 1rem;
    border-radius: 15px;
    background: rgba(255,191,75,.13);
    border: 1px solid rgba(210,155,65,.20);
}

.stButton > button, .stDownloadButton > button {
    border-radius: 13px;
    font-weight: 700;
    border: 1px solid rgba(124,92,255,.16);
}

button[kind="primary"] {
    background: linear-gradient(90deg, #ff4f91, #7c5cff) !important;
    color: white !important;
    border: none !important;
}

div[data-testid="stMetric"] {
    padding: 1rem;
    border-radius: 18px;
    background: rgba(255,255,255,.76);
    border: 1px solid var(--border);
}

div[data-testid="stChatMessage"] {
    border-radius: 18px;
    border: 1px solid rgba(105,87,165,.10);
    background: rgba(255,255,255,.69);
    padding: .45rem .65rem;
    margin-bottom: .45rem;
}

footer { visibility: hidden; }

@media (max-width: 850px) {
    .hero { padding: 1.6rem 1.3rem; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def card(title, body, icon="✨"):
    st.markdown(
        f'<div class="card"><div style="font-size:1.08rem;font-weight:800;margin-bottom:.35rem;">'
        f'{icon} {title}</div><div style="color:#61677d;line-height:1.6;">{body}</div></div>',
        unsafe_allow_html=True,
    )


def section_header(kicker, title, body=None):
    st.caption(kicker.upper())
    st.header(title)
    if body:
        st.write(body)


def resolve_api_key():
    try:
        value = st.secrets.get("OPENAI_API_KEY", "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "").strip()


def resolve_model():
    try:
        value = st.secrets.get("OPENAI_MODEL", "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


OPENAI_API_KEY = resolve_api_key()
OPENAI_MODEL = resolve_model()
AI_AVAILABLE = bool(OPENAI_API_KEY and OpenAI is not None)


def call_ai(messages, instructions):
    if not AI_AVAILABLE:
        raise RuntimeError("AI is not configured.")

    if st.session_state.ai_calls >= MAX_AI_MESSAGES:
        raise RuntimeError("This demo reached its AI message limit for this session.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    recent = messages[-16:]
    api_input = [
        {"role": item["role"], "content": item["content"]}
        for item in recent
        if item["role"] in {"user", "assistant"}
    ]

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=api_input,
    )

    text = getattr(response, "output_text", "")
    if not text:
        raise RuntimeError("The AI returned an empty response.")

    st.session_state.ai_calls += 1
    return text.strip()


def guide_instructions():
    profile_text = "No saved profile is available."
    if st.session_state.saved_profile:
        ranked = sorted(
            st.session_state.saved_profile.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:4]
        profile_text = "Saved preference priorities: " + ", ".join(
            f"{FEATURES[k]} {v}/5" for k, v in ranked
        )

    return f"""
You are PartnerLens Guide, a natural conversational relationship-reflection assistant for adults 18+.
Answer like a polished chat assistant: understand context, remember earlier turns, respond directly,
and do not repeat generic stock answers.

Important behavior:
- Treat short messages like "hurt", "sad", "angry", "confused", "why?", or "what should I say?" as meaningful follow-ups.
- Use the prior conversation context.
- Be warm, practical, concise, and specific.
- When useful, provide a sample message the user could send.
- Never claim to know another person's hidden thoughts.
- Do not diagnose people or casually label them narcissistic, toxic, avoidant, abusive, etc.
- Do not predict whether a relationship will definitely succeed or fail.
- If the user describes threats, violence, coercion, stalking, or immediate danger, prioritize real-world safety.
- This is a reflection and communication tool, not therapy or medical care.

Optional profile context:
{profile_text}
""".strip()


def simulator_instructions(scenario, style, difficulty):
    return f"""
You are role-playing a hypothetical adult partner in a communication-practice simulator.

Scenario:
{SCENARIOS[scenario]}

Partner style:
{style} — {PARTNER_STYLES[style]}

Difficulty:
{difficulty}

Rules:
- Stay in character as the hypothetical partner.
- Respond naturally to the user's exact latest message and remember the conversation.
- If the user says only "hurt", "sad", "angry", "confused", etc., acknowledge that specific feeling and continue naturally.
- Do not give coaching scores while in character.
- Usually answer in 1-4 sentences.
- Gentle = warm and collaborative.
- Realistic = balanced; do not automatically agree.
- Challenging = respectful but firmer about your own needs.
- Never insult, threaten, shame, manipulate, or pressure the user.
- Never pretend to be a real human outside this simulation.
""".strip()


def local_guide_reply(text):
    lower = text.lower().strip()
    if any(word in lower for word in ["hurt", "sad", "upset", "angry", "frustrated"]):
        return (
            "That sounds emotionally important. Try separating what happened from what it made you feel. "
            "A simple way to say it is: “When that happened, I felt hurt. I want to understand what happened, "
            "and I’d like us to talk about what we can do differently next time.”"
        )
    if "boundary" in lower:
        return (
            "A useful boundary is specific, respectful, and focused on what you will do. "
            "For example: “I’m willing to talk about this, but I’m not comfortable continuing the conversation "
            "if we start insulting each other. We can take a break and come back to it.”"
        )
    if "compatib" in lower:
        return (
            "A difference matters most when it affects a core value, repeated daily behavior, or a decision neither "
            "person can comfortably compromise on. Look at the pattern, not just one disagreement."
        )
    return (
        "Tell me what happened, how you felt, and what you want to happen next. "
        "I can help you turn that into a clear message or think through the situation."
    )


def collect_profile(prefix, defaults=None):
    profile = {}
    for key, question in QUESTIONS.items():
        value = 3 if not defaults else int(defaults.get(key, 3))
        profile[key] = st.slider(
            f"{FEATURE_ICONS[key]} {question}",
            min_value=1,
            max_value=5,
            value=value,
            key=f"{prefix}_{key}",
        )
    return profile


def compatibility_score(a, b):
    distances = [abs(a[k] - b[k]) / 4 for k in FEATURES]
    return round((1 - np.mean(distances)) * 100, 1)


def profile_dataframe(profile):
    return pd.DataFrame(
        [{"Dimension": FEATURES[k], "Score": v, "Percent": v * 20} for k, v in profile.items()]
    )


def radar_chart(profile, title):
    labels = [FEATURES[k] for k in FEATURES]
    values = [profile[k] for k in FEATURES]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values = values + values[:1]
    angles = angles + angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2.3)
    ax.fill(angles, values, alpha=.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5)
    ax.set_title(title, pad=22, fontweight="bold")
    plt.tight_layout()
    return fig


def message_analysis(text):
    lower = text.lower()
    score = 60
    positives = []
    improvements = []

    if any(x in lower for x in ["i feel", "i felt", "i'm hurt", "i am hurt", "i'm upset", "i am upset"]):
        score += 12
        positives.append("You name your emotional experience.")
    else:
        improvements.append("Name the feeling you experienced.")

    if any(x in lower for x in ["can we", "could we", "would you", "i need", "i would like"]):
        score += 12
        positives.append("You include a request or next step.")
    else:
        improvements.append("Add a clear request or next step.")

    if any(x in lower for x in ["always", "never", "whatever", "your fault", "you make me"]):
        score -= 18
        improvements.append("Replace absolute or blaming wording with the specific behavior.")

    if "?" in text:
        score += 6
        positives.append("You invite dialogue.")

    score = int(np.clip(score, 0, 100))
    return score, positives, improvements


def synthetic_profiles(n=700, seed=42):
    rng = np.random.default_rng(seed)
    base = rng.uniform(1.3, 4.7, size=(n, len(FEATURES)))
    return pd.DataFrame(base, columns=list(FEATURES.keys()))


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div style="padding:.35rem 0 .7rem 0;">'
    '<div style="font-family:Manrope;font-size:1.45rem;font-weight:800;">💞 PartnerLens AI</div>'
    '<div style="font-size:.78rem;color:#777b8f;margin-top:.2rem;">Relationship self-discovery studio</div>'
    '</div>',
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Explore",
    [
        "🏠 Home",
        "🧠 Partner Blueprint",
        "🎭 Scenario Discovery",
        "🤝 Compatibility Lab",
        "🧭 AI Relationship Guide",
        "💬 Conversation Coach",
        "🗨️ AI Chat Simulator",
        "🚦 Relationship Reflection",
        "🤖 ML Intelligence",
        "📊 My Profile Center",
    ],
)

st.sidebar.divider()

if AI_AVAILABLE:
    st.sidebar.success(f"● AI Online · {OPENAI_MODEL}")
else:
    st.sidebar.warning("○ AI Offline · add OpenAI key")

st.sidebar.caption(f"AI messages this session: {st.session_state.ai_calls}/{MAX_AI_MESSAGES}")

if st.sidebar.button("🧹 Clear entire session", use_container_width=True):
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value
    st.rerun()


# ============================================================
# GLOBAL HERO
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-kicker">AI-powered relationship self-discovery</div>'
    '<div class="hero-title">💞 PartnerLens AI</div>'
    '<p class="hero-copy">Understand yourself, compare relationship preferences, practice healthier '
    'communication, and explore patterns with explainable machine learning and conversational AI.</p>'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "For adults 18+. PartnerLens supports reflection and communication practice. "
    "It does not diagnose people or guarantee relationship outcomes."
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":
    section_header(
        "Welcome",
        "A smarter relationship reflection experience",
        "Build a preference profile, compare compatibility, practice conversations, and explore explainable patterns.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Preference dimensions", "10")
    c2.metric("AI conversation", "Live" if AI_AVAILABLE else "Setup needed")
    c3.metric("ML tools", "K-Means + PCA")
    c4.metric("Privacy", "Session based")

    st.divider()

    a, b, c = st.columns(3)
    with a:
        card("Partner Blueprint", "Score the relationship qualities that matter most to you.", "🧠")
        card("Compatibility Lab", "Compare two profiles and see where preferences align or differ.", "🤝")
    with b:
        card("AI Relationship Guide", "Ask natural follow-up questions and keep conversation context.", "🧭")
        card("Conversation Coach", "Improve difficult messages using clear communication signals.", "💬")
    with c:
        card("AI Chat Simulator", "Practice a realistic conversation with different partner styles.", "🗨️")
        card("ML Intelligence", "Explore synthetic preference clusters with PCA and K-Means.", "🤖")

    if not AI_AVAILABLE:
        st.warning(
            "The AI pages are currently using a local fallback. To make them answer like a real conversational AI, "
            "add your OPENAI_API_KEY in Streamlit Cloud → Manage app → Settings → Secrets."
        )


# ============================================================
# BLUEPRINT
# ============================================================

elif page == "🧠 Partner Blueprint":
    section_header(
        "Self-discovery",
        "🧠 Build your Partner Blueprint",
        "Rate how important each quality is to you from 1 to 5.",
    )

    profile_name = st.text_input("Profile name", st.session_state.saved_profile_name)
    profile = collect_profile("blueprint", st.session_state.saved_profile)

    if st.button("💾 Save Blueprint", type="primary"):
        st.session_state.saved_profile = profile.copy()
        st.session_state.saved_profile_name = profile_name.strip() or "My Profile"
        st.success("Blueprint saved for this session.")

    if st.session_state.saved_profile:
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(profile_dataframe(st.session_state.saved_profile), use_container_width=True, hide_index=True)
        with right:
            fig = radar_chart(st.session_state.saved_profile, "Your Relationship Preference Radar")
            st.pyplot(fig)
            plt.close(fig)


# ============================================================
# SCENARIO DISCOVERY
# ============================================================

elif page == "🎭 Scenario Discovery":
    section_header(
        "Behavioral reflection",
        "🎭 Scenario Discovery",
        "Choose the response that feels closest to you. There is no right answer.",
    )

    questions = [
        ("After a stressful day, what usually helps most?", ["Space first", "Talk it through", "Physical comfort", "Do something fun"]),
        ("For a free weekend, what sounds best?", ["Stay home together", "Big social event", "A few close friends", "Separate activities then reconnect"]),
        ("During a disagreement, what do you prefer?", ["Talk immediately", "Cool off then talk", "Write/text first", "Let small things go"]),
        ("If one partner gets a job in another state?", ["Move together", "Compare both careers", "Try long distance", "Keep my current plan"]),
    ]

    answers = []
    for i, (question, options) in enumerate(questions):
        answers.append(st.radio(question, options, key=f"scenario_{i}"))

    if st.button("✨ Analyze my pattern", type="primary"):
        text = "Your choices suggest you may value "
        tags = []
        if "Talk it through" in answers or "Talk immediately" in answers:
            tags.append("direct communication")
        if "Space first" in answers or "Cool off then talk" in answers:
            tags.append("emotional space")
        if "Move together" in answers or "Stay home together" in answers:
            tags.append("closeness")
        if "Compare both careers" in answers or "Separate activities then reconnect" in answers:
            tags.append("independence and balance")
        if not tags:
            tags = ["flexibility and context"]
        st.success(text + ", ".join(tags) + ".")


# ============================================================
# COMPATIBILITY
# ============================================================

elif page == "🤝 Compatibility Lab":
    section_header(
        "Preference alignment",
        "🤝 Compatibility Lab",
        "Compare two preference profiles. This is an alignment score, not a prediction of relationship success.",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Person A")
        if st.session_state.saved_profile:
            use_saved = st.checkbox("Use my saved Blueprint", value=True)
        else:
            use_saved = False
        a_profile = st.session_state.saved_profile if use_saved else collect_profile("compat_a")

    with col2:
        st.subheader("Person B")
        b_profile = collect_profile("compat_b")

    if st.button("🤝 Compare profiles", type="primary"):
        score = compatibility_score(a_profile, b_profile)
        st.metric("Overall preference alignment", f"{score}%")

        rows = []
        for key in FEATURES:
            diff = abs(a_profile[key] - b_profile[key])
            rows.append({
                "Dimension": FEATURES[key],
                "Person A": a_profile[key],
                "Person B": b_profile[key],
                "Alignment %": round((1 - diff / 4) * 100),
            })
        df = pd.DataFrame(rows).sort_values("Alignment %", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download comparison CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="partnerlens_compatibility.csv",
            mime="text/csv",
        )


# ============================================================
# AI GUIDE
# ============================================================

elif page == "🧭 AI Relationship Guide":
    section_header(
        "Conversational AI",
        "🧭 AI Relationship Guide",
        "Chat naturally. Ask follow-ups, give short emotional replies, or ask for help writing what to say.",
    )

    if AI_AVAILABLE:
        st.markdown(
            f'<div class="status-online"><b>● AI is online</b><br>'
            f'<span style="color:#687082;">Model: {OPENAI_MODEL}. Conversation context is active.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-offline"><b>○ AI is not configured</b><br>'
            '<span style="color:#687082;">Add OPENAI_API_KEY to Streamlit Secrets for real AI responses. '
            'Until then, a limited local fallback is used.</span></div>',
            unsafe_allow_html=True,
        )

    top1, top2 = st.columns([1, 3])
    with top1:
        if st.button("🧹 New chat", use_container_width=True):
            st.session_state.guide_messages = []
            st.rerun()
    with top2:
        st.caption('Try: “I feel hurt” → “why?” → “what should I say to them?”')

    st.divider()

    for message in st.session_state.guide_messages:
        avatar = "🧭" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    prompt = st.chat_input("Message PartnerLens Guide...")

    if prompt:
        st.session_state.guide_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        try:
            if AI_AVAILABLE:
                with st.spinner("Thinking..."):
                    reply = call_ai(st.session_state.guide_messages, guide_instructions())
            else:
                reply = local_guide_reply(prompt)
        except Exception as exc:
            reply = (
                "I couldn't reach the AI service for this message. "
                "Please check your Streamlit secret and API account, then try again.\n\n"
                f"`{type(exc).__name__}: {exc}`"
            )

        st.session_state.guide_messages.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant", avatar="🧭"):
            st.markdown(reply)


# ============================================================
# CONVERSATION COACH
# ============================================================

elif page == "💬 Conversation Coach":
    section_header(
        "Explainable communication analysis",
        "💬 Conversation Coach",
        "Draft a difficult message and get transparent feedback on clarity, feelings, requests, and blame language.",
    )

    situation = st.selectbox("Situation", list(SCENARIOS.keys()))
    card("Scenario", SCENARIOS[situation], "🎬")

    draft = st.text_area(
        "Write the message you want to send",
        height=180,
        placeholder="Example: I felt disappointed when our plans changed at the last minute. Can we talk about how to handle this next time?",
    )

    if st.button("🔍 Analyze message", type="primary"):
        if not draft.strip():
            st.warning("Write a message first.")
        else:
            score, positives, improvements = message_analysis(draft)
            st.metric("Communication quality", f"{score}%")

            left, right = st.columns(2)
            with left:
                st.subheader("What works")
                if positives:
                    for item in positives:
                        st.success(item)
                else:
                    st.info("Your message has a starting point. Add more detail to strengthen it.")

            with right:
                st.subheader("What to improve")
                if improvements:
                    for item in improvements:
                        st.warning(item)
                else:
                    st.success("The message already has strong communication signals.")

            if AI_AVAILABLE:
                if st.button("✨ Rewrite with AI"):
                    messages = [{"role": "user", "content": f"Rewrite this message to be calm, clear, respectful, and natural:\n\n{draft}"}]
                    try:
                        reply = call_ai(
                            messages,
                            "You are a communication editor. Preserve the user's meaning. Return one concise, natural message only.",
                        )
                        st.subheader("Suggested rewrite")
                        st.write(reply)
                    except Exception as exc:
                        st.error(f"AI rewrite failed: {exc}")


# ============================================================
# AI CHAT SIMULATOR
# ============================================================

elif page == "🗨️ AI Chat Simulator":
    section_header(
        "Interactive practice",
        "🗨️ AI Chat Simulator",
        "Practice a realistic conversation with a hypothetical partner. The AI should respond to context instead of repeating the same answer.",
    )

    c1, c2, c3 = st.columns(3)
    scenario = c1.selectbox("Scenario", list(SCENARIOS.keys()))
    style = c2.selectbox("Partner style", list(PARTNER_STYLES.keys()))
    difficulty = c3.selectbox("Difficulty", ["Gentle", "Realistic", "Challenging"], index=1)

    signature = (scenario, style, difficulty)

    if st.session_state.sim_signature != signature:
        st.session_state.sim_messages = []
        st.session_state.sim_signature = signature

    if AI_AVAILABLE:
        st.success(f"● Natural AI role-play online · {OPENAI_MODEL}")
    else:
        st.warning("○ AI key missing. The simulator will use a simple local fallback.")

    card(style, f"{PARTNER_STYLES[style]}<br><br><b>Scenario:</b> {SCENARIOS[scenario]}", "🎭")

    if st.button("🔄 Start / Reset Conversation", type="primary"):
        st.session_state.sim_messages = [
            {
                "role": "assistant",
                "content": {
                    "Canceled plans because of work": "I know we had plans tonight, but work got complicated and I need to stay late. I'm sorry.",
                    "Different texting needs": "I've noticed we communicate differently during the day. I like fewer messages than you seem to.",
                    "Need for personal space": "I care about us, but tonight I really need some time by myself to reset.",
                    "Career move to another state": "I got the job offer. It's a big opportunity, but it would mean moving to another state.",
                    "Money and spending disagreement": "I know we see spending differently. I don't want money to become something we keep fighting about.",
                    "Friendship and jealousy boundary": "I want to be honest. I sometimes feel uncomfortable about how much time you spend with that friend.",
                }[scenario],
            }
        ]
        st.rerun()

    if not st.session_state.sim_messages:
        st.session_state.sim_messages = [{"role": "assistant", "content": "I want to talk about this. What are you thinking?"}]

    for message in st.session_state.sim_messages:
        avatar = "💗" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    sim_prompt = st.chat_input("Reply as yourself...")

    if sim_prompt:
        st.session_state.sim_messages.append({"role": "user", "content": sim_prompt})

        with st.chat_message("user", avatar="👤"):
            st.markdown(sim_prompt)

        try:
            if AI_AVAILABLE:
                with st.spinner("Partner is replying..."):
                    reply = call_ai(
                        st.session_state.sim_messages,
                        simulator_instructions(scenario, style, difficulty),
                    )
            else:
                lower = sim_prompt.lower().strip()
                if any(x in lower for x in ["hurt", "sad", "upset", "angry", "frustrated"]):
                    reply = "I hear that this hurt you. I don't want to brush that off. Tell me what part affected you the most."
                elif "why" in lower:
                    reply = "Because I was trying to handle what I needed in the moment, but I can see that it affected you too. What did you need from me?"
                elif "space" in lower:
                    reply = "I can respect space, but I also think we should agree on when we reconnect so neither of us feels ignored."
                else:
                    reply = "I hear you. I want to understand your side, but I also want room for mine. What would feel like a fair next step?"

        except Exception as exc:
            reply = f"I couldn't generate the AI reply. Please check the API configuration. `{type(exc).__name__}: {exc}`"

        st.session_state.sim_messages.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant", avatar="💗"):
            st.markdown(reply)


# ============================================================
# REFLECTION
# ============================================================

elif page == "🚦 Relationship Reflection":
    section_header(
        "Personal reflection",
        "🚦 Relationship Reflection",
        "Use this as a structured check-in, not as a diagnosis or relationship verdict.",
    )

    items = {
        "I feel heard when I raise concerns.": st.slider("I feel heard when I raise concerns.", 1, 5, 3),
        "We can disagree without insults or threats.": st.slider("We can disagree without insults or threats.", 1, 5, 3),
        "My boundaries are generally respected.": st.slider("My boundaries are generally respected.", 1, 5, 3),
        "I can be myself without constantly walking on eggshells.": st.slider("I can be myself without constantly walking on eggshells.", 1, 5, 3),
        "Important decisions are discussed fairly.": st.slider("Important decisions are discussed fairly.", 1, 5, 3),
    }

    avg = np.mean(list(items.values()))
    st.metric("Reflection average", f"{avg:.1f}/5")

    if avg >= 4:
        st.success("Your answers suggest several strong relationship-process indicators.")
    elif avg >= 3:
        st.info("Your answers are mixed. It may help to look at which specific areas feel weakest.")
    else:
        st.warning("Several areas feel difficult. Focus on concrete patterns, boundaries, and whether conversations feel safe and respectful.")


# ============================================================
# ML INTELLIGENCE
# ============================================================

elif page == "🤖 ML Intelligence":
    section_header(
        "Explainable machine learning",
        "🤖 ML Intelligence",
        "Explore synthetic relationship-preference patterns using K-Means clustering and PCA.",
    )

    k = st.slider("Number of clusters", 2, 7, 4)
    df = synthetic_profiles()

    scaler = StandardScaler()
    X = scaler.fit_transform(df[list(FEATURES.keys())])

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)
    st.metric("Silhouette score", f"{score:.3f}")

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Cluster": labels,
    })

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cluster in sorted(plot_df["Cluster"].unique()):
        part = plot_df[plot_df["Cluster"] == cluster]
        ax.scatter(part["PC1"], part["PC2"], s=18, alpha=.65, label=f"Cluster {cluster + 1}")
    ax.set_title("Synthetic preference clusters (PCA projection)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.caption("Synthetic data is used only to demonstrate the ML workflow. These clusters are not psychological diagnoses.")


# ============================================================
# PROFILE CENTER
# ============================================================

elif page == "📊 My Profile Center":
    section_header(
        "Saved session data",
        "📊 My Profile Center",
        "Review or export the Blueprint currently saved in this browser session.",
    )

    if not st.session_state.saved_profile:
        st.info("No Blueprint has been saved yet. Go to Partner Blueprint and save one first.")
    else:
        st.subheader(st.session_state.saved_profile_name)
        df = profile_dataframe(st.session_state.saved_profile)
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig = radar_chart(st.session_state.saved_profile, "Saved Blueprint")
        st.pyplot(fig)
        plt.close(fig)

        st.download_button(
            "⬇️ Download Blueprint CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="partnerlens_blueprint.csv",
            mime="text/csv",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    f"💞 {APP_NAME} · Python · Streamlit · Pandas · NumPy · scikit-learn · OpenAI Responses API · "
    f"{datetime.now().year}"
)
