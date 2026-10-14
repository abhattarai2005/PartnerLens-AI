import os
import math
from io import StringIO

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
# PARTNERLENS AI — COMPLETE PORTFOLIO EDITION
# Self-contained Streamlit app
# ============================================================

st.set_page_config(
    page_title="PartnerLens AI",
    page_icon="💞",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

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

INSIGHTS = {
    "communication": "You may value a partner who communicates clearly, checks in consistently, and is willing to discuss problems openly.",
    "emotional_maturity": "You may work best with someone who can stay grounded during conflict, listen carefully, and take responsibility.",
    "independence": "Personal space and individual goals appear important. A partner with their own interests may fit you well.",
    "ambition": "Growth and motivation appear meaningful to you. You may appreciate someone with clear personal or professional goals.",
    "family_orientation": "Family connection appears meaningful in your picture of long-term compatibility.",
    "affection": "Emotional closeness, reassurance, and visible affection appear important to you.",
    "social_lifestyle": "Shared social experiences may play an important role in how connected you feel.",
    "fitness": "An active lifestyle may be an important part of the life you want to share.",
    "financial_planning": "Financial responsibility and planning may strongly influence your sense of long-term compatibility.",
    "adventure": "New experiences, travel, and spontaneity may help keep a relationship exciting for you.",
}

SCENARIO_MAPS = [
    {
        "Give me space first": {"independence": 2, "emotional_maturity": 1},
        "Ask me what happened": {"communication": 2, "emotional_maturity": 1},
        "Comfort me physically": {"affection": 2, "communication": 1},
        "Distract me with something fun": {"adventure": 2, "social_lifestyle": 1},
    },
    {
        "Stay home together": {"affection": 2, "family_orientation": 1},
        "Go to a big social event": {"social_lifestyle": 2, "adventure": 1},
        "Spend time with a few close friends": {"social_lifestyle": 1, "communication": 1},
        "Do separate activities and reconnect later": {"independence": 2},
    },
    {
        "Talk about it immediately": {"communication": 2},
        "Take time and discuss it later": {"emotional_maturity": 2, "independence": 1},
        "Write or text first": {"communication": 1, "emotional_maturity": 1},
        "Let small disagreements go": {"independence": 1, "emotional_maturity": 1},
    },
    {
        "Strongly consider moving together": {"family_orientation": 1, "affection": 1},
        "Discuss both careers equally": {"ambition": 2, "communication": 1},
        "Try long-distance first": {"independence": 2, "communication": 1},
        "Keep my location/career first": {"independence": 2, "ambition": 1},
    },
    {
        "Frequent messages throughout the day": {"communication": 2, "affection": 1},
        "A few meaningful check-ins": {"communication": 2, "emotional_maturity": 1},
        "Mostly talk after work or school": {"independence": 1, "communication": 1},
        "Lots of personal space": {"independence": 2},
    },
]

SIMULATOR_SCENARIOS = {
    "Canceled plans because of work": {
        "context": "You were looking forward to spending time together, but your partner canceled because of work.",
        "starter": "I know we had plans tonight, but work got complicated and I need to stay late. I'm sorry.",
    },
    "Different texting needs": {
        "context": "One person wants more frequent communication while the other prefers more space.",
        "starter": "I've noticed we communicate very differently during the day. I like checking in more often than you seem to.",
    },
    "Need for personal space": {
        "context": "Your partner asks for personal space after a stressful day.",
        "starter": "I care about us, but tonight I really need some time by myself to reset.",
    },
    "Career move to another state": {
        "context": "A career opportunity creates a major decision for the relationship.",
        "starter": "I got the job offer. It's a big opportunity, but it would mean moving to another state.",
    },
    "Money and spending disagreement": {
        "context": "You and your partner have different attitudes toward spending and saving.",
        "starter": "I know we see spending differently. I don't want money to become something we keep fighting about.",
    },
    "Friendship and jealousy boundary": {
        "context": "A friendship is creating insecurity and a need for a boundary conversation.",
        "starter": "I want to be honest. I sometimes feel uncomfortable about how much time you spend with that friend.",
    },
}

PARTNER_STYLES = {
    "Calm Communicator": "Listens, asks questions, and tries to resolve conflict collaboratively.",
    "Independent / Needs Space": "Values autonomy, personal time, and calm conversations without constant contact.",
    "Affectionate Reassurer": "Values emotional closeness, reassurance, warmth, and frequent connection.",
    "Career-Driven / Busy": "Cares about the relationship while placing strong importance on goals and scheduling.",
    "Conflict-Avoidant": "Finds tension uncomfortable and may initially pull back or minimize disagreements.",
    "Direct Communicator": "Prefers straightforward discussion over hints or indirect communication.",
}


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "saved_profile": None,
    "saved_profile_name": "My Profile",
    "scenario_profile": None,
    "compatibility_last": None,
    "guide_messages": [],
    "sim_chat": [],
    "sim_debrief": False,
    "theme_mode": "Aurora",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GLOBAL DESIGN / CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@700;800&display=swap');

:root {
    --pink: #ff4f91;
    --rose: #ff6b6b;
    --purple: #7c5cff;
    --violet: #9b5cff;
    --blue: #45a3ff;
    --cyan: #34d7d2;
    --gold: #ffbf4b;
    --ink: #1f2333;
    --muted: #6c7085;
    --card: rgba(255,255,255,0.74);
    --card-strong: rgba(255,255,255,0.92);
    --border: rgba(93, 73, 147, 0.13);
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 10%, rgba(255, 121, 171, .22), transparent 24%),
        radial-gradient(circle at 92% 13%, rgba(125, 92, 255, .20), transparent 25%),
        radial-gradient(circle at 78% 84%, rgba(52, 215, 210, .16), transparent 24%),
        linear-gradient(135deg, #fff8fb 0%, #f8f6ff 48%, #f5fbff 100%);
    color: var(--ink);
}

.block-container {
    max-width: 1380px;
    padding-top: 1.25rem;
    padding-bottom: 4rem;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(255,255,255,.95), rgba(250,247,255,.94));
    border-right: 1px solid rgba(100, 85, 150, 0.10);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

h1, h2, h3 {
    font-family: "Manrope", "Inter", sans-serif;
    letter-spacing: -0.03em;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 2.35rem 2.55rem;
    border-radius: 30px;
    background:
        radial-gradient(circle at 13% 20%, rgba(255,255,255,.95), transparent 21%),
        linear-gradient(115deg, rgba(255,104,157,.22), rgba(122,92,255,.19) 48%, rgba(65,199,255,.19));
    border: 1px solid rgba(122,92,255,.12);
    box-shadow: 0 24px 70px rgba(76, 47, 128, .10);
    margin-bottom: 1.1rem;
}

.hero:before {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    right: -70px;
    top: -90px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(255,79,145,.45), rgba(122,92,255,.35));
    filter: blur(2px);
}

.hero:after {
    content: "";
    position: absolute;
    width: 130px;
    height: 130px;
    right: 135px;
    bottom: -85px;
    border-radius: 999px;
    background: rgba(52,215,210,.25);
}

.hero-kicker {
    font-size: .78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .15em;
    color: #7859d7;
    margin-bottom: .45rem;
}

.hero h1 {
    position: relative;
    z-index: 1;
    font-size: clamp(2.6rem, 5vw, 4.7rem);
    line-height: .98;
    margin: 0 0 .8rem 0;
    background: linear-gradient(90deg, #ff4f91, #8b57ff, #2f9fe8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    position: relative;
    z-index: 1;
    font-size: 1.08rem;
    line-height: 1.75;
    color: #555a70;
    max-width: 880px;
    margin: 0;
}

.glass-card {
    padding: 1.15rem 1.2rem;
    border-radius: 22px;
    background: var(--card);
    border: 1px solid var(--border);
    box-shadow: 0 14px 38px rgba(64, 52, 108, .065);
    backdrop-filter: blur(14px);
    margin-bottom: .9rem;
}

.glass-card strong {
    color: #29263b;
}

.gradient-card {
    position: relative;
    overflow: hidden;
    padding: 1.3rem 1.35rem;
    min-height: 170px;
    border-radius: 24px;
    border: 1px solid rgba(130, 100, 190, .13);
    background: linear-gradient(145deg, rgba(255,255,255,.92), rgba(250,247,255,.72));
    box-shadow: 0 16px 42px rgba(77, 59, 121, .07);
}

.gradient-card:after {
    content:"";
    position:absolute;
    width:90px;height:90px;
    border-radius:50%;
    right:-30px;bottom:-35px;
    background:linear-gradient(135deg,rgba(255,79,145,.18),rgba(122,92,255,.20));
}

.feature-icon {
    font-size: 1.8rem;
    margin-bottom: .4rem;
}

.section-kicker {
    font-size: .75rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .14em;
    color: #7d62d7;
    margin-top: .4rem;
    margin-bottom: .15rem;
}

.score-ring {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 132px;
    height: 132px;
    border-radius: 50%;
    margin: .4rem auto;
    background:
      radial-gradient(circle at center, white 0 54%, transparent 55%),
      conic-gradient(#ff4f91, #8d5cff, #38b7e9, #ff4f91);
    box-shadow: 0 12px 35px rgba(118,86,210,.18);
}

.score-ring span {
    font-size: 1.55rem;
    font-weight: 800;
}

.pill {
    display: inline-block;
    padding: .36rem .72rem;
    margin: .18rem .22rem .18rem 0;
    border-radius: 999px;
    background: rgba(255,255,255,.78);
    border: 1px solid rgba(122,92,255,.15);
    font-size: .78rem;
    color: #615b79;
    box-shadow: 0 5px 14px rgba(80,60,130,.04);
}

.badge {
    display:inline-block;
    padding:.28rem .6rem;
    border-radius:999px;
    background:linear-gradient(90deg,rgba(255,79,145,.11),rgba(122,92,255,.11));
    color:#6f54ca;
    font-weight:700;
    font-size:.76rem;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,.77);
    border: 1px solid rgba(115, 92, 170, .12);
    border-radius: 20px;
    padding: 1rem 1.05rem;
    box-shadow: 0 12px 30px rgba(59, 45, 90, .05);
}

div[data-testid="stMetricValue"] {
    font-family: "Manrope", sans-serif;
}

.stButton > button, .stDownloadButton > button {
    border-radius: 14px;
    border: 1px solid rgba(122,92,255,.15);
    font-weight: 700;
    transition: .2s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 24px rgba(122,92,255,.13);
}

button[kind="primary"] {
    background: linear-gradient(90deg, #ff4f91, #8b5cff) !important;
    border: none !important;
    color: white !important;
}

div[data-baseweb="slider"] > div > div {
    border-radius: 999px;
}

div[data-testid="stChatMessage"] {
    background: rgba(255,255,255,.67);
    border: 1px solid rgba(115,92,170,.10);
    border-radius: 18px;
    padding: .4rem .8rem;
    margin-bottom: .55rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: .4rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: .5rem .8rem;
}

hr {
    border-color: rgba(100,80,150,.10);
}

.small-muted {
    color:#777b8f;
    font-size:.9rem;
}

footer {visibility: hidden;}

@media (max-width: 850px) {
    .hero {padding: 1.7rem 1.35rem;}
    .hero h1 {font-size: 2.7rem;}
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def section_header(kicker, title, body=None):
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.header(title)
    if body:
        st.write(body)


def info_card(title, body, icon="✨"):
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="font-size:1.25rem;font-weight:800;margin-bottom:.35rem;">{icon} {title}</div>
            <div style="color:#62667b;line-height:1.65;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(icon, title, body):
    st.markdown(
        f"""
        <div class="gradient-card">
            <div class="feature-icon">{icon}</div>
            <div style="font-size:1.08rem;font-weight:800;margin-bottom:.35rem;">{title}</div>
            <div style="color:#666b80;line-height:1.55;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def collect_profile(prefix, default_profile=None):
    profile = {}
    for key, question in QUESTIONS.items():
        default = 3 if default_profile is None else int(default_profile.get(key, 3))
        profile[key] = st.slider(
            f"{FEATURE_ICONS[key]} {question}",
            1, 5, default,
            key=f"{prefix}_{key}",
            help="1 = low importance • 5 = extremely important",
        )
    return profile


def profile_percentages(profile):
    return {k: int(v * 20) for k, v in profile.items()}


def ranked_profile(profile):
    return sorted(profile_percentages(profile).items(), key=lambda x: x[1], reverse=True)


def relationship_style(profile):
    top = [k for k, _ in ranked_profile(profile)[:3]]
    s = set(top)

    if {"independence", "ambition"}.issubset(s):
        return "Independent Achiever", "You may thrive with someone who respects goals, autonomy, and personal growth."
    if {"family_orientation", "emotional_maturity"}.issubset(s):
        return "Grounded Connector", "You may prioritize emotional steadiness, long-term connection, and family values."
    if {"affection", "communication"}.issubset(s):
        return "Expressive Communicator", "You may value closeness, reassurance, and frequent emotional connection."
    if {"adventure", "social_lifestyle"}.issubset(s):
        return "Social Explorer", "You may value novelty, shared experiences, and an energetic social life."
    if {"financial_planning", "ambition"}.issubset(s):
        return "Future Builder", "You may care strongly about stability, planning, and building toward long-term goals."
    return "Balanced Builder", "Your priorities are distributed across several dimensions rather than dominated by one theme."


def compatibility_score(a, b):
    diffs = [abs(a[k] - b[k]) / 4 for k in FEATURES]
    return round((1 - np.mean(diffs)) * 100, 1)


def compatibility_details(a, b):
    rows = []
    for k in FEATURES:
        diff = abs(a[k] - b[k])
        alignment = round((1 - diff / 4) * 100)
        rows.append({
            "Dimension": FEATURES[k],
            "Person A": a[k],
            "Person B": b[k],
            "Alignment %": alignment,
            "Difference": diff,
        })
    return pd.DataFrame(rows).sort_values("Alignment %", ascending=False)


def radar_chart(profile, title="Relationship Preference Radar"):
    labels = [FEATURES[k] for k in FEATURES]
    values = [profile[k] for k in FEATURES]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2.4)
    ax.fill(angles, values, alpha=.14)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5)
    ax.set_title(title, pad=24, fontweight="bold")
    plt.tight_layout()
    return fig


def comparison_radar(a, b):
    labels = [FEATURES[k] for k in FEATURES]
    va = [a[k] for k in FEATURES] + [a[list(FEATURES.keys())[0]]]
    vb = [b[k] for k in FEATURES] + [b[list(FEATURES.keys())[0]]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, va, linewidth=2.3, label="Person A")
    ax.fill(angles, va, alpha=.08)
    ax.plot(angles, vb, linewidth=2.3, label="Person B")
    ax.fill(angles, vb, alpha=.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5)
    ax.set_title("Preference Comparison", pad=24, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.11))
    plt.tight_layout()
    return fig


def donut(score):
    score = float(np.clip(score, 0, 100))
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.pie([score, 100-score], startangle=90, wedgeprops={"width": .28})
    ax.text(0, .06, f"{score:.1f}%", ha="center", va="center", fontsize=26, fontweight="bold")
    ax.text(0, -.15, "overall alignment", ha="center", va="center", fontsize=10)
    ax.set_aspect("equal")
    ax.set_title("Compatibility Alignment", pad=16, fontweight="bold")
    plt.tight_layout()
    return fig


def reflection_radar(values):
    labels = list(values.keys())
    vals = list(values.values()) + [list(values.values())[0]]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6.2, 6.2))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals, linewidth=2.3)
    ax.fill(angles, vals, alpha=.14)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([1,2,3,4,5])
    ax.set_ylim(0,5)
    ax.set_title("Relationship Reflection Radar", pad=20, fontweight="bold")
    plt.tight_layout()
    return fig


def profile_csv(profile, name):
    return pd.DataFrame([{"profile_name": name, **profile}]).to_csv(index=False).encode("utf-8")


def analyze_scenarios(answers):
    scores = {k: 0 for k in FEATURES}
    for answer, mapping in zip(answers, SCENARIO_MAPS):
        for key, points in mapping[answer].items():
            scores[key] += points
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def message_signals(text):
    lower = text.lower().strip()
    return {
        "feeling": any(x in lower for x in ["i feel", "i felt", "i'm feeling", "i am feeling", "i was hurt", "i was disappointed", "i was worried", "i was upset"]),
        "request": any(x in lower for x in ["can we", "could we", "would you", "i need", "i would like", "can you", "what do you think"]),
        "boundary": any(x in lower for x in ["not comfortable", "my boundary", "i need space", "i need some time", "i can't agree", "i cannot agree"]),
        "empathy": any(x in lower for x in ["i understand", "i can see", "i get why", "that makes sense", "i hear you"]),
        "absolute": any(x in lower for x in ["always", "never", "every single time"]),
        "dismissive": any(x in lower for x in ["whatever", "i don't care", "do whatever", "you're impossible", "you are impossible"]),
        "blame": any(x in lower for x in ["you make me", "all your fault", "because of you", "you ruin"]),
        "question": "?" in text,
        "apology": any(x in lower for x in ["i'm sorry", "i am sorry", "i apologize", "sorry about"]),
    }


def advanced_message_analysis(text):
    sig = message_signals(text)
    words = text.split()

    clarity = 55
    respect = 82
    feelings = 45
    solution = 45
    boundary = 45

    if len(words) >= 8:
        clarity += 12
    if len(words) >= 18:
        clarity += 5
    if sig["feeling"]:
        feelings += 38
        clarity += 7
    if sig["request"]:
        solution += 38
        clarity += 8
    if sig["boundary"]:
        boundary += 42
    if sig["empathy"]:
        respect += 10
        solution += 8
    if sig["question"]:
        solution += 8
    if sig["apology"]:
        respect += 6
    if sig["absolute"]:
        respect -= 18
        clarity -= 8
    if sig["dismissive"]:
        respect -= 28
        solution -= 18
    if sig["blame"]:
        respect -= 20
        clarity -= 8

    scores = {
        "Clarity": int(np.clip(clarity, 0, 100)),
        "Respect": int(np.clip(respect, 0, 100)),
        "Feeling Expression": int(np.clip(feelings, 0, 100)),
        "Solution Focus": int(np.clip(solution, 0, 100)),
        "Boundary Expression": int(np.clip(boundary, 0, 100)),
    }

    strengths = []
    improve = []

    if sig["feeling"]:
        strengths.append("You name your own feelings rather than only criticizing the other person.")
    else:
        improve.append("Name your feeling directly: disappointed, worried, hurt, overwhelmed, or frustrated.")

    if sig["request"]:
        strengths.append("You include a clear request or next step.")
    else:
        improve.append("Add a concrete request, such as “Can we talk about this tonight?”")

    if sig["empathy"]:
        strengths.append("You acknowledge the other person's perspective.")
    else:
        improve.append("Acknowledge the other person's perspective before stating your need.")

    if sig["boundary"]:
        strengths.append("You communicate a personal limit or boundary.")
    else:
        improve.append("If a boundary matters, state it calmly and specifically.")

    if sig["absolute"]:
        improve.append("Replace “always/never” with the specific behavior or situation.")
    if sig["dismissive"]:
        improve.append("Avoid dismissive wording that can shut down the conversation.")
    if sig["blame"]:
        improve.append("Shift from blame toward observation + feeling + need.")

    overall = round(np.mean(list(scores.values())))
    return {"overall": overall, "scores": scores, "strengths": strengths, "improve": improve, "signals": sig}


def suggested_reply(scenario):
    return {
        "Canceled plans because of work":
            "I understand work can change unexpectedly. I was disappointed because I was looking forward to our time. Can we choose another time and protect it?",
        "Different texting needs":
            "I care about staying connected, but constant texting can feel overwhelming to me. Can we agree on a few meaningful check-ins that work for both of us?",
        "Need for personal space":
            "I can respect that you need space. It would help me if we agree on when we'll reconnect so I don't feel shut out.",
        "Career move to another state":
            "I want to support your opportunity, and I also need us to consider my goals. Can we compare our options before either of us makes a final decision?",
        "Money and spending disagreement":
            "I don't expect us to spend exactly the same way. I do want us to agree on shared priorities and which financial decisions we make together.",
        "Friendship and jealousy boundary":
            "I want to understand your concern without giving up healthy friendships. Can we talk about the specific behavior that is bothering you and what boundary would feel respectful to both of us?",
    }[scenario]


def simulator_starter(scenario, style):
    openers = {
        "Calm Communicator": "I want to talk about this honestly and hear how you feel. ",
        "Independent / Needs Space": "I care about us, and I also need us to respect each other's space. ",
        "Affectionate Reassurer": "I care about you a lot, and I don't want this to make you feel unimportant. ",
        "Career-Driven / Busy": "I care about us, but I'm balancing a lot of responsibility right now. ",
        "Conflict-Avoidant": "I'm a little nervous about this conversation because I don't want us to fight. ",
        "Direct Communicator": "I'd rather be clear so we can deal with this directly. ",
    }
    return openers[style] + SIMULATOR_SCENARIOS[scenario]["starter"]


def local_simulator_reply(style, scenario, difficulty, text, turn):
    sig = message_signals(text)
    lower = text.lower().strip()
    wc = len(text.split())

    if wc <= 2 or lower in {"what", "huh", "why", "idk", "okay", "ok", "fine"}:
        return {
            "Calm Communicator": "I may not have explained that clearly. What part do you want me to clarify?",
            "Independent / Needs Space": "Are you asking what I need, or why I need some space?",
            "Affectionate Reassurer": "Tell me what felt confusing and I'll explain it better.",
            "Career-Driven / Busy": "Are you asking why the plan changed, or what I want us to do next?",
            "Conflict-Avoidant": "What part are you unsure about? We can take it one piece at a time.",
            "Direct Communicator": "Which part do you mean: what happened, what I need, or what I want us to change?",
        }[style]

    if sig["dismissive"] or sig["blame"]:
        tone = "I want to keep talking, but that wording feels sharp. Can we focus on the specific issue instead of attacking each other?"
    elif sig["feeling"] and sig["request"]:
        tone = "Thank you for telling me both how you feel and what you need. That gives me something concrete to respond to."
    elif sig["feeling"]:
        tone = "I hear that this really affected you, and I don't want to brush that aside."
    elif sig["request"] or sig["question"]:
        tone = "That's helpful because you're being specific about what you want to discuss."
    else:
        tone = "I want to understand what you mean before I assume too much."

    style_lines = {
        "Calm Communicator": "What would feel fair to both of us from here?",
        "Independent / Needs Space": "I can meet you halfway, but I also need some room to process things.",
        "Affectionate Reassurer": "I want us to solve this without either of us feeling unimportant.",
        "Career-Driven / Busy": "I need a solution that respects the relationship and the responsibilities I have.",
        "Conflict-Avoidant": "Can we focus on one issue at a time so the conversation doesn't become overwhelming?",
        "Direct Communicator": "Tell me exactly what you need to be different, and I'll tell you what I can realistically agree to.",
    }

    difficulty_line = {
        "Gentle": "I'm open to meeting you halfway.",
        "Realistic": "I want both of our needs to matter here.",
        "Challenging": "I'm willing to work on this, but I don't think one person's preferred solution automatically becomes the answer.",
    }[difficulty]

    return f"{tone} {style_lines[style]} {difficulty_line}"


def chat_debrief(history):
    messages = [x["content"] for x in history if x["role"] == "user"]
    if not messages:
        return None
    analyses = [advanced_message_analysis(x) for x in messages]
    result = {}
    for key in ["Clarity", "Respect", "Feeling Expression", "Solution Focus", "Boundary Expression"]:
        result[key] = round(np.mean([a["scores"][key] for a in analyses]))

    penalty = 0
    for a in analyses:
        penalty += 8 if a["signals"]["absolute"] else 0
        penalty += 12 if a["signals"]["dismissive"] else 0
        penalty += 10 if a["signals"]["blame"] else 0

    result["Escalation Risk"] = int(np.clip(round(30 + penalty / max(1, len(analyses))), 0, 100))
    result["Overall"] = round(np.mean([
        result["Clarity"],
        result["Respect"],
        result["Feeling Expression"],
        result["Solution Focus"],
        result["Boundary Expression"],
        100 - result["Escalation Risk"],
    ]))
    return result


def resolve_api_key(manual=""):
    manual = manual.strip()
    if manual:
        return manual
    try:
        val = st.secrets.get("OPENAI_API_KEY", "")
        if val:
            return str(val).strip()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "").strip()


def openai_reply(api_key, model, system_prompt, messages):
    if not api_key:
        raise RuntimeError("No API key supplied.")
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")
    client = OpenAI(api_key=api_key)
    payload = [{"role": "system", "content": system_prompt}] + messages
    response = client.responses.create(model=model, input=payload)
    text = getattr(response, "output_text", "")
    if not text:
        raise RuntimeError("The model returned an empty response.")
    return text.strip()


def guide_fallback(message):
    lower = message.lower()
    if "boundary" in lower:
        return "A healthy boundary is specific and describes what you will or will not participate in. Example: “I'm happy to discuss this, but I won't continue while we're insulting each other. Let's come back to it when we're calmer.”"
    if "compat" in lower:
        return "Compatibility is less about being identical and more about whether your major values, goals, communication needs, lifestyle, and boundaries can coexist without one person constantly sacrificing."
    if "qualities" in lower or "partner" in lower:
        return "Focus first on emotional maturity, honesty, communication, respect for boundaries, reliability, shared life goals, and the ability to repair conflict. Then separate true needs from nice-to-have preferences."
    return "I can help you reflect on communication, compatibility, boundaries, trust, dating, conflict, or partner qualities. Add an API key for fully open-ended AI answers."


@st.cache_data
def synthetic_profiles(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    archetypes = np.array([
        [4.6,4.4,2.8,3.4,4.7,4.5,3.0,3.1,4.0,2.9],  # grounded/family
        [3.6,3.7,4.7,4.6,2.7,2.8,3.4,4.0,4.5,4.1],  # independent achiever
        [4.3,3.8,3.0,3.3,3.2,4.5,4.6,3.5,3.0,4.7],  # social explorer
        [3.8,4.5,3.8,3.7,3.6,3.8,2.8,3.2,4.6,3.3],  # stable planner
        [4.7,4.6,3.4,4.3,3.6,4.0,3.3,3.8,4.1,3.8],  # balanced growth
    ])

    labels = rng.integers(0, len(archetypes), size=n)
    X = archetypes[labels] + rng.normal(0, .55, size=(n, len(FEATURES)))
    X = np.clip(X, 1, 5)

    df = pd.DataFrame(X, columns=list(FEATURES.keys()))
    df.insert(0, "profile_id", np.arange(1, n + 1))
    return df


@st.cache_data
def prepare_ml(df, k=5):
    feature_cols = list(FEATURES.keys())
    X = df[feature_cols]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(Xs)
    clustered = df.copy()
    clustered["cluster"] = labels
    centers = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=feature_cols
    )
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(Xs)
    viz = pd.DataFrame({"PC1":xy[:,0], "PC2":xy[:,1], "cluster":labels})
    return clustered, centers, scaler, model, pca, viz


@st.cache_data
def cluster_diagnostics(df):
    X = StandardScaler().fit_transform(df[list(FEATURES.keys())])
    rows = []
    for k in range(2, 9):
        m = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = m.fit_predict(X)
        rows.append({"k":k, "inertia":m.inertia_, "silhouette":silhouette_score(X, labels)})
    return pd.DataFrame(rows)


def cluster_names(centers):
    names = []
    used = set()
    fallback = [
        "Grounded Connectors",
        "Independent Achievers",
        "Social Explorers",
        "Future Planners",
        "Balanced Growth Partners",
    ]
    for i, row in centers.iterrows():
        top = set(row.sort_values(ascending=False).index[:3])
        if {"family_orientation","affection"}.issubset(top):
            name = "Grounded Connectors"
        elif {"independence","ambition"}.issubset(top):
            name = "Independent Achievers"
        elif {"adventure","social_lifestyle"}.issubset(top):
            name = "Social Explorers"
        elif {"financial_planning","emotional_maturity"}.issubset(top):
            name = "Future Planners"
        else:
            name = "Balanced Growth Partners"
        if name in used:
            for alt in fallback:
                if alt not in used:
                    name = alt
                    break
        used.add(name)
        names.append(name)
    return names


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding:.35rem 0 .7rem 0;">
        <div style="font-family:Manrope;font-size:1.45rem;font-weight:800;">💞 PartnerLens AI</div>
        <div style="font-size:.78rem;color:#777b8f;margin-top:.2rem;">Relationship self-discovery studio</div>
    </div>
    """,
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

if st.session_state.saved_profile:
    st.sidebar.success("✓ Blueprint saved this session")

if st.sidebar.button("🧹 Clear session", use_container_width=True):
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.rerun()

st.sidebar.markdown(
    """
    <div style="font-size:.75rem;color:#8b8ea0;line-height:1.65;margin-top:1rem;">
    Python • Streamlit • Pandas • NumPy • scikit-learn • K-Means • PCA • Responsible AI
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GLOBAL HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">AI-powered relationship self-discovery</div>
        <h1>💞 PartnerLens AI</h1>
        <p>
            Understand yourself, compare relationship preferences, practice healthier communication,
            and discover patterns with explainable machine learning — all in one colorful, privacy-conscious experience.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info("For adults 18+. PartnerLens supports reflection and preference-alignment analysis. It does not diagnose people or predict whether a relationship will succeed.")


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":
    section_header(
        "A smarter relationship reflection experience",
        "Know what matters before asking who matches.",
        "PartnerLens combines self-discovery, scenario-based reflection, communication coaching, compatibility analysis, interactive role-play, and explainable machine learning."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Preference Dimensions", "10")
    c2.metric("Synthetic ML Profiles", "2,000")
    c3.metric("Interactive Experiences", "9")
    c4.metric("Privacy", "Session-first")

    st.write("")

    row1 = st.columns(3)
    with row1[0]:
        feature_card("🧠", "Partner Blueprint", "Discover the qualities you value most and visualize your relationship preference profile.")
    with row1[1]:
        feature_card("🎭", "Scenario Discovery", "Reveal priorities through realistic situations instead of relying only on direct rating questions.")
    with row1[2]:
        feature_card("🤝", "Compatibility Lab", "Compare two profiles across ten dimensions and see the strongest alignments and discussion areas.")

    st.write("")
    row2 = st.columns(3)
    with row2[0]:
        feature_card("💬", "Conversation Coach", "Analyze clarity, respect, emotional expression, boundaries, and solution focus.")
    with row2[1]:
        feature_card("🗨️", "AI Chat Simulator", "Practice difficult conversations with several hypothetical partner styles and receive a debrief.")
    with row2[2]:
        feature_card("🤖", "ML Intelligence", "Explore K-Means, PCA, silhouette scores, and nearest-cluster assignment on synthetic data.")

    st.write("")
    info_card(
        "Portfolio story",
        "This project demonstrates product design, data analysis, machine learning, explainable scoring, session-state management, visualization, optional LLM integration, and responsible-AI guardrails.",
        "🚀",
    )

    st.markdown(
        """
        <span class="pill">Python</span>
        <span class="pill">Streamlit</span>
        <span class="pill">Pandas</span>
        <span class="pill">NumPy</span>
        <span class="pill">Matplotlib</span>
        <span class="pill">scikit-learn</span>
        <span class="pill">K-Means</span>
        <span class="pill">PCA</span>
        <span class="pill">Silhouette Score</span>
        <span class="pill">OpenAI API optional</span>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BLUEPRINT
# ============================================================

elif page == "🧠 Partner Blueprint":
    section_header(
        "Self-discovery",
        "🧠 Build Your Partner Blueprint",
        "Rate what genuinely matters to you. Your result is a transparent preference profile, not a psychological assessment."
    )

    name = st.text_input("Profile name", value=st.session_state.saved_profile_name)
    profile = collect_profile("bp", st.session_state.saved_profile)

    if st.button("✨ Generate My Blueprint", type="primary", use_container_width=True):
        st.session_state.saved_profile = profile
        st.session_state.saved_profile_name = name.strip() or "My Profile"

        ranked = ranked_profile(profile)
        style_name, style_desc = relationship_style(profile)

        st.divider()
        a, b = st.columns([1, 1.05])

        with a:
            info_card(style_name, style_desc, "💎")
            st.subheader("🏆 Your Top 5 Priorities")
            for i, (key, score) in enumerate(ranked[:5], 1):
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <span class="badge">#{i}</span>
                        <div style="font-size:1.08rem;font-weight:800;margin-top:.45rem;">
                        {FEATURE_ICONS[key]} {FEATURES[key]} — {score}%
                        </div>
                        <div style="color:#686c80;margin-top:.4rem;">{INSIGHTS[key]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with b:
            st.pyplot(radar_chart(profile), use_container_width=True)

        st.subheader("Full Preference Profile")
        chart_df = pd.DataFrame(
            {"Importance %":[score for _, score in ranked]},
            index=[FEATURES[k] for k, _ in ranked]
        )
        st.bar_chart(chart_df)

        st.download_button(
            "⬇️ Download My Profile",
            profile_csv(profile, st.session_state.saved_profile_name),
            "partnerlens_profile.csv",
            "text/csv",
            use_container_width=True,
        )


# ============================================================
# SCENARIO DISCOVERY
# ============================================================

elif page == "🎭 Scenario Discovery":
    section_header(
        "Situational reflection",
        "🎭 What Do You Actually Want?",
        "Scenario choices can reveal what you prioritize when tradeoffs become real."
    )

    q1 = st.radio("After a difficult day, what would you prefer your partner to do?", list(SCENARIO_MAPS[0].keys()))
    q2 = st.radio("It's Saturday night. What sounds best?", list(SCENARIO_MAPS[1].keys()))
    q3 = st.radio("You disagree. What feels best?", list(SCENARIO_MAPS[2].keys()))
    q4 = st.radio("Your partner gets a great job offer in another state. What feels closest to you?", list(SCENARIO_MAPS[3].keys()))
    q5 = st.radio("How much communication feels best during a normal day?", list(SCENARIO_MAPS[4].keys()))

    if st.button("🧠 Analyze My Scenario Choices", type="primary", use_container_width=True):
        results = analyze_scenarios([q1,q2,q3,q4,q5])
        st.session_state.scenario_profile = results
        max_points = max(v for _, v in results) or 1

        st.divider()
        st.subheader("Your strongest scenario signals")

        cols = st.columns(2)
        for i, (key, points) in enumerate(results[:6]):
            pct = round(points / max_points * 100)
            with cols[i % 2]:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div style="font-size:1.05rem;font-weight:800;">{FEATURE_ICONS[key]} {FEATURES[key]}</div>
                        <div style="font-size:1.8rem;font-weight:800;margin:.25rem 0;">{pct}%</div>
                        <div style="color:#6b6f82;">{INSIGHTS[key]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.caption("Scenario signals are reflective patterns from your choices, not psychological labels.")


# ============================================================
# COMPATIBILITY
# ============================================================

elif page == "🤝 Compatibility Lab":
    section_header(
        "Two-person analysis",
        "🤝 Compatibility Lab",
        "Compare two preference profiles across ten dimensions. The score measures stated preference alignment — not relationship success."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("👤 Person A")
        a = collect_profile("ca")
    with right:
        st.subheader("👤 Person B")
        b = collect_profile("cb")

    if st.button("💞 Analyze Our Compatibility", type="primary", use_container_width=True):
        score = compatibility_score(a, b)
        details = compatibility_details(a, b)
        st.session_state.compatibility_last = {"score":score, "details":details}

        st.divider()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Overall Alignment", f"{score}%")
        c2.metric("Strong Alignments", int((details["Alignment %"] >= 75).sum()))
        c3.metric("Exact Matches", int((details["Alignment %"] == 100).sum()))
        c4.metric("Discussion Areas", int((details["Alignment %"] <= 50).sum()))

        if score >= 85:
            st.success("Your stated preferences are highly aligned. The next useful question is how these values appear in real behavior.")
        elif score >= 70:
            st.info("You share many preferences, with a few meaningful differences worth discussing.")
        elif score >= 50:
            st.warning("You have a mix of similarities and meaningful differences. Clear expectations may matter most in lower-alignment areas.")
        else:
            st.warning("Your stated preferences differ considerably in several areas. Differences are not automatically bad, but may require more negotiation.")

        x, y = st.columns([.85, 1.15])
        with x:
            st.pyplot(donut(score), use_container_width=True)
        with y:
            st.pyplot(comparison_radar(a,b), use_container_width=True)

        st.subheader("🏆 Strongest alignments")
        best_cols = st.columns(4)
        for col, (_, row) in zip(best_cols, details.head(4).iterrows()):
            with col:
                info_card(row["Dimension"], f"<b>{int(row['Alignment %'])}% alignment</b><br>Person A: {int(row['Person A'])}/5 • Person B: {int(row['Person B'])}/5", "✨")

        st.subheader("💬 Areas worth discussing")
        for _, row in details.sort_values("Alignment %").head(4).iterrows():
            info_card(
                row["Dimension"],
                f"Alignment: <b>{int(row['Alignment %'])}%</b> • Person A rated {int(row['Person A'])}/5 • Person B rated {int(row['Person B'])}/5",
                "🧭",
            )

        st.subheader("Detailed comparison")
        st.dataframe(details.drop(columns=["Difference"]), use_container_width=True, hide_index=True)

        report = details.drop(columns=["Difference"]).copy()
        report.insert(0, "Overall Alignment %", score)
        st.download_button(
            "⬇️ Download Compatibility Report",
            report.to_csv(index=False).encode("utf-8"),
            "partnerlens_compatibility_report.csv",
            "text/csv",
            use_container_width=True,
        )


# ============================================================
# AI GUIDE
# ============================================================

elif page == "🧭 AI Relationship Guide":
    section_header(
        "Optional LLM integration",
        "🧭 AI Relationship Guide",
        "Ask questions about dating, compatibility, boundaries, conflict, communication, or partner qualities."
    )

    with st.expander("⚙️ AI connection settings"):
        api_input = st.text_input("OpenAI API key", type="password", key="guide_key", help="Optional. You can also set OPENAI_API_KEY in Streamlit secrets.")
        model = st.text_input("Model name", value="gpt-4.1-mini", key="guide_model")
        st.caption("If your API account uses a different supported model, replace the model name above.")

    api_key = resolve_api_key(api_input)

    if api_key:
        st.success("✓ AI connection available for this session")
    else:
        st.info("Local reflection mode is active. Add an API key for open-ended AI responses.")

    col1,col2,col3 = st.columns(3)
    quick_prompts = [
        "What qualities should I look for in a long-term partner?",
        "How can I tell if a compatibility difference really matters?",
        "Help me prepare for a difficult relationship conversation.",
    ]
    selected_quick = None
    for col, text in zip([col1,col2,col3], quick_prompts):
        with col:
            if st.button(text, use_container_width=True):
                selected_quick = text

    if st.button("🧹 Clear guide chat"):
        st.session_state.guide_messages = []
        st.rerun()

    for item in st.session_state.guide_messages:
        with st.chat_message(item["role"], avatar="🧭" if item["role"] == "assistant" else "👤"):
            st.markdown(item["content"])

    prompt = st.chat_input("Ask PartnerLens anything...")
    if selected_quick:
        prompt = selected_quick

    if prompt:
        st.session_state.guide_messages.append({"role":"user","content":prompt})

        profile_context = "No saved blueprint."
        if st.session_state.saved_profile:
            top = ranked_profile(st.session_state.saved_profile)[:5]
            profile_context = ", ".join(f"{FEATURES[k]} {v}%" for k,v in top)

        system = f"""
You are PartnerLens Guide, a practical and respectful relationship reflection assistant for adults.
Answer the user's actual question first. Be specific, concise, and useful.
Never pretend to know another person's hidden motives. Do not diagnose people.
Do not predict relationship success. Do not automatically tell people to break up.
When useful, give a concrete example of what the user could say.
If there are threats, stalking, coercion, violence, or immediate danger, prioritize safety and real-world support.
Saved blueprint context, only when relevant: {profile_context}
""".strip()

        try:
            if api_key:
                reply = openai_reply(api_key, model, system, st.session_state.guide_messages[-12:])
            else:
                reply = guide_fallback(prompt)
        except Exception as exc:
            reply = f"I couldn't reach the AI model, so I switched to local guidance. Technical detail: {exc}\n\n{guide_fallback(prompt)}"

        st.session_state.guide_messages.append({"role":"assistant","content":reply})
        st.rerun()


# ============================================================
# CONVERSATION COACH
# ============================================================

elif page == "💬 Conversation Coach":
    section_header(
        "Explainable communication analysis",
        "💬 Conversation Coach",
        "Draft a message for a difficult conversation and receive transparent feedback on clarity, respect, feelings, boundaries, and solution focus."
    )

    scenario = st.selectbox("Situation", list(SIMULATOR_SCENARIOS.keys()))
    info_card("Scenario", SIMULATOR_SCENARIOS[scenario]["context"], "🎬")

    text = st.text_area(
        "What would you say?",
        placeholder="Example: I was disappointed when our plans changed. Can we talk about how we handle this next time?",
        height=180,
    )

    if st.button("✨ Analyze My Message", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Write a message first.")
        else:
            result = advanced_message_analysis(text)
            st.divider()

            cols = st.columns(5)
            for col, (label, score) in zip(cols, result["scores"].items()):
                col.metric(label, f"{score}%")

            st.progress(result["overall"]/100)
            info_card("Overall communication quality", f"<b style='font-size:1.6rem'>{result['overall']}%</b>", "📈")

            a,b = st.columns(2)
            with a:
                st.subheader("✅ What works")
                if result["strengths"]:
                    for item in result["strengths"]:
                        st.write("• " + item)
                else:
                    st.write("• Your message communicates the issue, but it can be made more specific and collaborative.")
            with b:
                st.subheader("🔧 What to improve")
                if result["improve"]:
                    for item in result["improve"]:
                        st.write("• " + item)
                else:
                    st.write("• No major issues detected in this draft.")

            st.subheader("A stronger structure")
            st.markdown("**Observation → Feeling → Need/Boundary → Request → Invitation to respond**")
            info_card("Coach rewrite", suggested_reply(scenario), "✨")


# ============================================================
# CHAT SIMULATOR
# ============================================================

elif page == "🗨️ AI Chat Simulator":
    section_header(
        "Conversation practice",
        "🗨️ AI Chat Simulator",
        "Practice difficult conversations with hypothetical partner styles, then review how you communicated."
    )

    a,b,c = st.columns(3)
    with a:
        scenario = st.selectbox("Scenario", list(SIMULATOR_SCENARIOS.keys()), key="sim_scenario")
    with b:
        style = st.selectbox("Partner style", list(PARTNER_STYLES.keys()), key="sim_style")
    with c:
        difficulty = st.selectbox("Difficulty", ["Gentle","Realistic","Challenging"], index=1, key="sim_diff")

    info_card(style, f"{PARTNER_STYLES[style]}<br><br><b>Scenario:</b> {SIMULATOR_SCENARIOS[scenario]['context']}", "🎭")

    with st.expander("⚙️ Optional AI mode"):
        sim_key_input = st.text_input("OpenAI API key", type="password", key="sim_key")
        sim_model = st.text_input("Model name", value="gpt-4.1-mini", key="sim_model")
        st.caption("Without an API key, the built-in explainable simulator still works.")

    sim_key = resolve_api_key(sim_key_input)

    reset_col, debrief_col = st.columns(2)
    with reset_col:
        if st.button("🔄 Start / Reset Conversation", use_container_width=True):
            st.session_state.sim_chat = [{"role":"assistant","content":simulator_starter(scenario, style)}]
            st.session_state.sim_debrief = False
            st.rerun()
    with debrief_col:
        if st.button("📊 End & Debrief", use_container_width=True):
            st.session_state.sim_debrief = True

    if not st.session_state.sim_chat:
        st.session_state.sim_chat = [{"role":"assistant","content":simulator_starter(scenario, style)}]

    for message in st.session_state.sim_chat:
        with st.chat_message(message["role"], avatar="💗" if message["role"]=="assistant" else "👤"):
            st.write(message["content"])
            if message["role"] == "user":
                analysis = advanced_message_analysis(message["content"])
                with st.expander("🧠 Live coach note"):
                    st.write(f"Message quality: **{analysis['overall']}%**")
                    if analysis["strengths"]:
                        st.write("**Strength:** " + analysis["strengths"][0])
                    if analysis["improve"]:
                        st.write("**Try:** " + analysis["improve"][0])

    prompt = st.chat_input("Reply as yourself...")
    if prompt:
        st.session_state.sim_chat.append({"role":"user","content":prompt})
        user_turn = sum(1 for x in st.session_state.sim_chat if x["role"]=="user")

        if sim_key:
            system = f"""
You are role-playing a hypothetical adult partner in a communication-practice simulator.
Scenario: {SIMULATOR_SCENARIOS[scenario]['context']}
Partner style: {style} — {PARTNER_STYLES[style]}
Difficulty: {difficulty}
Stay in character. Reply naturally in 1-4 sentences.
React directly to what the user said and remember recent context.
Do not manipulate, threaten, shame, diagnose, or pretend to be a real human.
Move the conversation forward.
""".strip()
            try:
                reply = openai_reply(sim_key, sim_model, system, st.session_state.sim_chat[-12:])
            except Exception as exc:
                reply = local_simulator_reply(style, scenario, difficulty, prompt, user_turn)
                reply += f"\n\n_(AI connection unavailable; local simulator used.)_"
        else:
            reply = local_simulator_reply(style, scenario, difficulty, prompt, user_turn)

        st.session_state.sim_chat.append({"role":"assistant","content":reply})
        st.session_state.sim_debrief = False
        st.rerun()

    if st.session_state.sim_debrief:
        result = chat_debrief(st.session_state.sim_chat)
        st.divider()
        st.subheader("📊 Conversation Debrief")
        if result is None:
            st.info("Send at least one reply first.")
        else:
            x,y,z = st.columns(3)
            x.metric("Overall Skill", f"{result['Overall']}%")
            y.metric("Solution Focus", f"{result['Solution Focus']}%")
            z.metric("Escalation Risk", f"{result['Escalation Risk']}%")

            skills = pd.DataFrame(
                {"Score":[result[k] for k in ["Clarity","Respect","Feeling Expression","Solution Focus","Boundary Expression"]]},
                index=["Clarity","Respect","Feeling Expression","Solution Focus","Boundary Expression"]
            )
            st.bar_chart(skills)
            st.caption("These are heuristic coaching signals for practice, not clinical measurements.")


# ============================================================
# RELATIONSHIP REFLECTION
# ============================================================

elif page == "🚦 Relationship Reflection":
    section_header(
        "Behavior and boundary reflection",
        "🚦 Relationship Reflection",
        "Reflect on observable behaviors instead of relying on labels like “toxic” or “perfect.”"
    )

    situation = st.text_area(
        "Describe the situation",
        placeholder="Example: The person I'm dating supports my career, but gets upset whenever I spend time with friends.",
        height=150
    )

    a,b = st.columns(2)
    with a:
        boundaries = st.slider("Boundary respect",1,5,3)
        communication = st.slider("Safe, honest communication",1,5,3)
        independence = st.slider("Respect for independence",1,5,3)
    with b:
        reliability = st.slider("Reliability and follow-through",1,5,3)
        trust = st.slider("Trust and good faith",1,5,3)
        repair = st.slider("Ability to repair conflict",1,5,3)

    if st.button("🔎 Reflect on This Situation", type="primary", use_container_width=True):
        ratings = {
            "Boundaries":boundaries,
            "Communication":communication,
            "Independence":independence,
            "Reliability":reliability,
            "Trust":trust,
            "Repair":repair,
        }
        avg = float(np.mean(list(ratings.values())))
        pct = round(avg/5*100)

        x,y = st.columns([.8,1.2])
        with x:
            st.metric("Reflection Score", f"{pct}%")
            if avg >= 4:
                st.success("Your ratings suggest several supportive relationship behaviors.")
            elif avg >= 3:
                st.info("Your ratings are mixed. Some areas may be worth discussing directly.")
            else:
                st.warning("Several areas scored low. Focus on boundaries, communication, reliability, and whether both people's needs are respected.")
        with y:
            st.pyplot(reflection_radar(ratings), use_container_width=True)

        lows = [k for k,v in ratings.items() if v <= 2]
        highs = [k for k,v in ratings.items() if v >= 4]
        if highs:
            st.success("Positive signals: " + ", ".join(highs))
        if lows:
            st.warning("Areas needing attention: " + ", ".join(lows))

        st.subheader("Conversation starters")
        st.write("• What specific behavior made this situation feel supportive or difficult?")
        st.write("• What boundary or expectation feels unclear?")
        st.write("• What would a fair and respectful change look like?")
        st.write("• Is this occasional, or is it a repeated pattern?")
        st.write("• Can both people discuss the concern without punishment, fear, or intimidation?")


# ============================================================
# ML INTELLIGENCE
# ============================================================

elif page == "🤖 ML Intelligence":
    section_header(
        "Explainable machine learning",
        "🤖 ML Intelligence Lab",
        "Explore how unsupervised learning discovers patterns across 2,000 synthetic relationship-preference profiles."
    )

    df = synthetic_profiles()
    clustered, centers, scaler, model, pca, viz = prepare_ml(df, 5)
    diagnostics = cluster_diagnostics(df)
    names = cluster_names(centers)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Synthetic Profiles", f"{len(df):,}")
    c2.metric("Preference Features", len(FEATURES))
    c3.metric("K-Means Clusters", 5)
    c4.metric("PCA Dimensions", 2)

    tab1,tab2,tab3 = st.tabs(["📈 Model Selection","🗺️ PCA Map","🧩 Cluster Explorer"])

    with tab1:
        x,y = st.columns(2)
        with x:
            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(diagnostics["k"], diagnostics["inertia"], marker="o")
            ax.set_title("Elbow Method")
            ax.set_xlabel("k")
            ax.set_ylabel("Inertia")
            ax.grid(alpha=.2)
            st.pyplot(fig, use_container_width=True)
        with y:
            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(diagnostics["k"], diagnostics["silhouette"], marker="o")
            ax.set_title("Silhouette Score")
            ax.set_xlabel("k")
            ax.set_ylabel("Silhouette")
            ax.grid(alpha=.2)
            st.pyplot(fig, use_container_width=True)

        best_idx = diagnostics["silhouette"].idxmax()
        st.info(f"Highest silhouette score in this synthetic dataset: k={int(diagnostics.loc[best_idx,'k'])} with score {diagnostics.loc[best_idx,'silhouette']:.3f}. The app keeps k=5 to demonstrate a stable five-group product experience.")

    with tab2:
        fig, ax = plt.subplots(figsize=(10,6))
        for cid in sorted(viz["cluster"].unique()):
            sub = viz[viz["cluster"]==cid]
            ax.scatter(sub["PC1"], sub["PC2"], s=18, alpha=.55, label=f"{cid+1}: {names[cid]}")
        ax.set_title("PCA Projection of 2,000 Synthetic Profiles")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
        st.pyplot(fig, use_container_width=True)
        st.caption("PCA compresses the 10-dimensional feature space into 2 dimensions for visualization.")

    with tab3:
        counts = clustered["cluster"].value_counts().sort_index()
        rows = []
        for i, center in centers.iterrows():
            top3 = center.sort_values(ascending=False).index[:3]
            rows.append({
                "Cluster":i+1,
                "Name":names[i],
                "Profiles":int(counts.loc[i]),
                "Top Signals":", ".join(FEATURES[k] for k in top3),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        selected = st.selectbox("Inspect cluster", range(5), format_func=lambda i: f"Cluster {i+1} — {names[i]}")
        center = centers.iloc[selected].sort_values(ascending=False)
        st.bar_chart(pd.DataFrame({"Average Score":center.values}, index=[FEATURES[k] for k in center.index]))

    st.subheader("Which ML group am I closest to?")
    if not st.session_state.saved_profile:
        st.info("Create and save a Partner Blueprint first, then return here.")
    else:
        arr = np.array([[st.session_state.saved_profile[k] for k in FEATURES]])
        scaled = scaler.transform(arr)
        cid = int(model.predict(scaled)[0])
        distances = model.transform(scaled)[0]
        sorted_dist = np.sort(distances)
        fit = 1 - (sorted_dist[0] / (sorted_dist[-1] + 1e-9))
        fit = float(np.clip(fit, 0, 1))

        a,b = st.columns(2)
        a.metric("Closest ML Group", names[cid])
        b.metric("Relative Cluster Fit", f"{fit*100:.0f}%")
        st.caption("Relative cluster fit is a distance visualization, not a probability.")

    st.caption("All ML data here is synthetic and generated inside the app for learning and portfolio demonstration.")


# ============================================================
# PROFILE CENTER
# ============================================================

elif page == "📊 My Profile Center":
    section_header(
        "Session dashboard",
        "📊 My Profile Center",
        "Review the profile information and insights you created during this session."
    )

    if not st.session_state.saved_profile:
        st.info("You have not created a Partner Blueprint yet.")
        if st.button("Go to Partner Blueprint"):
            st.info("Choose **Partner Blueprint** from the left sidebar.")
    else:
        profile = st.session_state.saved_profile
        ranked = ranked_profile(profile)
        style_name, style_desc = relationship_style(profile)

        c1,c2,c3 = st.columns(3)
        c1.metric("Profile", st.session_state.saved_profile_name)
        c2.metric("Top Priority", FEATURES[ranked[0][0]])
        c3.metric("Preference Style", style_name)

        info_card(style_name, style_desc, "💎")
        st.pyplot(radar_chart(profile), use_container_width=True)

        st.subheader("Top priorities")
        cols = st.columns(5)
        for col, (key, score) in zip(cols, ranked[:5]):
            with col:
                st.metric(FEATURES[key], f"{score}%")

        st.download_button(
            "⬇️ Download Profile",
            profile_csv(profile, st.session_state.saved_profile_name),
            "partnerlens_profile.csv",
            "text/csv",
            use_container_width=True,
        )

        st.warning("Session data is temporary and may reset when the app restarts.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="
        text-align:center;
        padding:1.2rem;
        border-top:1px solid rgba(100,80,150,.10);
        color:#85889a;
        font-size:.82rem;">
        💞 PartnerLens AI • Built with Python, Streamlit, scikit-learn and responsible AI design
    </div>
    """,
    unsafe_allow_html=True,
)
