import os
import random
import hashlib
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
# PARTNERLENS AI — FINAL CHATGPT-STYLE PORTFOLIO EDITION
# ============================================================

st.set_page_config(
    page_title="PartnerLens AI",
    page_icon="💞",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APP CONSTANTS
# ============================================================

APP_NAME = "PartnerLens AI"
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
MAX_AI_CALLS_PER_SESSION = 40
RECENT_UI_MESSAGES = 30

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
    "communication":
        "You may value a partner who communicates clearly, checks in consistently, and is willing to discuss problems openly.",
    "emotional_maturity":
        "You may work best with someone who can stay grounded during conflict, listen carefully, and take responsibility.",
    "independence":
        "Personal space and individual goals appear important. A partner with their own interests may fit you well.",
    "ambition":
        "Growth and motivation appear meaningful to you. You may appreciate someone with clear personal or professional goals.",
    "family_orientation":
        "Family connection appears meaningful in your picture of long-term compatibility.",
    "affection":
        "Emotional closeness, reassurance, and visible affection appear important to you.",
    "social_lifestyle":
        "Shared social experiences may play an important role in how connected you feel.",
    "fitness":
        "An active lifestyle may be an important part of the life you want to share.",
    "financial_planning":
        "Financial responsibility and planning may strongly influence your sense of long-term compatibility.",
    "adventure":
        "New experiences, travel, and spontaneity may help keep a relationship exciting for you.",
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
        "context":
            "You were looking forward to spending time together, but your partner canceled because of work.",
        "starter":
            "I know we had plans tonight, but work got complicated and I need to stay late. I'm sorry.",
    },
    "Different texting needs": {
        "context":
            "One person wants more frequent communication while the other prefers more space.",
        "starter":
            "I've noticed we communicate very differently during the day. I like checking in more often than you seem to.",
    },
    "Need for personal space": {
        "context":
            "Your partner asks for personal space after a stressful day.",
        "starter":
            "I care about us, but tonight I really need some time by myself to reset.",
    },
    "Career move to another state": {
        "context":
            "A career opportunity creates a major decision for the relationship.",
        "starter":
            "I got the job offer. It's a big opportunity, but it would mean moving to another state.",
    },
    "Money and spending disagreement": {
        "context":
            "You and your partner have different attitudes toward spending and saving.",
        "starter":
            "I know we see spending differently. I don't want money to become something we keep fighting about.",
    },
    "Friendship and jealousy boundary": {
        "context":
            "A friendship is creating insecurity and a need for a boundary conversation.",
        "starter":
            "I want to be honest. I sometimes feel uncomfortable about how much time you spend with that friend.",
    },
}

PARTNER_STYLES = {
    "Calm Communicator":
        "Listens, explains feelings, asks questions, and tries to resolve conflict collaboratively.",
    "Independent / Needs Space":
        "Values autonomy, personal time, and calm conversations without constant contact.",
    "Affectionate Reassurer":
        "Values emotional closeness, reassurance, warmth, and frequent connection.",
    "Career-Driven / Busy":
        "Cares about the relationship while placing strong importance on goals, work, and scheduling.",
    "Conflict-Avoidant":
        "Finds tension uncomfortable and may initially pull back or minimize disagreements.",
    "Direct Communicator":
        "Prefers straightforward discussion over hints or indirect communication.",
}


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "saved_profile": None,
    "saved_profile_name": "My Profile",
    "scenario_profile": None,
    "compatibility_last": None,

    # ChatGPT-style relationship guide
    "guide_messages": [],
    "guide_conversation_id": None,

    # ChatGPT-style roleplay simulator
    "sim_chat": [],
    "sim_conversation_id": None,
    "sim_debrief": False,
    "sim_config_signature": None,

    # API usage guard
    "ai_calls": 0,

    # quick prompt staging
    "guide_pending": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@700;800&display=swap');

:root {
    --pink: #ff4f91;
    --rose: #ff6b7a;
    --purple: #7c5cff;
    --violet: #9b5cff;
    --blue: #45a3ff;
    --cyan: #34d7d2;
    --gold: #ffbf4b;
    --ink: #1f2333;
    --muted: #6c7085;
    --card: rgba(255,255,255,0.76);
    --card-strong: rgba(255,255,255,0.94);
    --border: rgba(93,73,147,0.13);
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 6% 8%, rgba(255,121,171,.24), transparent 25%),
        radial-gradient(circle at 92% 10%, rgba(125,92,255,.22), transparent 27%),
        radial-gradient(circle at 75% 88%, rgba(52,215,210,.16), transparent 26%),
        linear-gradient(135deg, #fff9fb 0%, #f8f6ff 50%, #f4fbff 100%);
    color: var(--ink);
}

.block-container {
    max-width: 1380px;
    padding-top: 1.15rem;
    padding-bottom: 4rem;
}

section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 0% 0%, rgba(255,79,145,.09), transparent 30%),
        linear-gradient(180deg, rgba(255,255,255,.97), rgba(249,247,255,.96));
    border-right: 1px solid rgba(100,85,150,.10);
}

h1, h2, h3 {
    font-family: "Manrope", "Inter", sans-serif;
    letter-spacing: -0.035em;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 2.45rem 2.6rem;
    border-radius: 32px;
    background:
        radial-gradient(circle at 13% 20%, rgba(255,255,255,.96), transparent 22%),
        linear-gradient(115deg, rgba(255,104,157,.24), rgba(122,92,255,.20) 48%, rgba(65,199,255,.20));
    border: 1px solid rgba(122,92,255,.12);
    box-shadow: 0 24px 70px rgba(76,47,128,.10);
    margin-bottom: 1rem;
}

.hero:before {
    content: "";
    position: absolute;
    width: 250px;
    height: 250px;
    right: -78px;
    top: -98px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(255,79,145,.46), rgba(122,92,255,.36));
}

.hero:after {
    content: "";
    position: absolute;
    width: 145px;
    height: 145px;
    right: 140px;
    bottom: -98px;
    border-radius: 999px;
    background: rgba(52,215,210,.24);
}

.hero-kicker {
    position: relative;
    z-index: 2;
    font-size: .78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .15em;
    color: #7859d7;
    margin-bottom: .5rem;
}

.hero h1 {
    position: relative;
    z-index: 2;
    font-size: clamp(2.65rem, 5vw, 4.8rem);
    line-height: .98;
    margin: 0 0 .85rem 0;
    background: linear-gradient(90deg, #ff4f91, #8b57ff, #2f9fe8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    position: relative;
    z-index: 2;
    font-size: 1.08rem;
    line-height: 1.75;
    color: #555a70;
    max-width: 900px;
    margin: 0;
}

.glass-card {
    padding: 1.15rem 1.22rem;
    border-radius: 22px;
    background: rgba(255,255,255,.74);
    border: 1px solid var(--border);
    box-shadow: 0 14px 38px rgba(64,52,108,.06);
    backdrop-filter: blur(14px);
    margin-bottom: .9rem;
}

.gradient-card {
    position: relative;
    overflow: hidden;
    padding: 1.3rem 1.35rem;
    min-height: 170px;
    border-radius: 24px;
    border: 1px solid rgba(130,100,190,.13);
    background: linear-gradient(145deg, rgba(255,255,255,.94), rgba(250,247,255,.74));
    box-shadow: 0 16px 42px rgba(77,59,121,.07);
}

.gradient-card:after {
    content:"";
    position:absolute;
    width:90px;
    height:90px;
    border-radius:50%;
    right:-30px;
    bottom:-35px;
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
    margin-top: .35rem;
    margin-bottom: .15rem;
}

.pill {
    display: inline-block;
    padding: .36rem .72rem;
    margin: .18rem .22rem .18rem 0;
    border-radius: 999px;
    background: rgba(255,255,255,.80);
    border: 1px solid rgba(122,92,255,.15);
    font-size: .78rem;
    color: #615b79;
}

.badge {
    display:inline-block;
    padding:.28rem .62rem;
    border-radius:999px;
    background:linear-gradient(90deg,rgba(255,79,145,.11),rgba(122,92,255,.11));
    color:#6f54ca;
    font-weight:700;
    font-size:.76rem;
}

.ai-online {
    padding:.75rem 1rem;
    border-radius:16px;
    background:linear-gradient(90deg,rgba(52,215,210,.13),rgba(69,163,255,.10));
    border:1px solid rgba(52,180,190,.18);
    margin-bottom:.8rem;
}

.ai-offline {
    padding:.75rem 1rem;
    border-radius:16px;
    background:linear-gradient(90deg,rgba(255,191,75,.13),rgba(255,107,122,.09));
    border:1px solid rgba(215,150,55,.18);
    margin-bottom:.8rem;
}

.chat-shell {
    border-radius:24px;
    padding:.9rem;
    background:rgba(255,255,255,.42);
    border:1px solid rgba(112,88,168,.10);
    box-shadow:0 15px 40px rgba(70,55,110,.045);
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,.78);
    border: 1px solid rgba(115,92,170,.12);
    border-radius: 20px;
    padding: 1rem 1.05rem;
    box-shadow: 0 12px 30px rgba(59,45,90,.05);
}

div[data-testid="stChatMessage"] {
    background: rgba(255,255,255,.70);
    border: 1px solid rgba(115,92,170,.10);
    border-radius: 19px;
    padding: .45rem .85rem;
    margin-bottom: .55rem;
    box-shadow:0 8px 24px rgba(70,55,110,.035);
}

div[data-testid="stChatInput"] {
    border-radius: 18px;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 14px;
    border: 1px solid rgba(122,92,255,.15);
    font-weight: 700;
    transition: .2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 24px rgba(122,92,255,.13);
}

button[kind="primary"] {
    background: linear-gradient(90deg, #ff4f91, #8b5cff) !important;
    border: none !important;
    color: white !important;
}

hr {
    border-color: rgba(100,80,150,.10);
}

footer {
    visibility: hidden;
}

@media (max-width: 850px) {
    .hero {
        padding:1.75rem 1.35rem;
    }
    .hero h1 {
        font-size:2.7rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def section_header(kicker, title, body=None):
    st.markdown(
        f'<div class="section-kicker">{kicker}</div>',
        unsafe_allow_html=True,
    )
    st.header(title)
    if body:
        st.write(body)


def info_card(title, body, icon="✨"):
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="font-size:1.25rem;font-weight:800;margin-bottom:.35rem;">
                {icon} {title}
            </div>
            <div style="color:#62667b;line-height:1.65;">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(icon, title, body):
    st.markdown(
        f"""
        <div class="gradient-card">
            <div class="feature-icon">{icon}</div>
            <div style="font-size:1.08rem;font-weight:800;margin-bottom:.35rem;">
                {title}
            </div>
            <div style="color:#666b80;line-height:1.55;">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def collect_profile(prefix, default_profile=None):
    profile = {}

    for key, question in QUESTIONS.items():
        default = 3

        if default_profile:
            default = int(default_profile.get(key, 3))

        profile[key] = st.slider(
            f"{FEATURE_ICONS[key]} {question}",
            min_value=1,
            max_value=5,
            value=default,
            key=f"{prefix}_{key}",
            help="1 = low importance • 5 = extremely important",
        )

    return profile


# ============================================================
# PROFILE / COMPATIBILITY HELPERS
# ============================================================

def profile_percentages(profile):
    return {
        key: int(value * 20)
        for key, value in profile.items()
    }


def ranked_profile(profile):
    return sorted(
        profile_percentages(profile).items(),
        key=lambda x: x[1],
        reverse=True,
    )


def relationship_style(profile):
    top = [key for key, _ in ranked_profile(profile)[:3]]
    top_set = set(top)

    if {"independence", "ambition"}.issubset(top_set):
        return (
            "Independent Achiever",
            "You may thrive with someone who respects goals, autonomy, and personal growth.",
        )

    if {"family_orientation", "emotional_maturity"}.issubset(top_set):
        return (
            "Grounded Connector",
            "You may prioritize emotional steadiness, long-term connection, and family values.",
        )

    if {"affection", "communication"}.issubset(top_set):
        return (
            "Expressive Communicator",
            "You may value closeness, reassurance, and frequent emotional connection.",
        )

    if {"adventure", "social_lifestyle"}.issubset(top_set):
        return (
            "Social Explorer",
            "You may value novelty, shared experiences, and an energetic social life.",
        )

    if {"financial_planning", "ambition"}.issubset(top_set):
        return (
            "Future Builder",
            "You may care strongly about stability, planning, and building toward long-term goals.",
        )

    return (
        "Balanced Builder",
        "Your priorities are distributed across several relationship dimensions rather than dominated by one theme.",
    )


def compatibility_score(person_a, person_b):
    distances = [
        abs(person_a[key] - person_b[key]) / 4
        for key in FEATURES
    ]

    return round(
        (1 - np.mean(distances)) * 100,
        1,
    )


def compatibility_details(person_a, person_b):
    rows = []

    for key in FEATURES:
        difference = abs(person_a[key] - person_b[key])
        alignment = round((1 - difference / 4) * 100)

        rows.append({
            "Dimension": FEATURES[key],
            "Person A": person_a[key],
            "Person B": person_b[key],
            "Alignment %": alignment,
            "Difference": difference,
        })

    return pd.DataFrame(rows).sort_values(
        "Alignment %",
        ascending=False,
    )


def profile_csv(profile, profile_name):
    return pd.DataFrame([
        {
            "profile_name": profile_name,
            **profile,
        }
    ]).to_csv(
        index=False
    ).encode("utf-8")


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def radar_chart(profile, title="Relationship Preference Radar"):
    labels = [
        FEATURES[key]
        for key in FEATURES
    ]

    values = [
        profile[key]
        for key in FEATURES
    ]

    values = values + values[:1]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(labels),
        endpoint=False,
    ).tolist()

    angles += angles[:1]

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = plt.subplot(111, polar=True)

    ax.plot(
        angles,
        values,
        linewidth=2.4,
    )

    ax.fill(
        angles,
        values,
        alpha=0.14,
    )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5)
    ax.set_title(
        title,
        pad=24,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def comparison_radar(person_a, person_b):
    labels = [
        FEATURES[key]
        for key in FEATURES
    ]

    first_key = list(FEATURES.keys())[0]

    values_a = [
        person_a[key]
        for key in FEATURES
    ] + [person_a[first_key]]

    values_b = [
        person_b[key]
        for key in FEATURES
    ] + [person_b[first_key]]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(labels),
        endpoint=False,
    ).tolist()

    angles += angles[:1]

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = plt.subplot(111, polar=True)

    ax.plot(
        angles,
        values_a,
        linewidth=2.3,
        label="Person A",
    )

    ax.fill(
        angles,
        values_a,
        alpha=0.08,
    )

    ax.plot(
        angles,
        values_b,
        linewidth=2.3,
        label="Person B",
    )

    ax.fill(
        angles,
        values_b,
        alpha=0.08,
    )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5)
    ax.set_title(
        "Preference Comparison",
        pad=24,
        fontweight="bold",
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.20, 1.11),
    )

    plt.tight_layout()
    return fig


def donut(score):
    score = float(
        np.clip(score, 0, 100)
    )

    fig, ax = plt.subplots(
        figsize=(5.6, 5.6)
    )

    ax.pie(
        [score, 100 - score],
        startangle=90,
        wedgeprops={"width": 0.28},
    )

    ax.text(
        0,
        0.06,
        f"{score:.1f}%",
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
    )

    ax.text(
        0,
        -0.15,
        "overall alignment",
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.set_aspect("equal")
    ax.set_title(
        "Compatibility Alignment",
        pad=16,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def reflection_radar(values_dict):
    labels = list(values_dict.keys())
    values = list(values_dict.values())

    values = values + values[:1]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(labels),
        endpoint=False,
    ).tolist()

    angles += angles[:1]

    fig = plt.figure(figsize=(6.2, 6.2))
    ax = plt.subplot(111, polar=True)

    ax.plot(
        angles,
        values,
        linewidth=2.3,
    )

    ax.fill(
        angles,
        values,
        alpha=0.14,
    )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5)
    ax.set_title(
        "Relationship Reflection Radar",
        pad=20,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


# ============================================================
# SCENARIO HELPERS
# ============================================================

def analyze_scenarios(answers):
    scores = {
        key: 0
        for key in FEATURES
    }

    for answer, mapping in zip(
        answers,
        SCENARIO_MAPS,
    ):
        for key, points in mapping[answer].items():
            scores[key] += points

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )


# ============================================================
# MESSAGE ANALYSIS
# ============================================================

def message_signals(text):
    lower = text.lower().strip()

    feeling_phrases = [
        "i feel",
        "i felt",
        "i'm feeling",
        "i am feeling",
        "i was hurt",
        "i'm hurt",
        "i am hurt",
        "i was disappointed",
        "i was worried",
        "i was upset",
        "i'm sad",
        "i am sad",
        "i'm angry",
        "i am angry",
        "i'm frustrated",
        "i am frustrated",
    ]

    request_phrases = [
        "can we",
        "could we",
        "would you",
        "i need",
        "i would like",
        "can you",
        "could you",
        "what do you think",
        "would it be possible",
    ]

    boundary_phrases = [
        "not comfortable",
        "my boundary",
        "i need space",
        "i need some time",
        "i can't agree",
        "i cannot agree",
        "i won't",
        "i will not",
    ]

    empathy_phrases = [
        "i understand",
        "i can see",
        "i get why",
        "that makes sense",
        "i hear you",
        "i understand why",
    ]

    return {
        "feeling": any(
            phrase in lower
            for phrase in feeling_phrases
        ),

        "request": any(
            phrase in lower
            for phrase in request_phrases
        ),

        "boundary": any(
            phrase in lower
            for phrase in boundary_phrases
        ),

        "empathy": any(
            phrase in lower
            for phrase in empathy_phrases
        ),

        "absolute": any(
            phrase in lower
            for phrase in [
                "always",
                "never",
                "every single time",
            ]
        ),

        "dismissive": any(
            phrase in lower
            for phrase in [
                "whatever",
                "i don't care",
                "do whatever",
                "you're impossible",
                "you are impossible",
            ]
        ),

        "blame": any(
            phrase in lower
            for phrase in [
                "you make me",
                "all your fault",
                "because of you",
                "you ruin",
            ]
        ),

        "question": "?" in text,

        "apology": any(
            phrase in lower
            for phrase in [
                "i'm sorry",
                "i am sorry",
                "i apologize",
                "sorry about",
            ]
        ),
    }


def detect_emotion(text):
    lower = text.lower().strip()

    emotion_map = {
        "hurt": ["hurt", "pain", "painful"],
        "sad": ["sad", "down", "unhappy"],
        "angry": ["angry", "mad", "furious"],
        "upset": ["upset"],
        "disappointed": ["disappointed"],
        "confused": ["confused", "confusing"],
        "worried": ["worried", "anxious", "nervous"],
        "scared": ["scared", "afraid", "fear"],
        "lonely": ["lonely", "alone"],
        "ignored": ["ignored", "unheard"],
        "unimportant": ["unimportant", "second choice", "not important"],
        "frustrated": ["frustrated", "annoyed"],
        "overwhelmed": ["overwhelmed", "too much"],
        "jealous": ["jealous", "jealousy"],
    }

    for emotion, keywords in emotion_map.items():
        if any(
            keyword in lower
            for keyword in keywords
        ):
            return emotion

    return None


def advanced_message_analysis(text):
    signals = message_signals(text)
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

    if signals["feeling"]:
        feelings += 38
        clarity += 7

    if detect_emotion(text):
        feelings += 20

    if signals["request"]:
        solution += 38
        clarity += 8

    if signals["boundary"]:
        boundary += 42

    if signals["empathy"]:
        respect += 10
        solution += 8

    if signals["question"]:
        solution += 8

    if signals["apology"]:
        respect += 6

    if signals["absolute"]:
        respect -= 18
        clarity -= 8

    if signals["dismissive"]:
        respect -= 28
        solution -= 18

    if signals["blame"]:
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
    opportunities = []

    if signals["feeling"] or detect_emotion(text):
        strengths.append(
            "You communicate an emotional experience instead of only describing the other person's behavior."
        )
    else:
        opportunities.append(
            "Try naming your feeling: disappointed, worried, hurt, overwhelmed, or frustrated."
        )

    if signals["request"]:
        strengths.append(
            "You include a request or next step."
        )
    else:
        opportunities.append(
            "Add a clear request such as “Can we talk about this tonight?”"
        )

    if signals["empathy"]:
        strengths.append(
            "You acknowledge the other person's perspective."
        )
    else:
        opportunities.append(
            "Consider briefly acknowledging the other person's perspective before stating your need."
        )

    if signals["boundary"]:
        strengths.append(
            "You communicate a personal limit or boundary."
        )
    else:
        opportunities.append(
            "If a boundary matters here, state it calmly and specifically."
        )

    if signals["absolute"]:
        opportunities.append(
            "Replace “always/never” with the specific behavior or recent situation."
        )

    if signals["dismissive"]:
        opportunities.append(
            "Avoid dismissive wording that can shut down the conversation."
        )

    if signals["blame"]:
        opportunities.append(
            "Shift from blame toward observation + feeling + need."
        )

    overall = round(
        np.mean(
            list(scores.values())
        )
    )

    return {
        "overall": overall,
        "scores": scores,
        "strengths": strengths,
        "opportunities": opportunities,
        "signals": signals,
    }


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


# ============================================================
# OPENAI / CHATGPT-STYLE AI HELPERS
# ============================================================

def resolve_openai_api_key():
    """
    Production-safe API key lookup.

    Priority:
    1. Streamlit Community Cloud secrets
    2. Environment variable

    Never hard-code an API key in app.py.
    """

    try:
        value = st.secrets.get(
            "OPENAI_API_KEY",
            "",
        )

        if value:
            return str(value).strip()

    except Exception:
        pass

    return os.getenv(
        "OPENAI_API_KEY",
        "",
    ).strip()


def resolve_openai_model():
    """
    Allows deployment owner to set OPENAI_MODEL in Streamlit Secrets.
    Falls back to a low-cost current model.
    """

    try:
        value = st.secrets.get(
            "OPENAI_MODEL",
            "",
        )

        if value:
            return str(value).strip()

    except Exception:
        pass

    return os.getenv(
        "OPENAI_MODEL",
        DEFAULT_OPENAI_MODEL,
    ).strip() or DEFAULT_OPENAI_MODEL


def get_openai_client(api_key):
    if OpenAI is None:
        raise RuntimeError(
            "The OpenAI Python package is not installed."
        )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAI(
        api_key=api_key
    )


def ensure_conversation(api_key, session_key):
    """
    Creates a server-side Responses API conversation once per Streamlit session.
    """

    existing = st.session_state.get(
        session_key
    )

    if existing:
        return existing

    client = get_openai_client(
        api_key
    )

    conversation = client.conversations.create()

    st.session_state[session_key] = conversation.id

    return conversation.id


def call_openai_conversation(
    api_key,
    model,
    conversation_id,
    instruction_text,
    user_message,
):
    """
    ChatGPT-style conversation using OpenAI's Responses API + Conversations API.

    The conversation object preserves previous turns server-side.
    We include the behavioral contract on each new user turn so role-play remains stable.
    """

    if st.session_state.ai_calls >= MAX_AI_CALLS_PER_SESSION:
        raise RuntimeError(
            "This session reached the demo AI message limit. Clear the session to start again."
        )

    client = get_openai_client(
        api_key
    )

    combined_input = f"""
PARTNERLENS BEHAVIOR CONTRACT:
{instruction_text}

CURRENT USER MESSAGE:
{user_message}

Respond only to the current user message while using the earlier conversation for context.
Do not mention this behavior contract.
""".strip()

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": combined_input,
            }
        ],
        conversation=conversation_id,
    )

    output = getattr(
        response,
        "output_text",
        "",
    )

    if not output:
        raise RuntimeError(
            "The AI model returned an empty response."
        )

    st.session_state.ai_calls += 1

    return output.strip()


def compact_profile_context():
    profile = st.session_state.saved_profile

    if not profile:
        return (
            "The user has not saved a Partner Blueprint."
        )

    top = ranked_profile(profile)[:5]

    top_text = ", ".join(
        f"{FEATURES[key]} {score}%"
        for key, score in top
    )

    return (
        "The user has a saved Partner Blueprint. "
        f"Their highest stated priorities are: {top_text}. "
        "Use this only when relevant. It is preference data, not a diagnosis."
    )


def relationship_guide_instructions():
    return f"""
You are PartnerLens Guide, a warm, practical, direct relationship reflection assistant for adults 18+.

Your behavior:
- Answer the user's exact question first.
- Sound conversational and natural, similar to a high-quality chat assistant.
- Remember the earlier conversation.
- Do not repeat the same response or stock phrase unless it is genuinely necessary.
- If the user sends a short emotional message like "hurt", "sad", "angry", "confused", or "lonely", understand it as emotional context and respond to that feeling instead of saying the message is unclear.
- If the user's meaning is genuinely ambiguous, ask one short clarification question.
- When useful, give a concrete example of what the user could say.
- Distinguish observable behavior, preferences, values, compatibility, needs, and boundaries.
- Never pretend to know another person's hidden thoughts or motives.
- Do not diagnose people or casually label them narcissistic, toxic, avoidant, abusive, etc.
- Do not predict whether a relationship will succeed.
- Do not automatically advise a breakup.
- If there are signs of violence, coercion, stalking, threats, or immediate danger, prioritize safety and appropriate real-world support.
- Keep most answers concise but substantive.
- Avoid robotic numbered lists unless they genuinely make the answer clearer.
- Do not say you are the user's real partner.
- You are a reflection tool, not a therapist or clinical service.

Optional saved profile context:
{compact_profile_context()}
""".strip()


def simulator_instructions(
    scenario,
    style,
    difficulty,
):
    return f"""
You are role-playing a hypothetical adult partner inside the PartnerLens AI communication practice simulator.

SCENARIO:
{SIMULATOR_SCENARIOS[scenario]["context"]}

PARTNER STYLE:
{style}
{PARTNER_STYLES[style]}

DIFFICULTY:
{difficulty}

ROLE-PLAY RULES:
- Stay in character as the hypothetical partner.
- Sound natural, responsive, and emotionally aware.
- Use the full earlier conversation as context.
- React directly to what the user just said.
- Never repeat the same clarification response simply because the user's message is short.
- A short emotional message such as "hurt", "sad", "angry", "mad", "upset", "lonely", "ignored", "confused", "scared", or "frustrated" should be treated as meaningful emotional information.
- If the user says only "hurt", respond naturally to the fact that they feel hurt and move the conversation forward.
- If the user makes a clear request, answer the request.
- If the user asks a question, answer it in character.
- If the user expresses a feeling, acknowledge the specific feeling.
- If the message is truly unclear (for example only "huh?" or "what?"), ask one natural clarification question.
- Partner style must noticeably influence wording and firmness.
- Difficulty changes how easily the partner compromises:
  * Gentle = warm and highly collaborative.
  * Realistic = balanced, reasonable, and not automatically agreeing.
  * Challenging = respectful but firmer and more likely to defend their own needs.
- Usually respond in 1 to 4 sentences.
- Move toward understanding, negotiation, or a concrete next step.
- Do not become a generic relationship coach.
- Do not give the user communication scores inside the role-play.
- Do not manipulate, threaten, shame, insult, sexually pressure, or emotionally punish the user.
- Do not diagnose the user.
- Never claim to be a real human partner.
""".strip()


# ============================================================
# LOCAL FALLBACK SIMULATOR
# ============================================================

def deterministic_choice(options, seed_text):
    """
    Gives variation without changing randomly on every Streamlit rerun.
    """

    digest = hashlib.sha256(
        seed_text.encode("utf-8")
    ).hexdigest()

    index = int(
        digest[:8],
        16,
    ) % len(options)

    return options[index]


def local_emotional_reply(
    style,
    emotion,
    scenario,
    turn,
):
    style_responses = {
        "Calm Communicator": [
            f"I hear that you're feeling {emotion}. I don't want to dismiss that. What part of this hurt the most?",
            f"Thank you for telling me you're feeling {emotion}. I want to understand what landed badly for you.",
            f"I can hear that you're {emotion}. Tell me what you need me to understand about it.",
        ],

        "Independent / Needs Space": [
            f"I hear that you're {emotion}. Needing some space doesn't mean I want to ignore how this affected you.",
            f"I understand you're feeling {emotion}. I still need room to process, but I do want to hear what affected you.",
            f"I don't want my need for space to make your {emotion} feelings seem unimportant. Tell me what hurt.",
        ],

        "Affectionate Reassurer": [
            f"I'm sorry you're feeling {emotion}. I care about you, and I want to understand what made you feel that way.",
            f"I hate knowing you're feeling {emotion}. Talk to me—what part affected you most?",
            f"I hear that you're {emotion}. I want you to feel cared for while we figure this out together.",
        ],

        "Career-Driven / Busy": [
            f"I hear that you're {emotion}. I don't want my responsibilities to make you feel like your feelings come second.",
            f"I understand you're feeling {emotion}. Work matters to me, but I don't want to ignore what this did to you.",
            f"I can see this left you feeling {emotion}. Tell me what you needed from me in that moment.",
        ],

        "Conflict-Avoidant": [
            f"I didn't realize you were feeling that {emotion}. These talks are hard for me, but I do want to understand.",
            f"I hear that you're {emotion}. I don't want to shut down just because the conversation is uncomfortable.",
            f"Knowing you're feeling {emotion} makes me want to understand this better, even if I need us to go slowly.",
        ],

        "Direct Communicator": [
            f"I hear you. You're feeling {emotion}. Tell me specifically what I did or said that caused it.",
            f"Okay—you feel {emotion}. Let's identify the exact behavior that made you feel that way.",
            f"I understand you're {emotion}. Be direct with me about what you want me to understand or change.",
        ],
    }

    return deterministic_choice(
        style_responses[style],
        f"{style}|{emotion}|{scenario}|{turn}",
    )


def local_simulator_reply(
    style,
    scenario,
    difficulty,
    text,
    turn,
):
    """
    Improved rule-based fallback.
    Unlike the earlier version, short emotional messages are NOT treated as vague.
    """

    raw = text.strip()
    lower = raw.lower()
    word_count = len(raw.split())

    emotion = detect_emotion(
        raw
    )

    if emotion:
        base = local_emotional_reply(
            style,
            emotion,
            scenario,
            turn,
        )

        if difficulty == "Challenging":
            return (
                base
                + " I also want room for my side of this, not only one interpretation."
            )

        if difficulty == "Gentle":
            return (
                base
                + " I'm open to slowing down and working through it with you."
            )

        return base

    truly_vague = {
        "what",
        "what?",
        "huh",
        "huh?",
        "idk",
        "i don't know",
        "dont know",
        "?",
    }

    if lower in truly_vague:
        clarifiers = {
            "Calm Communicator":
                "I may not have explained myself clearly. Which part do you want me to explain?",
            "Independent / Needs Space":
                "Are you asking why I need space, or what I want to happen next?",
            "Affectionate Reassurer":
                "Tell me what part felt unclear and I'll explain it.",
            "Career-Driven / Busy":
                "Are you asking why the situation changed, or how I think we should handle it?",
            "Conflict-Avoidant":
                "I might not have explained myself well. Which part should we focus on?",
            "Direct Communicator":
                "Which part do you mean: what happened, what I need, or what I want us to change?",
        }

        return clarifiers[style]

    signals = message_signals(
        raw
    )

    if signals["dismissive"] or signals["blame"]:
        opening = (
            "I want to keep talking, but that wording feels sharp. "
            "I'd rather focus on the specific issue than attack each other."
        )

    elif signals["feeling"] and signals["request"]:
        opening = (
            "Thank you for telling me both how you feel and what you're asking for. "
            "That gives me something concrete to respond to."
        )

    elif signals["feeling"]:
        opening = (
            "I hear that this affected you emotionally, and I don't want to brush that aside."
        )

    elif signals["request"] or signals["question"]:
        opening = (
            "That's helpful because you're being specific about what you want to discuss."
        )

    elif word_count <= 3:
        opening_options = [
            "I'm listening. Say a little more about what you mean.",
            "I want to understand that better. What are you trying to tell me?",
            "Okay. Help me understand what you're feeling or asking for.",
        ]

        opening = deterministic_choice(
            opening_options,
            f"{raw}|{turn}|{style}",
        )

    else:
        opening = (
            "I want to understand your point before I assume too much."
        )

    style_lines = {
        "Calm Communicator":
            "What would feel fair to both of us from here?",
        "Independent / Needs Space":
            "I can work with you on this, but I also need some room to process.",
        "Affectionate Reassurer":
            "I want us to solve this without either of us feeling unimportant.",
        "Career-Driven / Busy":
            "I need a solution that respects the relationship and the responsibilities I have.",
        "Conflict-Avoidant":
            "Can we focus on one issue at a time so this doesn't become overwhelming?",
        "Direct Communicator":
            "Tell me exactly what you need to be different, and I'll tell you what I can realistically agree to.",
    }

    difficulty_lines = {
        "Gentle":
            "I'm very open to meeting you halfway.",
        "Realistic":
            "I want both of our needs to matter here.",
        "Challenging":
            "I'm willing to work on it, but I don't automatically agree that your preferred solution is the only fair one.",
    }

    scenario_lines = {
        "Canceled plans because of work":
            "I know changing plans can make our time together feel less protected.",
        "Different texting needs":
            "We probably need a communication rhythm that doesn't leave one person ignored and the other pressured.",
        "Need for personal space":
            "For me, needing space is about resetting, not caring less about the relationship.",
        "Career move to another state":
            "This decision affects both of our futures, so neither person's goals should disappear.",
        "Money and spending disagreement":
            "We may spend differently, but we still need shared expectations for important financial decisions.",
        "Friendship and jealousy boundary":
            "I think trust and boundaries matter more here than trying to control who either person can be friends with.",
    }

    return (
        f"{opening} "
        f"{scenario_lines[scenario]} "
        f"{style_lines[style]} "
        f"{difficulty_lines[difficulty]}"
    )


# ============================================================
# CHAT DEBRIEF
# ============================================================

def chat_debrief(history):
    user_messages = [
        item["content"]
        for item in history
        if item["role"] == "user"
    ]

    if not user_messages:
        return None

    analyses = [
        advanced_message_analysis(message)
        for message in user_messages
    ]

    result = {}

    for key in [
        "Clarity",
        "Respect",
        "Feeling Expression",
        "Solution Focus",
        "Boundary Expression",
    ]:
        result[key] = round(
            np.mean(
                [
                    analysis["scores"][key]
                    for analysis in analyses
                ]
            )
        )

    escalation_penalty = 0

    for analysis in analyses:
        escalation_penalty += (
            8
            if analysis["signals"]["absolute"]
            else 0
        )

        escalation_penalty += (
            12
            if analysis["signals"]["dismissive"]
            else 0
        )

        escalation_penalty += (
            10
            if analysis["signals"]["blame"]
            else 0
        )

    result["Escalation Risk"] = int(
        np.clip(
            round(
                30
                + escalation_penalty
                / max(1, len(analyses))
            ),
            0,
            100,
        )
    )

    result["Overall"] = round(
        np.mean([
            result["Clarity"],
            result["Respect"],
            result["Feeling Expression"],
            result["Solution Focus"],
            result["Boundary Expression"],
            100 - result["Escalation Risk"],
        ])
    )

    return result


# ============================================================
# SYNTHETIC ML DATA
# ============================================================

@st.cache_data
def synthetic_profiles(
    n=2000,
    seed=42,
):
    rng = np.random.default_rng(
        seed
    )

    archetypes = np.array([
        [4.6, 4.4, 2.8, 3.4, 4.7, 4.5, 3.0, 3.1, 4.0, 2.9],
        [3.6, 3.7, 4.7, 4.6, 2.7, 2.8, 3.4, 4.0, 4.5, 4.1],
        [4.3, 3.8, 3.0, 3.3, 3.2, 4.5, 4.6, 3.5, 3.0, 4.7],
        [3.8, 4.5, 3.8, 3.7, 3.6, 3.8, 2.8, 3.2, 4.6, 3.3],
        [4.7, 4.6, 3.4, 4.3, 3.6, 4.0, 3.3, 3.8, 4.1, 3.8],
    ])

    labels = rng.integers(
        0,
        len(archetypes),
        size=n,
    )

    X = (
        archetypes[labels]
        + rng.normal(
            0,
            0.55,
            size=(n, len(FEATURES)),
        )
    )

    X = np.clip(
        X,
        1,
        5,
    )

    df = pd.DataFrame(
        X,
        columns=list(FEATURES.keys()),
    )

    df.insert(
        0,
        "profile_id",
        np.arange(1, n + 1),
    )

    return df


@st.cache_data
def prepare_ml(
    df,
    k=5,
):
    feature_cols = list(
        FEATURES.keys()
    )

    X = df[
        feature_cols
    ]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    labels = model.fit_predict(
        X_scaled
    )

    clustered = df.copy()

    clustered["cluster"] = labels

    centers = pd.DataFrame(
        scaler.inverse_transform(
            model.cluster_centers_
        ),
        columns=feature_cols,
    )

    pca = PCA(
        n_components=2,
        random_state=42,
    )

    xy = pca.fit_transform(
        X_scaled
    )

    viz = pd.DataFrame({
        "PC1": xy[:, 0],
        "PC2": xy[:, 1],
        "cluster": labels,
    })

    return (
        clustered,
        centers,
        scaler,
        model,
        pca,
        viz,
    )


@st.cache_data
def cluster_diagnostics(df):
    X = StandardScaler().fit_transform(
        df[
            list(FEATURES.keys())
        ]
    )

    rows = []

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        )

        labels = model.fit_predict(
            X
        )

        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(
                X,
                labels,
            ),
        })

    return pd.DataFrame(
        rows
    )


def cluster_names(centers):
    names = []
    used = set()

    fallbacks = [
        "Grounded Connectors",
        "Independent Achievers",
        "Social Explorers",
        "Future Planners",
        "Balanced Growth Partners",
    ]

    for _, row in centers.iterrows():
        top = set(
            row.sort_values(
                ascending=False
            ).index[:3]
        )

        if {
            "family_orientation",
            "affection",
        }.issubset(top):
            name = "Grounded Connectors"

        elif {
            "independence",
            "ambition",
        }.issubset(top):
            name = "Independent Achievers"

        elif {
            "adventure",
            "social_lifestyle",
        }.issubset(top):
            name = "Social Explorers"

        elif {
            "financial_planning",
            "emotional_maturity",
        }.issubset(top):
            name = "Future Planners"

        else:
            name = "Balanced Growth Partners"

        if name in used:
            for alternative in fallbacks:
                if alternative not in used:
                    name = alternative
                    break

        used.add(
            name
        )

        names.append(
            name
        )

    return names


# ============================================================
# AI STATUS
# ============================================================

OPENAI_API_KEY = resolve_openai_api_key()
OPENAI_MODEL = resolve_openai_model()
AI_AVAILABLE = bool(
    OPENAI_API_KEY
    and OpenAI is not None
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding:.35rem 0 .7rem 0;">
        <div style="font-family:Manrope;font-size:1.45rem;font-weight:800;">
            💞 PartnerLens AI
        </div>
        <div style="font-size:.78rem;color:#777b8f;margin-top:.2rem;">
            Relationship self-discovery studio
        </div>
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

if AI_AVAILABLE:
    st.sidebar.success(
        f"● AI Online · {OPENAI_MODEL}"
    )
else:
    st.sidebar.warning(
        "○ AI Offline · Local fallback"
    )

st.sidebar.caption(
    f"AI messages this session: "
    f"{st.session_state.ai_calls}/{MAX_AI_CALLS_PER_SESSION}"
)

if st.session_state.saved_profile:
    st.sidebar.success(
        "✓ Blueprint saved"
    )

if st.sidebar.button(
    "🧹 Clear entire session",
    use_container_width=True,
):
    for key, value in DEFAULTS.items():
        st.session_state[key] = value

    st.rerun()

st.sidebar.markdown(
    """
    <div style="font-size:.75rem;color:#8b8ea0;line-height:1.65;margin-top:1rem;">
        Python • Streamlit • Pandas • NumPy • scikit-learn •
        K-Means • PCA • OpenAI Responses API • Responsible AI
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
        <div class="hero-kicker">
            AI-powered relationship self-discovery
        </div>

        <h1>
            💞 PartnerLens AI
        </h1>

        <p>
            Understand yourself, compare relationship preferences,
            practice healthier communication, and explore patterns
            with explainable machine learning and conversational AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "For adults 18+. PartnerLens supports reflection and preference-alignment analysis. "
    "It does not diagnose people or predict whether a relationship will succeed."
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":
    section_header(
        "A smarter relationship reflection experience",
        "Know what matters before asking who matches.",
        "PartnerLens combines self-discovery, scenario-based reflection, compatibility analysis, communication coaching, realistic AI conversation practice, and explainable machine learning.",
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Preference Dimensions",
        "10",
    )

    c2.metric(
        "Synthetic ML Profiles",
        "2,000",
    )

    c3.metric(
        "Interactive Experiences",
        "9",
    )

    c4.metric(
        "Conversational AI",
        "Live" if AI_AVAILABLE else "Fallback",
    )

    st.write("")

    row1 = st.columns(3)

    with row1[0]:
        feature_card(
            "🧠",
            "Partner Blueprint",
            "Discover the qualities you value most and visualize your relationship preference profile.",
        )

    with row1[1]:
        feature_card(
            "🎭",
            "Scenario Discovery",
            "Reveal priorities through realistic situations instead of relying only on direct rating questions.",
        )

    with row1[2]:
        feature_card(
            "🤝",
            "Compatibility Lab",
            "Compare two profiles across ten dimensions and identify strong alignments and discussion areas.",
        )

    st.write("")

    row2 = st.columns(3)

    with row2[0]:
        feature_card(
            "💬",
            "Conversation Coach",
            "Analyze clarity, respect, emotional expression, boundaries, and solution focus.",
        )

    with row2[1]:
        feature_card(
            "🗨️",
            "ChatGPT-Style Simulator",
            "Practice natural conversations with hypothetical partner styles using persistent conversational context.",
        )

    with row2[2]:
        feature_card(
            "🤖",
            "ML Intelligence",
            "Explore K-Means, PCA, silhouette scores, and nearest-cluster assignment on synthetic preference data.",
        )

    st.write("")

    info_card(
        "Portfolio story",
        "This project demonstrates product design, data analysis, machine learning, explainable scoring, session-state management, data visualization, conversational AI integration, secure secret handling, and responsible-AI guardrails.",
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
        <span class="pill">OpenAI Responses API</span>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PARTNER BLUEPRINT
# ============================================================

elif page == "🧠 Partner Blueprint":
    section_header(
        "Self-discovery",
        "🧠 Build Your Partner Blueprint",
        "Rate what genuinely matters to you. Your result is a transparent preference profile, not a psychological assessment.",
    )

    profile_name = st.text_input(
        "Profile name",
        value=st.session_state.saved_profile_name,
    )

    profile = collect_profile(
        "blueprint",
        st.session_state.saved_profile,
    )

    if st.button(
        "✨ Generate My Blueprint",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.saved_profile = profile
        st.session_state.saved_profile_name = (
            profile_name.strip()
            or "My Profile"
        )

        ranked = ranked_profile(
            profile
        )

        style_name, style_description = relationship_style(
            profile
        )

        st.divider()

        left, right = st.columns(
            [1, 1.05]
        )

        with left:
            info_card(
                style_name,
                style_description,
                "💎",
            )

            st.subheader(
                "🏆 Your Top 5 Priorities"
            )

            for index, (key, score) in enumerate(
                ranked[:5],
                start=1,
            ):
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <span class="badge">#{index}</span>

                        <div style="font-size:1.08rem;font-weight:800;margin-top:.45rem;">
                            {FEATURE_ICONS[key]} {FEATURES[key]} — {score}%
                        </div>

                        <div style="color:#686c80;margin-top:.4rem;">
                            {INSIGHTS[key]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with right:
            st.pyplot(
                radar_chart(profile),
                use_container_width=True,
            )

        st.subheader(
            "Full Preference Profile"
        )

        chart_df = pd.DataFrame(
            {
                "Importance %": [
                    score
                    for _, score in ranked
                ]
            },
            index=[
                FEATURES[key]
                for key, _ in ranked
            ],
        )

        st.bar_chart(
            chart_df
        )

        st.download_button(
            "⬇️ Download My Profile",
            data=profile_csv(
                profile,
                st.session_state.saved_profile_name,
            ),
            file_name="partnerlens_profile.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# SCENARIO DISCOVERY
# ============================================================

elif page == "🎭 Scenario Discovery":
    section_header(
        "Situational reflection",
        "🎭 What Do You Actually Want?",
        "Scenario choices can reveal what you prioritize when tradeoffs become real.",
    )

    q1 = st.radio(
        "After a difficult day, what would you prefer your partner to do?",
        list(
            SCENARIO_MAPS[0].keys()
        ),
    )

    q2 = st.radio(
        "It's Saturday night. What sounds best?",
        list(
            SCENARIO_MAPS[1].keys()
        ),
    )

    q3 = st.radio(
        "You disagree. What feels best?",
        list(
            SCENARIO_MAPS[2].keys()
        ),
    )

    q4 = st.radio(
        "Your partner gets a great job offer in another state. What feels closest to you?",
        list(
            SCENARIO_MAPS[3].keys()
        ),
    )

    q5 = st.radio(
        "How much communication feels best during a normal day?",
        list(
            SCENARIO_MAPS[4].keys()
        ),
    )

    if st.button(
        "🧠 Analyze My Scenario Choices",
        type="primary",
        use_container_width=True,
    ):
        results = analyze_scenarios(
            [q1, q2, q3, q4, q5]
        )

        st.session_state.scenario_profile = results

        max_points = max(
            value
            for _, value in results
        ) or 1

        st.divider()

        st.subheader(
            "Your strongest scenario signals"
        )

        columns = st.columns(2)

        for index, (key, points) in enumerate(
            results[:6]
        ):
            pct = round(
                points
                / max_points
                * 100
            )

            with columns[
                index % 2
            ]:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div style="font-size:1.05rem;font-weight:800;">
                            {FEATURE_ICONS[key]} {FEATURES[key]}
                        </div>

                        <div style="font-size:1.8rem;font-weight:800;margin:.25rem 0;">
                            {pct}%
                        </div>

                        <div style="color:#6b6f82;">
                            {INSIGHTS[key]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.caption(
            "Scenario signals are reflective patterns from your choices, not psychological labels."
        )


# ============================================================
# COMPATIBILITY LAB
# ============================================================

elif page == "🤝 Compatibility Lab":
    section_header(
        "Two-person analysis",
        "🤝 Compatibility Lab",
        "Compare two preference profiles across ten dimensions. The score measures stated preference alignment — not relationship success.",
    )

    left, right = st.columns(2)

    with left:
        st.subheader(
            "👤 Person A"
        )

        person_a = collect_profile(
            "compat_a"
        )

    with right:
        st.subheader(
            "👤 Person B"
        )

        person_b = collect_profile(
            "compat_b"
        )

    if st.button(
        "💞 Analyze Our Compatibility",
        type="primary",
        use_container_width=True,
    ):
        score = compatibility_score(
            person_a,
            person_b,
        )

        details = compatibility_details(
            person_a,
            person_b,
        )

        st.session_state.compatibility_last = {
            "score": score,
            "details": details,
        }

        st.divider()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Overall Alignment",
            f"{score}%",
        )

        c2.metric(
            "Strong Alignments",
            int(
                (
                    details["Alignment %"]
                    >= 75
                ).sum()
            ),
        )

        c3.metric(
            "Exact Matches",
            int(
                (
                    details["Alignment %"]
                    == 100
                ).sum()
            ),
        )

        c4.metric(
            "Discussion Areas",
            int(
                (
                    details["Alignment %"]
                    <= 50
                ).sum()
            ),
        )

        if score >= 85:
            st.success(
                "Your stated preferences are highly aligned. The next useful question is how those values appear in real behavior."
            )

        elif score >= 70:
            st.info(
                "You share many preferences, with a few meaningful differences worth discussing."
            )

        elif score >= 50:
            st.warning(
                "You have a mix of similarities and meaningful differences. Clear expectations may matter most in lower-alignment areas."
            )

        else:
            st.warning(
                "Your stated preferences differ considerably in several areas. Differences are not automatically bad, but they may require more negotiation."
            )

        chart_left, chart_right = st.columns(
            [0.85, 1.15]
        )

        with chart_left:
            st.pyplot(
                donut(score),
                use_container_width=True,
            )

        with chart_right:
            st.pyplot(
                comparison_radar(
                    person_a,
                    person_b,
                ),
                use_container_width=True,
            )

        st.subheader(
            "🏆 Strongest alignments"
        )

        best_columns = st.columns(4)

        for column, (_, row) in zip(
            best_columns,
            details.head(4).iterrows(),
        ):
            with column:
                info_card(
                    row["Dimension"],
                    f"<b>{int(row['Alignment %'])}% alignment</b><br>"
                    f"Person A: {int(row['Person A'])}/5 • "
                    f"Person B: {int(row['Person B'])}/5",
                    "✨",
                )

        st.subheader(
            "💬 Areas worth discussing"
        )

        for _, row in details.sort_values(
            "Alignment %"
        ).head(4).iterrows():
            info_card(
                row["Dimension"],
                f"Alignment: <b>{int(row['Alignment %'])}%</b> • "
                f"Person A rated {int(row['Person A'])}/5 • "
                f"Person B rated {int(row['Person B'])}/5",
                "🧭",
            )

        st.subheader(
            "Detailed comparison"
        )

        st.dataframe(
            details.drop(
                columns=["Difference"]
            ),
            use_container_width=True,
            hide_index=True,
        )

        report = details.drop(
            columns=["Difference"]
        ).copy()

        report.insert(
            0,
            "Overall Alignment %",
            score,
        )

        st.download_button(
            "⬇️ Download Compatibility Report",
            data=report.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="partnerlens_compatibility_report.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# CHATGPT-STYLE AI RELATIONSHIP GUIDE
# ============================================================

elif page == "🧭 AI Relationship Guide":
    section_header(
        "Conversational AI",
        "🧭 AI Relationship Guide",
        "Ask natural follow-up questions about dating, compatibility, boundaries, conflict, communication, or partner qualities. The AI remembers the conversation within this session.",
    )

    if AI_AVAILABLE:
        st.markdown(
            f"""
            <div class="ai-online">
                <b>● AI mode is online</b><br>
                <span style="color:#687082;">
                    Powered by {OPENAI_MODEL}. Conversation memory is active for this session.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div class="ai-offline">
                <b>○ AI mode is not configured yet</b><br>
                <span style="color:#687082;">
                    The app will use a small local fallback until OPENAI_API_KEY is added to Streamlit Secrets.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.saved_profile:
        st.caption(
            "✓ Your saved Partner Blueprint can be used as optional context when relevant."
        )

    quick_columns = st.columns(3)

    quick_prompts = [
        "What qualities should I look for in a long-term partner?",
        "How do I know if a compatibility difference really matters?",
        "Help me prepare for a difficult relationship conversation.",
    ]

    for index, (
        column,
        quick_prompt,
    ) in enumerate(
        zip(
            quick_columns,
            quick_prompts,
        )
    ):
        with column:
            if st.button(
                quick_prompt,
                use_container_width=True,
                key=f"guide_quick_{index}",
            ):
                st.session_state.guide_pending = quick_prompt

    action_left, action_right = st.columns(
        [1, 2]
    )

    with action_left:
        if st.button(
            "🧹 New conversation",
            use_container_width=True,
        ):
            st.session_state.guide_messages = []
            st.session_state.guide_conversation_id = None
            st.session_state.guide_pending = None
            st.rerun()

    with action_right:
        st.caption(
            "Tip: try short follow-ups such as “why?”, “I feel hurt”, or “what should I say?” — the AI uses prior conversation context."
        )

    st.divider()

    for item in st.session_state.guide_messages[
        -RECENT_UI_MESSAGES:
    ]:
        avatar = (
            "🧭"
            if item["role"] == "assistant"
            else "👤"
        )

        with st.chat_message(
            item["role"],
            avatar=avatar,
        ):
            st.markdown(
                item["content"]
            )

    guide_prompt = st.chat_input(
        "Message PartnerLens Guide..."
    )

    if st.session_state.guide_pending:
        guide_prompt = st.session_state.guide_pending
        st.session_state.guide_pending = None

    if guide_prompt:
        st.session_state.guide_messages.append({
            "role": "user",
            "content": guide_prompt,
        })

        if AI_AVAILABLE:
            try:
                conversation_id = ensure_conversation(
                    OPENAI_API_KEY,
                    "guide_conversation_id",
                )

                with st.spinner(
                    "PartnerLens is thinking..."
                ):
                    reply = call_openai_conversation(
                        api_key=OPENAI_API_KEY,
                        model=OPENAI_MODEL,
                        conversation_id=conversation_id,
                        instruction_text=relationship_guide_instructions(),
                        user_message=guide_prompt,
                    )

            except Exception as exc:
                reply = (
                    "I couldn't reach the AI model for that message, "
                    "so I switched to local guidance.\n\n"
                    f"{local_guide_fallback(guide_prompt) if 'local_guide_fallback' in globals() else 'Tell me a little more about what you want to work through.'}"
                    f"\n\n_Technical detail: {exc}_"
                )

        else:
            lower = guide_prompt.lower()

            if detect_emotion(guide_prompt):
                emotion = detect_emotion(
                    guide_prompt
                )

                reply = (
                    f"It sounds like you're feeling {emotion}. "
                    "Tell me what happened right before you felt that way, "
                    "and what you hoped the other person would do instead."
                )

            elif "boundary" in lower:
                reply = (
                    "A useful boundary is specific and focuses on what you will do. "
                    "For example: “I'm willing to talk about this, but I won't continue while we're insulting each other. "
                    "Let's come back to it when we can speak respectfully.”"
                )

            elif "compat" in lower:
                reply = (
                    "Compatibility is less about being identical and more about whether your major values, goals, communication needs, lifestyle, and boundaries can coexist without one person constantly sacrificing."
                )

            elif "qualities" in lower or "partner" in lower:
                reply = (
                    "Start with emotional maturity, honesty, communication, respect for boundaries, reliability, shared life goals, and the ability to repair conflict. Then separate true needs from nice-to-have preferences."
                )

            else:
                reply = (
                    "I can help with communication, compatibility, boundaries, trust, conflict, dating, or partner qualities. "
                    "For fully open-ended ChatGPT-style answers, connect OPENAI_API_KEY in Streamlit Secrets."
                )

        st.session_state.guide_messages.append({
            "role": "assistant",
            "content": reply,
        })

        st.rerun()


# ============================================================
# CONVERSATION COACH
# ============================================================

elif page == "💬 Conversation Coach":
    section_header(
        "Explainable communication analysis",
        "💬 Conversation Coach",
        "Draft a difficult message and get transparent feedback on clarity, respect, emotions, boundaries, and solution focus.",
    )

    scenario = st.selectbox(
        "Situation",
        list(
            SIMULATOR_SCENARIOS.keys()
        ),
    )

    info_card(
        "Scenario",
        SIMULATOR_SCENARIOS[scenario]["context"],
        "🎬",
    )

    text = st.text_area(
        "What would you say?",
        placeholder=(
            "Example: I was disappointed when our plans changed. "
            "Can we talk about how we handle this next time?"
        ),
        height=180,
    )

    if st.button(
        "✨ Analyze My Message",
        type="primary",
        use_container_width=True,
    ):
        if not text.strip():
            st.warning(
                "Write a message first."
            )

        else:
            result = advanced_message_analysis(
                text
            )

            st.divider()

            score_columns = st.columns(5)

            for column, (
                label,
                score,
            ) in zip(
                score_columns,
                result["scores"].items(),
            ):
                column.metric(
                    label,
                    f"{score}%",
                )

            st.progress(
                result["overall"] / 100
            )

            info_card(
                "Overall communication quality",
                f"<b style='font-size:1.6rem'>{result['overall']}%</b>",
                "📈",
            )

            left, right = st.columns(2)

            with left:
                st.subheader(
                    "✅ What works"
                )

                if result["strengths"]:
                    for item in result["strengths"]:
                        st.write(
                            "• " + item
                        )

                else:
                    st.write(
                        "• Your message communicates the issue, but it could be more specific and collaborative."
                    )

            with right:
                st.subheader(
                    "🔧 What to improve"
                )

                if result["opportunities"]:
                    for item in result["opportunities"]:
                        st.write(
                            "• " + item
                        )

                else:
                    st.write(
                        "• No major communication issues were detected in this draft."
                    )

            st.subheader(
                "A stronger structure"
            )

            st.markdown(
                "**Observation → Feeling → Need/Boundary → Request → Invitation to respond**"
            )

            info_card(
                "Coach rewrite",
                suggested_reply(
                    scenario
                ),
                "✨",
            )


# ============================================================
# CHATGPT-STYLE PARTNER SIMULATOR
# ============================================================

elif page == "🗨️ AI Chat Simulator":
    section_header(
        "Interactive conversation practice",
        "🗨️ ChatGPT-Style Partner Simulator",
        "Practice a realistic conversation with a hypothetical partner. With AI enabled, the simulator remembers the earlier conversation and responds naturally to short emotional messages like “hurt” instead of repeating a generic clarification.",
    )

    first, second, third = st.columns(3)

    with first:
        scenario = st.selectbox(
            "Scenario",
            list(
                SIMULATOR_SCENARIOS.keys()
            ),
            key="sim_scenario",
        )

    with second:
        style = st.selectbox(
            "Partner style",
            list(
                PARTNER_STYLES.keys()
            ),
            key="sim_style",
        )

    with third:
        difficulty = st.selectbox(
            "Difficulty",
            [
                "Gentle",
                "Realistic",
                "Challenging",
            ],
            index=1,
            key="sim_difficulty",
        )

    current_signature = (
        scenario,
        style,
        difficulty,
    )

    if (
        st.session_state.sim_config_signature
        is not None
        and st.session_state.sim_config_signature
        != current_signature
    ):
        st.info(
            "You changed the simulation settings. Click **Start / Reset Conversation** so the new scenario and partner style begin with a fresh memory."
        )

    if AI_AVAILABLE:
        st.markdown(
            f"""
            <div class="ai-online">
                <b>● Natural AI role-play is enabled</b><br>
                <span style="color:#687082;">
                    {OPENAI_MODEL} • persistent conversation context • emotion-aware short-message handling
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div class="ai-offline">
                <b>○ Local simulator mode</b><br>
                <span style="color:#687082;">
                    The improved fallback now understands short emotions such as “hurt”, “sad”, and “angry”.
                    Add OPENAI_API_KEY to Streamlit Secrets for fully open-ended AI role-play.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    info_card(
        style,
        f"{PARTNER_STYLES[style]}<br><br>"
        f"<b>Scenario:</b> {SIMULATOR_SCENARIOS[scenario]['context']}",
        "🎭",
    )

    reset_column, debrief_column = st.columns(2)

    with reset_column:
        if st.button(
            "🔄 Start / Reset Conversation",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.sim_chat = [
                {
                    "role": "assistant",
                    "content": SIMULATOR_SCENARIOS[scenario]["starter"],
                }
            ]

            st.session_state.sim_conversation_id = None
            st.session_state.sim_debrief = False
            st.session_state.sim_config_signature = current_signature

            st.rerun()

    with debrief_column:
        if st.button(
            "📊 End & Debrief",
            use_container_width=True,
        ):
            st.session_state.sim_debrief = True

    if not st.session_state.sim_chat:
        st.session_state.sim_chat = [
            {
                "role": "assistant",
                "content": SIMULATOR_SCENARIOS[scenario]["starter"],
            }
        ]

        st.session_state.sim_config_signature = current_signature

    st.divider()

    st.subheader(
        "Conversation"
    )

    for message in st.session_state.sim_chat[
        -RECENT_UI_MESSAGES:
    ]:
        avatar = (
            "💗"
            if message["role"] == "assistant"
            else "👤"
        )

        with st.chat_message(
            message["role"],
            avatar=avatar,
        ):
            st.markdown(
                message["content"]
            )

            if message["role"] == "user":
                analysis = advanced_message_analysis(
                    message["content"]
                )

                with st.expander(
                    "🧠 Live coach note"
                ):
                    st.write(
                        f"Message quality: **{analysis['overall']}%**"
                    )

                    if analysis["strengths"]:
                        st.write(
                            "**Strength:** "
                            + analysis["strengths"][0]
                        )

                    if analysis["opportunities"]:
                        st.write(
                            "**Try:** "
                            + analysis["opportunities"][0]
                        )

    sim_prompt = st.chat_input(
        "Reply as yourself..."
    )

    if sim_prompt:
        st.session_state.sim_chat.append({
            "role": "user",
            "content": sim_prompt,
        })

        user_turn = sum(
            1
            for item in st.session_state.sim_chat
            if item["role"] == "user"
        )

        if AI_AVAILABLE:
            try:
                conversation_id = ensure_conversation(
                    OPENAI_API_KEY,
                    "sim_conversation_id",
                )

                with st.spinner(
                    "Partner is responding..."
                ):
                    reply = call_openai_conversation(
                        api_key=OPENAI_API_KEY,
                        model=OPENAI_MODEL,
                        conversation_id=conversation_id,
                        instruction_text=simulator_instructions(
                            scenario,
                            style,
                            difficulty,
                        ),
                        user_message=sim_prompt,
                    )

            except Exception as exc:
                reply = local_simulator_reply(
                    style,
                    scenario,
                    difficulty,
                    sim_prompt,
                    user_turn,
                )

                reply += (
                    "\n\n_AI connection was unavailable for this turn, "
                    "so the improved local simulator answered instead._"
                )

        else:
            reply = local_simulator_reply(
                style,
                scenario,
                difficulty,
                sim_prompt,
                user_turn,
            )

        st.session_state.sim_chat.append({
            "role": "assistant",
            "content": reply,
        })

        st.session_state.sim_debrief = False

        st.rerun()

    if any(
        item["role"] == "user"
        for item in st.session_state.sim_chat
    ):
        with st.expander(
            "💡 Need help with your next reply?"
        ):
            st.write(
                suggested_reply(
                    scenario
                )
            )

            st.caption(
                "Use this as inspiration rather than a script you must copy."
            )

    if st.session_state.sim_debrief:
        result = chat_debrief(
            st.session_state.sim_chat
        )

        st.divider()

        st.subheader(
            "📊 Conversation Debrief"
        )

        if result is None:
            st.info(
                "Send at least one reply first."
            )

        else:
            first_metric, second_metric, third_metric = st.columns(3)

            first_metric.metric(
                "Overall Skill",
                f"{result['Overall']}%",
            )

            second_metric.metric(
                "Solution Focus",
                f"{result['Solution Focus']}%",
            )

            third_metric.metric(
                "Escalation Risk",
                f"{result['Escalation Risk']}%",
            )

            skills = pd.DataFrame(
                {
                    "Score": [
                        result["Clarity"],
                        result["Respect"],
                        result["Feeling Expression"],
                        result["Solution Focus"],
                        result["Boundary Expression"],
                    ]
                },
                index=[
                    "Clarity",
                    "Respect",
                    "Feeling Expression",
                    "Solution Focus",
                    "Boundary Expression",
                ],
            )

            st.bar_chart(
                skills
            )

            st.caption(
                "These are heuristic communication coaching signals for practice, not clinical measurements."
            )


# ============================================================
# RELATIONSHIP REFLECTION
# ============================================================

elif page == "🚦 Relationship Reflection":
    section_header(
        "Behavior and boundary reflection",
        "🚦 Relationship Reflection",
        "Reflect on observable behaviors instead of relying on labels like “toxic” or “perfect.”",
    )

    situation = st.text_area(
        "Describe the situation",
        placeholder=(
            "Example: The person I'm dating supports my career, "
            "but gets upset whenever I spend time with friends."
        ),
        height=150,
    )

    left, right = st.columns(2)

    with left:
        boundaries = st.slider(
            "Boundary respect",
            1,
            5,
            3,
        )

        communication = st.slider(
            "Safe, honest communication",
            1,
            5,
            3,
        )

        independence = st.slider(
            "Respect for independence",
            1,
            5,
            3,
        )

    with right:
        reliability = st.slider(
            "Reliability and follow-through",
            1,
            5,
            3,
        )

        trust = st.slider(
            "Trust and good faith",
            1,
            5,
            3,
        )

        repair = st.slider(
            "Ability to repair conflict",
            1,
            5,
            3,
        )

    if st.button(
        "🔎 Reflect on This Situation",
        type="primary",
        use_container_width=True,
    ):
        ratings = {
            "Boundaries": boundaries,
            "Communication": communication,
            "Independence": independence,
            "Reliability": reliability,
            "Trust": trust,
            "Repair": repair,
        }

        average = float(
            np.mean(
                list(
                    ratings.values()
                )
            )
        )

        percentage = round(
            average
            / 5
            * 100
        )

        chart_left, chart_right = st.columns(
            [0.8, 1.2]
        )

        with chart_left:
            st.metric(
                "Reflection Score",
                f"{percentage}%",
            )

            if average >= 4:
                st.success(
                    "Your ratings suggest several supportive relationship behaviors."
                )

            elif average >= 3:
                st.info(
                    "Your ratings are mixed. Some areas may be worth discussing directly."
                )

            else:
                st.warning(
                    "Several areas scored low. Focus on boundaries, communication, reliability, and whether both people's needs are respected."
                )

        with chart_right:
            st.pyplot(
                reflection_radar(
                    ratings
                ),
                use_container_width=True,
            )

        lows = [
            key
            for key, value in ratings.items()
            if value <= 2
        ]

        highs = [
            key
            for key, value in ratings.items()
            if value >= 4
        ]

        if highs:
            st.success(
                "Positive signals: "
                + ", ".join(highs)
            )

        if lows:
            st.warning(
                "Areas needing attention: "
                + ", ".join(lows)
            )

        st.subheader(
            "Conversation starters"
        )

        st.write(
            "• What specific behavior made this situation feel supportive or difficult?"
        )

        st.write(
            "• What boundary or expectation feels unclear?"
        )

        st.write(
            "• What would a fair and respectful change look like?"
        )

        st.write(
            "• Is this occasional, or is it a repeated pattern?"
        )

        st.write(
            "• Can both people discuss the concern without punishment, fear, or intimidation?"
        )


# ============================================================
# ML INTELLIGENCE
# ============================================================

elif page == "🤖 ML Intelligence":
    section_header(
        "Explainable machine learning",
        "🤖 ML Intelligence Lab",
        "Explore how unsupervised learning discovers patterns across 2,000 synthetic relationship-preference profiles.",
    )

    df = synthetic_profiles()

    (
        clustered,
        centers,
        scaler,
        model,
        pca,
        viz,
    ) = prepare_ml(
        df,
        5,
    )

    diagnostics = cluster_diagnostics(
        df
    )

    names = cluster_names(
        centers
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Synthetic Profiles",
        f"{len(df):,}",
    )

    c2.metric(
        "Preference Features",
        len(FEATURES),
    )

    c3.metric(
        "K-Means Clusters",
        5,
    )

    c4.metric(
        "PCA Dimensions",
        2,
    )

    tab1, tab2, tab3 = st.tabs([
        "📈 Model Selection",
        "🗺️ PCA Map",
        "🧩 Cluster Explorer",
    ])

    with tab1:
        left, right = st.columns(2)

        with left:
            fig, ax = plt.subplots(
                figsize=(7, 4)
            )

            ax.plot(
                diagnostics["k"],
                diagnostics["inertia"],
                marker="o",
            )

            ax.set_title(
                "Elbow Method"
            )

            ax.set_xlabel(
                "k"
            )

            ax.set_ylabel(
                "Inertia"
            )

            ax.grid(
                alpha=0.2
            )

            st.pyplot(
                fig,
                use_container_width=True,
            )

        with right:
            fig, ax = plt.subplots(
                figsize=(7, 4)
            )

            ax.plot(
                diagnostics["k"],
                diagnostics["silhouette"],
                marker="o",
            )

            ax.set_title(
                "Silhouette Score"
            )

            ax.set_xlabel(
                "k"
            )

            ax.set_ylabel(
                "Silhouette"
            )

            ax.grid(
                alpha=0.2
            )

            st.pyplot(
                fig,
                use_container_width=True,
            )

        best_index = diagnostics[
            "silhouette"
        ].idxmax()

        best_k = int(
            diagnostics.loc[
                best_index,
                "k",
            ]
        )

        best_score = float(
            diagnostics.loc[
                best_index,
                "silhouette",
            ]
        )

        st.info(
            f"Highest silhouette score in this synthetic dataset: "
            f"k={best_k} with score {best_score:.3f}. "
            "The product keeps k=5 to demonstrate a stable five-group experience."
        )

    with tab2:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        for cluster_id in sorted(
            viz["cluster"].unique()
        ):
            subset = viz[
                viz["cluster"]
                == cluster_id
            ]

            ax.scatter(
                subset["PC1"],
                subset["PC2"],
                s=18,
                alpha=0.55,
                label=(
                    f"{cluster_id + 1}: "
                    f"{names[cluster_id]}"
                ),
            )

        ax.set_title(
            "PCA Projection of 2,000 Synthetic Profiles"
        )

        ax.set_xlabel(
            "Principal Component 1"
        )

        ax.set_ylabel(
            "Principal Component 2"
        )

        ax.legend(
            fontsize=8
        )

        ax.grid(
            alpha=0.15
        )

        st.pyplot(
            fig,
            use_container_width=True,
        )

        st.caption(
            "PCA compresses the 10-dimensional preference space into 2 dimensions for visualization."
        )

    with tab3:
        counts = clustered[
            "cluster"
        ].value_counts().sort_index()

        rows = []

        for index, center in centers.iterrows():
            top3 = center.sort_values(
                ascending=False
            ).index[:3]

            rows.append({
                "Cluster": index + 1,
                "Name": names[index],
                "Profiles": int(
                    counts.loc[index]
                ),
                "Top Signals": ", ".join(
                    FEATURES[key]
                    for key in top3
                ),
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        selected = st.selectbox(
            "Inspect cluster",
            range(5),
            format_func=lambda index: (
                f"Cluster {index + 1} — "
                f"{names[index]}"
            ),
        )

        center = centers.iloc[
            selected
        ].sort_values(
            ascending=False
        )

        st.bar_chart(
            pd.DataFrame(
                {
                    "Average Score":
                        center.values
                },
                index=[
                    FEATURES[key]
                    for key in center.index
                ],
            )
        )

    st.subheader(
        "Which ML group am I closest to?"
    )

    if not st.session_state.saved_profile:
        st.info(
            "Create and save a Partner Blueprint first, then return here."
        )

    else:
        user_array = np.array([
            [
                st.session_state.saved_profile[key]
                for key in FEATURES
            ]
        ])

        user_scaled = scaler.transform(
            user_array
        )

        cluster_id = int(
            model.predict(
                user_scaled
            )[0]
        )

        distances = model.transform(
            user_scaled
        )[0]

        sorted_distances = np.sort(
            distances
        )

        fit = 1 - (
            sorted_distances[0]
            / (
                sorted_distances[-1]
                + 1e-9
            )
        )

        fit = float(
            np.clip(
                fit,
                0,
                1,
            )
        )

        left, right = st.columns(2)

        left.metric(
            "Closest ML Group",
            names[cluster_id],
        )

        right.metric(
            "Relative Cluster Fit",
            f"{fit * 100:.0f}%",
        )

        st.caption(
            "Relative cluster fit is a distance visualization, not a calibrated probability."
        )

    st.caption(
        "All ML data here is synthetic and generated inside the app for learning and portfolio demonstration."
    )


# ============================================================
# PROFILE CENTER
# ============================================================

elif page == "📊 My Profile Center":
    section_header(
        "Session dashboard",
        "📊 My Profile Center",
        "Review the profile information and insights you created during this session.",
    )

    if not st.session_state.saved_profile:
        st.info(
            "You have not created a Partner Blueprint yet."
        )

    else:
        profile = st.session_state.saved_profile

        ranked = ranked_profile(
            profile
        )

        style_name, style_description = relationship_style(
            profile
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Profile",
            st.session_state.saved_profile_name,
        )

        c2.metric(
            "Top Priority",
            FEATURES[
                ranked[0][0]
            ],
        )

        c3.metric(
            "Preference Style",
            style_name,
        )

        info_card(
            style_name,
            style_description,
            "💎",
        )

        st.pyplot(
            radar_chart(
                profile
            ),
            use_container_width=True,
        )

        st.subheader(
            "Top priorities"
        )

        columns = st.columns(5)

        for column, (
            key,
            score,
        ) in zip(
            columns,
            ranked[:5],
        ):
            with column:
                st.metric(
                    FEATURES[key],
                    f"{score}%",
                )

        st.download_button(
            "⬇️ Download Profile",
            data=profile_csv(
                profile,
                st.session_state.saved_profile_name,
            ),
            file_name="partnerlens_profile.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.warning(
            "Session data is temporary and may reset when the app restarts."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "<br><br>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        text-align:center;
        padding:1.2rem;
        border-top:1px solid rgba(100,80,150,.10);
        color:#85889a;
        font-size:.82rem;
    ">
        💞 PartnerLens AI • Python • Streamlit • scikit-learn • OpenAI Responses API • Responsible AI
    </div>
    """,
    unsafe_allow_html=True,
)
