# PartnerLens AI
# A polished Streamlit relationship self-discovery and communication practice app.
#
# IMPORTANT:
# - This app is educational and reflective. It is not therapy or diagnosis.
# - AI features require an Groq API key in Streamlit Secrets.
# - The AI guide and simulator preserve conversation context in st.session_state.
# - The app is designed to stay usable even when the API is unavailable.

from __future__ import annotations

import io
import json
import math
import os
import random
import statistics
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="PartnerLens AI",
    page_icon="💞",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "PartnerLens AI is a relationship self-discovery and communication "
            "practice portfolio project built with Python, Streamlit, data analysis, "
            "explainable scoring, and OpenAI conversational AI."
        )
    },
)


# =============================================================================
# 2. APP CONSTANTS
# =============================================================================

APP_NAME = "PartnerLens AI"
APP_TAGLINE = "Relationship self-discovery, compatibility reflection, and communication practice."
APP_VERSION = "3.0"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
MAX_AI_CALLS_PER_SESSION = 60
MAX_HISTORY_MESSAGES = 24
MAX_MESSAGE_CHARACTERS = 7000

PAGE_HOME = "🏠 Home"
PAGE_BLUEPRINT = "🧠 Partner Blueprint"
PAGE_SCENARIOS = "🎭 Scenario Discovery"
PAGE_COMPATIBILITY = "🤝 Compatibility Lab"
PAGE_GUIDE = "🧭 AI Relationship Guide"
PAGE_COACH = "💬 Conversation Coach"
PAGE_SIMULATOR = "🗨️ AI Chat Simulator"
PAGE_REFLECTION = "🚦 Relationship Reflection"
PAGE_ML = "🤖 ML Intelligence"
PAGE_PROFILE = "📊 My Profile Center"

PAGES = [
    PAGE_HOME,
    PAGE_BLUEPRINT,
    PAGE_SCENARIOS,
    PAGE_COMPATIBILITY,
    PAGE_GUIDE,
    PAGE_COACH,
    PAGE_SIMULATOR,
    PAGE_REFLECTION,
    PAGE_ML,
    PAGE_PROFILE,
]


# =============================================================================
# 3. RELATIONSHIP DIMENSIONS
# =============================================================================

FEATURES: Dict[str, str] = {
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

FEATURE_ICONS: Dict[str, str] = {
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

FEATURE_QUESTIONS: Dict[str, str] = {
    "communication": "How important is open and consistent communication?",
    "emotional_maturity": "How important is calm, emotionally mature conflict handling?",
    "independence": "How important are personal space and independent goals?",
    "ambition": "How important are career growth and personal ambition?",
    "family_orientation": "How important is being family-oriented?",
    "affection": "How important are affection, warmth, and reassurance?",
    "social_lifestyle": "How important is an active social lifestyle?",
    "fitness": "How important is an active and healthy lifestyle?",
    "financial_planning": "How important is responsible financial planning?",
    "adventure": "How important are travel, novelty, and spontaneity?",
}

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "communication": (
        "How much you value clear discussion, regular check-ins, active listening, "
        "and direct conversations about concerns."
    ),
    "emotional_maturity": (
        "How much you value accountability, emotional regulation, patience, repair, "
        "and respectful disagreement."
    ),
    "independence": (
        "How much you value personal time, individual friendships, separate interests, "
        "and autonomy inside a relationship."
    ),
    "ambition": (
        "How much you value goal setting, career development, achievement, discipline, "
        "and personal growth."
    ),
    "family_orientation": (
        "How much family relationships, traditions, long-term family plans, and family "
        "involvement matter to you."
    ),
    "affection": (
        "How much you value verbal reassurance, physical affection, appreciation, "
        "warmth, and visible closeness."
    ),
    "social_lifestyle": (
        "How much you value events, friends, social plans, community, and an active "
        "shared social life."
    ),
    "fitness": (
        "How much you value movement, exercise, nutrition awareness, and an active "
        "day-to-day lifestyle."
    ),
    "financial_planning": (
        "How much you value budgeting, saving, financial responsibility, transparency, "
        "and planning for shared goals."
    ),
    "adventure": (
        "How much you value travel, trying new things, novelty, spontaneity, and "
        "exploration."
    ),
}


# =============================================================================
# 4. SCENARIOS AND ROLE-PLAY DATA
# =============================================================================

SCENARIOS: Dict[str, Dict[str, str]] = {
    "Canceled plans because of work": {
        "icon": "💼",
        "context": (
            "You were looking forward to spending time together, but your partner "
            "canceled because work became unexpectedly busy."
        ),
        "starter": (
            "I know we had plans tonight, but work got complicated and I need to stay "
            "late. I'm sorry."
        ),
    },
    "Different texting needs": {
        "icon": "📱",
        "context": (
            "One person wants more frequent communication during the day while the "
            "other prefers fewer check-ins."
        ),
        "starter": (
            "I've noticed we communicate differently during the day. I prefer fewer "
            "messages than you seem to."
        ),
    },
    "Need for personal space": {
        "icon": "🌙",
        "context": (
            "Your partner asks for personal space after a stressful day, and you want "
            "to understand what that means for the relationship."
        ),
        "starter": (
            "I care about us, but tonight I really need some time by myself to reset."
        ),
    },
    "Career move to another state": {
        "icon": "🧳",
        "context": (
            "A major career opportunity would require one partner to move, creating a "
            "decision that affects both people's goals."
        ),
        "starter": (
            "I got the job offer. It's a big opportunity, but it would mean moving to "
            "another state."
        ),
    },
    "Money and spending disagreement": {
        "icon": "💳",
        "context": (
            "You and your partner have different attitudes toward saving, spending, "
            "shared expenses, and long-term financial planning."
        ),
        "starter": (
            "I know we see spending differently. I don't want money to become something "
            "we keep fighting about."
        ),
    },
    "Friendship and jealousy boundary": {
        "icon": "🤝",
        "context": (
            "A friendship is creating insecurity and a need for a respectful boundary "
            "conversation without controlling each other."
        ),
        "starter": (
            "I want to be honest. I sometimes feel uncomfortable about how much time "
            "you spend with that friend."
        ),
    },
    "Household responsibility imbalance": {
        "icon": "🧹",
        "context": (
            "One person feels they are carrying more of the household planning and "
            "chores than the other."
        ),
        "starter": (
            "I've been feeling overwhelmed with how much of the household stuff I've "
            "been handling lately."
        ),
    },
    "Family boundary disagreement": {
        "icon": "🏡",
        "context": (
            "One partner wants stronger boundaries with extended family while the other "
            "feels pressure to keep family closely involved."
        ),
        "starter": (
            "I know family matters to both of us, but I think we need clearer boundaries "
            "around our private decisions."
        ),
    },
    "Different social energy": {
        "icon": "🎉",
        "context": (
            "One person enjoys frequent social plans while the other needs more quiet "
            "time and smaller gatherings."
        ),
        "starter": (
            "I like going out with friends often, but I know it can feel like too much "
            "for you sometimes."
        ),
    },
    "Long-distance communication": {
        "icon": "✈️",
        "context": (
            "Distance makes communication routines, reassurance, scheduling, and "
            "expectations more important."
        ),
        "starter": (
            "I miss you, but I also think we need a routine that works better for both "
            "of us while we're apart."
        ),
    },
}

PARTNER_STYLES: Dict[str, str] = {
    "Calm Communicator": (
        "Listens, explains feelings, asks questions, and tries to resolve conflict "
        "collaboratively."
    ),
    "Independent / Needs Space": (
        "Values autonomy, personal time, and calm conversations without constant contact."
    ),
    "Affectionate Reassurer": (
        "Values emotional closeness, reassurance, warmth, and frequent connection."
    ),
    "Career-Driven / Busy": (
        "Cares about the relationship while placing strong importance on goals, work, "
        "and scheduling."
    ),
    "Conflict-Avoidant": (
        "Finds tension uncomfortable and may initially pull back, minimize, or need more "
        "time before engaging."
    ),
    "Direct Communicator": (
        "Prefers straightforward discussion, concrete examples, and clear requests over "
        "hints or indirect communication."
    ),
    "Practical Problem Solver": (
        "Focuses on concrete next steps, logistics, tradeoffs, and agreements."
    ),
    "Emotionally Expressive": (
        "Shares feelings openly and values emotional acknowledgment before problem solving."
    ),
}

DIFFICULTY_DESCRIPTIONS: Dict[str, str] = {
    "Gentle": (
        "Warm, patient, and highly collaborative. The partner tries hard to understand "
        "and meet halfway."
    ),
    "Realistic": (
        "Balanced and reasonable. The partner listens but also has genuine preferences "
        "and will not automatically agree."
    ),
    "Challenging": (
        "Still respectful, but firmer. The partner defends their needs and may require "
        "clearer communication before compromising."
    ),
}


# =============================================================================
# 5. SCENARIO DISCOVERY QUESTIONS
# =============================================================================

DISCOVERY_QUESTIONS: List[Dict[str, Any]] = [
    {
        "question": "After a stressful day, what usually helps you most?",
        "options": {
            "Give me some space first": {
                "independence": 3,
                "emotional_maturity": 1,
            },
            "Ask me what happened and listen": {
                "communication": 3,
                "emotional_maturity": 2,
            },
            "Comfort me and stay close": {
                "affection": 3,
                "communication": 1,
            },
            "Distract me with something fun": {
                "adventure": 2,
                "social_lifestyle": 1,
            },
        },
    },
    {
        "question": "Your ideal free Saturday is closest to...",
        "options": {
            "A relaxed day at home together": {
                "affection": 2,
                "family_orientation": 1,
            },
            "A big event with friends": {
                "social_lifestyle": 3,
                "adventure": 1,
            },
            "A workout or active plan": {
                "fitness": 3,
                "adventure": 1,
            },
            "Separate interests, then reconnect later": {
                "independence": 3,
                "communication": 1,
            },
        },
    },
    {
        "question": "During a disagreement, what feels most natural?",
        "options": {
            "Talk about it right away": {
                "communication": 3,
            },
            "Take time, then discuss it calmly": {
                "emotional_maturity": 3,
                "independence": 1,
            },
            "Write or text first so I can organize my thoughts": {
                "communication": 2,
                "emotional_maturity": 1,
            },
            "Let small disagreements go": {
                "independence": 1,
                "emotional_maturity": 1,
            },
        },
    },
    {
        "question": "If one partner gets a major opportunity far away, you prefer to...",
        "options": {
            "Strongly consider moving together": {
                "family_orientation": 2,
                "affection": 2,
            },
            "Compare both careers and make a joint plan": {
                "ambition": 3,
                "communication": 2,
            },
            "Try long distance before making a permanent choice": {
                "independence": 2,
                "communication": 2,
            },
            "Protect my own location and career first": {
                "independence": 3,
                "ambition": 2,
            },
        },
    },
    {
        "question": "What communication rhythm sounds best?",
        "options": {
            "Frequent messages throughout the day": {
                "communication": 3,
                "affection": 2,
            },
            "A few meaningful check-ins": {
                "communication": 3,
                "emotional_maturity": 1,
            },
            "Mostly talk after work or school": {
                "independence": 2,
                "communication": 1,
            },
            "Very little texting; connect in person": {
                "independence": 3,
            },
        },
    },
    {
        "question": "When thinking about money in a serious relationship...",
        "options": {
            "I want detailed planning and shared goals": {
                "financial_planning": 3,
                "communication": 1,
            },
            "I want some shared planning but also separate freedom": {
                "financial_planning": 2,
                "independence": 2,
            },
            "I prefer flexibility over strict budgets": {
                "adventure": 1,
                "independence": 1,
            },
            "I rarely think about money compatibility": {
                "financial_planning": 0,
            },
        },
    },
    {
        "question": "How do you feel about trying new things together?",
        "options": {
            "I love travel, surprises, and novelty": {
                "adventure": 3,
                "social_lifestyle": 1,
            },
            "Some novelty is great, but I also like routine": {
                "adventure": 2,
                "emotional_maturity": 1,
            },
            "I prefer familiar plans and predictability": {
                "family_orientation": 1,
                "financial_planning": 1,
            },
            "It depends completely on the situation": {
                "emotional_maturity": 1,
            },
        },
    },
    {
        "question": "How important is physical activity as a shared lifestyle?",
        "options": {
            "Very important; I want an active partner": {
                "fitness": 3,
            },
            "Nice to share, but not essential": {
                "fitness": 2,
            },
            "I care more about overall health than exercise": {
                "fitness": 1,
                "emotional_maturity": 1,
            },
            "Not important to me": {
                "fitness": 0,
            },
        },
    },
]


# =============================================================================
# 6. SESSION STATE DEFAULTS
# =============================================================================

STATE_DEFAULTS: Dict[str, Any] = {
    "saved_profile": None,
    "saved_profile_name": "My Profile",
    "scenario_profile": None,
    "scenario_answers": [],
    "compatibility_last": None,
    "guide_messages": [],
    "sim_messages": [],
    "sim_signature": None,
    "ai_calls": 0,
    "coach_last": None,
    "reflection_last": None,
    "flash_message": None,
    "api_last_error": None,
}

for _key, _value in STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value


# =============================================================================
# 7. GLOBAL CSS
# =============================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

    :root {
        --pl-pink: #ff4f91;
        --pl-rose: #ff6b7a;
        --pl-purple: #7756ff;
        --pl-violet: #a05cff;
        --pl-blue: #3c9df2;
        --pl-cyan: #29c7c2;
        --pl-gold: #ffb84d;
        --pl-ink: #20243a;
        --pl-muted: #6d7287;
        --pl-border: rgba(105, 87, 165, .14);
        --pl-card: rgba(255,255,255,.78);
        --pl-card-strong: rgba(255,255,255,.93);
    }

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 7% 5%, rgba(255,79,145,.14), transparent 24%),
            radial-gradient(circle at 93% 8%, rgba(119,86,255,.17), transparent 27%),
            radial-gradient(circle at 79% 91%, rgba(41,199,194,.12), transparent 27%),
            linear-gradient(135deg, #fffafd 0%, #f8f6ff 50%, #f4fbff 100%);
        color: var(--pl-ink);
    }

    .block-container {
        max-width: 1420px;
        padding-top: 1.20rem;
        padding-bottom: 4rem;
    }

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 0% 0%, rgba(255,79,145,.08), transparent 30%),
            linear-gradient(180deg, rgba(255,255,255,.96), rgba(249,247,255,.96));
        border-right: 1px solid rgba(105,87,165,.11);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: .6rem;
    }

    .pl-brand {
        padding: .45rem .1rem .7rem .1rem;
    }

    .pl-brand-title {
        font-family: "Manrope", sans-serif;
        font-size: 1.52rem;
        font-weight: 800;
        letter-spacing: -.035em;
        color: #25283d;
    }

    .pl-brand-sub {
        margin-top: .22rem;
        color: #7a7e91;
        font-size: .79rem;
        line-height: 1.5;
    }

    .pl-hero {
        position: relative;
        overflow: hidden;
        padding: 2.35rem 2.55rem;
        border-radius: 32px;
        background:
            radial-gradient(circle at 12% 15%, rgba(255,255,255,.98), transparent 23%),
            linear-gradient(115deg, rgba(255,104,157,.22), rgba(122,92,255,.20) 48%, rgba(65,199,255,.19));
        border: 1px solid rgba(122,92,255,.12);
        box-shadow: 0 24px 70px rgba(76,47,128,.10);
        margin-bottom: 1rem;
    }

    .pl-hero::before {
        content: "";
        position: absolute;
        width: 270px;
        height: 270px;
        right: -84px;
        top: -105px;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(255,79,145,.40), rgba(122,92,255,.34));
    }

    .pl-hero::after {
        content: "";
        position: absolute;
        width: 150px;
        height: 150px;
        right: 150px;
        bottom: -100px;
        border-radius: 999px;
        background: rgba(52,215,210,.23);
    }

    .pl-kicker {
        position: relative;
        z-index: 2;
        font-size: .77rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .16em;
        color: #795bd6;
        margin-bottom: .55rem;
    }

    .pl-hero-title {
        position: relative;
        z-index: 2;
        font-family: "Manrope", sans-serif;
        font-size: clamp(2.65rem, 5vw, 4.85rem);
        font-weight: 800;
        line-height: .98;
        letter-spacing: -.055em;
        margin: 0 0 .85rem 0;
        background: linear-gradient(90deg, #ff4f91, #8758ff, #2e9ce7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .pl-hero-copy {
        position: relative;
        z-index: 2;
        max-width: 900px;
        color: #565b70;
        font-size: 1.06rem;
        line-height: 1.75;
        margin: 0;
    }

    .pl-card {
        padding: 1.15rem 1.2rem;
        border-radius: 22px;
        background: var(--pl-card);
        border: 1px solid var(--pl-border);
        box-shadow: 0 13px 38px rgba(64,52,108,.055);
        backdrop-filter: blur(13px);
        margin-bottom: .9rem;
    }

    .pl-card-title {
        font-size: 1.06rem;
        font-weight: 800;
        margin-bottom: .38rem;
        color: #2c3045;
    }

    .pl-card-body {
        color: #62677c;
        line-height: 1.66;
        font-size: .94rem;
    }

    .pl-feature-card {
        position: relative;
        overflow: hidden;
        padding: 1.3rem 1.35rem;
        min-height: 175px;
        border-radius: 24px;
        border: 1px solid rgba(130,100,190,.13);
        background: linear-gradient(145deg, rgba(255,255,255,.94), rgba(250,247,255,.76));
        box-shadow: 0 16px 42px rgba(77,59,121,.065);
        margin-bottom: 1rem;
    }

    .pl-feature-card::after {
        content:"";
        position:absolute;
        width:90px;
        height:90px;
        border-radius:50%;
        right:-30px;
        bottom:-35px;
        background:linear-gradient(135deg,rgba(255,79,145,.16),rgba(122,92,255,.18));
    }

    .pl-feature-icon {
        font-size: 1.82rem;
        margin-bottom: .45rem;
    }

    .pl-feature-title {
        position: relative;
        z-index: 2;
        font-size: 1.08rem;
        font-weight: 800;
        color: #292d42;
        margin-bottom: .35rem;
    }

    .pl-feature-copy {
        position: relative;
        z-index: 2;
        color: #666b80;
        line-height: 1.55;
        font-size: .92rem;
    }

    .pl-section-kicker {
        font-size: .75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .14em;
        color: #7d62d7;
        margin-top: .35rem;
        margin-bottom: .15rem;
    }

    .pl-pill {
        display: inline-block;
        padding: .36rem .72rem;
        margin: .18rem .22rem .18rem 0;
        border-radius: 999px;
        background: rgba(255,255,255,.80);
        border: 1px solid rgba(122,92,255,.15);
        font-size: .78rem;
        color: #615b79;
    }

    .pl-badge {
        display: inline-block;
        padding: .3rem .65rem;
        border-radius: 999px;
        background: linear-gradient(90deg,rgba(255,79,145,.11),rgba(122,92,255,.11));
        color:#6f54ca;
        font-weight:700;
        font-size:.76rem;
    }

    .pl-ai-online {
        padding:.78rem 1rem;
        border-radius:16px;
        background:linear-gradient(90deg,rgba(52,215,210,.13),rgba(69,163,255,.10));
        border:1px solid rgba(52,180,190,.18);
        margin-bottom:.8rem;
    }

    .pl-ai-offline {
        padding:.78rem 1rem;
        border-radius:16px;
        background:linear-gradient(90deg,rgba(255,191,75,.13),rgba(255,107,122,.09));
        border:1px solid rgba(215,150,55,.18);
        margin-bottom:.8rem;
    }

    .pl-score-good {
        padding: .85rem 1rem;
        border-radius: 16px;
        background: rgba(52, 199, 145, .10);
        border: 1px solid rgba(52, 170, 130, .17);
    }

    .pl-score-mid {
        padding: .85rem 1rem;
        border-radius: 16px;
        background: rgba(255, 184, 77, .11);
        border: 1px solid rgba(220, 150, 60, .17);
    }

    .pl-score-low {
        padding: .85rem 1rem;
        border-radius: 16px;
        background: rgba(255, 107, 122, .10);
        border: 1px solid rgba(220, 90, 105, .17);
    }

    .pl-chat-note {
        padding: .72rem .9rem;
        border-radius: 14px;
        background: rgba(119,86,255,.07);
        border: 1px solid rgba(119,86,255,.12);
        color: #646079;
        margin-bottom: .8rem;
        font-size: .88rem;
    }

    .pl-mini-stat {
        text-align:center;
        padding: 1rem .8rem;
        border-radius: 18px;
        background: rgba(255,255,255,.78);
        border: 1px solid var(--pl-border);
        box-shadow: 0 10px 28px rgba(60,45,100,.05);
    }

    .pl-mini-stat-value {
        font-family: "Manrope", sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        color: #302a50;
    }

    .pl-mini-stat-label {
        margin-top: .15rem;
        font-size: .78rem;
        color: #7a7e91;
    }

    h1, h2, h3 {
        font-family: "Manrope", "Inter", sans-serif;
        letter-spacing: -.035em;
        color: #25293e;
    }

    p {
        color: #4e5469;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.78);
        border: 1px solid rgba(115,92,170,.12);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        box-shadow: 0 12px 30px rgba(59,45,90,.05);
    }

    div[data-testid="stChatMessage"] {
        background: rgba(255,255,255,.73);
        border: 1px solid rgba(115,92,170,.10);
        border-radius: 20px;
        padding: .48rem .85rem;
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
        transition: .18s ease;
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

    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        border-radius: 13px !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 13px !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(124,92,255,.10);
    }

    details {
        border-radius: 15px !important;
        border: 1px solid rgba(124,92,255,.10) !important;
        background: rgba(255,255,255,.60);
    }

    hr {
        border-color: rgba(100,80,150,.10);
    }

    footer {
        visibility: hidden;
    }

    @media (max-width: 850px) {
        .pl-hero {
            padding: 1.7rem 1.35rem;
        }

        .pl-hero-title {
            font-size: 2.65rem;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 8. GENERIC DISPLAY HELPERS
# =============================================================================

def section_header(kicker: str, title: str, body: Optional[str] = None) -> None:
    st.markdown(
        f'<div class="pl-section-kicker">{kicker}</div>',
        unsafe_allow_html=True,
    )
    st.header(title)
    if body:
        st.write(body)


def info_card(title: str, body: str, icon: str = "✨") -> None:
    st.markdown(
        f"""
        <div class="pl-card">
            <div class="pl-card-title">{icon} {title}</div>
            <div class="pl-card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(icon: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="pl-feature-card">
            <div class="pl-feature-icon">{icon}</div>
            <div class="pl-feature-title">{title}</div>
            <div class="pl-feature-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str) -> str:
    return f'<span class="pl-pill">{text}</span>'


def safe_rerun() -> None:
    st.rerun()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def as_percent(value_1_to_5: float) -> int:
    return int(round(clamp(value_1_to_5, 1, 5) * 20))


def short_text(text: str, max_chars: int = 240) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"


# =============================================================================
# 9. GROQ FREE-TIER AI CONFIGURATION
# =============================================================================

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        if value is None:
            return default
        return str(value).strip()
    except Exception:
        return default


def resolve_groq_api_key() -> str:
    return get_secret(
        "GROQ_API_KEY",
        os.getenv("GROQ_API_KEY", ""),
    )


def resolve_groq_model() -> str:
    return (
        get_secret(
            "GROQ_MODEL",
            os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        )
        or DEFAULT_GROQ_MODEL
    )


GROQ_API_KEY = resolve_groq_api_key()
GROQ_MODEL = resolve_groq_model()

AI_AVAILABLE = bool(
    GROQ_API_KEY
    and OpenAI is not None
)


def ai_status_message() -> None:
    if AI_AVAILABLE:
        st.markdown(
            f"""
            <div class="pl-ai-online">
                <b>● Free AI Online</b><br>
                <span style="color:#667087;">
                    Groq · {GROQ_MODEL}. Conversation memory is active for this session.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="pl-ai-offline">
                <b>○ Free AI not connected yet</b><br>
                <span style="color:#667087;">
                    Add GROQ_API_KEY in Streamlit Cloud → Manage app → Settings → Secrets.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def sanitize_history(
    messages: Sequence[Dict[str, str]],
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> List[Dict[str, str]]:
    clean: List[Dict[str, str]] = []

    for item in messages[-max_messages:]:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()

        if role not in {"user", "assistant"}:
            continue
        if not content:
            continue
        if len(content) > MAX_MESSAGE_CHARACTERS:
            content = content[:MAX_MESSAGE_CHARACTERS]

        clean.append({"role": role, "content": content})

    return clean


def call_ai(
    messages: Sequence[Dict[str, str]],
    instructions: str,
    *,
    max_output_tokens: int = 900,
) -> str:
    if not AI_AVAILABLE:
        raise RuntimeError(
            "Groq AI is not configured. Add GROQ_API_KEY to Streamlit Secrets."
        )

    if st.session_state.ai_calls >= MAX_AI_CALLS_PER_SESSION:
        raise RuntimeError(
            "This app session reached its AI-call safety limit. "
            "Use Clear entire session to start fresh."
        )

    history = sanitize_history(messages)
    if not history:
        raise ValueError("No conversation message was provided.")

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
    )

    api_messages: List[Dict[str, str]] = [
        {"role": "system", "content": instructions}
    ]
    api_messages.extend(history)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=api_messages,
            temperature=0.65,
            max_completion_tokens=max_output_tokens,
            top_p=0.95,
        )
    except Exception as exc:
        st.session_state.api_last_error = f"{type(exc).__name__}: {exc}"
        raise

    try:
        answer = response.choices[0].message.content or ""
    except Exception:
        answer = ""

    answer = answer.strip()
    if not answer:
        raise RuntimeError("The Groq model returned an empty response.")

    st.session_state.ai_calls += 1
    st.session_state.api_last_error = None
    return answer


# =============================================================================
# 10. PROFILE HELPERS
# =============================================================================

def collect_profile(
    prefix: str,
    default_profile: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    profile: Dict[str, int] = {}

    for key, question in FEATURE_QUESTIONS.items():
        default = 3

        if default_profile:
            default = int(default_profile.get(key, 3))

        profile[key] = st.slider(
            f"{FEATURE_ICONS[key]} {question}",
            min_value=1,
            max_value=5,
            value=default,
            key=f"{prefix}_{key}",
            help=FEATURE_DESCRIPTIONS[key],
        )

    return profile


def profile_dataframe(profile: Dict[str, int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for key, value in profile.items():
        rows.append(
            {
                "Dimension": FEATURES[key],
                "Score (1–5)": int(value),
                "Importance %": as_percent(value),
                "Description": FEATURE_DESCRIPTIONS[key],
            }
        )

    return pd.DataFrame(rows)


def ranked_profile(profile: Dict[str, int]) -> List[Tuple[str, int]]:
    return sorted(
        profile.items(),
        key=lambda pair: pair[1],
        reverse=True,
    )


def relationship_style(profile: Dict[str, int]) -> Tuple[str, str]:
    ranked = ranked_profile(profile)
    top = {key for key, _ in ranked[:4]}

    if {"communication", "affection"}.issubset(top):
        return (
            "Expressive Connector",
            "You may value frequent emotional connection, clear communication, and visible warmth.",
        )

    if {"independence", "ambition"}.issubset(top):
        return (
            "Independent Achiever",
            "You may thrive with someone who respects autonomy, personal goals, and growth.",
        )

    if {"family_orientation", "financial_planning"}.issubset(top):
        return (
            "Future Builder",
            "Long-term stability, family, and practical planning appear especially meaningful.",
        )

    if {"adventure", "social_lifestyle"}.issubset(top):
        return (
            "Social Explorer",
            "Shared experiences, novelty, energy, and an active social life may matter strongly.",
        )

    if {"emotional_maturity", "communication"}.issubset(top):
        return (
            "Grounded Communicator",
            "You may place high value on thoughtful conversation, repair, and respectful conflict handling.",
        )

    if {"fitness", "adventure"}.issubset(top):
        return (
            "Active Explorer",
            "Movement, experiences, and an energetic lifestyle may be important parts of connection.",
        )

    return (
        "Balanced Builder",
        "Your priorities are spread across several dimensions rather than being dominated by one theme.",
    )


def profile_context_for_ai() -> str:
    profile = st.session_state.saved_profile

    if not profile:
        return "No saved Partner Blueprint is available."

    ranked = ranked_profile(profile)
    top = ranked[:5]

    return (
        "The user has a saved Partner Blueprint. Their highest stated preference "
        "scores are: "
        + ", ".join(
            f"{FEATURES[key]} {score}/5"
            for key, score in top
        )
        + ". Use this only when relevant. These are preferences, not diagnoses."
    )


def compatibility_score(
    person_a: Dict[str, int],
    person_b: Dict[str, int],
) -> float:
    differences = [
        abs(person_a[key] - person_b[key]) / 4
        for key in FEATURES
    ]

    return round(
        (1 - float(np.mean(differences))) * 100,
        1,
    )


def compatibility_dataframe(
    person_a: Dict[str, int],
    person_b: Dict[str, int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for key in FEATURES:
        difference = abs(person_a[key] - person_b[key])
        alignment = round((1 - difference / 4) * 100)

        rows.append(
            {
                "Dimension": FEATURES[key],
                "Person A": person_a[key],
                "Person B": person_b[key],
                "Difference": difference,
                "Alignment %": alignment,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "Alignment %",
        ascending=False,
    )


# =============================================================================
# 11. VISUALIZATION HELPERS
# =============================================================================

def radar_chart(
    profile: Dict[str, int],
    title: str = "Preference Radar",
):
    labels = [FEATURES[key] for key in FEATURES]
    values = [profile[key] for key in FEATURES]

    values = values + values[:1]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(labels),
        endpoint=False,
    ).tolist()
    angles = angles + angles[:1]

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = plt.subplot(111, polar=True)

    ax.plot(
        angles,
        values,
        linewidth=2.4,
    )

    ax.fill(
        angles,
        values,
        alpha=0.13,
    )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8.4)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5)
    ax.set_title(
        title,
        pad=24,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def comparison_radar(
    person_a: Dict[str, int],
    person_b: Dict[str, int],
):
    labels = [FEATURES[key] for key in FEATURES]

    values_a = [person_a[key] for key in FEATURES]
    values_b = [person_b[key] for key in FEATURES]

    values_a = values_a + values_a[:1]
    values_b = values_b + values_b[:1]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(labels),
        endpoint=False,
    ).tolist()
    angles = angles + angles[:1]

    fig = plt.figure(figsize=(7.4, 7.4))
    ax = plt.subplot(111, polar=True)

    ax.plot(
        angles,
        values_a,
        linewidth=2.2,
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
        linewidth=2.2,
        label="Person B",
    )
    ax.fill(
        angles,
        values_b,
        alpha=0.08,
    )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8.2)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5)
    ax.set_title(
        "Preference Comparison",
        pad=24,
        fontweight="bold",
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.22, 1.12),
    )

    plt.tight_layout()
    return fig


def donut_chart(score: float):
    score = float(clamp(score, 0, 100))

    fig, ax = plt.subplots(figsize=(5.7, 5.7))

    ax.pie(
        [score, 100 - score],
        startangle=90,
        wedgeprops={"width": 0.27},
    )

    ax.text(
        0,
        0.06,
        f"{score:.1f}%",
        ha="center",
        va="center",
        fontsize=27,
        fontweight="bold",
    )

    ax.text(
        0,
        -0.15,
        "preference alignment",
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


# =============================================================================
# 12. SCENARIO DISCOVERY HELPERS
# =============================================================================

def analyze_discovery_answers(
    answers: Sequence[str],
) -> List[Tuple[str, int]]:
    scores = {
        key: 0
        for key in FEATURES
    }

    for question_data, answer in zip(
        DISCOVERY_QUESTIONS,
        answers,
    ):
        mapping = question_data["options"].get(answer, {})

        for key, points in mapping.items():
            scores[key] += int(points)

    return sorted(
        scores.items(),
        key=lambda pair: pair[1],
        reverse=True,
    )


def scenario_profile_from_scores(
    ranked_scores: Sequence[Tuple[str, int]],
) -> Dict[str, int]:
    raw = dict(ranked_scores)

    if not raw:
        return {key: 3 for key in FEATURES}

    max_score = max(raw.values()) or 1

    profile: Dict[str, int] = {}

    for key in FEATURES:
        normalized = 1 + 4 * (raw.get(key, 0) / max_score)
        profile[key] = int(round(clamp(normalized, 1, 5)))

    return profile


# =============================================================================
# 13. MESSAGE ANALYSIS
# =============================================================================

FEELING_WORDS = {
    "hurt",
    "sad",
    "angry",
    "mad",
    "upset",
    "disappointed",
    "frustrated",
    "anxious",
    "worried",
    "scared",
    "lonely",
    "ignored",
    "overwhelmed",
    "confused",
    "unimportant",
    "jealous",
}

REQUEST_PHRASES = {
    "can we",
    "could we",
    "would you",
    "can you",
    "could you",
    "i need",
    "i would like",
    "i'd like",
    "what do you think",
    "would it be possible",
}

EMPATHY_PHRASES = {
    "i understand",
    "i can see",
    "i hear you",
    "that makes sense",
    "i get why",
    "i understand why",
}

BOUNDARY_PHRASES = {
    "i need space",
    "i need some time",
    "i'm not comfortable",
    "i am not comfortable",
    "my boundary",
    "i won't",
    "i will not",
    "i can't agree",
    "i cannot agree",
}

BLAME_PHRASES = {
    "you make me",
    "your fault",
    "because of you",
    "you ruin",
    "you always",
    "you never",
}

DISMISSIVE_PHRASES = {
    "whatever",
    "i don't care",
    "do whatever",
    "you're impossible",
    "you are impossible",
    "shut up",
}


def detect_emotion(text: str) -> Optional[str]:
    lower = text.lower()

    for word in FEELING_WORDS:
        if word in lower:
            return word

    return None


def message_signals(text: str) -> Dict[str, bool]:
    lower = text.lower().strip()

    return {
        "feeling": (
            "i feel" in lower
            or "i felt" in lower
            or detect_emotion(text) is not None
        ),
        "request": any(
            phrase in lower
            for phrase in REQUEST_PHRASES
        ),
        "empathy": any(
            phrase in lower
            for phrase in EMPATHY_PHRASES
        ),
        "boundary": any(
            phrase in lower
            for phrase in BOUNDARY_PHRASES
        ),
        "blame": any(
            phrase in lower
            for phrase in BLAME_PHRASES
        ),
        "dismissive": any(
            phrase in lower
            for phrase in DISMISSIVE_PHRASES
        ),
        "question": "?" in text,
        "absolute": (
            " always " in f" {lower} "
            or " never " in f" {lower} "
        ),
        "apology": (
            "i'm sorry" in lower
            or "i am sorry" in lower
            or "i apologize" in lower
        ),
    }


def analyze_message(text: str) -> Dict[str, Any]:
    signals = message_signals(text)
    words = text.split()

    clarity = 50
    respect = 80
    feeling_expression = 40
    solution_focus = 42
    boundary_expression = 42

    if len(words) >= 8:
        clarity += 10

    if len(words) >= 18:
        clarity += 6

    if signals["feeling"]:
        feeling_expression += 38
        clarity += 6

    if signals["request"]:
        solution_focus += 38
        clarity += 8

    if signals["empathy"]:
        respect += 10
        solution_focus += 8

    if signals["boundary"]:
        boundary_expression += 42

    if signals["question"]:
        solution_focus += 7

    if signals["apology"]:
        respect += 5

    if signals["absolute"]:
        respect -= 14
        clarity -= 6

    if signals["blame"]:
        respect -= 22
        clarity -= 8

    if signals["dismissive"]:
        respect -= 30
        solution_focus -= 20

    scores = {
        "Clarity": int(clamp(clarity, 0, 100)),
        "Respect": int(clamp(respect, 0, 100)),
        "Feeling Expression": int(clamp(feeling_expression, 0, 100)),
        "Solution Focus": int(clamp(solution_focus, 0, 100)),
        "Boundary Expression": int(clamp(boundary_expression, 0, 100)),
    }

    strengths: List[str] = []
    improvements: List[str] = []

    if signals["feeling"]:
        strengths.append(
            "You name an emotional experience rather than only describing the other person's behavior."
        )
    else:
        improvements.append(
            "Name the feeling you experienced, such as disappointed, hurt, worried, or frustrated."
        )

    if signals["request"]:
        strengths.append(
            "You include a clear request or next step."
        )
    else:
        improvements.append(
            "Add a concrete request, such as “Can we talk about this tonight?”"
        )

    if signals["empathy"]:
        strengths.append(
            "You acknowledge the other person's perspective."
        )
    else:
        improvements.append(
            "If appropriate, briefly acknowledge the other person's perspective."
        )

    if signals["boundary"]:
        strengths.append(
            "You communicate a personal limit or boundary."
        )

    if signals["absolute"]:
        improvements.append(
            "Replace “always” or “never” with the specific behavior or recent situation."
        )

    if signals["blame"]:
        improvements.append(
            "Shift from blame toward observation + feeling + need + request."
        )

    if signals["dismissive"]:
        improvements.append(
            "Avoid dismissive wording that can shut down the conversation."
        )

    overall = round(
        float(np.mean(list(scores.values())))
    )

    return {
        "overall": overall,
        "scores": scores,
        "strengths": strengths,
        "improvements": improvements,
        "signals": signals,
    }


# =============================================================================
# 14. AI PROMPTS
# =============================================================================

def guide_instructions() -> str:
    return f"""
You are PartnerLens Guide, a warm, practical, natural relationship-reflection assistant for adults 18+.

Your job is to have a real conversation with the user, similar to a polished general-purpose chat assistant,
while staying focused on relationship communication, self-reflection, compatibility, values, needs, and boundaries.

CONVERSATION BEHAVIOR:
- Answer the user's exact latest message first.
- Use the earlier messages in the conversation as context.
- Never act like each user message is a brand-new unrelated question.
- If the user sends a short follow-up such as "hurt", "why?", "what do I say?", "then what?", "she ignored me",
  or "I don't know", infer the intended connection to the earlier conversation when it is reasonably clear.
- Do not repeat the same generic paragraph over and over.
- Ask at most one clarification question when the meaning is genuinely unclear.
- If the user expresses a feeling, acknowledge that specific feeling naturally.
- If the user asks what to say, provide a realistic sample message they could send.
- If the user asks what something might mean, give multiple plausible interpretations rather than pretending
  to know another person's hidden thoughts.
- Keep most answers concise but substantive. Expand only when the question needs detail.
- Sound human, calm, and conversational. Do not sound like a worksheet or rubric unless the user asks for one.

BOUNDARIES AND SAFETY:
- Do not diagnose the user or another person.
- Do not casually label someone narcissistic, toxic, abusive, avoidant, bipolar, etc.
- Do not promise whether a relationship will succeed or fail.
- Do not automatically advise ending a relationship.
- If the user describes violence, threats, coercion, stalking, sexual pressure, or immediate danger,
  prioritize safety and encourage appropriate real-world support.
- You are a reflection and communication tool, not a therapist, doctor, lawyer, or emergency service.

OPTIONAL SAVED PROFILE CONTEXT:
{profile_context_for_ai()}
""".strip()


def coach_rewrite_instructions() -> str:
    return """
You are a communication editor inside PartnerLens AI.

Rewrite the user's message so it sounds natural, calm, specific, respectful, and emotionally clear.
Preserve the user's real meaning.
Do not make the user excessively apologetic.
Do not make the message robotic.
Prefer observation + feeling + need + concrete request when that structure fits naturally.
Return:
1. A polished version the user could actually send.
2. One short sentence explaining the most important improvement.
""".strip()


def simulator_instructions(
    scenario: str,
    style: str,
    difficulty: str,
) -> str:
    return f"""
You are role-playing a hypothetical adult partner inside PartnerLens AI.

SCENARIO:
{SCENARIOS[scenario]["context"]}

PARTNER STYLE:
{style}
{PARTNER_STYLES[style]}

DIFFICULTY:
{difficulty}
{DIFFICULTY_DESCRIPTIONS[difficulty]}

ROLE-PLAY RULES:
- Stay in character as the hypothetical partner.
- Respond directly to the user's latest message and remember the earlier role-play conversation.
- Do not become a relationship coach while role-playing.
- Do not give scores, analysis, or communication tips unless the user explicitly exits role-play and asks for a debrief.
- A short message such as "hurt", "sad", "why?", "really?", "I need space", or "that's not fair" is meaningful.
  Respond naturally based on the conversation instead of asking the same generic clarification question.
- If the user expresses a feeling, acknowledge the exact feeling.
- If the user asks a question, answer the question in character.
- If the user makes a request, respond to whether the partner can agree, partially agree, or needs to negotiate.
- The partner style should noticeably affect wording and priorities.
- Difficulty affects willingness to compromise:
  Gentle = highly warm and collaborative.
  Realistic = balanced and not automatically agreeing.
  Challenging = respectful but firmer and more protective of the partner's own needs.
- Usually respond in 1–5 sentences.
- Keep the conversation moving toward understanding, negotiation, or a concrete next step.
- Never insult, threaten, shame, manipulate, punish, or sexually pressure the user.
- Never claim to be the user's real partner or a real human.
""".strip()


def debrief_instructions() -> str:
    return """
You are a communication coach reviewing a short practice conversation.

Give a compact, constructive debrief:
- one thing the user did well,
- one thing that could be clearer,
- one alternative sentence the user could try,
- one question the user should consider.

Do not diagnose anyone. Do not decide whether the relationship should continue.
""".strip()


# =============================================================================
# 15. LOCAL FALLBACK CONVERSATION ENGINE
# =============================================================================

def local_guide_reply(
    user_text: str,
    history: Sequence[Dict[str, str]],
) -> str:
    lower = user_text.lower().strip()
    emotion = detect_emotion(user_text)

    previous_user = ""

    for item in reversed(history[:-1]):
        if item.get("role") == "user":
            previous_user = item.get("content", "")
            break

    if emotion:
        if previous_user:
            return (
                f"It sounds like you're feeling {emotion} about what you described. "
                "A useful next step is to name the specific moment that affected you and what you needed instead. "
                "You could say: “When that happened, I felt "
                f"{emotion}. I want to understand what happened, and I'd like us to talk about what we can do differently.”"
            )

        return (
            f"I hear that you're feeling {emotion}. What happened right before you started feeling that way?"
        )

    if lower in {"why", "why?", "how", "how?", "really", "really?"}:
        if previous_user:
            return (
                "There could be more than one explanation, and we shouldn't assume their motive without asking. "
                "Focus on what you actually observed, then ask a direct question about it."
            )

        return (
            "Tell me the situation you're referring to and I'll help you think through the most likely explanations."
        )

    if "what should i say" in lower or "what do i say" in lower or "write" in lower:
        return (
            "Try: “I want to talk about this without blaming each other. When that happened, I felt hurt and confused. "
            "Can you tell me how you saw it, and can we agree on what we should do differently next time?”"
        )

    if "boundary" in lower:
        return (
            "A strong boundary is specific and focused on what you will do. "
            "For example: “I'm willing to discuss this, but I'm not comfortable continuing if we start insulting each other. "
            "If that happens, I'm going to pause the conversation and come back later.”"
        )

    if "compatib" in lower:
        return (
            "Compatibility is less about matching on every preference and more about whether important differences can be discussed "
            "and managed without either person repeatedly giving up something essential."
        )

    return (
        "Tell me what happened, how it affected you, and what you want to happen next. "
        "I can help you make sense of it or turn it into a clear message."
    )


def local_simulator_reply(
    user_text: str,
    scenario: str,
    style: str,
    difficulty: str,
) -> str:
    lower = user_text.lower().strip()
    emotion = detect_emotion(user_text)

    if emotion:
        base = (
            f"I hear that you're feeling {emotion}. I don't want to dismiss that. "
            "Tell me what part of this affected you the most."
        )

        if style == "Independent / Needs Space":
            base = (
                f"I hear that you're {emotion}. My need for space isn't meant to make your feelings unimportant. "
                "I want to understand what affected you, even if I still need some room."
            )

        elif style == "Direct Communicator":
            base = (
                f"I hear that you're {emotion}. Tell me specifically what I did or said that caused that feeling."
            )

        elif style == "Affectionate Reassurer":
            base = (
                f"I'm sorry you're feeling {emotion}. I care about you and I want to understand what made you feel that way."
            )

        if difficulty == "Challenging":
            base += " I also need room for my side of this."

        return base

    if lower in {"why", "why?", "how", "how?"}:
        return (
            "Because I was trying to handle what I needed in that moment, but I can see that it affected you too. "
            "I don't want to assume we see it the same way."
        )

    if "sorry" in lower:
        return (
            "I appreciate you saying that. I don't need perfection; I want us to be honest about what happened and what we do next."
        )

    if "space" in lower:
        return (
            "I can respect space, but I think we should be clear about when we reconnect so it doesn't feel like disappearing."
        )

    if difficulty == "Gentle":
        return (
            "I hear what you're saying. I want us to slow this down and find something that works for both of us. "
            "What would help you feel understood right now?"
        )

    if difficulty == "Challenging":
        return (
            "I hear your point, but I don't automatically agree with your solution. "
            "I want us to be specific about what each of us needs and what we're actually willing to change."
        )

    return (
        "I hear you. I want to understand your side, and I also want room for mine. "
        "What would feel like a fair next step for both of us?"
    )


# =============================================================================
# 16. SIMPLE ML HELPERS (NUMPY ONLY)
# =============================================================================

@st.cache_data
def synthetic_profiles(
    n: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    archetypes = np.array(
        [
            [4.6, 4.4, 2.8, 3.4, 4.7, 4.5, 3.0, 3.1, 4.0, 2.9],
            [3.6, 3.7, 4.7, 4.6, 2.7, 2.8, 3.4, 4.0, 4.5, 4.1],
            [4.3, 3.8, 3.0, 3.3, 3.2, 4.5, 4.6, 3.5, 3.0, 4.7],
            [3.8, 4.5, 3.8, 3.7, 3.6, 3.8, 2.8, 3.2, 4.6, 3.3],
            [4.7, 4.6, 3.4, 4.3, 3.6, 4.0, 3.3, 3.8, 4.1, 3.8],
        ]
    )

    labels = rng.integers(
        0,
        len(archetypes),
        size=n,
    )

    matrix = (
        archetypes[labels]
        + rng.normal(
            0,
            0.52,
            size=(n, len(FEATURES)),
        )
    )

    matrix = np.clip(
        matrix,
        1,
        5,
    )

    df = pd.DataFrame(
        matrix,
        columns=list(FEATURES.keys()),
    )

    df.insert(
        0,
        "profile_id",
        np.arange(1, n + 1),
    )

    return df


def standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)

    stds = np.where(
        stds == 0,
        1,
        stds,
    )

    return (matrix - means) / stds


def numpy_kmeans(
    matrix: np.ndarray,
    k: int,
    seed: int = 42,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(matrix),
        size=k,
        replace=False,
    )

    centroids = matrix[indices].copy()
    labels = np.zeros(len(matrix), dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(
            matrix[:, None, :] - centroids[None, :, :],
            axis=2,
        )

        new_labels = distances.argmin(axis=1)

        if np.array_equal(labels, new_labels):
            labels = new_labels
            break

        labels = new_labels

        new_centroids: List[np.ndarray] = []

        for cluster in range(k):
            members = matrix[labels == cluster]

            if len(members) == 0:
                new_centroids.append(
                    matrix[
                        rng.integers(0, len(matrix))
                    ]
                )
            else:
                new_centroids.append(
                    members.mean(axis=0)
                )

        centroids = np.array(new_centroids)

    return labels, centroids


def pca_two_components(
    matrix: np.ndarray,
) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0)

    _, _, vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    components = vt[:2].T

    return centered @ components


def approximate_cluster_separation(
    matrix: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> float:
    if len(centroids) < 2:
        return 0.0

    within: List[float] = []

    for cluster_index, centroid in enumerate(centroids):
        members = matrix[labels == cluster_index]

        if len(members):
            within.append(
                float(
                    np.mean(
                        np.linalg.norm(
                            members - centroid,
                            axis=1,
                        )
                    )
                )
            )

    between: List[float] = []

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between.append(
                float(
                    np.linalg.norm(
                        centroids[i] - centroids[j]
                    )
                )
            )

    if not within or not between:
        return 0.0

    within_mean = float(np.mean(within))
    between_mean = float(np.mean(between))

    if within_mean == 0:
        return 100.0

    ratio = between_mean / within_mean

    return round(
        clamp(ratio * 25, 0, 100),
        1,
    )


# =============================================================================
# 17. SIDEBAR
# =============================================================================

st.sidebar.markdown(
    """
    <div class="pl-brand">
        <div class="pl-brand-title">💞 PartnerLens AI</div>
        <div class="pl-brand-sub">Relationship self-discovery studio</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Explore",
    PAGES,
    label_visibility="collapsed",
)

st.sidebar.divider()

if AI_AVAILABLE:
    st.sidebar.success(
        f"● Free AI Online · {GROQ_MODEL}"
    )
else:
    st.sidebar.warning(
        "○ Free AI not connected"
    )

st.sidebar.caption(
    f"AI messages this session: "
    f"{st.session_state.ai_calls}/{MAX_AI_CALLS_PER_SESSION}"
)

if st.sidebar.button(
    "🧹 Clear entire session",
    use_container_width=True,
):
    for state_key, default_value in STATE_DEFAULTS.items():
        st.session_state[state_key] = default_value

    safe_rerun()

with st.sidebar.expander("⚙️ Free AI setup"):
    st.write(
        "Create a free Groq API key, then add these in "
        "**Manage app → Settings → Secrets**:"
    )

    st.code(
        'GROQ_API_KEY = "gsk_your_key_here"\\nGROQ_MODEL = "openai/gpt-oss-20b"',
        language="toml",
    )

    st.caption(
        "Keep the key private. Do not paste it into app.py or commit it to GitHub."
    )

    if st.session_state.api_last_error:
        st.error(
            short_text(
                st.session_state.api_last_error,
                600,
            )
        )

# =============================================================================
# 18. GLOBAL HERO
# =============================================================================

st.markdown(
    f"""
    <div class="pl-hero">
        <div class="pl-kicker">AI-powered relationship self-discovery</div>
        <div class="pl-hero-title">💞 {APP_NAME}</div>
        <p class="pl-hero-copy">
            Understand your preferences, compare relationship alignment, practice
            difficult conversations, and explore patterns with explainable analytics
            and conversational AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "For adults 18+. PartnerLens supports reflection and communication practice. "
    "It does not diagnose people, read another person's mind, or predict whether a relationship will succeed."
)


# =============================================================================
# 19. HOME PAGE
# =============================================================================

def render_home() -> None:
    section_header(
        "Welcome",
        "A complete relationship reflection workspace",
        (
            "PartnerLens combines self-discovery, explainable scoring, communication "
            "practice, data visualization, and contextual AI chat in one Streamlit app."
        ),
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Preference dimensions",
            "10",
            help="Ten relationship preference dimensions are used throughout the app.",
        )

    with m2:
        st.metric(
            "AI status",
            "Online" if AI_AVAILABLE else "Setup needed",
        )

    with m3:
        st.metric(
            "Conversation memory",
            "Active",
            help="Recent messages are kept in session state and sent with each AI turn.",
        )

    with m4:
        st.metric(
            "ML demo",
            "K-Means + PCA",
        )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        feature_card(
            "🧠",
            "Partner Blueprint",
            (
                "Rate the relationship qualities that matter most to you and save a "
                "personal preference profile."
            ),
        )

        feature_card(
            "🤝",
            "Compatibility Lab",
            (
                "Compare two preference profiles dimension-by-dimension. The score "
                "shows alignment, not a prediction."
            ),
        )

        feature_card(
            "🎭",
            "Scenario Discovery",
            (
                "Use everyday choices to reveal which relationship themes tend to "
                "matter most to you."
            ),
        )

    with c2:
        feature_card(
            "🧭",
            "AI Relationship Guide",
            (
                "Have a contextual conversation. Short follow-ups like “hurt,” “why?” "
                "and “what should I say?” use the earlier chat."
            ),
        )

        feature_card(
            "💬",
            "Conversation Coach",
            (
                "Analyze difficult messages for clarity, respect, emotional expression, "
                "boundaries, and solution focus."
            ),
        )

        feature_card(
            "🗨️",
            "AI Chat Simulator",
            (
                "Practice with different hypothetical partner styles and difficulty "
                "levels using conversational AI."
            ),
        )

    with c3:
        feature_card(
            "🚦",
            "Relationship Reflection",
            (
                "Check in on communication, boundaries, fairness, trust, and emotional "
                "safety without turning the result into a diagnosis."
            ),
        )

        feature_card(
            "🤖",
            "ML Intelligence",
            (
                "Explore synthetic preference clusters using a transparent NumPy "
                "implementation of K-Means and PCA."
            ),
        )

        feature_card(
            "📊",
            "Profile Center",
            (
                "Review, visualize, and export the profile you saved during the current "
                "session."
            ),
        )

    st.divider()

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("How the AI conversation works")

        st.markdown(
            """
            PartnerLens sends the **recent conversation history** with every AI request.
            That means the assistant can understand a sequence such as:

            **You:** “My partner canceled our plans again.”

            **Guide:** responds to the situation.

            **You:** “hurt”

            **Guide:** understands that *hurt* refers to the canceled plans.

            **You:** “what should I say?”

            **Guide:** can write a message based on the conversation instead of starting over.
            """
        )

    with right:
        if AI_AVAILABLE:
            st.success(
                "Your Groq key is configured. The Guide and Simulator can use real conversational AI."
            )
        else:
            st.warning(
                "The app is working, but the AI pages are in fallback mode until you add GROQ_API_KEY."
            )

        info_card(
            "Recommended model",
            (
                "The default is <b>openai/gpt-oss-20b</b> because it is fast and cost-efficient "
                "for conversational workloads. You can change GROQ_MODEL in Streamlit Secrets."
            ),
            "⚡",
        )

    st.divider()

    st.subheader("Quick start")

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        st.markdown(
            f"{pill('1')} Build your Blueprint",
            unsafe_allow_html=True,
        )

    with q2:
        st.markdown(
            f"{pill('2')} Compare preferences",
            unsafe_allow_html=True,
        )

    with q3:
        st.markdown(
            f"{pill('3')} Practice a message",
            unsafe_allow_html=True,
        )

    with q4:
        st.markdown(
            f"{pill('4')} Chat with the Guide",
            unsafe_allow_html=True,
        )


# =============================================================================
# 20. PARTNER BLUEPRINT PAGE
# =============================================================================

def render_blueprint() -> None:
    section_header(
        "Self-discovery",
        "🧠 Build your Partner Blueprint",
        (
            "Rate each dimension from 1 to 5. These are preferences, not a test of "
            "whether your relationship is good or bad."
        ),
    )

    name = st.text_input(
        "Profile name",
        value=st.session_state.saved_profile_name,
        placeholder="My Profile",
    )

    with st.expander(
        "What do these dimensions mean?",
        expanded=False,
    ):
        for key in FEATURES:
            st.markdown(
                f"**{FEATURE_ICONS[key]} {FEATURES[key]}** — "
                f"{FEATURE_DESCRIPTIONS[key]}"
            )

    profile = collect_profile(
        "blueprint",
        st.session_state.saved_profile,
    )

    save_col, reset_col = st.columns([1, 1])

    with save_col:
        if st.button(
            "💾 Save Blueprint",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.saved_profile = profile.copy()
            st.session_state.saved_profile_name = (
                name.strip()
                or "My Profile"
            )

            st.success(
                "Blueprint saved for this session."
            )

    with reset_col:
        if st.button(
            "↩️ Reset saved Blueprint",
            use_container_width=True,
        ):
            st.session_state.saved_profile = None
            st.session_state.saved_profile_name = "My Profile"
            safe_rerun()

    st.divider()

    if st.session_state.saved_profile:
        saved = st.session_state.saved_profile
        style_name, style_text = relationship_style(saved)

        st.subheader(
            f"Your current pattern: {style_name}"
        )
        st.write(style_text)

        left, right = st.columns(
            [1.12, 1]
        )

        with left:
            table = profile_dataframe(saved)
            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
            )

            top_three = ranked_profile(saved)[:3]

            st.markdown(
                "**Highest stated priorities:** "
                + " ".join(
                    pill(
                        f"{FEATURE_ICONS[key]} {FEATURES[key]} {score}/5"
                    )
                    for key, score in top_three
                ),
                unsafe_allow_html=True,
            )

        with right:
            fig = radar_chart(
                saved,
                "Your Relationship Preference Radar",
            )
            st.pyplot(fig)
            plt.close(fig)

        csv_data = table.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Blueprint CSV",
            data=csv_data,
            file_name="partnerlens_blueprint.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "Move the sliders and click **Save Blueprint** to generate your saved profile."
        )


# =============================================================================
# 21. SCENARIO DISCOVERY PAGE
# =============================================================================

def render_scenario_discovery() -> None:
    section_header(
        "Behavioral reflection",
        "🎭 Scenario Discovery",
        (
            "Choose the response that feels closest to you. The purpose is to surface "
            "themes, not to label your personality."
        ),
    )

    answers: List[str] = []

    for index, item in enumerate(
        DISCOVERY_QUESTIONS,
        start=1,
    ):
        answer = st.radio(
            f"{index}. {item['question']}",
            list(item["options"].keys()),
            key=f"discovery_{index}",
        )

        answers.append(answer)

    if st.button(
        "✨ Analyze my scenario pattern",
        type="primary",
    ):
        ranked = analyze_discovery_answers(answers)
        profile = scenario_profile_from_scores(ranked)

        st.session_state.scenario_profile = profile
        st.session_state.scenario_answers = answers

    if st.session_state.scenario_profile:
        profile = st.session_state.scenario_profile

        st.divider()
        st.subheader("Your scenario-based pattern")

        top = ranked_profile(profile)[:5]

        cols = st.columns(5)

        for col, (key, score) in zip(cols, top):
            with col:
                st.metric(
                    f"{FEATURE_ICONS[key]} {FEATURES[key]}",
                    f"{score}/5",
                )

        left, right = st.columns([1, 1])

        with left:
            st.write(
                "Your answers most strongly pointed toward:"
            )

            for key, score in top:
                st.markdown(
                    f"- **{FEATURES[key]}:** {FEATURE_DESCRIPTIONS[key]}"
                )

        with right:
            fig = radar_chart(
                profile,
                "Scenario Discovery Radar",
            )
            st.pyplot(fig)
            plt.close(fig)

        if st.button(
            "💾 Save this as my Blueprint",
        ):
            st.session_state.saved_profile = profile.copy()
            st.session_state.saved_profile_name = "Scenario Profile"
            st.success(
                "Scenario profile saved as your Blueprint."
            )


# =============================================================================
# 22. COMPATIBILITY LAB PAGE
# =============================================================================

def render_compatibility() -> None:
    section_header(
        "Preference alignment",
        "🤝 Compatibility Lab",
        (
            "Compare two sets of stated preferences. A high score means the selected "
            "preferences are similar; it does not predict relationship success."
        ),
    )

    person_a_col, person_b_col = st.columns(2)

    with person_a_col:
        st.subheader("Person A")

        use_saved = False

        if st.session_state.saved_profile:
            use_saved = st.checkbox(
                "Use my saved Blueprint",
                value=True,
                key="compat_use_saved",
            )

        if use_saved:
            person_a = dict(
                st.session_state.saved_profile
            )

            st.success(
                f"Using saved profile: {st.session_state.saved_profile_name}"
            )

            for key, value in person_a.items():
                st.caption(
                    f"{FEATURE_ICONS[key]} {FEATURES[key]}: {value}/5"
                )
        else:
            person_a = collect_profile(
                "compat_a",
            )

    with person_b_col:
        st.subheader("Person B")
        person_b = collect_profile(
            "compat_b",
        )

    if st.button(
        "🤝 Compare profiles",
        type="primary",
        use_container_width=True,
    ):
        score = compatibility_score(
            person_a,
            person_b,
        )

        details = compatibility_dataframe(
            person_a,
            person_b,
        )

        st.session_state.compatibility_last = {
            "score": score,
            "details": details.to_dict(
                orient="records"
            ),
            "person_a": person_a,
            "person_b": person_b,
        }

    if st.session_state.compatibility_last:
        result = st.session_state.compatibility_last

        st.divider()

        score = float(
            result["score"]
        )

        r1, r2 = st.columns(
            [0.8, 1.2]
        )

        with r1:
            fig = donut_chart(score)
            st.pyplot(fig)
            plt.close(fig)

        with r2:
            st.subheader(
                "What the score means"
            )

            if score >= 85:
                st.success(
                    "The two selected preference profiles are highly similar overall."
                )
            elif score >= 70:
                st.info(
                    "The profiles have substantial alignment with some meaningful differences."
                )
            elif score >= 55:
                st.warning(
                    "The profiles are mixed. Several differences may deserve discussion."
                )
            else:
                st.warning(
                    "The selected preferences differ considerably. The useful question is whether those differences can be managed respectfully."
                )

            st.write(
                "Compatibility is not only similarity. Different preferences can work when "
                "people communicate clearly, negotiate fairly, and do not repeatedly sacrifice essential needs."
            )

        details_df = pd.DataFrame(
            result["details"]
        )

        st.dataframe(
            details_df,
            use_container_width=True,
            hide_index=True,
        )

        fig = comparison_radar(
            result["person_a"],
            result["person_b"],
        )
        st.pyplot(fig)
        plt.close(fig)

        st.download_button(
            "⬇️ Download comparison CSV",
            data=details_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="partnerlens_compatibility.csv",
            mime="text/csv",
        )


# =============================================================================
# 23. AI RELATIONSHIP GUIDE PAGE
# =============================================================================

def render_ai_guide() -> None:
    section_header(
        "Contextual conversational AI",
        "🧭 AI Relationship Guide",
        (
            "Talk naturally. The Guide keeps recent messages in context, so your short "
            "follow-ups can refer to what you already said."
        ),
    )

    ai_status_message()

    st.markdown(
        """
        <div class="pl-chat-note">
            Try a real conversation: “My partner canceled our plans again.” → “hurt” →
            “why would they do that?” → “what should I say?”
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls_left, controls_right = st.columns(
        [1, 3]
    )

    with controls_left:
        if st.button(
            "🧹 New Guide chat",
            use_container_width=True,
        ):
            st.session_state.guide_messages = []
            safe_rerun()

    with controls_right:
        st.caption(
            "Recent conversation history is sent with each AI request so the Guide can respond in context."
        )

    if not st.session_state.guide_messages:
        with st.chat_message(
            "assistant",
            avatar="🧭",
        ):
            st.markdown(
                "Hi — tell me what is happening, how you're feeling, or what conversation you're trying to handle."
            )

    for message in st.session_state.guide_messages:
        avatar = (
            "🧭"
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

    prompt = st.chat_input(
        "Message PartnerLens Guide…",
        key="guide_chat_input",
    )

    if prompt:
        prompt = prompt.strip()

        if not prompt:
            return

        if len(prompt) > MAX_MESSAGE_CHARACTERS:
            prompt = prompt[:MAX_MESSAGE_CHARACTERS]

        st.session_state.guide_messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message(
            "user",
            avatar="👤",
        ):
            st.markdown(prompt)

        try:
            if AI_AVAILABLE:
                with st.spinner(
                    "Thinking…"
                ):
                    reply = call_ai(
                        st.session_state.guide_messages,
                        guide_instructions(),
                        max_output_tokens=900,
                    )
            else:
                reply = local_guide_reply(
                    prompt,
                    st.session_state.guide_messages,
                )
        except Exception as exc:
            st.session_state.api_last_error = f"{type(exc).__name__}: {exc}"
            reply = local_guide_reply(
                prompt,
                st.session_state.guide_messages,
            )
            reply += (
                "\n\n*Free Groq AI was temporarily unavailable, so PartnerLens "
                "used its built-in fallback for this reply.*"
            )

        st.session_state.guide_messages.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )

        with st.chat_message(
            "assistant",
            avatar="🧭",
        ):
            st.markdown(reply)

        safe_rerun()


# =============================================================================
# 24. CONVERSATION COACH PAGE
# =============================================================================

def render_conversation_coach() -> None:
    section_header(
        "Explainable communication analysis",
        "💬 Conversation Coach",
        (
            "Draft a difficult message and get transparent feedback on clarity, respect, "
            "emotional expression, boundaries, and solution focus."
        ),
    )

    scenario = st.selectbox(
        "Situation",
        list(SCENARIOS.keys()),
    )

    info_card(
        "Scenario",
        SCENARIOS[scenario]["context"],
        SCENARIOS[scenario]["icon"],
    )

    draft = st.text_area(
        "Write the message you want to send",
        height=190,
        placeholder=(
            "Example: I felt disappointed when our plans changed at the last minute. "
            "Can we talk about how to handle this next time?"
        ),
    )

    analyze_col, sample_col = st.columns(2)

    with analyze_col:
        analyze_clicked = st.button(
            "🔍 Analyze message",
            type="primary",
            use_container_width=True,
        )

    with sample_col:
        if st.button(
            "📝 Load an example",
            use_container_width=True,
        ):
            st.info(
                "Example: “I know work got complicated. I was disappointed because I was looking forward to our time together. Can we choose another time and protect it?”"
            )

    if analyze_clicked:
        if not draft.strip():
            st.warning(
                "Write a message first."
            )
        else:
            st.session_state.coach_last = analyze_message(
                draft
            )

    if st.session_state.coach_last:
        analysis = st.session_state.coach_last

        st.divider()

        st.metric(
            "Overall communication score",
            f"{analysis['overall']}%",
        )

        metric_cols = st.columns(
            len(analysis["scores"])
        )

        for col, (name, value) in zip(
            metric_cols,
            analysis["scores"].items(),
        ):
            with col:
                st.metric(
                    name,
                    f"{value}%",
                )

        strength_col, improvement_col = st.columns(2)

        with strength_col:
            st.subheader("What works")

            if analysis["strengths"]:
                for item in analysis["strengths"]:
                    st.success(item)
            else:
                st.info(
                    "Your message has a starting point. Add more detail to make the intent clearer."
                )

        with improvement_col:
            st.subheader("What to improve")

            if analysis["improvements"]:
                for item in analysis["improvements"]:
                    st.warning(item)
            else:
                st.success(
                    "The message already contains strong communication signals."
                )

        st.divider()

        st.subheader("AI rewrite")

        if AI_AVAILABLE:
            if st.button(
                "✨ Rewrite this message with AI",
                type="primary",
            ):
                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"Scenario: {SCENARIOS[scenario]['context']}\n\n"
                            f"Message to improve:\n{draft}"
                        ),
                    }
                ]

                try:
                    with st.spinner(
                        "Rewriting…"
                    ):
                        rewrite = call_ai(
                            messages,
                            coach_rewrite_instructions(),
                            max_output_tokens=650,
                        )

                    st.success(
                        "Suggested rewrite"
                    )
                    st.markdown(rewrite)

                except Exception as exc:
                    st.error(
                        f"AI rewrite failed: {type(exc).__name__}: {exc}"
                    )
        else:
            st.info(
                "Add your Groq API key to enable a contextual AI rewrite."
            )


# =============================================================================
# 25. AI CHAT SIMULATOR PAGE
# =============================================================================

def render_ai_simulator() -> None:
    section_header(
        "Interactive communication practice",
        "🗨️ AI Chat Simulator",
        (
            "Practice a conversation with a hypothetical partner. With AI enabled, each "
            "reply uses the earlier role-play conversation as context."
        ),
    )

    settings_1, settings_2, settings_3 = st.columns(3)

    with settings_1:
        scenario = st.selectbox(
            "Scenario",
            list(SCENARIOS.keys()),
            key="sim_scenario",
        )

    with settings_2:
        style = st.selectbox(
            "Partner style",
            list(PARTNER_STYLES.keys()),
            key="sim_style",
        )

    with settings_3:
        difficulty = st.selectbox(
            "Difficulty",
            list(DIFFICULTY_DESCRIPTIONS.keys()),
            index=1,
            key="sim_difficulty",
        )

    signature = (
        scenario,
        style,
        difficulty,
    )

    if st.session_state.sim_signature != signature:
        st.session_state.sim_signature = signature
        st.session_state.sim_messages = [
            {
                "role": "assistant",
                "content": SCENARIOS[scenario]["starter"],
            }
        ]

    ai_status_message()

    info_card(
        style,
        (
            f"{PARTNER_STYLES[style]}<br><br>"
            f"<b>Difficulty:</b> {difficulty} — {DIFFICULTY_DESCRIPTIONS[difficulty]}"
        ),
        "🎭",
    )

    reset_col, debrief_col = st.columns(
        [1, 1]
    )

    with reset_col:
        if st.button(
            "🔄 Reset role-play",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.sim_messages = [
                {
                    "role": "assistant",
                    "content": SCENARIOS[scenario]["starter"],
                }
            ]

            safe_rerun()

    with debrief_col:
        debrief_clicked = st.button(
            "📋 Debrief my conversation",
            use_container_width=True,
        )

    if debrief_clicked:
        user_messages = [
            item
            for item in st.session_state.sim_messages
            if item["role"] == "user"
        ]

        if not user_messages:
            st.info(
                "Send at least one message in the role-play before requesting a debrief."
            )

        elif AI_AVAILABLE:
            try:
                with st.spinner(
                    "Preparing debrief…"
                ):
                    debrief = call_ai(
                        st.session_state.sim_messages,
                        debrief_instructions(),
                        max_output_tokens=700,
                    )

                st.info(debrief)

            except Exception as exc:
                st.error(
                    f"Debrief failed: {type(exc).__name__}: {exc}"
                )

        else:
            st.info(
                "Local debrief: try naming your feeling, describing the specific behavior, "
                "and making one clear request."
            )

    st.divider()

    for message in st.session_state.sim_messages:
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

    prompt = st.chat_input(
        "Reply as yourself…",
        key="sim_chat_input",
    )

    if prompt:
        prompt = prompt.strip()

        if not prompt:
            return

        if len(prompt) > MAX_MESSAGE_CHARACTERS:
            prompt = prompt[:MAX_MESSAGE_CHARACTERS]

        st.session_state.sim_messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message(
            "user",
            avatar="👤",
        ):
            st.markdown(prompt)

        try:
            if AI_AVAILABLE:
                with st.spinner(
                    "Partner is replying…"
                ):
                    reply = call_ai(
                        st.session_state.sim_messages,
                        simulator_instructions(
                            scenario,
                            style,
                            difficulty,
                        ),
                        max_output_tokens=700,
                    )
            else:
                reply = local_simulator_reply(
                    prompt,
                    scenario,
                    style,
                    difficulty,
                )

        except Exception as exc:
            st.session_state.api_last_error = f"{type(exc).__name__}: {exc}"
            reply = local_simulator_reply(
                prompt,
                scenario,
                style,
                difficulty,
            )
            reply += (
                "\n\n*Free Groq AI was temporarily unavailable, so the simulator "
                "used its built-in fallback for this turn.*"
            )

        st.session_state.sim_messages.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )

        with st.chat_message(
            "assistant",
            avatar="💗",
        ):
            st.markdown(reply)

        safe_rerun()


# =============================================================================
# 26. RELATIONSHIP REFLECTION PAGE
# =============================================================================

REFLECTION_ITEMS: Dict[str, Tuple[str, str]] = {
    "heard": (
        "I generally feel heard when I raise a concern.",
        "communication",
    ),
    "respectful_conflict": (
        "We can disagree without insults, intimidation, or threats.",
        "emotional_safety",
    ),
    "boundaries": (
        "My reasonable boundaries are generally respected.",
        "boundaries",
    ),
    "authenticity": (
        "I can be myself without constantly walking on eggshells.",
        "emotional_safety",
    ),
    "fair_decisions": (
        "Important decisions are discussed in a way that gives both people a voice.",
        "fairness",
    ),
    "repair": (
        "After conflict, we can usually return to the issue and repair the conversation.",
        "repair",
    ),
    "trust": (
        "I generally feel I can trust what my partner tells me.",
        "trust",
    ),
    "support": (
        "We support each other's important goals without one person's goals always dominating.",
        "mutual_support",
    ),
}


def render_reflection() -> None:
    section_header(
        "Structured check-in",
        "🚦 Relationship Reflection",
        (
            "Rate how true each statement feels right now. The result is a reflection aid, "
            "not a diagnosis or verdict about your relationship."
        ),
    )

    values: Dict[str, int] = {}

    for key, (question, _) in REFLECTION_ITEMS.items():
        values[key] = st.slider(
            question,
            min_value=1,
            max_value=5,
            value=3,
            key=f"reflect_{key}",
            help="1 = rarely true · 5 = consistently true",
        )

    if st.button(
        "🚦 Review my reflection",
        type="primary",
    ):
        average = float(
            np.mean(
                list(values.values())
            )
        )

        st.session_state.reflection_last = {
            "values": values.copy(),
            "average": average,
        }

    if st.session_state.reflection_last:
        result = st.session_state.reflection_last
        average = float(result["average"])

        st.divider()

        st.metric(
            "Reflection average",
            f"{average:.1f}/5",
        )

        if average >= 4.2:
            st.success(
                "Your answers suggest several strong relationship-process indicators right now."
            )
        elif average >= 3.2:
            st.info(
                "Your answers are mixed. The most useful next step is to look at which specific areas feel weakest."
            )
        else:
            st.warning(
                "Several areas feel difficult. Focus on concrete patterns, boundaries, fairness, and whether conversations feel safe and respectful."
            )

        values = result["values"]

        weakest = sorted(
            values.items(),
            key=lambda pair: pair[1],
        )[:3]

        strongest = sorted(
            values.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )[:3]

        left, right = st.columns(2)

        with left:
            st.subheader("Current strengths")

            for key, value in strongest:
                label = REFLECTION_ITEMS[key][0]
                st.success(
                    f"{value}/5 — {label}"
                )

        with right:
            st.subheader("Areas to reflect on")

            for key, value in weakest:
                label = REFLECTION_ITEMS[key][0]
                st.warning(
                    f"{value}/5 — {label}"
                )

        st.caption(
            "If you are dealing with threats, violence, coercion, stalking, or immediate danger, "
            "a scoring tool is not the right resource; prioritize real-world safety and support."
        )


# =============================================================================
# 27. ML INTELLIGENCE PAGE
# =============================================================================

def render_ml_intelligence() -> None:
    section_header(
        "Explainable machine learning",
        "🤖 ML Intelligence",
        (
            "Explore synthetic relationship-preference data using a transparent NumPy "
            "implementation of K-Means clustering and two-component PCA."
        ),
    )

    st.warning(
        "This page uses synthetic demonstration data. The clusters are not psychological types or diagnoses."
    )

    controls_1, controls_2 = st.columns(2)

    with controls_1:
        cluster_count = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=7,
            value=5,
        )

    with controls_2:
        sample_count = st.slider(
            "Synthetic sample size",
            min_value=300,
            max_value=2000,
            value=1200,
            step=100,
        )

    df = synthetic_profiles(
        sample_count,
        42,
    )

    feature_matrix = df[
        list(FEATURES.keys())
    ].to_numpy()

    standardized = standardize_matrix(
        feature_matrix
    )

    labels, centroids = numpy_kmeans(
        standardized,
        cluster_count,
        seed=42,
    )

    coords = pca_two_components(
        standardized
    )

    separation = approximate_cluster_separation(
        standardized,
        labels,
        centroids,
    )

    metric_1, metric_2, metric_3 = st.columns(3)

    with metric_1:
        st.metric(
            "Profiles",
            f"{len(df):,}",
        )

    with metric_2:
        st.metric(
            "Clusters",
            cluster_count,
        )

    with metric_3:
        st.metric(
            "Separation indicator",
            f"{separation:.1f}/100",
            help=(
                "A simple descriptive ratio of between-cluster centroid distance "
                "to average within-cluster distance. It is not a clinical metric."
            ),
        )

    plot_df = pd.DataFrame(
        {
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Cluster": labels,
        }
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.8)
    )

    for cluster in sorted(
        plot_df["Cluster"].unique()
    ):
        subset = plot_df[
            plot_df["Cluster"] == cluster
        ]

        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=18,
            alpha=.62,
            label=f"Cluster {cluster + 1}",
        )

    ax.set_title(
        "Synthetic preference clusters — PCA projection"
    )
    ax.set_xlabel(
        "Principal Component 1"
    )
    ax.set_ylabel(
        "Principal Component 2"
    )
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader(
        "Cluster preference averages"
    )

    clustered_df = df.copy()
    clustered_df["cluster"] = labels

    averages = (
        clustered_df
        .groupby("cluster")[
            list(FEATURES.keys())
        ]
        .mean()
        .round(2)
    )

    averages.index = [
        f"Cluster {int(index) + 1}"
        for index in averages.index
    ]

    averages = averages.rename(
        columns=FEATURES
    )

    st.dataframe(
        averages,
        use_container_width=True,
    )

    with st.expander(
        "How this ML page works"
    ):
        st.markdown(
            """
            1. PartnerLens generates **synthetic** 1–5 preference values around several
               designed archetype centers.
            2. The features are standardized.
            3. K-Means assigns profiles to a chosen number of clusters.
            4. PCA compresses the 10-dimensional data into two dimensions for plotting.
            5. Cluster averages are shown so the grouping remains interpretable.

            This is a portfolio demonstration of an ML workflow, not a model trained to
            make claims about real people.
            """
        )


# =============================================================================
# 28. PROFILE CENTER PAGE
# =============================================================================

def render_profile_center() -> None:
    section_header(
        "Saved session data",
        "📊 My Profile Center",
        (
            "Review your saved Blueprint, inspect your strongest preferences, and export "
            "the data during the current browser session."
        ),
    )

    if not st.session_state.saved_profile:
        st.info(
            "No Blueprint is saved yet. Go to **Partner Blueprint**, choose your ratings, and click Save Blueprint."
        )
        return

    profile = st.session_state.saved_profile
    name = st.session_state.saved_profile_name

    st.subheader(name)

    style_name, style_text = relationship_style(
        profile
    )

    st.success(
        f"Current pattern: **{style_name}** — {style_text}"
    )

    top = ranked_profile(profile)

    top_cols = st.columns(5)

    for col, (key, score) in zip(
        top_cols,
        top[:5],
    ):
        with col:
            st.metric(
                f"{FEATURE_ICONS[key]} {FEATURES[key]}",
                f"{score}/5",
            )

    st.divider()

    left, right = st.columns(
        [1.08, 1]
    )

    with left:
        df = profile_dataframe(
            profile
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

    with right:
        fig = radar_chart(
            profile,
            "Saved Blueprint",
        )
        st.pyplot(fig)
        plt.close(fig)

    export_payload = {
        "profile_name": name,
        "saved_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "profile": profile,
    }

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "⬇️ Download CSV",
            data=profile_dataframe(
                profile
            ).to_csv(
                index=False
            ).encode("utf-8"),
            file_name="partnerlens_profile.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(
                export_payload,
                indent=2,
            ).encode("utf-8"),
            file_name="partnerlens_profile.json",
            mime="application/json",
            use_container_width=True,
        )


# =============================================================================
# 29. PAGE ROUTER
# =============================================================================

if page == PAGE_HOME:
    render_home()

elif page == PAGE_BLUEPRINT:
    render_blueprint()

elif page == PAGE_SCENARIOS:
    render_scenario_discovery()

elif page == PAGE_COMPATIBILITY:
    render_compatibility()

elif page == PAGE_GUIDE:
    render_ai_guide()

elif page == PAGE_COACH:
    render_conversation_coach()

elif page == PAGE_SIMULATOR:
    render_ai_simulator()

elif page == PAGE_REFLECTION:
    render_reflection()

elif page == PAGE_ML:
    render_ml_intelligence()

elif page == PAGE_PROFILE:
    render_profile_center()

else:
    st.error(
        "Unknown page."
    )


# =============================================================================
# 30. FOOTER
# =============================================================================

st.divider()

st.caption(
    f"💞 {APP_NAME} v{APP_VERSION} · Python · Streamlit · Pandas · NumPy · "
    f"Matplotlib · Groq Free AI · Responsible reflection"
)



