# PartnerLens AI

PartnerLens AI is an AI-powered relationship self-discovery and communication-practice web application built with Python and Streamlit.

The project is designed for adults (18+) to explore relationship preferences, compare compatibility, practice difficult conversations, and receive contextual AI-assisted guidance.

## Features

- Partner preference questionnaire
- Personal relationship blueprint
- Scenario-based self-discovery
- Two-person compatibility comparison
- Explainable compatibility scoring
- AI Relationship Guide
- Conversation Coach
- AI Chat Simulator
- Relationship reflection tools
- Synthetic relationship-profile dataset
- K-Means clustering
- PCA visualization
- Profile export to CSV and JSON
- Responsive Streamlit web interface
- Session-based conversation memory
- Local fallback responses if AI service is unavailable

## AI Features

PartnerLens AI includes conversational AI features using Groq's free API tier.

The AI Relationship Guide:
- Remembers recent conversation context
- Responds to follow-up questions
- Helps users reflect on feelings, needs, communication, and boundaries
- Can suggest example messages for difficult conversations

The AI Chat Simulator:
- Simulates different partner communication styles
- Supports multiple relationship scenarios
- Allows realistic back-and-forth communication practice
- Includes Gentle, Realistic, and Challenging difficulty levels

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Groq API
- OpenAI-compatible Python SDK
- HTML/CSS
- GitHub
- Streamlit Community Cloud

## Machine Learning

PartnerLens AI also demonstrates basic machine learning concepts using synthetic relationship-preference data.

Current ML features include:

- Synthetic profile generation
- Data standardization
- K-Means clustering
- PCA dimensionality reduction
- Cluster visualization
- Explainable cluster summaries

The ML section is intended for educational and portfolio purposes and does not diagnose or classify real people.

## Privacy and Responsible AI

PartnerLens AI is designed as a reflection and communication-practice tool.

- It does not diagnose mental-health conditions
- It does not predict whether a relationship will succeed or fail
- It does not claim to know another person's thoughts
- User profile data is stored only in the active Streamlit session
- API keys are stored securely using Streamlit Secrets
- No API keys should be committed to GitHub

## Run Locally

Create a virtual environment:

```bash
python -m venv .venv
