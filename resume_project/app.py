"""Resume screening: rank PDF resumes against a job description.

Run with:  streamlit run app.py
Needs:     streamlit, pypdf, scikit-learn, pandas  (see requirements.txt)
"""
from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from html import escape

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SKILL_WEIGHT = 0.7  # share of the final score that comes from skill match
TEXT_WEIGHT = 1 - SKILL_WEIGHT
MAX_CHIPS = 15

# Resource limits. Every upload is untrusted input on a shared server.
MAX_FILES = 25              # resumes per run
MAX_FILE_MB = 5             # per file; also set server.maxUploadSize (.streamlit/config.toml)
MAX_TOTAL_MB = 40           # all files in one run
MAX_PAGES = 30              # per resume
MAX_TEXT_CHARS = 100_000    # extracted text kept per resume
PARSE_TIMEOUT_S = 10        # per resume, checked between pages
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
MAX_TOTAL_BYTES = MAX_TOTAL_MB * 1024 * 1024

SKILLS = {
    # Programming languages
    "python", "java", "cpp", "csharp", "javascript",
    "typescript", "php", "ruby", "swift", "kotlin",
    "golang", "rust", "matlab",
    # Web development
    "html", "css", "react", "angular", "vue",
    "nodejs", "express", "django", "flask",
    "streamlit", "bootstrap", "tailwind",
    # Databases
    "sql", "mysql", "postgresql", "mongodb",
    "sqlite", "oracle", "firebase",
    # Data science / AI
    "machinelearning", "deeplearning", "tensorflow",
    "keras", "pytorch", "numpy", "pandas",
    "matplotlib", "seaborn", "opencv",
    "nlp", "scikit", "sklearn",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker",
    "kubernetes", "jenkins", "linux",
    "git", "github", "gitlab",
    # BI / analytics
    "powerbi", "tableau", "excel",
    "analytics", "visualization",
    # Mobile
    "android", "ios", "flutter", "reactnative",
    # Cybersecurity
    "cybersecurity", "penetration", "testing", "networking",
    # Software engineering
    "oop", "api", "restapi", "microservices",
    # General tech
    "ai", "automation", "cloud", "devops",
}

# Spellings that would otherwise be destroyed by punctuation stripping
# ("c++" -> "c", "node.js" -> "node js") or that name the same skill.
ALIASES = [
    (r"c\+\+", " cpp "),
    (r"c#", " csharp "),
    (r"\bnode[\s.-]*js\b", " nodejs "),
    (r"\breact[\s.-]*js\b", " react "),
    (r"\bvue[\s.-]*js\b", " vue "),
    (r"\bexpress[\s.-]*js\b", " express "),
    (r"react[\s-]*native", " reactnative "),
    (r"power[\s-]*bi", " powerbi "),
    (r"scikit[\s-]*learn", " sklearn "),
    (r"postgres(?!ql)", " postgresql "),
    (r"\bgo[\s-]*lang\b", " golang "),
    (r"\brest(?:ful)?[\s-]*apis?\b", " restapi api "),
    (r"machine[\s-]*learning|\bml\b", " machinelearning "),
    (r"deep[\s-]*learning", " deeplearning "),
]

# Readable names for skills stored as a single token
DISPLAY = {
    "machinelearning": "machine learning", "deeplearning": "deep learning",
    "cpp": "c++", "csharp": "c#", "nodejs": "node.js", "powerbi": "power bi",
    "reactnative": "react native", "sklearn": "scikit-learn",
    "golang": "go", "restapi": "rest api",
}

# (minimum score, label, colour), highest first
VERDICTS = [
    (80, "Almost perfect match", "#22c55e"),
    (60, "Excellent match", "#2dd4bf"),
    (30, "Good match", "#60a5fa"),
    (10, "Average match", "#fbbf24"),
    (0, "Low match", "#f87171"),
]

# Deeper variants of the verdict colours for the light theme
LIGHT_TONE = {"#22c55e": "#15803d", "#2dd4bf": "#0f766e", "#60a5fa": "#1d4ed8", "#fbbf24": "#b45309", "#f87171": "#b91c1c"}

# Verdict colour -> dot used where only text fits (expander labels)
DOTS = {"#22c55e": "🟢", "#2dd4bf": "🟢", "#60a5fa": "🔵", "#fbbf24": "🟠", "#f87171": "🔴"}

TOP_CARDS = 3          # candidates shown as full cards; the rest are collapsed
SHORTLIST_SCORE = 60   # "Excellent match" and above

SORTS = {
    "Score": lambda c: c.score,
    "Skill match": lambda c: c.skill_score,
    "Experience": lambda c: c.experience,
}

# st.dataframe / st.button switched from use_container_width to width="stretch"
_VERSION = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
STRETCH = {"width": "stretch"} if _VERSION >= (1, 50) else {"use_container_width": True}


# ----------------------------------------------------------------------------
# Core logic (no Streamlit UI calls below this line until render functions)
# ----------------------------------------------------------------------------
@dataclass
class Candidate:
    name: str
    score: float
    skill_score: float
    text_score: float
    experience: int
    matched: list[str]
    missing: list[str]


def verdict_for(score: float) -> tuple[str, str]:
    for threshold, label, colour in VERDICTS:
        if score >= threshold:
            return label, colour
    return VERDICTS[-1][1], VERDICTS[-1][2]


class UploadRejected(ValueError):
    """An upload failed validation. The message is safe to show to the user."""


def check_upload(file) -> bytes:
    """Validate an uploaded file before any parser sees it. Returns its bytes."""
    if file.size > MAX_FILE_BYTES:
        raise UploadRejected(f"is larger than {MAX_FILE_MB} MB.")
    data = file.getvalue()
    # The extension is user-controlled; check the actual content.
    if not data.startswith(b"%PDF-"):
        raise UploadRejected("isn't a PDF file.")
    return data


# Bounded cache: unbounded caching of upload bytes is a memory leak.
@st.cache_data(show_spinner=False, max_entries=32, ttl=3600)
def read_pdf(data: bytes) -> str:
    """Extract text from PDF bytes, within the page, text and time limits."""
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise UploadRejected("is password-protected.")
    if len(reader.pages) > MAX_PAGES:
        raise UploadRejected(f"has more than {MAX_PAGES} pages.")

    deadline = time.monotonic() + PARSE_TIMEOUT_S
    parts: list[str] = []
    total = 0
    for page in reader.pages:
        if time.monotonic() > deadline:
            raise UploadRejected("took too long to read.")
        chunk = page.extract_text() or ""
        parts.append(chunk)
        total += len(chunk)
        if total >= MAX_TEXT_CHARS:
            break
    return " ".join(parts)[:MAX_TEXT_CHARS]


_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")


def md_escape(text: str) -> str:
    """Escape Markdown so a filename can't render as a link or formatting in st.warning."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$&])", r"\\\1", text)


def csv_safe(value):
    """Neutralise spreadsheet formulas: a leading ' makes Excel/Sheets treat the cell as text."""
    if isinstance(value, str) and value.lstrip(" ").startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def clean_text(text: str) -> str:
    text = text.lower()
    for pattern, replacement in ALIASES:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_experience(text: str) -> int:
    """Largest 'N years' / 'N yrs' figure mentioned, or 0."""
    matches = re.findall(r"(\d+)\s*(?:years?|yrs?)\b", text)
    return max(map(int, matches)) if matches else 0


def has_usable_words(cleaned_text: str) -> bool:
    """True if the text has at least one word the vectorizer won't discard."""
    return any(len(w) >= 2 and w not in ENGLISH_STOP_WORDS for w in cleaned_text.split())


def score_candidates(job_clean: str, resumes: dict[str, str]) -> tuple[list[Candidate], bool]:
    """Rank resumes. Returns (candidates sorted best-first, skills_detected).

    Both the job text and the resume texts must already be passed through clean_text().
    """
    job_words = set(job_clean.split())
    required = sorted(SKILLS & job_words)
    skills_detected = bool(required)

    names = list(resumes)
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(
            [resumes[n] for n in names] + [job_clean]
        )
        similarity = cosine_similarity(matrix[-1], matrix[:-1])[0] * 100
    except ValueError:
        # "empty vocabulary": every document is only stop words or one-letter tokens
        similarity = [0.0] * len(names)

    # If the job mentions no recognised skills, fall back to text similarity only
    # rather than awarding everyone a free 100% skill score.
    skill_w, text_w = (SKILL_WEIGHT, TEXT_WEIGHT) if skills_detected else (0.0, 1.0)

    candidates = []
    for name, text_score in zip(names, similarity):
        text_score = float(text_score)
        resume_words = set(resumes[name].split())
        matched = [s for s in required if s in resume_words]
        missing = [s for s in required if s not in resume_words]
        skill_score = len(matched) / len(required) * 100 if required else 0.0

        candidates.append(
            Candidate(
                name=name,
                score=skill_w * skill_score + text_w * text_score,
                skill_score=skill_score,
                text_score=text_score,
                experience=extract_experience(resumes[name]),
                matched=matched,
                missing=missing,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates, skills_detected


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
# Colours that differ between themes are written as light-dark(<light>, <dark>). The browser
# resolves them from the colour-scheme Streamlit sets on .stApp, so the page follows the
# Streamlit theme (Settings menu or system setting) instantly, with no Python-side detection.
CSS_BASE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Space+Grotesk:wght@500;700&display=swap');

.stApp {
    --muted:      light-dark(rgba(27,29,54,.62), rgba(230,232,245,.62));
    --line:       light-dark(rgba(124,58,237,.16), rgba(255,255,255,.09));
    --soft:       light-dark(rgba(27,29,54,.06), rgba(255,255,255,.07));
    --violet:     light-dark(#7c3aed, #8b5cf6);
    --cyan:       light-dark(#0891b2, #22d3ee);
    --panel-a:    light-dark(rgba(255,255,255,.90), rgba(255,255,255,.06));
    --panel-b:    light-dark(rgba(255,255,255,.60), rgba(255,255,255,.02));
    --shadow:     light-dark(rgba(76,29,149,.10), rgba(0,0,0,.35));
    --hover-line: light-dark(rgba(124,58,237,.45), rgba(139,92,246,.50));
    --hover-glow: light-dark(rgba(124,58,237,.18), rgba(139,92,246,.18));
    --tile:       light-dark(rgba(255,255,255,.80), rgba(255,255,255,.04));
    --tile-line:  light-dark(rgba(124,58,237,.14), rgba(255,255,255,.08));
    --track:      light-dark(rgba(27,29,54,.10), rgba(255,255,255,.10));
    --grid:       light-dark(rgba(80,70,160,.07), rgba(255,255,255,.035));
    --tint:       light-dark(rgba(124,58,237,.12), rgba(139,92,246,.16));
}

html, body, .stApp, [class*="css"], .stMarkdown, label, button { font-family: 'Manrope', system-ui, sans-serif; }
h2, h3, h4 { font-family: 'Space Grotesk', 'Manrope', sans-serif; letter-spacing: -0.02em; }

/* Backdrop: soft glows plus a faint grid that fades out down the page */
.stApp {
    background:
        radial-gradient(900px 520px at 10% -8%, light-dark(rgba(167,139,250,.42), rgba(139,92,246,.24)), transparent 60%),
        radial-gradient(800px 480px at 98% 0%, light-dark(rgba(125,211,252,.42), rgba(34,211,238,.14)), transparent 55%),
        radial-gradient(800px 520px at 50% 108%, light-dark(rgba(253,186,116,.30), transparent), transparent 60%),
        light-dark(#f6f5ff, #0b0d17);
}
.stApp::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(var(--grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid) 1px, transparent 1px);
    background-size: 48px 48px;
    -webkit-mask-image: radial-gradient(ellipse at 50% 0%, #000 15%, transparent 72%);
            mask-image: radial-gradient(ellipse at 50% 0%, #000 15%, transparent 72%);
}
[data-testid="stHeader"] { background: transparent; }
footer, [data-testid="stDecoration"] { display: none; }   /* the main menu stays: it holds the theme switch */
.block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem; position: relative; z-index: 1; }

/* Hero */
.hero { padding: 1.2rem 0 1.6rem; }
.eyebrow {
    display: inline-flex; align-items: center; gap: .55rem; padding: .28rem .8rem;
    border: 1px solid light-dark(rgba(124,58,237,.32), rgba(167,139,250,.35)); border-radius: 999px;
    background: light-dark(rgba(124,58,237,.10), rgba(139,92,246,.10));
    font-size: .76rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase;
    color: light-dark(#6d28d9, #c4b5fd);
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--cyan); animation: pulse 2s infinite; }
.hero-title {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: -.03em; line-height: 1.04;
    font-size: clamp(2.3rem, 5.2vw, 3.5rem); margin: .9rem 0 .7rem;
    background: linear-gradient(92deg, light-dark(#4c1d95, #fff) 8%, light-dark(#7c3aed, #c4b5fd) 45%, light-dark(#0891b2, #67e8f9) 92%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: var(--muted); max-width: 56ch; font-size: 1.05rem; margin: 0; }

/* Glass panels (containers are given keys in the Python code) */
.st-key-job_panel, .st-key-upload_panel, [class*="st-key-cand_"] {
    background: linear-gradient(160deg, var(--panel-a), var(--panel-b));
    border: 1px solid var(--line); border-radius: 18px; padding: 1.3rem 1.4rem;
    -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
    box-shadow: 0 12px 40px var(--shadow);
    transition: border-color .25s, box-shadow .25s, transform .25s;
}
.st-key-job_panel:hover, .st-key-upload_panel:hover, [class*="st-key-cand_"]:hover {
    border-color: var(--hover-line);
    box-shadow: 0 0 0 1px var(--hover-glow), 0 16px 48px var(--hover-glow);
    transform: translateY(-2px);
}
.st-key-cand_1 {
    border-color: light-dark(rgba(124,58,237,.55), rgba(139,92,246,.60));
    background: linear-gradient(160deg, var(--tint), light-dark(rgba(8,145,178,.06), rgba(34,211,238,.05)) 60%, light-dark(rgba(255,255,255,.65), rgba(255,255,255,.02)));
    box-shadow: 0 0 60px var(--hover-glow), 0 12px 40px var(--shadow);
}
[class*="st-key-cand_"] { animation: rise .6s cubic-bezier(.2,.8,.2,1) backwards; }

.panel-title { display: flex; align-items: center; gap: .65rem; font-size: 1.08rem; font-weight: 800; margin: 0 0 .2rem; }
.num {
    display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 50%;
    font-size: .8rem; font-weight: 800; color: #fff; background: linear-gradient(135deg, #7c3aed, #06b6d4);
}
.section-hint { color: var(--muted); font-size: .9rem; margin: 0 0 .8rem 2.3rem; }

/* Inputs */
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within {
    box-shadow: 0 0 0 1px var(--violet), 0 0 18px var(--hover-glow);
}
[data-testid="stFileUploaderDropzone"] {
    background: light-dark(rgba(255,255,255,.60), rgba(255,255,255,.03));
    border: 1.5px dashed light-dark(rgba(124,58,237,.45), rgba(167,139,250,.45));
    border-radius: 14px; transition: border-color .25s, background .25s;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--cyan); background: light-dark(rgba(8,145,178,.06), rgba(34,211,238,.05)); }

/* Buttons */
.stButton, [data-testid="stFormSubmitButton"],
[data-testid="stElementContainer"]:has(> .stButton),
[data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]),
[data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]) > div { width: 100%; }
:is(.stButton, [data-testid="stFormSubmitButton"]) button:is([kind="primary"], [kind="primaryFormSubmit"], [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"]) {
    width: 100%; color: #fff !important; border: 0; font-weight: 800; letter-spacing: .01em;
    padding: .85rem 1rem; border-radius: 12px;
    background: linear-gradient(95deg, #7c3aed, #06b6d4);
    box-shadow: 0 8px 30px rgba(124,58,237,.35);
    transition: transform .2s, box-shadow .2s, filter .2s;
}
.stButton button:is([kind="primary"], [data-testid="stBaseButton-primary"]):hover:not(:disabled),
[data-testid="stFormSubmitButton"] button:is([kind="primaryFormSubmit"], [data-testid="stBaseButton-primaryFormSubmit"]):hover:not(:disabled) {
    transform: translateY(-1px); filter: brightness(1.1); box-shadow: 0 12px 38px rgba(34,211,238,.35);
}
:is(.stButton, [data-testid="stFormSubmitButton"]) button:disabled { opacity: .5; filter: saturate(.35); box-shadow: none; }

/* Candidate header */
.cand { display: flex; align-items: center; gap: 1.2rem; margin-bottom: .9rem; flex-wrap: wrap; }
.cand-info { min-width: 0; }
.cand-rank { display: flex; align-items: center; gap: .55rem; font-size: .8rem; font-weight: 700; color: var(--muted); letter-spacing: .05em; text-transform: uppercase; }
.cand-name { font-size: 1.2rem; font-weight: 800; line-height: 1.25; word-break: break-word; margin-top: .15rem; }

.rank-badge {
    display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 50%;
    font-size: .8rem; font-weight: 800; color: var(--muted); letter-spacing: 0;
    background: var(--soft); border: 1px solid light-dark(rgba(27,29,54,.15), rgba(255,255,255,.14));
}
.rank-badge.r1 { color: #1c1303; border: 0; background: linear-gradient(135deg, #fde68a, #f59e0b); box-shadow: 0 0 16px rgba(245,158,11,.5); }
.rank-badge.r2 { color: #111827; border: 0; background: linear-gradient(135deg, #f3f4f6, #9ca3af); box-shadow: 0 0 14px rgba(156,163,175,.45); }
.rank-badge.r3 { color: #1f1108; border: 0; background: linear-gradient(135deg, #fdba74, #c2410c); box-shadow: 0 0 14px rgba(234,88,12,.35); }

/* Score ring: masked conic gradient, animates from 0 to the score.
   Verdict colour is passed as a light and a dark variant (--cl / --cd). */
@property --p { syntax: '<number>'; inherits: false; initial-value: 0; }
.ring-wrap, .pill { --c: light-dark(var(--cl), var(--cd)); }
.ring-wrap { position: relative; width: 88px; height: 88px; flex: none; filter: drop-shadow(0 0 8px color-mix(in srgb, var(--c) 45%, transparent)); }
.ring {
    --w: 9px; --p: var(--t); position: absolute; inset: 0; border-radius: 50%;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), var(--track) 0);
    -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - var(--w)), #000 calc(100% - var(--w) + 1px));
            mask: radial-gradient(farthest-side, transparent calc(100% - var(--w)), #000 calc(100% - var(--w) + 1px));
    animation: fill 1.3s cubic-bezier(.2,.8,.2,1) backwards;
}
.ring-wrap b { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-family: 'Space Grotesk', sans-serif; font-size: 1.45rem; font-weight: 700; }
.ring-wrap b small { font-size: .72rem; opacity: .7; margin-left: 1px; }

.pill {
    display: inline-block; margin-top: .4rem; padding: .18rem .7rem; border-radius: 999px;
    font-size: .78rem; font-weight: 800; color: var(--c);
    background: color-mix(in srgb, var(--c) 14%, transparent);
    border: 1px solid color-mix(in srgb, var(--c) 35%, transparent);
}
.badge {
    margin-left: .2rem; padding: .12rem .6rem; border-radius: 999px; font-size: .7rem; font-weight: 800;
    letter-spacing: .06em; color: #fff; background: linear-gradient(95deg, #7c3aed, #06b6d4);
}

/* Skill chips */
.chips-label { font-size: .8rem; font-weight: 800; margin: 1rem 0 .4rem; color: var(--muted); letter-spacing: .06em; text-transform: uppercase; }
.chip { display: inline-block; padding: .22rem .7rem; margin: 0 .4rem .4rem 0; border-radius: 8px; font-size: .84rem; font-weight: 600; border: 1px solid transparent; }
.chip.hit  {
    color: light-dark(#166534, #86efac); background: light-dark(rgba(34,197,94,.14), rgba(34,197,94,.12));
    border-color: light-dark(rgba(22,163,74,.40), rgba(34,197,94,.35)); box-shadow: 0 0 14px light-dark(transparent, rgba(34,197,94,.12));
}
.chip.miss {
    color: light-dark(#991b1b, #fca5a5); background: rgba(239,68,68,.12);
    border-color: light-dark(rgba(220,38,38,.35), rgba(239,68,68,.35)); box-shadow: 0 0 14px light-dark(transparent, rgba(239,68,68,.12));
}
.chip.none { color: var(--muted); background: var(--soft); }

/* Metrics, callout, tabs, expanders */
[data-testid="stMetric"] { background: var(--tile); border: 1px solid var(--tile-line); border-radius: 14px; padding: .8rem 1rem; }
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; font-weight: 700; }
.callout {
    border-left: 3px solid var(--violet); border-radius: 10px; padding: .8rem 1.1rem; margin: .8rem 0 1.1rem;
    background: linear-gradient(90deg, var(--tint), transparent);
}
.stTabs [data-baseweb="tab"] { font-weight: 800; }
[data-testid="stExpander"] details { background: light-dark(rgba(255,255,255,.65), rgba(255,255,255,.035)); border: 1px solid var(--line); border-radius: 14px; }
[data-testid="stExpander"] details:hover { border-color: var(--hover-line); }

/* Empty state */
.empty {
    border: 1.5px dashed light-dark(rgba(124,58,237,.35), rgba(167,139,250,.35)); border-radius: 18px;
    padding: 2.2rem 1.6rem; margin-top: 1.6rem; text-align: center;
    background: radial-gradient(500px 200px at 50% 0%, var(--tint), transparent 70%);
}
.empty > b { font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; }
.empty > p { color: var(--muted); margin: .45rem auto 0; max-width: 56ch; }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .9rem; margin-top: 1.4rem; text-align: left; }
.step { padding: 1rem 1.1rem; border-radius: 14px; background: var(--tile); border: 1px solid var(--tile-line); }
.step b { display: flex; align-items: center; gap: .6rem; margin-bottom: .35rem; }
.step span { color: var(--muted); font-size: .88rem; }
.foot { margin-top: 2.5rem; text-align: center; color: var(--muted); font-size: .82rem; }

@keyframes fill  { from { --p: 0; } to { --p: var(--t); } }
@keyframes rise  { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(34,211,238,.6); } 70% { box-shadow: 0 0 0 9px rgba(34,211,238,0); } 100% { box-shadow: 0 0 0 0 rgba(34,211,238,0); } }
@media (prefers-reduced-motion: reduce) { *, *::before { animation: none !important; transition: none !important; } }
"""


def build_css() -> str:
    # Cards fade in one after another
    stagger = "".join(f".st-key-cand_{n} {{ animation-delay: {n * 0.07:.2f}s; }}\n" for n in range(1, 13))
    return CSS_BASE + stagger + "</style>"


CSS = build_css()


def chips(items: list[str], kind: str, empty_text: str) -> str:
    if not items:
        return f'<span class="chip none">{escape(empty_text)}</span>'
    shown = "".join(f'<span class="chip {kind}">{escape(DISPLAY.get(i, i))}</span>' for i in items[:MAX_CHIPS])
    extra = len(items) - MAX_CHIPS
    if extra > 0:
        shown += f'<span class="chip none">+{extra} more</span>'
    return shown


# ----------------------------------------------------------------------------
# UI sections
# ----------------------------------------------------------------------------
def render_inputs() -> tuple[str, list, bool]:
    """Draw the input form. Returns (job_text, uploaded_files, submitted).

    Everything is inside st.form, so Streamlit re-runs the script only when
    "Rank candidates" is pressed, not after each field.
    """
    with st.form("job_form", border=False):
        left, right = st.columns([3, 2], gap="large")

        with left, st.container(key="job_panel"):
            st.markdown('<div class="panel-title"><span class="num">1</span>Job description</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="section-hint">The more specific the skills, the better the ranking.</p>',
                unsafe_allow_html=True,
            )
            role = st.text_input("Job role", placeholder="e.g. Data Analyst", max_chars=100)
            skills = st.text_area("Required skills", placeholder="Python, SQL, Machine Learning, AWS", height=90, max_chars=2000)
            quals = st.text_area("Qualifications", placeholder="B.Tech in Computer Science", height=90, max_chars=2000)

        with right, st.container(key="upload_panel"):
            st.markdown('<div class="panel-title"><span class="num">2</span>Resumes</div>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="section-hint">PDF only. Up to {MAX_FILES} files, {MAX_FILE_MB} MB each.</p>',
                unsafe_allow_html=True,
            )
            files = st.file_uploader(
                "Upload resumes", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
            )

        submitted = st.form_submit_button("Rank candidates", type="primary")

    return f"{role}\n{skills}\n{quals}", files or [], submitted


def run_ranking(job_text: str, files: list) -> dict | None:
    """Read PDFs and score them. Returns a result bundle, or None after showing an error."""
    job_clean = clean_text(job_text)
    if not has_usable_words(job_clean):
        st.error("Add a job role, required skills, or qualifications with at least one meaningful word.")
        return None
    if not files:
        st.error("Upload at least one resume PDF.")
        return None

    if len(files) > MAX_FILES:
        st.error(f"Upload at most {MAX_FILES} resumes at a time (you added {len(files)}).")
        return None
    if sum(f.size for f in files) > MAX_TOTAL_BYTES:
        st.error(f"The files add up to more than {MAX_TOTAL_MB} MB. Upload fewer or smaller files.")
        return None

    resumes: dict[str, str] = {}
    skipped: list[str] = []
    progress = st.progress(0.0, text="Reading resumes...")
    for i, file in enumerate(files, 1):
        try:
            text = clean_text(read_pdf(check_upload(file)))
        except UploadRejected as reason:
            skipped.append(f"{md_escape(file.name)} {reason}")
        except Exception:
            # Don't surface parser internals to the user
            skipped.append(f"{md_escape(file.name)} couldn't be read. It may be corrupted.")
        else:
            if text:
                # keep names unique if two files share a name
                name = file.name if file.name not in resumes else f"{file.name} ({i})"
                resumes[name] = text
            else:
                skipped.append(f"{md_escape(file.name)} has no readable text (scanned PDFs aren't supported).")
        progress.progress(i / len(files), text=f"Reading resumes... {i} of {len(files)}")
    progress.empty()

    for message in skipped:
        st.warning(message)
    if not resumes:
        st.error("None of the uploaded PDFs contained readable text.")
        return None

    candidates, skills_detected = score_candidates(job_clean, resumes)
    return {"candidates": candidates, "skills_detected": skills_detected}


def results_table(candidates: list[Candidate]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Rank": range(1, len(candidates) + 1),
            "Candidate": [c.name for c in candidates],
            "Score": [round(c.score, 1) for c in candidates],
            "Skill match %": [round(c.skill_score, 1) for c in candidates],
            "Text similarity %": [round(c.text_score, 1) for c in candidates],
            "Experience (yrs)": [c.experience for c in candidates],
        }
    )


def render_summary(candidates: list[Candidate], min_exp: int) -> None:
    meets = sum(c.experience >= min_exp for c in candidates)
    average = sum(c.score for c in candidates) / len(candidates)

    cols = st.columns(4)
    cols[0].metric("Resumes analyzed", len(candidates))
    cols[1].metric("Top score", f"{candidates[0].score:.0f}%")
    cols[2].metric("Average score", f"{average:.0f}%")
    cols[3].metric("Meet experience", f"{meets} of {len(candidates)}")

    shortlist = sum(c.score >= SHORTLIST_SCORE and c.experience >= min_exp for c in candidates)
    requirement = f" and have {min_exp}+ years of experience" if min_exp else ""
    st.markdown(
        f'<div class="callout"><b>{shortlist} of {len(candidates)}</b> candidates score '
        f"{SHORTLIST_SCORE}% or higher{requirement}.</div>",
        unsafe_allow_html=True,
    )

    table = results_table(candidates)
    # Filenames are user-controlled: neutralise formulas in the exported copy only
    csv_table = table.assign(Candidate=table["Candidate"].map(csv_safe))

    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f%%"),
        },
        **STRETCH,
    )
    st.download_button(
        "Download results (CSV)",
        csv_table.to_csv(index=False).encode("utf-8"),
        file_name="resume_ranking.csv",
        mime="text/csv",
    )


def render_candidate_details(c: Candidate, min_exp: int, skills_detected: bool) -> None:
    m1, m2, m3 = st.columns(3)
    m1.metric("Skill match", f"{c.skill_score:.0f}%" if skills_detected else "n/a")
    m2.metric("Text similarity", f"{c.text_score:.0f}%")
    m3.metric("Experience found", f"{c.experience} yrs")

    if c.experience >= min_exp:
        kind, text = "hit", f"Meets the {min_exp}+ year requirement"
    else:
        kind, text = "miss", f"Below the {min_exp}+ year requirement (found {c.experience})"
    st.markdown(f'<span class="chip {kind}">{escape(text)}</span>', unsafe_allow_html=True)

    if skills_detected:
        st.markdown(
            '<div class="chips-label">Skills found</div>' + chips(c.matched, "hit", "No required skills found"),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="chips-label">Skills missing</div>' + chips(c.missing, "miss", "Nothing missing"),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Score = {SKILL_WEIGHT:.0%} × skill match ({c.skill_score:.1f}%) "
            f"+ {TEXT_WEIGHT:.0%} × text similarity ({c.text_score:.1f}%) = {c.score:.1f}%"
        )
    else:
        st.caption(f"No known skills in the job description, so the score is text similarity alone: {c.score:.1f}%")


def render_candidate(rank: int, c: Candidate, min_exp: int, skills_detected: bool) -> None:
    """Full card with the animated score ring. Used for the top candidates."""
    label, colour = verdict_for(c.score)
    pct = round(c.score)
    medal = f" r{rank}" if rank <= 3 else ""
    badge = '<span class="badge">TOP RANKED</span>' if rank == 1 else ""

    # Key gives the container a stable CSS class (st-key-cand_<rank>) for the glass styling
    with st.container(key=f"cand_{rank}"):
        st.markdown(
            f'<div class="cand">'
            f'<div class="ring-wrap" style="--t:{min(pct, 100)};--cd:{colour};--cl:{LIGHT_TONE[colour]}" role="img" aria-label="Score {pct} percent">'
            f'<div class="ring"></div><b>{pct}<small>%</small></b></div>'
            f'<div class="cand-info"><div class="cand-rank">'
            f'<span class="rank-badge{medal}">{rank}</span>Rank {rank}{badge}</div>'
            f'<div class="cand-name">{escape(c.name)}</div>'
            f'<span class="pill" style="--cd:{colour};--cl:{LIGHT_TONE[colour]}">{label}</span></div></div>',
            unsafe_allow_html=True,
        )
        render_candidate_details(c, min_exp, skills_detected)


def render_candidates(candidates: list[Candidate], min_exp: int, skills_detected: bool) -> None:
    f1, f2, f3 = st.columns([2, 3, 4], gap="large", vertical_alignment="bottom")
    sort_by = f1.selectbox("Sort by", list(SORTS))
    min_score = f2.slider("Minimum score", 0, 100, 0, format="%d%%")
    only_meeting = f3.toggle("Only candidates who meet the experience requirement")

    # Rank always means rank by score, whatever the current sort order
    shown = [
        (rank, c)
        for rank, c in enumerate(candidates, 1)
        if c.score >= min_score and (not only_meeting or c.experience >= min_exp)
    ]
    shown.sort(key=lambda rc: SORTS[sort_by](rc[1]), reverse=True)

    if not shown:
        st.info("No candidates match these filters. Lower the minimum score or turn off the experience filter.")
        return

    st.caption(f"Showing {len(shown)} of {len(candidates)} candidates")
    for rank, c in shown[:TOP_CARDS]:
        render_candidate(rank, c, min_exp, skills_detected)

    rest = shown[TOP_CARDS:]
    if rest:
        st.markdown(f"#### More candidates ({len(rest)})")
        for rank, c in rest:
            _, colour = verdict_for(c.score)
            with st.expander(f"{DOTS[colour]} #{rank} {md_escape(c.name)}: {round(c.score)}%"):
                render_candidate_details(c, min_exp, skills_detected)


def render_results(state: dict) -> None:
    """Draw saved results. The experience requirement is read here, outside the form: it is
    only a display comparison, so it updates live without re-ranking."""
    candidates: list[Candidate] = state["candidates"]
    skills_detected: bool = state["skills_detected"]

    st.write("")
    st.markdown("## Results")
    if not skills_detected:
        st.info(
            "None of the skills in the job description are in the known skills list, "
            "so candidates are ranked by text similarity only."
        )

    box, _ = st.columns([1, 3])
    min_exp = int(box.number_input(
        "Minimum experience (years)", min_value=0, max_value=50, value=0, key="min_exp",
        help="Flags candidates below this. It doesn't change their scores.",
    ))

    overview, details = st.tabs(["Overview", "Candidates"])
    with overview:
        render_summary(candidates, min_exp)
    with details:
        render_candidates(candidates, min_exp, skills_detected)


def render_empty_state() -> None:
    st.markdown(
        '<div class="empty"><b>No results yet</b>'
        "<p>Candidates are ordered by how many required skills they have and how closely "
        "their resume reads like the job description.</p>"
        '<div class="steps">'
        '<div class="step"><b><span class="num">1</span>Describe the role</b>'
        "<span>Add a title and the skills you need.</span></div>"
        '<div class="step"><b><span class="num">2</span>Upload resumes</b>'
        "<span>PDF files, up to " + str(MAX_FILES) + " at a time.</span></div>"
        '<div class="step"><b><span class="num">3</span>Review the ranking</b>'
        "<span>Filter, sort and export the results.</span></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def hero_html(status: str) -> str:
    return (
        f'<div class="hero"><span class="eyebrow"><span class="dot"></span>{escape(status)}</span>'
        '<div class="hero-title">Resume screening</div>'
        '<p class="hero-sub">Describe the role, upload resumes as PDFs, and get '
        "candidates ranked by how well they match.</p></div>"
    )


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Resume Screening", page_icon="📄", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    hero = st.empty()  # filled in at the end, once we know the status

    job_text, files, submitted = render_inputs()

    if submitted:
        # Keep results in session state so they survive reruns (e.g. the CSV download)
        st.session_state["results"] = run_ranking(job_text, files)

    results = st.session_state.get("results")
    if results:
        render_results(results)
        status = f"{len(results['candidates'])} resumes ranked"
    else:
        render_empty_state()
        status = f"{len(files)} resumes loaded" if files else "Ready"

    hero.markdown(hero_html(status), unsafe_allow_html=True)
    st.markdown(
        f'<div class="foot">Score = {SKILL_WEIGHT:.0%} skill match + {TEXT_WEIGHT:.0%} text similarity.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
