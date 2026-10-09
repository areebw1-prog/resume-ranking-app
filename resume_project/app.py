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
    (80, "Almost perfect match", "#16a34a"),
    (60, "Excellent match", "#0d9488"),
    (30, "Good match", "#2563eb"),
    (10, "Average match", "#d97706"),
    (0, "Low match", "#dc2626"),
]

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
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stButton button, label {
    font-family: 'Manrope', system-ui, sans-serif;
}
.block-container { max-width: 1100px; padding-top: 2.5rem; padding-bottom: 4rem; }

.app-title { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; }
.app-sub   { opacity: .7; margin: .25rem 0 1.75rem; font-size: 1.02rem; max-width: 60ch; }
.section-title { font-size: 1.05rem; font-weight: 700; margin: 0 0 .25rem; }
.section-hint  { opacity: .65; font-size: .9rem; margin-bottom: .75rem; }

.stButton { width: 100%; }
.stButton button[kind="primary"] {
    width: 100%; background: #4f46e5; border: 0; font-weight: 700;
    padding: .7rem 1rem; border-radius: 10px;
}
.stButton button[kind="primary"]:hover { background: #4338ca; }

.cand { display: flex; align-items: center; gap: 1.1rem; margin-bottom: .75rem; }
.cand-rank { font-size: .85rem; opacity: .6; font-weight: 700; }
.cand-name { font-size: 1.15rem; font-weight: 800; line-height: 1.25; word-break: break-word; }

/* Ring is masked, so it works on any theme background */
.ring {
    width: 76px; height: 76px; flex: none; border-radius: 50%;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(128,128,128,.25) 0);
    -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 8px), #000 calc(100% - 7px));
            mask: radial-gradient(farthest-side, transparent calc(100% - 8px), #000 calc(100% - 7px));
}
.ring-wrap { position: relative; width: 76px; height: 76px; flex: none; }
.ring-wrap .ring { position: absolute; inset: 0; }
.ring-wrap b { position: absolute; inset: 0; display: grid; place-items: center; font-size: 1.05rem; font-weight: 800; }

.pill {
    display: inline-block; margin-top: .3rem; padding: .15rem .65rem;
    border-radius: 999px; font-size: .8rem; font-weight: 700;
    color: color-mix(in srgb, var(--c) 78%, currentColor);
    background: color-mix(in srgb, var(--c) 16%, transparent);
}

.chips-label { font-size: .85rem; font-weight: 700; margin: .9rem 0 .35rem; opacity: .8; }
.chip {
    display: inline-block; padding: .2rem .65rem; margin: 0 .35rem .35rem 0;
    border-radius: 6px; font-size: .84rem; font-weight: 500;
}
.chip.hit  { color: color-mix(in srgb, #16a34a 75%, currentColor); background: rgba(22,163,74,.15); }
.chip.miss { color: color-mix(in srgb, #dc2626 75%, currentColor); background: rgba(220,38,38,.14); }
.chip.none { background: rgba(128,128,128,.15); }

[data-testid="stMetricValue"] { font-weight: 800; }
</style>
"""


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
def render_inputs() -> tuple[str, int, list]:
    """Draw the form. Returns (job_text, minimum_experience, uploaded_files)."""
    left, right = st.columns([3, 2], gap="large")

    with left, st.container(border=True):
        st.markdown('<p class="section-title">Job description</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-hint">The more specific the skills, the better the ranking.</p>',
            unsafe_allow_html=True,
        )
        role = st.text_input("Job role", placeholder="e.g. Data Analyst", max_chars=100)
        skills = st.text_area("Required skills", placeholder="Python, SQL, Machine Learning, AWS", height=90, max_chars=2000)
        quals = st.text_area("Qualifications", placeholder="B.Tech in Computer Science", height=90, max_chars=2000)
        min_exp = st.number_input("Minimum experience (years)", min_value=0, max_value=50, value=0)

    with right, st.container(border=True):
        st.markdown('<p class="section-title">Resumes</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="section-hint">PDF only. Up to {MAX_FILES} files, {MAX_FILE_MB} MB each.</p>',
            unsafe_allow_html=True,
        )
        files = st.file_uploader(
            "Upload resumes", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
        )
        if files:
            st.caption(f"{len(files)} resume(s) ready")

    job_text = f"{role}\n{skills}\n{quals}"
    return job_text, int(min_exp), files or []


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


def render_summary(candidates: list[Candidate], min_exp: int) -> None:
    meets = sum(c.experience >= min_exp for c in candidates)
    average = sum(c.score for c in candidates) / len(candidates)

    cols = st.columns(4)
    cols[0].metric("Resumes analyzed", len(candidates))
    cols[1].metric("Top score", f"{candidates[0].score:.0f}%")
    cols[2].metric("Average score", f"{average:.0f}%")
    cols[3].metric("Meet experience", f"{meets} of {len(candidates)}")

    table = pd.DataFrame(
        {
            "Rank": range(1, len(candidates) + 1),
            "Candidate": [c.name for c in candidates],
            "Score": [round(c.score, 1) for c in candidates],
            "Skill match %": [round(c.skill_score, 1) for c in candidates],
            "Text similarity %": [round(c.text_score, 1) for c in candidates],
            "Experience (yrs)": [c.experience for c in candidates],
        }
    )
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


def render_candidate(rank: int, c: Candidate, min_exp: int, skills_detected: bool) -> None:
    label, colour = verdict_for(c.score)
    pct = round(c.score)

    with st.container(border=True):
        st.markdown(
            f'<div class="cand">'
            f'<div class="ring-wrap" role="img" aria-label="Score {pct} percent">'
            f'<div class="ring" style="--p:{min(pct, 100)};--c:{colour}"></div><b>{pct}%</b></div>'
            f'<div><div class="cand-rank">Rank {rank}</div>'
            f'<div class="cand-name">{escape(c.name)}</div>'
            f'<span class="pill" style="--c:{colour}">{label}</span></div></div>',
            unsafe_allow_html=True,
        )

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
                '<div class="chips-label">Skills found</div>'
                + chips(c.matched, "hit", "No required skills found"),
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="chips-label">Skills missing</div>'
                + chips(c.missing, "miss", "Nothing missing"),
                unsafe_allow_html=True,
            )

        with st.expander("How this score is calculated"):
            if skills_detected:
                st.write(
                    f"Skill match {c.skill_score:.1f}% × {SKILL_WEIGHT} + "
                    f"text similarity {c.text_score:.1f}% × {TEXT_WEIGHT:.1f} = **{c.score:.1f}%**"
                )
            else:
                st.write(f"No known skills in the job description, so the score is text similarity alone: **{c.score:.1f}%**")


def render_results(state: dict, min_exp: int) -> None:
    """Draw saved results. min_exp is the *current* form value: the experience check is
    a display comparison, so it updates live without re-ranking."""
    candidates: list[Candidate] = state["candidates"]
    skills_detected: bool = state["skills_detected"]

    st.write("")
    st.markdown("## Results")
    if not skills_detected:
        st.info(
            "None of the skills in the job description are in the known skills list, "
            "so candidates are ranked by text similarity only."
        )

    render_summary(candidates, min_exp)

    st.write("")
    st.markdown("### Candidate details")
    for rank, candidate in enumerate(candidates, 1):
        render_candidate(rank, candidate, min_exp, skills_detected)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Resume Screening", page_icon="📄", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown('<p class="app-title">Resume screening</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-sub">Describe the role, upload resumes as PDFs, and get '
        "candidates ranked by how well they match.</p>",
        unsafe_allow_html=True,
    )

    job_text, min_exp, files = render_inputs()

    st.write("")
    if st.button("Rank candidates", type="primary"):
        bundle = run_ranking(job_text, files)
        # Keep results in session state so they survive reruns (e.g. the CSV download)
        st.session_state["results"] = bundle

    if st.session_state.get("results"):
        render_results(st.session_state["results"], min_exp)


if __name__ == "__main__":
    main()
