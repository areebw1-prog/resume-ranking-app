import re
from html import escape

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Resume Screening",
    page_icon="📄",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
ignore_words = {
    "experience", "work", "worked", "team", "project", "projects",
    "using", "used", "developer", "application", "system", "data",
    "year", "years",
}

skills_list = {
    # Programming languages
    "python", "java", "c", "cpp", "csharp", "javascript",
    "typescript", "php", "ruby", "swift", "kotlin",
    "go", "rust", "r", "matlab",
    # Web development
    "html", "css", "react", "angular", "vue",
    "nodejs", "express", "django", "flask",
    "streamlit", "bootstrap", "tailwind",
    # Databases
    "sql", "mysql", "postgresql", "mongodb",
    "sqlite", "oracle", "firebase",
    # Data science / AI
    "machine", "learning", "deep", "tensorflow",
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
    "oop", "api", "rest", "microservices",
    # General tech
    "ai", "ml", "automation", "cloud", "devops",
}

# (minimum score, label, colour)
VERDICTS = [
    (80, "Almost perfect match", "#16a34a"),
    (60, "Excellent match", "#0d9488"),
    (30, "Good match", "#2563eb"),
    (10, "Average match", "#d97706"),
    (0, "Low match", "#dc2626"),
]


def verdict_for(score):
    for threshold, label, color in VERDICTS:
        if score >= threshold:
            return label, color
    return VERDICTS[-1][1], VERDICTS[-1][2]


# ----------------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------------
def extract_text_from_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text)


def extract_experience(text):
    matches = re.findall(r"(\d+)\s+years", text)
    return max(int(n) for n in matches) if matches else 0


def split_skills(job_words, resume_words):
    """Return (matched, missing) skills, matching whole words only."""
    required = sorted(skills_list & job_words)
    matched = [s for s in required if s in resume_words]
    missing = [s for s in required if s not in resume_words]
    return matched, missing


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
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

/* Primary button */
.stButton button[kind="primary"] {
    background: #4f46e5; border: 0; font-weight: 700;
    padding: .7rem 1rem; border-radius: 10px;
}
.stButton button[kind="primary"]:hover { background: #4338ca; }

/* Candidate header */
.cand { display: flex; align-items: center; gap: 1.1rem; margin-bottom: .75rem; }
.cand-rank { font-size: .85rem; opacity: .6; font-weight: 700; }
.cand-name { font-size: 1.15rem; font-weight: 800; line-height: 1.25; word-break: break-word; }
.ring {
    --size: 76px;
    width: var(--size); height: var(--size); flex: none;
    border-radius: 50%;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(128,128,128,.22) 0);
    display: grid; place-items: center; position: relative;
    font-weight: 800; font-size: 1.05rem;
}
.ring::before {
    content: ""; position: absolute; inset: 7px; border-radius: 50%;
    background: var(--bg, #0e1117);
}
.ring span { position: relative; }
.pill {
    display: inline-block; margin-top: .3rem; padding: .15rem .65rem;
    border-radius: 999px; font-size: .8rem; font-weight: 700;
    color: var(--c); background: color-mix(in srgb, var(--c) 16%, transparent);
}

/* Skill chips */
.chips-label { font-size: .85rem; font-weight: 700; margin: .9rem 0 .35rem; opacity: .8; }
.chip {
    display: inline-block; padding: .2rem .65rem; margin: 0 .35rem .35rem 0;
    border-radius: 6px; font-size: .84rem; font-weight: 500;
}
.chip.hit  { background: rgba(22,163,74,.16);  color: #22c55e; }
.chip.miss { background: rgba(220,38,38,.16);  color: #f87171; }
.chip.none { background: rgba(128,128,128,.15); }

[data-testid="stMetricValue"] { font-weight: 800; }
</style>
""",
    unsafe_allow_html=True,
)


def chips(items, kind, empty_text):
    if not items:
        return f'<span class="chip none">{escape(empty_text)}</span>'
    return "".join(f'<span class="chip {kind}">{escape(i)}</span>' for i in items)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<p class="app-title">Resume screening</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Describe the role, upload resumes as PDFs, and get '
    "candidates ranked by how well they match.</p>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<p class="section-title">Job description</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-hint">The more specific the skills, the better the ranking.</p>',
            unsafe_allow_html=True,
        )

        job_role = st.text_input("Job role", placeholder="e.g. Data Analyst")

        required_skills = st.text_area(
            "Required skills",
            placeholder="Python, SQL, Machine Learning, AWS",
            height=90,
        )

        qualifications = st.text_area(
            "Qualifications",
            placeholder="B.Tech in Computer Science",
            height=90,
        )

        experience_required = st.number_input(
            "Minimum experience (years)",
            min_value=0,
            max_value=50,
            value=0,
        )

with right:
    with st.container(border=True):
        st.markdown('<p class="section-title">Resumes</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-hint">PDF only. Upload as many as you need.</p>',
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Upload resumes",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} resume(s) ready")

job_desc = f"""
Job Role:
{job_role}

Required Skills:
{required_skills}

Qualifications:
{qualifications}

Minimum Experience Required:
{experience_required} years
"""

st.write("")
rank_button = st.button("Rank candidates", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------------
if rank_button:
    if not (job_role.strip() or required_skills.strip() or qualifications.strip()):
        st.error("Add a job role, required skills, or qualifications to compare against.")
        st.stop()
    if not uploaded_files:
        st.error("Upload at least one resume PDF.")
        st.stop()

    resumes, names = [], []
    with st.spinner("Reading resumes..."):
        for file in uploaded_files:
            try:
                text = clean_text(extract_text_from_pdf(file))
                if text.strip():
                    resumes.append(text)
                    names.append(file.name)
                else:
                    st.warning(f"{file.name}: no readable text. Scanned PDFs aren't supported.")
            except Exception:
                st.warning(f"{file.name}: couldn't be opened.")

    if not resumes:
        st.error("None of the uploaded PDFs contained readable text.")
        st.stop()

    cleaned_job_desc = clean_text(job_desc)
    job_words = set(cleaned_job_desc.split())

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(resumes + [cleaned_job_desc])
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    results = []
    for name, sim, resume_text in zip(names, similarities, resumes):
        resume_words = set(resume_text.split())
        matched, missing = split_skills(job_words, resume_words)

        total_required = len(matched) + len(missing)
        skill_score = 100 if total_required == 0 else len(matched) / total_required * 100
        tfidf_percent = sim * 100
        hybrid = 0.7 * skill_score + 0.3 * tfidf_percent

        results.append(
            {
                "name": name,
                "score": hybrid,
                "skill_score": skill_score,
                "tfidf_percent": tfidf_percent,
                "experience": extract_experience(resume_text),
                "matched": matched,
                "missing": missing,
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)

    # ---------------- Summary ----------------
    st.write("")
    st.markdown("## Results")

    meets_exp = sum(r["experience"] >= experience_required for r in results)
    avg_score = sum(r["score"] for r in results) / len(results)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Resumes analyzed", len(results))
    m2.metric("Top score", f"{results[0]['score']:.0f}%")
    m3.metric("Average score", f"{avg_score:.0f}%")
    m4.metric("Meet experience", f"{meets_exp} of {len(results)}")

    # ---------------- Overview table ----------------
    table = pd.DataFrame(
        {
            "Rank": range(1, len(results) + 1),
            "Candidate": [r["name"] for r in results],
            "Score": [round(r["score"], 1) for r in results],
            "Skill match %": [round(r["skill_score"], 1) for r in results],
            "Text similarity %": [round(r["tfidf_percent"], 1) for r in results],
            "Experience (yrs)": [r["experience"] for r in results],
        }
    )

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.0f%%"
            ),
        },
    )

    st.download_button(
        "Download results (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="resume_ranking.csv",
        mime="text/csv",
    )

    # ---------------- Candidate cards ----------------
    st.write("")
    st.markdown("### Candidate details")

    for rank, r in enumerate(results, 1):
        label, color = verdict_for(r["score"])
        pct = round(r["score"])

        with st.container(border=True):
            st.markdown(
                f'<div class="cand">'
                f'<div class="ring" style="--p:{min(pct, 100)};--c:{color}"><span>{pct}%</span></div>'
                f"<div>"
                f'<div class="cand-rank">Rank {rank}</div>'
                f'<div class="cand-name">{escape(r["name"])}</div>'
                f'<span class="pill" style="--c:{color}">{label}</span>'
                f"</div></div>",
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Skill match", f"{r['skill_score']:.0f}%")
            c2.metric("Text similarity", f"{r['tfidf_percent']:.0f}%")
            c3.metric("Experience found", f"{r['experience']} yrs")

            if r["experience"] >= experience_required:
                exp_chip = f"Meets the {experience_required}+ year requirement"
                exp_kind = "hit"
            else:
                exp_chip = (
                    f"Below the {experience_required}+ year requirement "
                    f"(found {r['experience']})"
                )
                exp_kind = "miss"
            st.markdown(
                f'<span class="chip {exp_kind}">{escape(exp_chip)}</span>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="chips-label">Skills found</div>'
                + chips(r["matched"][:15], "hit", "No required skills found"),
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="chips-label">Skills missing</div>'
                + chips(r["missing"][:15], "miss", "Nothing missing"),
                unsafe_allow_html=True,
            )

            with st.expander("How this score is calculated"):
                st.write(
                    f"Skill match {r['skill_score']:.1f}% × 0.7 + "
                    f"text similarity {r['tfidf_percent']:.1f}% × 0.3 "
                    f"= **{r['score']:.1f}%**"
                )
