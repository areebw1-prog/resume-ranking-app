import streamlit as st
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Ignore common meaningless words
ignore_words = {
    "experience",
    "work",
    "worked",
    "team",
    "project",
    "projects",
    "using",
    "used",
    "developer",
    "application",
    "system",
    "data",
    "year",
    "years"
}
skills_list = {

    # Programming Languages
    "python", "java", "c", "cpp", "csharp", "javascript",
    "typescript", "php", "ruby", "swift", "kotlin",
    "go", "rust", "r", "matlab",

    # Web Development
    "html", "css", "react", "angular", "vue",
    "nodejs", "express", "django", "flask",
    "streamlit", "bootstrap", "tailwind",

    # Databases
    "sql", "mysql", "postgresql", "mongodb",
    "sqlite", "oracle", "firebase",

    # Data Science / AI
    "machine", "learning", "deep", "tensorflow",
    "keras", "pytorch", "numpy", "pandas",
    "matplotlib", "seaborn", "opencv",
    "nlp", "scikit", "sklearn",

    # Cloud / DevOps
    "aws", "azure", "gcp", "docker",
    "kubernetes", "jenkins", "linux",
    "git", "github", "gitlab",

    # BI / Analytics
    "powerbi", "tableau", "excel",
    "analytics", "visualization",

    # Mobile Development
    "android", "ios", "flutter",
    "reactnative",

    # Cybersecurity
    "cybersecurity", "penetration",
    "testing", "networking",

    # Software Engineering
    "oop", "api", "rest", "microservices",

    # General Tech
    "ai", "ml", "automation",
    "cloud", "devops"
}

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""

    reader = PdfReader(file)

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


# Function to clean text
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text

def extract_experience(text):

    matches = re.findall(r'(\d+)\s+years', text)

    if matches:
        numbers = [int(num) for num in matches]
        return max(numbers)

    return 0

def calculate_skill_match_score(job_desc_text, resume_text, skills_set):
    """Calculate what percentage of required skills are present in resume"""
    
    # Extract skills that appear in job description
    job_lower = job_desc_text.lower()
    required_skills_found = []
    
    for skill in skills_set:
        if skill in job_lower:
            required_skills_found.append(skill)
    
    if not required_skills_found:
        return 100  # No specific skills required in JD
    
    # Count how many required skills are in resume
    resume_lower = resume_text.lower()
    matched_count = 0
    
    for skill in required_skills_found:
        # Use word boundary to avoid partial matches (e.g., "java" in "javascript")
        if re.search(r'\b' + re.escape(skill) + r'\b', resume_lower):
            matched_count += 1
    
    skill_score = (matched_count / len(required_skills_found)) * 100
    return skill_score

# Page Title
st.markdown(
    "<h1 style='color:red; text-align:center;'>AI RESUME SCREENING SYSTEM</h1>",
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# Job Description
st.markdown("<h3>📌 JOB DESCRIPTION</h3>", unsafe_allow_html=True)

# Job Role
job_role = st.text_input("💼 Job Role")

# Required Skills
required_skills = st.text_area(
    "🛠 Required Skills",
    placeholder="Example: Python, SQL, Machine Learning, AWS"
)

# Qualifications
qualifications = st.text_area(
    "🎓 Qualifications",
    placeholder="Example: B.Tech in Computer Science, 2+ years experience"
)

experience_required = st.number_input(
    "📅 Minimum Experience Required (Years)",
    min_value=0,
    max_value=50,
    value=0
)

# Full Job Description
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

# Upload Resume PDFs
st.markdown("<h3>📂 UPLOAD RESUMES</h3>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Select PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# Success Message
if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

# Center Button
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    rank_button = st.button(
        "🔍 RANK CANDIDATES",
        use_container_width=True
    )

# Ranking Logic
if rank_button:

    # Validation
    if not job_desc.strip():
        st.error("❌ Please enter job description")

    elif not uploaded_files:
        st.error("❌ Please upload resumes")

    else:
        st.info("Processing resumes... Please wait")

        resumes = []
        names = []

        # Extract Resume Text
        for file in uploaded_files:

            try:
                text = extract_text_from_pdf(file)
                text = clean_text(text)

                if text.strip():
                    resumes.append(text)
                    names.append(file.name)

            except Exception:
                st.warning(f"Could not read {file.name}")

        # Check if resumes contain text
        if len(resumes) == 0:
            st.error("❌ No readable text found in resumes")

        else:

            # Combine resumes + job description
            cleaned_job_desc = clean_text(job_desc)

            documents = resumes + [cleaned_job_desc]

            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(stop_words="english")

            tfidf_matrix = vectorizer.fit_transform(documents)

            # Cosine Similarity
            scores = cosine_similarity(
                tfidf_matrix[-1],
                tfidf_matrix[:-1]
            )

            # Store Results with Hybrid Scoring
            results = []
            for i, (name, tfidf_score, resume_text) in enumerate(zip(names, scores[0], resumes)):
                
                # Calculate skill match percentage
                skill_score = calculate_skill_match_score(cleaned_job_desc, resume_text, skills_list)
                
                # Convert TF-IDF to percentage
                tfidf_percent = tfidf_score * 100
                
                # Hybrid: 70% skill match + 30% TF-IDF
                hybrid_score = (0.7 * skill_score) + (0.3 * tfidf_percent)
                
                results.append({
                    'name': name,
                    'tfidf_score': tfidf_score,
                    'tfidf_percent': tfidf_percent,
                    'skill_score': skill_score,
                    'hybrid_score': hybrid_score,
                    'resume_text': resume_text
                })
            
            # Sort by hybrid score descending
            results.sort(key=lambda x: x['hybrid_score'], reverse=True)

            # Display Results
            st.markdown(
                "<h2>🏆 RANKING RESULTS</h2>",
                unsafe_allow_html=True
            )

            st.markdown("<hr>", unsafe_allow_html=True)
            top_score = round(results[0]['hybrid_score'], 2)

            st.info(f"""
            📄 Total Resumes Analyzed: {len(results)}
            🏆 Highest Match Score: {top_score}%
            """)

            for i, result in enumerate(results, 1):
                
                # Extract values from result dictionary
                name = result['name']
                hybrid_score = result['hybrid_score']
                tfidf_percent = result['tfidf_percent']
                skill_score = result['skill_score']
                resume_text = result['resume_text']
                
                percent = round(hybrid_score, 2)
                
                candidate_experience = extract_experience(resume_text)

                job_words = set(cleaned_job_desc.split())

                resume_words = set(resume_text.split())

                matched_words = job_words.intersection(resume_words)

                matched_words = [
                    word for word in matched_words
                    if word not in ignore_words and len(word) > 2
                    ]

                top_matches = [
                    word for word in matched_words
                    if word in skills_list
                ][:10]

                missing_skills = [
                    skill for skill in skills_list
                    if skill in job_words and skill not in resume_words
                ][:10]

                st.markdown(f"""
                <div style="
                    padding:15px;
                    border-radius:10px;
                    background-color:#1e1e1e;
                    border:1px solid #333;
                    margin-bottom:10px;
                ">
                    <h3 style=" color:white;">
                        🏆 #{i} — {name}
                    </h3>
                </div>
                """, unsafe_allow_html=True)

                st.write(f"**Final Match Score:** {percent}%")
                
                # Show score breakdown
                with st.expander("📊 View Score Breakdown"):
                    st.write(f"**Skill Match Score:** {round(skill_score, 2)}% (70% weight)")
                    st.write(f"**TF-IDF Similarity:** {round(tfidf_percent, 2)}% (30% weight)")
                    st.write(f"**Formula:** (0.7 × {round(skill_score, 2)}) + (0.3 × {round(tfidf_percent, 2)}) = {percent}%")

                st.write(f"**Experience Detected:** {candidate_experience} years")

                if candidate_experience >= experience_required:
                    st.success("✅ Experience Requirement Met")
                else:
                    st.warning("⚠️ Experience Requirement Not Met")

                st.write("**Top Matching Keywords:**")

                if top_matches:
                    st.write(", ".join(top_matches))
                else:
                    st.write("No major skill matches found")

                st.write("**Missing Skills:**")

                if missing_skills:
                    st.write(", ".join(missing_skills))
                else:
                    st.write("No major missing skills")

                if percent >= 80:
                    st.success("Almost Perfect Match")
                elif percent >= 60:
                    st.success("Excellent Match")
                elif percent >= 30:
                    st.info("Good Match")
                elif percent >= 10:
                    st.warning("Average Match")
                else:
                    st.error("Low Match")

                # Progress Bar (using hybrid_score as decimal 0-1)
                st.progress(hybrid_score / 100)

                st.markdown("<hr>", unsafe_allow_html=True)
