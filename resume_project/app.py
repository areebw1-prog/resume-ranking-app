import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# This should change the title color and add a big header
st.markdown("<h1 style='color: red;'>AI RESUME SCREENING SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<h3>📌 JOB DESCRIPTION</h3>", unsafe_allow_html=True)
job_desc = st.text_area("", height=150)

st.markdown("<h3>📂 UPLOAD RESUMES</h3>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("Select PDF files", type="pdf", accept_multiple_files=True)

# Show uploaded files
if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    rank_button = st.button("🔍 RANK CANDIDATES", use_container_width=True)

if rank_button:
    if not job_desc:
        st.error("❌ Please enter job description")
    elif not uploaded_files:
        st.error("❌ Please upload resumes")
    else:
        st.info("Processing... Please wait")
        
        resumes = []
        names = []
        
        for file in uploaded_files:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            if text.strip():
                resumes.append(text)
                names.append(file.name)
        
        if len(resumes) == 0:
            st.error("No text found in resumes")
        else:
            documents = resumes + [job_desc]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents)
            scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
            
            results = list(zip(names, scores[0]))
            results.sort(key=lambda x: x[1], reverse=True)
            
            st.markdown("<h2>🏆 RANKING RESULTS</h2>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            
            for i, (name, score) in enumerate(results, 1):
                st.markdown(f"### {i}. {name}")
                st.write(f"**Match Score:** `{round(score, 3)}`")
                # This creates a simple progress bar using HTML
                bar_width = int(score * 100)
                st.markdown(f"""
                <div style="background-color:#e0e0e0; border-radius:10px; height:20px; width:100%;">
                    <div style="background-color:#4CAF50; border-radius:10px; height:20px; width:{bar_width}%;"></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
