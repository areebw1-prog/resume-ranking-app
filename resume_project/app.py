import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Screening System")

job_desc = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader("Upload Resumes", accept_multiple_files=True)

if st.button("Rank Candidates"):

    resumes = []
    names = []

    for file in uploaded_files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        resumes.append(text)
        names.append(file.name)

    documents = resumes + [job_desc]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    results = list(zip(names, scores[0]))
    results.sort(key=lambda x: x[1], reverse=True)

    st.subheader("Ranking Results")

    for r in results:
        st.write(r[0], "Score:", round(r[1], 3))