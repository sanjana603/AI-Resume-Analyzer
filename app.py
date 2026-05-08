import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# -----------------------------
# NLTK Downloads
# -----------------------------
nltk.download("punkt")
nltk.download("stopwords")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# LOAD CSS
# -----------------------------
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<h1 style='text-align: center; color: white;'>
📄 AI Resume Analyzer
</h1>

<h4 style='text-align: center; color: #9CA3AF; text-align:center;'>
Smart ATS Resume Screening System using NLP & AI
</h4>
""", unsafe_allow_html=True)

# -----------------------------
# HERO SECTION
# -----------------------------
st.info("""
🚀 Analyze resumes instantly using AI-powered ATS scoring,
keyword extraction, and job description matching.
""")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.header("📌 Project Overview")

    st.info("""
This AI Resume Analyzer compares resumes with job descriptions using:

- NLP Techniques
- TF-IDF Vectorization
- Cosine Similarity
- Keyword Matching
- ATS-style Resume Analysis
""")

    st.header("⚙️ How It Works")

    st.write("""
1. Upload Resume PDF
2. Paste Job Description
3. Click Analyze Resume
4. View ATS Match Score
5. Check Missing Skills
""")

    st.header("✨ Features")

    st.write("""
✅ ATS Resume Score  
✅ Keyword Analysis  
✅ Missing Skills Detection  
✅ Resume Suggestions  
✅ Dashboard Visualization  
""")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def extract_text_from_pdf(uploaded_file):

    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        return text

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text):

    stop_words = set(stopwords.words('english'))

    words = word_tokenize(text)

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    return filtered_words


def calculate_similarity(resume_text, job_description):

    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform([
        resume_clean,
        jd_clean
    ])

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0] * 100

    return round(similarity, 2)


def extract_keywords(text, top_n=15):

    words = remove_stopwords(clean_text(text))

    freq = Counter(words)

    common_words = freq.most_common(top_n)

    return [word for word, count in common_words]


def get_missing_skills(resume_keywords, jd_keywords):

    missing = []

    for word in jd_keywords:

        if word not in resume_keywords:
            missing.append(word)

    return missing

# -----------------------------
# MAIN SECTION
# -----------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    " Paste Job Description",
    height=250
)

# -----------------------------
# CENTER BUTTON
# -----------------------------

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    analyze = st.button(" Analyze Resume")

# -----------------------------
# ANALYSIS
# -----------------------------

if analyze:

    if not uploaded_file:
        st.warning("Please upload your resume.")
        st.stop()

    if not job_description:
        st.warning("Please paste the job description.")
        st.stop()

    with st.spinner("Analyzing Resume..."):

        resume_text = extract_text_from_pdf(uploaded_file)

        if not resume_text:
            st.error("Could not extract text from PDF.")
            st.stop()

        # Similarity Score
        similarity_score = calculate_similarity(
            resume_text,
            job_description
        )

        # Keywords
        resume_keywords = extract_keywords(resume_text)
        jd_keywords = extract_keywords(job_description)

        # Missing Skills
        missing_skills = get_missing_skills(
            resume_keywords,
            jd_keywords
        )

        # -----------------------------
        # DASHBOARD
        # -----------------------------

        st.divider()

        st.markdown("## 📊 ATS Analysis Dashboard")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "ATS Score",
                f"{similarity_score}%"
            )

        with col2:
            st.metric(
                "Resume Keywords",
                len(resume_keywords)
            )

        with col3:
            st.metric(
                "Missing Skills",
                len(missing_skills)
            )

        # -----------------------------
        # PROGRESS BAR
        # -----------------------------

        st.progress(int(similarity_score))

        # -----------------------------
        # MATCH CHART
        # -----------------------------

        fig, ax = plt.subplots(figsize=(7, 1))

        if similarity_score < 40:
            color = "#ff4b4b"

        elif similarity_score < 70:
            color = "#ffa726"

        else:
            color = "#0f9d58"

        ax.barh([0], [similarity_score], color=color)

        ax.set_xlim(0, 100)

        ax.set_yticks([])

        ax.set_xlabel("ATS Match Percentage")

        ax.set_title("Resume vs Job Description Match")

        st.pyplot(fig)

        # -----------------------------
        # FEEDBACK
        # -----------------------------

        st.subheader(" Resume Feedback")

        if similarity_score < 40:

            st.error("""
Your resume has a low match with the job description.

Suggestions:
- Add more technical keywords
- Tailor resume for the specific role
- Include relevant projects & skills
""")

        elif similarity_score < 70:

            st.warning("""
Good match, but improvements can be made.

Suggestions:
- Add missing skills
- Improve project descriptions
- Include measurable achievements
""")

        else:

            st.success("""
Excellent match!

Your resume strongly aligns with the job requirements.
""")

        # -----------------------------
        # KEYWORDS
        # -----------------------------

        st.subheader(" Top Resume Keywords")

        st.write(", ".join(resume_keywords))

        # -----------------------------
        # MISSING SKILLS
        # -----------------------------

        st.subheader(" Missing Skills")

        if missing_skills:

            st.write(", ".join(missing_skills))

        else:

            st.success("No major missing skills detected.")

        # -----------------------------
        # FINAL RECOMMENDATIONS
        # -----------------------------

        st.subheader(" Final Recommendations")

        st.write("""
- Use role-specific keywords
- Add certifications if available
- Keep resume ATS-friendly
- Include projects with measurable impact
- Use proper formatting and clean sections
""")

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

st.markdown(
    "<center>Made with  using Streamlit & NLP</center>",
    unsafe_allow_html=True
)