import re
import PyPDF2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        return text

    except Exception as e:
        return ""


def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text):

    stop_words = set(stopwords.words('english'))

    words = word_tokenize(text)

    filtered_words = [word for word in words if word not in stop_words]

    return " ".join(filtered_words)


def calculate_similarity(resume_text, job_description):

    resume_processed = remove_stopwords(clean_text(resume_text))

    job_processed = remove_stopwords(clean_text(job_description))

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        [resume_processed, job_processed]
    )

    score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0] * 100

    return round(score, 2), resume_processed, job_processed