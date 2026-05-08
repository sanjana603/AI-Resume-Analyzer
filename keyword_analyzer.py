def extract_skills(text):

    skills = [
        "python",
        "sql",
        "machine learning",
        "flask",
        "streamlit",
        "pandas",
        "numpy",
        "java",
        "html",
        "css"
    ]

    found_skills = []

    for skill in skills:

        if skill in text.lower():

            found_skills.append(skill)

    return found_skills


def get_missing_keywords(resume, job_description):

    resume_words = set(resume.split())

    jd_words = set(job_description.split())

    missing = jd_words - resume_words

    return list(missing)[:15]