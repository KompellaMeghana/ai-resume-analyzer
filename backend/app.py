from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

@app.route("/")
def home():
    return "AI Resume Analyzer Backend Running"

@app.route("/analyze", methods=["POST"])
def analyze():
    resume_file = request.files["resume"]
    job_description = request.form["job_description"]

    resume_text = extract_text_from_pdf(resume_file)

    documents = [resume_text, job_description]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score = round(float(similarity[0][0]) * 100, 2)

    return jsonify({"match_score": score})

if __name__ == "__main__":
    app.run(debug=True)