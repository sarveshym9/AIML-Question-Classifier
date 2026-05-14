import os
from flask import Flask, render_template, request
from extractor import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_image
)
from preprocess import extract_questions, clean_text
from similarity import classify_questions

app = Flask(__name__)

UPLOAD_FOLDER = 'input_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('output', exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    repeated_questions = []
    rare_questions = []
    error_message = None

    if request.method == 'POST':
        files = request.files.getlist('files')

        original_questions = []
        cleaned_questions = []

        for file in files:
            if file.filename == '':
                continue

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            text = ""
            fname = file.filename.lower()

            if fname.endswith('.pdf'):
                text = extract_text_from_pdf(filepath)
            elif fname.endswith('.docx'):
                text = extract_text_from_docx(filepath)
            elif fname.endswith(('.jpg', '.jpeg', '.png')):
                text = extract_text_from_image(filepath)

            questions = extract_questions(text)
            for q in questions:
                cleaned = clean_text(q)
                if cleaned.strip():          # skip empty cleaned results
                    original_questions.append(q)
                    cleaned_questions.append(cleaned)

        if len(cleaned_questions) < 2:
            error_message = (
                "Please upload files containing at least 2 detectable questions."
            )
        else:
            # classify_questions returns (repeated_indexes, rare_indexes)
            repeated_idx, rare_idx = classify_questions(
                cleaned_questions, original_questions
            )
            repeated_questions = [original_questions[i] for i in repeated_idx]
            rare_questions     = [original_questions[i] for i in rare_idx]

    return render_template(
        'index.html',
        repeated_questions=repeated_questions,
        rare_questions=rare_questions,
        error_message=error_message
    )


if __name__ == '__main__':
    app.run(debug=True)
