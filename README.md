# Q-Lens — Repeat & Rare Question Classifier

An AI/ML web app that extracts questions from PDF, Word, and image files
and classifies them as repeated or unique using NLP and cosine similarity.

## Technologies
Python, Flask, NLTK, scikit-learn, pdfplumber, python-docx, Tesseract OCR, OpenCV
----------------------------------------------------------------------------------------------------
## How to run
pip install -r requirements.txt
python app.py

## Features
- Supports PDF, DOCX, and image uploads
- OCR for scanned question papers
- TF-IDF + Jaccard hybrid similarity scoring
- Exports results as CSV files
## Installation

### 1. Install Python dependencies
pip install -r requirements.txt

### 2. Install Tesseract OCR
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Mac: brew install tesseract
- Linux: sudo apt install tesseract-ocr

### 3. Run the app
python app.py
Then open http://127.0.0.1:5000 in your browser
