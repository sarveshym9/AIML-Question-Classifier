import sys
import pdfplumber
import docx
import pytesseract
import cv2
from PIL import Image


# ── Tesseract path (Windows only) ────────────────────────────────────────────
# On Linux/macOS tesseract is on PATH after `brew install tesseract` or
# `sudo apt install tesseract-ocr`, so no manual path is needed.
# On Windows, point this to your installation:
if sys.platform == "win32":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
# ─────────────────────────────────────────────────────────────────────────────


def extract_text_from_pdf(filepath: str) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"PDF extraction error ({filepath}): {e}")
    return text


def extract_text_from_docx(filepath: str) -> str:
    """Extract paragraph text from a Word (.docx) file."""
    text = ""
    try:
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"DOCX extraction error ({filepath}): {e}")
    return text


def extract_text_from_image(filepath: str) -> str:
    """
    Extract text from an image using OpenCV preprocessing + Tesseract OCR.
    Pipeline: load → grayscale → denoise → threshold → OCR
    """
    try:
        image = cv2.imread(filepath)
        if image is None:
            print(f"Image load error: could not read {filepath}")
            return ""

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Otsu thresholding (adaptive, better than fixed 150)
        _, thresh = cv2.threshold(
            denoised, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        pil_image = Image.fromarray(thresh)

        # PSM 6 = assume a uniform block of text (good for question papers)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(pil_image, config=custom_config)

    except Exception as e:
        print(f"OCR error ({filepath}): {e}")
        return ""

    return text
