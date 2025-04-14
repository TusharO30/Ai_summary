from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
from PIL import Image
import base64
import io
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv()
app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check-auth")
def check_auth():
    try:
        genai.list_models()
        return jsonify({"status": "OK", "message": "Authentication successful"})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 500

@app.route("/api/extract-text", methods=["POST"])
def extract_text():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400

    try:
        pdf_file = request.files['pdf']
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()

        if not text.strip():
            return jsonify({"error": "No text extracted from PDF"}), 400

        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/extract-images", methods=["POST"])
def extract_images():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400

    try:
        pdf_file = request.files['pdf']
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        images = []
        ocr_texts = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                images.append({
                    "page": page_index + 1,
                    "image_index": img_index + 1,
                    "ext": image_ext,
                    "base64": encoded_image
                })

                image = Image.open(io.BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(image)
                if ocr_text.strip():
                    ocr_texts.append(ocr_text.strip())

        return jsonify({"images": images, "ocr_text": "\n".join(ocr_texts)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "").strip()
    ocr_text = data.get("ocr_text", "").strip()
    length = data.get("length", "medium")

    combined_text = f"{text}\n\n[Text extracted from graphs/images:]\n{ocr_text}".strip()

    if len(combined_text) < 100:
        return jsonify({"error": "Combined text must be ≥100 characters"}), 400

    prompt = (
        "Summarize this research paper in bullet points focusing on key contributions, "
        "methodology, and results. Use academic language. "
    )

    if length == "short":
        prompt += "Make the summary concise (3-4 bullets)."
    elif length == "long":
        prompt += "Make the summary detailed and comprehensive (10+ bullets)."
    else:
        prompt += "Keep the summary medium in length (5-7 bullets)."

    prompt += f"\n\nHere's the text:\n\n{combined_text}"

    try:
        response = model.generate_content(prompt)
        return jsonify({"summary": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
