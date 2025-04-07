from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv() 
app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')
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
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        
        if not text.strip():
            return jsonify({"error": "No text extracted from PDF"}), 400
            
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "").strip()
    
    if len(text) < 100:
        return jsonify({"error": "Text must be ≥100 characters"}), 400
    
    try:
        response = model.generate_content(
            f"Summarize this research paper in bullet points focusing on key contributions, "
            f"methodology, and results. Use academic language. Here's the text:\n\n{text}"
        )
        
        return jsonify({"summary": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)