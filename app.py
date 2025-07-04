from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from analyzer import run_analysis
import os

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.csv')
    file.save(filepath)
    return jsonify({'message': 'File uploaded successfully'})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        params = request.json
        print("Received parameters:", params)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.csv')
        result = run_analysis(params, filepath)
        print("Analysis result keys:", result.keys())
        return jsonify(result)
    except Exception as e:
        print("Error in /analyze:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
