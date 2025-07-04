from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from analyzer import run_analysis  # <-- import your core logic

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/analyze', methods=['POST'])
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        params = request.json
        print("Received parameters:", params)
        result = run_analysis(params)
        print("Analysis result keys:", result.keys())
        return jsonify(result)
    except Exception as e:
        print("Error in /analyze:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Bind explicitly to localhost on port 5000 (default)
    app.run(debug=True, host='127.0.0.1', port=5000)

