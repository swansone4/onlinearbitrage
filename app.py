from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from analyzer import run_analysis
import os
import json
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup


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

@app.route('/presets', methods=['GET'])
def get_presets():
    try:
        presets_file = os.path.join(app.config['UPLOAD_FOLDER'], 'custom_presets.json')
        if os.path.exists(presets_file):
            with open(presets_file, 'r') as f:
                presets = json.load(f)
        else:
            presets = {}
        return jsonify(presets)
    except Exception as e:
        print("Error getting presets:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/presets', methods=['POST'])
def save_preset():
    try:
        data = request.json
        preset_name = data.get('name')
        preset_data = data.get('data')
        
        if not preset_name or not preset_data:
            return jsonify({'error': 'Missing preset name or data'}), 400
        
        presets_file = os.path.join(app.config['UPLOAD_FOLDER'], 'custom_presets.json')
        
        # Load existing presets
        if os.path.exists(presets_file):
            with open(presets_file, 'r') as f:
                presets = json.load(f)
        else:
            presets = {}
        
        # Save new preset
        presets[preset_name] = preset_data
        
        # Write back to file
        with open(presets_file, 'w') as f:
            json.dump(presets, f, indent=2)
        
        return jsonify({'message': 'Preset saved successfully'})
    except Exception as e:
        print("Error saving preset:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/presets/<preset_name>', methods=['DELETE'])
def delete_preset(preset_name):
    try:
        presets_file = os.path.join(app.config['UPLOAD_FOLDER'], 'custom_presets.json')
        
        if not os.path.exists(presets_file):
            return jsonify({'error': 'No presets file found'}), 404
        
        with open(presets_file, 'r') as f:
            presets = json.load(f)
        
        if preset_name not in presets:
            return jsonify({'error': 'Preset not found'}), 404
        
        # Delete the preset
        del presets[preset_name]
        
        # Write back to file
        with open(presets_file, 'w') as f:
            json.dump(presets, f, indent=2)
        
        return jsonify({'message': 'Preset deleted successfully'})
    except Exception as e:
        print("Error deleting preset:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/proxy')
def proxy():
    """Proxy endpoint to fetch external websites for iframe display"""
    try:
        url = request.args.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return jsonify({'error': 'Invalid URL'}), 400
        
        # Fetch the external website
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"Fetching URL: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        print(f"Response size: {len(response.content)} bytes")
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove problematic elements that could cause issues in iframe
        for script in soup.find_all('script'):
            script.decompose()
        
        for iframe in soup.find_all('iframe'):
            iframe.decompose()
        
        # Update relative URLs to absolute URLs
        for tag in soup.find_all(['a', 'img', 'link']):
            for attr in ['href', 'src']:
                if tag.get(attr):
                    if not tag[attr].startswith(('http://', 'https://', '//')):
                        tag[attr] = urljoin(url, tag[attr])
        
        # Add base tag to ensure relative URLs work
        base_tag = soup.new_tag('base', href=url)
        if soup.head:
            soup.head.insert(0, base_tag)
        elif soup.html:
            soup.html.insert(0, base_tag)
        
        # Add some basic styling to make it look better in iframe
        style_tag = soup.new_tag('style')
        style_tag.string = '''
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 10px; 
                background: white; 
            }
            .iframe-container { 
                max-width: 100%; 
                overflow-x: auto; 
            }
        '''
        if soup.head:
            soup.head.append(style_tag)
        elif soup.html:
            soup.html.insert(0, style_tag)
        
        # Create a simplified version for iframe display
        simplified_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>External Site: {url}</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background: white; 
                    color: #333;
                }}
                .header {{
                    background: #f8f9fa;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                }}
                .content {{
                    max-width: 100%;
                    overflow-x: auto;
                }}
                .external-link {{
                    display: inline-block;
                    background: #007bff;
                    color: white;
                    padding: 10px 20px;
                    text-decoration: none;
                    border-radius: 5px;
                    margin-top: 15px;
                }}
                .external-link:hover {{
                    background: #0056b3;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>External Website Preview</h2>
                <p><strong>URL:</strong> {url}</p>
                <p><strong>Status:</strong> ✅ Successfully loaded via proxy</p>
            </div>
            <div class="content">
                <p>This is a simplified preview of the external website. The full content has been processed for iframe display.</p>
                <p><strong>Original response size:</strong> {len(response.content)} bytes</p>
                <a href="{url}" target="_blank" class="external-link">Open Original Site in New Tab</a>
            </div>
        </body>
        </html>
        '''
        
        return simplified_html, 200, {'Content-Type': 'text/html'}
        
    except requests.exceptions.RequestException as e:
        print(f"Proxy error for {url}: {e}")
        return jsonify({'error': f'Failed to fetch URL: {str(e)}'}), 500
    except Exception as e:
        print(f"Unexpected proxy error: {e}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500



@app.route('/uploads/buy_list.csv')
def download_buy_list():
    try:
        buy_list_path = os.path.join(app.config['UPLOAD_FOLDER'], 'buy_list.csv')
        if os.path.exists(buy_list_path):
            return send_from_directory(app.config['UPLOAD_FOLDER'], 'buy_list.csv', as_attachment=True)
        else:
            return jsonify({'error': 'Buy list not found'}), 404
    except Exception as e:
        print("Error serving buy list:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
