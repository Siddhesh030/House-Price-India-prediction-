import os
import glob
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Automatically look for your pickle file, even if it's named model (2).pkl or model.pkl
def load_model():
    possible_files = glob.glob("model*.pkl")
    if possible_files:
        # Pick the first matching pickle file found
        model_path = possible_files[0]
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI House Valuator Pro</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --card-bg: rgba(255, 255, 255, 0.03);
            --border: rgba(255, 255, 255, 0.08);
            --accent: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --accent-glow: rgba(99, 102, 241, 0.15);
            --text: #f8fafc;
            --text-muted: #64748b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.15) 0px, transparent 50%);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 3rem 1.5rem;
        }

        .container {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 3rem;
            width: 100%;
            max-width: 1000px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 3rem;
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.05em;
            background: var(--accent);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 1rem;
        }

        /* Dynamic Error Display */
        .error-banner {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: #f87171;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
            font-weight: 500;
            animation: shake 0.4s linear;
        }

        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            75% { transform: translateX(5px); }
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.75rem;
        }

        .input-wrapper {
            position: relative;
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.85rem;
            font-weight: 500;
            color: #94a3b8;
            margin-bottom: 0.6rem;
            transition: color 0.2s ease;
        }

        input, select {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.85rem 1.2rem;
            color: #fff;
            font-size: 0.95rem;
            transition: all 0.25s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: #818cf8;
            background: rgba(255, 255, 255, 0.05);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
        }

        .input-wrapper:focus-within label {
            color: #a5b4fc;
        }

        .actions {
            grid-column: 1 / -1;
            display: flex;
            justify-content: center;
            margin-top: 2rem;
        }

        button {
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 1rem 3.5rem;
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px -5px rgba(99, 102, 241, 0.6);
        }

        /* Result Module Animation */
        .result-card {
            margin-top: 3rem;
            padding: 2.5rem;
            border-radius: 20px;
            background: radial-gradient(circle at top left, rgba(99, 102, 241, 0.08), transparent), var(--card-bg);
            border: 1px solid rgba(99, 102, 241, 0.2);
            text-align: center;
            box-shadow: 0 0 40px var(--accent-glow);
            animation: popIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        @keyframes popIn {
            from { transform: scale(0.92); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }

        .result-label {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #a5b4fc;
            margin-bottom: 0.5rem;
        }

        .result-amount {
            font-size: 3rem;
            font-weight: 700;
            color: #34d399;
            text-shadow: 0 0 30px rgba(52, 211, 153, 0.2);
        }
    </style>
</head>
<body>

    <div class="container">
        <div class="header">
            <h1>Property Valuation Engine</h1>
            <p class="subtitle">Provide features below to parse real-time statistical inference.</p>
        </div>

        <!-- Dynamic Warning if Pickle File completely missing -->
        {% if error_msg %}
        <div class="error-banner">
            ⚠️ {{ error_msg }}
        </div>
        {% endif %}
        
        <form method="POST" action="/predict" class="grid-form">
            <div class="input-wrapper">
                <label>Number of Bedrooms</label>
                <input type="number" name="bedrooms" value="{{inputs.get('bedrooms', 3)}}" required>
            </div>
            <div class="input-wrapper">
                <label>Number of Bathrooms</label>
                <input type="number" step="0.1" name="bathrooms" value="{{inputs.get('bathrooms', 2)}}" required>
            </div>
            <div class="input-wrapper">
                <label>Living Area (sqft)</label>
                <input type="number" name="living_area" value="{{inputs.get('living_area', 1800)}}" required>
            </div>
            <div class="input-wrapper">
                <label>Lot Area (sqft)</label>
                <input type="number" name="lot_area" value="{{inputs.get('lot_area', 5000)}}" required>
            </div>
            <div class="input-wrapper">
                <label>Number of Floors</label>
                <input type="number" step="0.5" name="floors" value="{{inputs.get('floors', 1)}}" required>
            </div>
            <div class="input-wrapper">
                <label>Waterfront Present</label>
                <select name="waterfront">
                    <option value="0" {% if inputs.get('waterfront') == '0' %}selected{% endif %}>No</option>
                    <option value="1" {% if inputs.get('waterfront') == '1' %}selected{% endif %}>Yes</option>
                </select>
            </div>
            <div class="input-wrapper">
                <label>Number of Views</label>
                <input type="number" name="views" min="0" max="4" value="{{inputs.get('views', 0)}}" required>
            </div>
            <div class="input-wrapper">
                <label>Condition Rating</label>
                <input type="number" name="condition" min="1" max="5" value="{{inputs.get('condition', 3)}}" required>
            </div>
            <div class="input-wrapper">
                <label>Grade Level</label>
                <input type="number" name="grade" min="1" max="13" value="{{inputs.get('grade', 7)}}" required>
            </div>
            <div class="input-wrapper">
                <label>Area Above Ground (sqft)</label>
                <input type="number" name="area_excl_basement" value="{{inputs.get('area_excl_basement', 1500)}}" required>
            </div>
            <div class="input-wrapper">
                <label>Basement Area (sqft)</label>
                <input type="number" name="basement_area" value="{{inputs.get('basement_area', 300)}}" required>
            </div>
            <div class="input-wrapper">
                <label>Year Built</label>
                <input type="number" name="built_year" value="{{inputs.get('built_year', 2000)}}" required>
            </div>
            <div class="input-wrapper">
                <label>Year Renovated</label>
                <input type="number" name="renovation_year" value="{{inputs.get('renovation_year', 0)}}" required>
            </div>
            <div class="input-wrapper">
                <label>Lot Area Renovated</label>
                <input type="number" name="lot_area_renov" value="{{inputs.get('lot_area_renov', 5000)}}" required>
            </div>
            <div class="input-wrapper">
                <label>Nearby Schools</label>
                <input type="number" name="schools_nearby" value="{{inputs.get('schools_nearby', 2)}}" required>
            </div>
            <div class="input-wrapper">
                <label>Airport Distance (km)</label>
                <input type="number" name="airport_dist" value="{{inputs.get('airport_dist', 15)}}" required>
            </div>

            <div class="actions">
                <button type="submit">Run Prediction Pipeline</button>
            </div>
        </form>

        {% if prediction %}
        <div class="result-card">
            <p class="result-label">Regressed Market Valuation</p>
            <p class="result-amount">{{ prediction }}</p>
        </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    error_msg = None if model else "Model file (model.pkl) not found in directory. Please upload your pickle file."
    return render_template_string(HTML_TEMPLATE, inputs={}, prediction=None, error_msg=error_msg)

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if not model:
        model = load_model()
        if not model:
            return render_template_string(HTML_TEMPLATE, inputs=request.form, prediction=None, error_msg="Prediction failed: Model file is missing.")
    
    try:
        features = [
            float(request.form.get("bedrooms")),
            float(request.form.get("bathrooms")),
            float(request.form.get("living_area")),
            float(request.form.get("lot_area")),
            float(request.form.get("floors")),
            float(request.form.get("waterfront")),
            float(request.form.get("views")),
            float(request.form.get("condition")),
            float(request.form.get("grade")),
            float(request.form.get("area_excl_basement")),
            float(request.form.get("basement_area")),
            float(request.form.get("built_year")),
            float(request.form.get("renovation_year")),
            float(request.form.get("lot_area_renov")),
            float(request.form.get("schools_nearby")),
            float(request.form.get("airport_dist"))
        ]
        
        prediction_val = model.predict([np.array(features)])[0]
        formatted_pred = f"${prediction_val:,.2f}"
        
        return render_template_string(HTML_TEMPLATE, inputs=request.form, prediction=formatted_pred, error_msg=None)
        
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, inputs=request.form, prediction=None, error_msg=f"Data processing error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
