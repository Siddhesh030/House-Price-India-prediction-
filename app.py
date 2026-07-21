import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the model safely using pickle
MODEL_PATH = "model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print(f"Warning: {MODEL_PATH} not found. Place your model file in the same directory.")

# HTML Template with beautiful CSS glassmorphism UI & micro-interactions
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Poppins', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            overflow-x: hidden;
        }

        /* Ambient floating background animations */
        .orb {
            position: absolute;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(79, 70, 229, 0.3) 0%, rgba(0,0,0,0) 70%);
            border-radius: 50%;
            z-index: 0;
            animation: float Orb 10s ease-in-out infinite alternate;
        }
        .orb-1 { top: -10%; left: -10%; }
        .orb-2 { bottom: -10%; right: -10%; animation-delay: -5s; }

        @keyframes floatOrb {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(50px, 50px) scale(1.1); }
        }

        .container {
            position: relative;
            z-index: 1;
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2.5rem;
            width: 100%;
            max-width: 900px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1 {
            text-align: center;
            font-weight: 600;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #6366f1, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-bottom: 2.5rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
            font-weight: 400;
            transition: color 0.3s;
        }

        input, select {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 1rem;
            outline: none;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.3);
            background: rgba(255, 255, 255, 0.12);
        }

        .input-group:focus-within label {
            color: #6366f1;
        }

        .btn-container {
            grid-column: 1 / -1;
            margin-top: 1.5rem;
            text-align: center;
        }

        button {
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 1rem 3rem;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
        }

        button:hover {
            background: var(--primary-hover);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
        }

        button:active {
            transform: translateY(1px);
        }

        .result-box {
            margin-top: 2.5rem;
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.2);
            text-align: center;
            display: {% if prediction %} block {% else %} none {% endif %};
            animation: scaleUp 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        @keyframes scaleUp {
            from { transform: scale(0.9); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }

        .result-title {
            font-size: 1rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .result-value {
            font-size: 2.5rem;
            font-weight: 600;
            color: #34d399;
            margin-top: 0.5rem;
            text-shadow: 0 0 20px rgba(52, 211, 153, 0.3);
        }
    </style>
</head>
<body>
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>

    <div class="container">
        <h1>House Value Estimator</h1>
        <p class="subtitle">Enter the metrics below to get an instant ML prediction model evaluation.</p>
        
        <form method="POST" action="/predict" class="grid-form">
            <!-- Dynamically generating the exact 16 inputs derived from your pickle file layout -->
            <div class="input-group">
                <label>Number of Bedrooms</label>
                <input type="number" name="bedrooms" min="0" value="{{inputs.get('bedrooms', 3)}}" required>
            </div>
            <div class="input-group">
                <label>Number of Bathrooms</label>
                <input type="number" step="0.1" name="bathrooms" min="0" value="{{inputs.get('bathrooms', 2)}}" required>
            </div>
            <div class="input-group">
                <label>Living Area (sqft)</label>
                <input type="number" name="living_area" min="0" value="{{inputs.get('living_area', 1800)}}" required>
            </div>
            <div class="input-group">
                <label>Lot Area (sqft)</label>
                <input type="number" name="lot_area" min="0" value="{{inputs.get('lot_area', 5000)}}" required>
            </div>
            <div class="input-group">
                <label>Number of Floors</label>
                <input type="number" step="0.5" name="floors" min="1" value="{{inputs.get('floors', 1)}}" required>
            </div>
            <div class="input-group">
                <label>Waterfront Present</label>
                <select name="waterfront">
                    <option value="0" {% if inputs.get('waterfront') == '0' %}selected{% endif %}>No</option>
                    <option value="1" {% if inputs.get('waterfront') == '1' %}selected{% endif %}>Yes</option>
                </select>
            </div>
            <div class="input-group">
                <label>Number of Views</label>
                <input type="number" name="views" min="0" max="4" value="{{inputs.get('views', 0)}}" required>
            </div>
            <div class="input-group">
                <label>Condition of the House</label>
                <input type="number" name="condition" min="1" max="5" value="{{inputs.get('condition', 3)}}" required>
            </div>
            <div class="input-group">
                <label>Grade of the House</label>
                <input type="number" name="grade" min="1" max="13" value="{{inputs.get('grade', 7)}}" required>
            </div>
            <div class="input-group">
                <label>Area (Excl. Basement)</label>
                <input type="number" name="area_excl_basement" min="0" value="{{inputs.get('area_excl_basement', 1500)}}" required>
            </div>
            <div class="input-group">
                <label>Area of the Basement</label>
                <input type="number" name="basement_area" min="0" value="{{inputs.get('basement_area', 3000)}}" required>
            </div>
            <div class="input-group">
                <label>Built Year</label>
                <input type="number" name="built_year" min="1800" max="2026" value="{{inputs.get('built_year', 2000)}}" required>
            </div>
            <div class="input-group">
                <label>Renovation Year (0 if none)</label>
                <input type="number" name="renovation_year" min="0" max="2026" value="{{inputs.get('renovation_year', 0)}}" required>
            </div>
            <div class="input-group">
                <label>Lot Area Renovated</label>
                <input type="number" name="lot_area_renov" min="0" value="{{inputs.get('lot_area_renov', 5000)}}" required>
            </div>
            <div class="input-group">
                <label>Number of Schools Nearby</label>
                <input type="number" name="schools_nearby" min="0" max="10" value="{{inputs.get('schools_nearby', 2)}}" required>
            </div>
            <div class="input-group">
                <label>Distance from Airport</label>
                <input type="number" name="airport_dist" min="0" value="{{inputs.get('airport_dist', 15)}}" required>
            </div>

            <div class="btn-container">
                <button type="submit">Calculate Valuation</button>
            </div>
        </form>

        {% if prediction %}
        <div class="result-box">
            <div class="result-title">Estimated Property Value</div>
            <div class="result-value">{{ prediction }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, inputs={}, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model structural layout missing or not found.", 500
    
    try:
        # Grabbing and formatting features strictly matching your model layout array
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
        
        # Structure the input arrays
        final_features = [np.array(features)]
        prediction_val = model.predict(final_features)[0]
        
        # Formatted valuation output strings
        formatted_pred = f"${prediction_val:,.2f}"
        
        return render_template_string(HTML_TEMPLATE, inputs=request.form, prediction=formatted_pred)
        
    except Exception as e:
        return f"Error executing predictions: {str(e)}", 400

if __name__ == "__main__":
    # Standard fallback configuration for local serving vs cloud deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
