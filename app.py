import streamlit as st
import pandas as pd
import joblib
import os
import time

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AutoValue — Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  —  modern glassmorphic dark UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
      radial-gradient(ellipse 80% 50% at 20% 0%, rgba(99,102,241,0.18) 0%, transparent 50%),
      radial-gradient(ellipse 60% 50% at 80% 10%, rgba(56,189,248,0.12) 0%, transparent 50%),
      radial-gradient(ellipse 50% 40% at 50% 100%, rgba(168,85,247,0.10) 0%, transparent 60%),
      #07080c;
    color: #e6e8ec;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1180px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, rgba(20,22,32,0.7) 0%, rgba(15,18,28,0.55) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 24px;
    padding: 2.75rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px -20px rgba(0,0,0,0.5);
}

.hero::before {
    content:''; position:absolute; top:-80px; right:-80px;
    width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, rgba(99,102,241,0.35) 0%, transparent 70%);
    filter: blur(20px);
}
.hero::after {
    content:''; position:absolute; bottom:-60px; left:30%;
    width: 200px; height: 200px; border-radius: 50%;
    background: radial-gradient(circle, rgba(56,189,248,0.25) 0%, transparent 70%);
    filter: blur(15px);
}
.hero-content { position: relative; z-index: 2; }
.hero-tag {
    display: inline-flex; align-items:center; gap:6px;
    background: rgba(99,102,241,0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 999px;
    font-size: 11px; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; padding: 6px 14px; margin-bottom: 1.25rem;
}
.hero-tag-dot { width:6px; height:6px; border-radius:50%; background:#a5b4fc;
    box-shadow: 0 0 10px #a5b4fc; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem; font-weight: 700;
    color: #f5f7fa; letter-spacing: -2px; line-height: 1.05;
    margin-bottom: 1rem;
}
.hero h1 .grad {
    background: linear-gradient(135deg, #818cf8 0%, #38bdf8 50%, #a78bfa 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { font-size: 16px; color: #9ca3b0; max-width: 560px; line-height: 1.7; }

.hero-stats {
    display: flex; gap: 2.5rem; margin-top: 2rem;
    padding-top: 1.75rem; border-top: 1px solid rgba(255,255,255,0.06);
}
.hero-stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem; font-weight: 700; color: #f5f7fa;
}
.hero-stat-label { font-size: 12px; color: #6b7280; margin-top: 4px; letter-spacing: 0.5px; }

/* ── Section header ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 1.25rem; margin-top: 0.5rem;
}
.section-dot {
    width: 6px; height: 24px; border-radius: 3px;
    background: linear-gradient(180deg, #818cf8, #38bdf8);
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14px; font-weight: 600; letter-spacing: 0.5px;
    color: #f5f7fa;
}
.section-sub { font-size: 12px; color: #6b7280; margin-left: auto; }

/* ── Card panels ── */
.card {
    background: rgba(20,22,32,0.55);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
    transition: all 0.25s ease;
}
.card:hover { border-color: rgba(129,140,248,0.25); transform: translateY(-1px); }

/* ── Widget labels ── */
.stNumberInput > label, .stSelectbox > label, .stSlider > label {
    font-size: 11px !important; font-weight: 600 !important;
    color: #9ca3b0 !important; text-transform: uppercase !important;
    letter-spacing: 0.8px !important; margin-bottom: 6px !important;
}

/* ── Inputs ── */
.stNumberInput input, .stSelectbox > div > div {
    background: rgba(15,17,25,0.6) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #f5f7fa !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
}
.stNumberInput input:focus, .stSelectbox > div > div:focus-within {
    border-color: rgba(129,140,248,0.6) !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.15) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] > div > div { background: linear-gradient(90deg,#818cf8,#38bdf8) !important; }

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #38bdf8 100%) !important;
    background-size: 200% 200% !important;
    color: white !important; border: none !important;
    border-radius: 14px !important;
    padding: 0.95rem 2rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important; font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 10px 40px -10px rgba(99,102,241,0.6) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    background-position: 100% 100% !important;
    box-shadow: 0 15px 50px -10px rgba(139,92,246,0.7) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(56,189,248,0.10) 100%);
    border: 1px solid rgba(129,140,248,0.3);
    border-radius: 20px;
    padding: 2rem 2.25rem;
    text-align: center;
    margin-top: 1rem;
    position: relative; overflow: hidden;
    backdrop-filter: blur(10px);
}
.result-box::before {
    content:''; position:absolute; top:-50px; left:50%;
    transform: translateX(-50%);
    width: 250px; height: 250px; border-radius: 50%;
    background: radial-gradient(circle, rgba(129,140,248,0.25) 0%, transparent 70%);
    pointer-events: none;
}
.result-content { position: relative; z-index: 2; }
.result-label {
    font-size: 11px; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: #a5b4fc; margin-bottom: 0.75rem;
}
.result-price {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem; font-weight: 700;
    background: linear-gradient(135deg, #f5f7fa 0%, #a5b4fc 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -2px; line-height: 1.1;
}
.result-sub { font-size: 14px; color: #9ca3b0; margin-top: 0.5rem; font-weight: 500; }
.result-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(34,197,94,0.12); color: #4ade80;
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 999px; font-size: 12px; font-weight: 600;
    padding: 5px 14px; margin-top: 1.25rem;
}

/* ── Empty state ── */
.empty-state {
    background: rgba(20,22,32,0.4);
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 2.5rem 2rem; text-align: center;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.7; }
.empty-text { font-size: 14px; color: #6b7280; line-height: 1.7; }
.empty-text b { color: #a5b4fc; font-weight: 600; }

/* ── Error box ── */
.error-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 14px;
    padding: 1.25rem 1.5rem; margin-top: 1rem;
    display: flex; gap: 12px; align-items: flex-start;
}
.error-icon { font-size: 20px; flex-shrink: 0; }
.error-title { font-weight: 600; color: #f87171; font-size: 14px; margin-bottom: 4px; }
.error-msg { font-size: 13px; color: #fca5a5; line-height: 1.5; }

/* ── Chips ── */
.chip-row { display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
    background: rgba(99,102,241,0.08);
    color: #c7d2fe;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 999px;
    padding: 5px 13px; font-size: 12px; font-weight: 500;
}

/* ── Tips ── */
.tip-card {
    background: rgba(56,189,248,0.06);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 14px;
    padding: 1.25rem 1.5rem; margin-top: 1rem;
}
.tip-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px; font-weight: 600; color: #7dd3fc;
    margin-bottom: 0.75rem; display: flex; align-items: center; gap: 6px;
}
.tip-item {
    font-size: 12.5px; color: #9ca3b0;
    margin-bottom: 6px; padding-left: 18px; position: relative; line-height: 1.5;
}
.tip-item::before {
    content:'→'; position:absolute; left:0; color:#7dd3fc; font-weight:600;
}

hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(20,22,32,0.5) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #c7d2fe !important; font-size: 13px !important; font-weight: 500 !important;
}
[data-testid="stExpander"] { border: none !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #818cf8 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL COMPONENTS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_components():
    model_dir = 'model_components'
    try:
        components = {
            "xgb_model":                     joblib.load(os.path.join(model_dir, 'xgb_model.pkl')),
            "scaler":                        joblib.load(os.path.join(model_dir, 'robust_scaler.pkl')),
            "brand_target_map":              joblib.load(os.path.join(model_dir, 'brand_target_map.pkl')),
            "model_target_map":              joblib.load(os.path.join(model_dir, 'model_target_map.pkl')),
            "x_train_columns":               joblib.load(os.path.join(model_dir, 'x_train_columns.pkl')),
            "car_brands_choices":            joblib.load(os.path.join(model_dir, 'car_brands_choices.pkl')),
            "car_models_choices":            joblib.load(os.path.join(model_dir, 'car_models_choices.pkl')),
            "seller_types_choices":          joblib.load(os.path.join(model_dir, 'seller_types_choices.pkl')),
            "fuel_types_choices":            joblib.load(os.path.join(model_dir, 'fuel_types_choices.pkl')),
            "transmission_types_choices":    joblib.load(os.path.join(model_dir, 'transmission_types_choices.pkl')),
            "valid_brand_model_combinations":joblib.load(os.path.join(model_dir, 'valid_brand_model_combinations.pkl')),
        }
        return components, None
    except FileNotFoundError as e:
        return None, f"Model file not found: {e}"
    except Exception as e:
        return None, f"Failed to load model: {e}"


# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
def predict_price(components, vehicle_age, km_driven, mileage, engine,
                  max_power, seats, car_brand, car_model,
                  seller_type, fuel_type, transmission_type):

    if (car_brand, car_model) not in components["valid_brand_model_combinations"]:
        raise ValueError(f"'{car_brand} {car_model}' is not a valid combination. "
                         "Please choose a model that belongs to the selected brand.")
    if vehicle_age < 0 or vehicle_age > 30:
        raise ValueError("Vehicle age must be between 0 and 30 years.")
    if km_driven < 0:
        raise ValueError("Kilometers driven cannot be negative.")
    if mileage <= 0:
        raise ValueError("Mileage must be a positive number.")
    if engine <= 0:
        raise ValueError("Engine CC must be a positive number.")
    if max_power <= 0:
        raise ValueError("Max power must be a positive number.")
    if seats < 2 or seats > 9:
        raise ValueError("Seats must be between 2 and 9.")

    input_data = pd.DataFrame([{
        'vehicle_age': vehicle_age, 'km_driven': km_driven, 'mileage': mileage,
        'engine': engine, 'max_power': max_power, 'seats': seats,
        'brand': car_brand, 'model': car_model,
        'seller_type': seller_type, 'fuel_type': fuel_type,
        'transmission_type': transmission_type,
    }])

    btm = components["brand_target_map"]; mtm = components["model_target_map"]
    input_data['brand_encoded'] = input_data['brand'].map(btm).fillna(btm.mean())
    input_data['model_encoded'] = input_data['model'].map(mtm).fillna(mtm.mean())
    input_data.drop(columns=['brand', 'model'], inplace=True)

    input_data = pd.get_dummies(input_data,
        columns=['seller_type', 'fuel_type', 'transmission_type'])

    for col in components["x_train_columns"]:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[components["x_train_columns"]]

    cols_to_scale = ['km_driven', 'mileage', 'engine', 'max_power',
                     'vehicle_age', 'brand_encoded', 'model_encoded']
    input_data[cols_to_scale] = components["scaler"].transform(input_data[cols_to_scale])

    price = components["xgb_model"].predict(input_data)[0]
    if price <= 0:
        raise ValueError("Model returned an invalid price. Please check your inputs.")
    return float(price)


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-content">
    <div class="hero-tag"><span class="hero-tag-dot"></span> XGBoost Powered · Live Model</div>
    <h1>Know your car's <span class="grad">true value</span><br>in seconds.</h1>
    <p>AutoValue uses machine learning trained on thousands of real Indian used-car
    transactions to give you an instant, data-driven price estimate.</p>
    <div class="hero-stats">
      <div>
        <div class="hero-stat-num">XGBoost</div>
        <div class="hero-stat-label">ML Algorithm</div>
      </div>
      <div>
        <div class="hero-stat-num">11</div>
        <div class="hero-stat-label">Input Features</div>
      </div>
      <div>
        <div class="hero-stat-num">₹ INR</div>
        <div class="hero-stat-label">Output Currency</div>
      </div>
      <div>
        <div class="hero-stat-num">&lt; 1s</div>
        <div class="hero-stat-label">Prediction Time</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner("Loading model components..."):
    components, load_error = load_model_components()

if load_error:
    st.markdown(f"""
    <div class="error-box">
      <div class="error-icon">⚠️</div>
      <div>
        <div class="error-title">Model Loading Failed</div>
        <div class="error-msg">{load_error}<br>
        Make sure the <b>model_components/</b> folder is in the same directory as this script.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    # Car Identity
    st.markdown("""
    <div class="section-header">
      <div class="section-dot"></div>
      <div class="section-title">Car Identity</div>
      <div class="section-sub">Make · Model · Type</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        car_brand = st.selectbox("Brand", options=components["car_brands_choices"],
            index=components["car_brands_choices"].index("Maruti")
                  if "Maruti" in components["car_brands_choices"] else 0)
    with c2:
        car_model = st.selectbox("Model", options=components["car_models_choices"],
            index=components["car_models_choices"].index("Alto")
                  if "Alto" in components["car_models_choices"] else 0)

    c3, c4 = st.columns(2)
    with c3:
        fuel_type = st.selectbox("Fuel Type", options=components["fuel_types_choices"])
    with c4:
        transmission_type = st.selectbox("Transmission", options=components["transmission_types_choices"])

    seller_type = st.selectbox("Seller Type", options=components["seller_types_choices"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Vehicle Specs
    st.markdown("""
    <div class="section-header">
      <div class="section-dot"></div>
      <div class="section-title">Vehicle Specs</div>
      <div class="section-sub">Engine · Power · Mileage</div>
    </div>
    """, unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        engine = st.number_input("Engine (CC)", 500, 5000, 1197, 50)
    with c6:
        max_power = st.number_input("Max Power (BHP)", 30.0, 600.0, 85.0, 0.5)

    c7, c8 = st.columns(2)
    with c7:
        mileage = st.number_input("Mileage (kmpl)", 5.0, 50.0, 20.5, 0.1)
    with c8:
        seats = st.number_input("Seats", 2, 9, 5, 1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Usage History
    st.markdown("""
    <div class="section-header">
      <div class="section-dot"></div>
      <div class="section-title">Usage History</div>
      <div class="section-sub">Age · Distance</div>
    </div>
    """, unsafe_allow_html=True)

    vehicle_age = st.slider("Vehicle Age (years)", 0, 30, 5)
    km_driven = st.number_input("Kilometers Driven", 0, 1_000_000, 45000, 1000)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("✨  Predict Selling Price", use_container_width=True)


with right_col:
    st.markdown("""
    <div class="section-header">
      <div class="section-dot"></div>
      <div class="section-title">Price Estimate</div>
    </div>
    """, unsafe_allow_html=True)

    result_placeholder = st.empty()
    result_placeholder.markdown("""
    <div class="empty-state">
      <div class="empty-icon">🚗</div>
      <div class="empty-text">
        Fill in the car details on the left,<br>
        then click <b>Predict Selling Price</b><br>
        to see the estimated value.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if predict_clicked:
        with st.spinner("Calculating estimate..."):
            time.sleep(0.4)
            try:
                price = predict_price(components, vehicle_age, km_driven, mileage,
                    engine, max_power, seats, car_brand, car_model,
                    seller_type, fuel_type, transmission_type)
                price_lakhs = price / 1_00_000
                result_placeholder.markdown(f"""
                <div class="result-box">
                  <div class="result-content">
                    <div class="result-label">Estimated Selling Price</div>
                    <div class="result-price">₹{price:,.0f}</div>
                    <div class="result-sub">≈ {price_lakhs:.2f} Lakhs</div>
                    <div class="result-badge">✓ Prediction Successful</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            except ValueError as e:
                result_placeholder.markdown(f"""
                <div class="error-box">
                  <div class="error-icon">⚠️</div>
                  <div>
                    <div class="error-title">Invalid Input</div>
                    <div class="error-msg">{e}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                result_placeholder.markdown(f"""
                <div class="error-box">
                  <div class="error-icon">❌</div>
                  <div>
                    <div class="error-title">Prediction Error</div>
                    <div class="error-msg">Something went wrong: {e}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <div class="section-dot"></div>
      <div class="section-title">Input Summary</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="chip-row">
      <span class="chip">🚗 {car_brand} {car_model}</span>
      <span class="chip">⛽ {fuel_type}</span>
      <span class="chip">⚙️ {transmission_type}</span>
      <span class="chip">📅 {vehicle_age} yr old</span>
      <span class="chip">🛣️ {km_driven:,} km</span>
      <span class="chip">🔧 {engine} CC</span>
      <span class="chip">💨 {max_power} BHP</span>
      <span class="chip">📊 {mileage} kmpl</span>
      <span class="chip">💺 {seats} seats</span>
      <span class="chip">👤 {seller_type}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tip-card">
      <div class="tip-title">💡 Tips for accurate results</div>
      <div class="tip-item">Match brand + model exactly as listed</div>
      <div class="tip-item">Enter actual odometer reading for km driven</div>
      <div class="tip-item">Engine CC and BHP are on the RC / owner's manual</div>
      <div class="tip-item">Vehicle age = current year − year of manufacture</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋  Quick example presets"):
        st.markdown("""
| Preset | Brand | Model | Age | KM |
|--------|-------|-------|-----|----|
| Budget hatchback | Maruti | Alto | 5 | 45,000 |
| Mid sedan | Honda | City | 2 | 15,000 |
| Popular SUV | Hyundai | Creta | 7 | 70,000 |

*Select the preset values manually in the form to test.*
        """)
