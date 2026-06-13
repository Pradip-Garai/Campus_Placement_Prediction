import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page config for a premium dashboard feel
st.set_page_config(
    page_title="Corporate Fit & Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Google Fonts and Inject Premium Custom Styles
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    
    <style>
    /* Global Styles */
    .main {
        background: #090d16;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f8fafc;
    }
    
    body {
        background-color: #090d16;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Header Card - Animated Background */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .header-card {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #111827, #020617);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        padding: 3rem 2.5rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .header-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 80% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .header-title {
        background: linear-gradient(90deg, #60a5fa, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.2rem;
        margin: 0;
        padding-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 400;
        margin: 0;
        max-width: 800px;
        line-height: 1.6;
    }
    
    /* Glassmorphic Cards & Forms */
    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        padding: 2.5rem !important;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Input Elements Overrides */
    div[data-baseweb="select"] > div {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
    }
    
    div[data-baseweb="input"] {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    
    input {
        color: #f8fafc !important;
    }
    
    label {
        font-weight: 600 !important;
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        margin-bottom: 0.4rem !important;
    }
    
    /* Subheading dividers */
    .form-section-header {
        font-family: 'Outfit', sans-serif;
        color: #818cf8;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 1.8rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .form-section-header::after {
        content: '';
        flex-grow: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(129, 140, 248, 0.3), transparent);
    }
    
    /* Action Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.02em !important;
        border: none !important;
        padding: 0.9rem 2rem !important;
        border-radius: 14px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25) !important;
        margin-top: 1.5rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.45) !important;
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(1px) !important;
    }
    
    /* Result Section Header */
    .result-section-header {
        color: #60a5fa;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.6rem;
    }
    
    /* Premium HTML Dashboard Cards */
    .dashboard-metrics {
        display: flex;
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    
    .result-card {
        flex: 1;
        background: rgba(30, 41, 59, 0.35);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
    }
    
    .card-prob::before { background: linear-gradient(90deg, #10b981, #34d399); }
    .card-sector::before { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    .card-sal::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
    
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.35);
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    .res-val {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    
    .val-high { color: #10b981; text-shadow: 0 0 20px rgba(16, 185, 129, 0.2); }
    .val-med { color: #fbbf24; text-shadow: 0 0 20px rgba(251, 191, 36, 0.2); }
    .val-low { color: #ef4444; text-shadow: 0 0 20px rgba(239, 68, 68, 0.2); }
    .val-blue { color: #3b82f6; text-shadow: 0 0 20px rgba(59, 130, 246, 0.2); }
    .val-purple { color: #8b5cf6; text-shadow: 0 0 20px rgba(139, 92, 246, 0.2); }
    
    .res-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    /* Bespoke Progress Bars Chart */
    .custom-chart-card {
        background: rgba(30, 41, 59, 0.25);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    .chart-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 1.5rem;
    }
    
    .custom-bar-row {
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 1.25rem;
        padding: 0.5rem;
        border-radius: 10px;
        transition: background-color 0.2s ease;
    }
    
    .custom-bar-row:hover {
        background-color: rgba(255, 255, 255, 0.02);
    }
    
    .custom-bar-label {
        width: 150px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #cbd5e1;
    }
    
    .custom-bar-track {
        flex-grow: 1;
        height: 10px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        overflow: hidden;
    }
    
    .custom-bar-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Unique gradients for each sector */
    .fill-tech { background: linear-gradient(90deg, #06b6d4, #3b82f6); }
    .fill-finance { background: linear-gradient(90deg, #10b981, #059669); }
    .fill-consulting { background: linear-gradient(90deg, #8b5cf6, #6366f1); }
    .fill-core { background: linear-gradient(90deg, #f59e0b, #d97706); }
    .fill-mkt { background: linear-gradient(90deg, #ec4899, #db2777); }
    
    .custom-bar-value {
        width: 50px;
        text-align: right;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        color: #f8fafc;
    }
    
    /* Actionable Recommendations Styles */
    .rec-card-container {
        background: rgba(30, 41, 59, 0.25);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
    }
    
    .rec-item {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.04);
        padding: 1.25rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        transition: all 0.2s ease;
    }
    
    .rec-item:hover {
        border-color: rgba(129, 140, 248, 0.2);
        transform: translateX(4px);
    }
    
    .rec-icon {
        font-size: 1.5rem;
        background: rgba(99, 102, 241, 0.1);
        padding: 0.5rem;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .rec-content {
        flex-grow: 1;
    }
    
    .rec-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #f8fafc;
        font-size: 1.05rem;
        margin-bottom: 0.3rem;
    }
    
    .rec-desc {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    .rec-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-left: auto;
    }
    
    .badge-high { background-color: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.2); }
    .badge-med { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.2); }
    .badge-low { background-color: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.2); }
    </style>
""", unsafe_allow_html=True)

# ----------------- Load Trained Models & Auto-Train Helper -----------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(_BASE_DIR, 'placement_models.pkl')

def train_and_save_models():
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    csv_path = os.path.join(_BASE_DIR, "Placement_Data_Expanded.csv")
    if not os.path.exists(csv_path):
        return None
        
    df = pd.read_csv(csv_path)
    
    encoders = {}
    categorical_cols = ['gender', 'degree', 'stream', 'workex']
    df_encoded = df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    encoder_classes = {col: list(le.classes_) for col, le in encoders.items()}
    
    feature_cols = [
        'gender', 'degree', 'stream', 'ssc_p', 'hsc_p', 'cgpa', 'workex',
        'coding_skills', 'communication_skills', 'analytical_skills', 'domain_knowledge',
        'projects', 'internships', 'certifications'
    ]
    
    # Train placement status model
    X_status = df_encoded[feature_cols]
    y_status = df_encoded['placed_status'].apply(lambda x: 1 if x == 'Placed' else 0)
    status_model = RandomForestClassifier(n_estimators=100, random_state=42)
    status_model.fit(X_status, y_status)
    
    # Train placed sector model (only Placed students)
    df_placed = df_encoded[df_encoded['placed_status'] == 'Placed'].copy()
    sector_encoder = LabelEncoder()
    df_placed['placed_sector_encoded'] = sector_encoder.fit_transform(df_placed['placed_sector'])
    encoders['placed_sector'] = sector_encoder
    
    X_sector = df_placed[feature_cols]
    y_sector = df_placed['placed_sector_encoded']
    sector_model = RandomForestClassifier(n_estimators=100, random_state=42)
    sector_model.fit(X_sector, y_sector)
    
    # Train salary regressor model (only Placed students)
    X_salary = df_placed[feature_cols]
    y_salary = df_placed['salary_lpa']
    salary_model = RandomForestRegressor(n_estimators=100, random_state=42)
    salary_model.fit(X_salary, y_salary)
    
    models_dict = {
        'status_model': status_model,
        'sector_model': sector_model,
        'salary_model': salary_model,
        'encoders': encoders,
        'feature_cols': feature_cols,
        'encoder_classes': encoder_classes
    }
    
    # Attempt to cache the trained models locally for future fast loads
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(models_dict, f)
    except Exception:
        # If running in a read-only container environment, bypass writing to disk
        pass
        
    return models_dict

@st.cache_resource
def load_models():
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
            
    # Auto-train if dataset is available
    with st.spinner("⏳ First run: Training predictive models on the fly..."):
        return train_and_save_models()

models_dict = load_models()

if models_dict is None:
    st.error("⚠️ Dataset file (`Placement_Data_Expanded.csv`) not found! Please ensure it is present in the repository.")
    st.stop()

status_model = models_dict['status_model']
sector_model = models_dict['sector_model']
salary_model = models_dict['salary_model']
encoders = models_dict['encoders']
feature_cols = models_dict['feature_cols']
encoder_classes = models_dict['encoder_classes']

# Streams mapping by degree
streams_map = {
    'B.Tech': ['Computer Science', 'Information Technology', 'Electronics', 'Mechanical', 'Civil'],
    'MCA': ['Computer Applications'],
    'BCA': ['Computer Applications'],
    'B.Sc': ['Computer Science', 'Mathematics', 'Physics', 'Biotech'],
    'B.Com': ['Accounting & Finance', 'General Commerce'],
    'BBA': ['Marketing', 'Finance', 'Human Resources', 'General Management'],
    'MBA': ['Marketing', 'Finance', 'Human Resources', 'Operations', 'Business Analytics']
}

# Header UI
st.markdown("""
    <div class="header-card">
        <h1 class="header-title">🎓 CareerFit Predictive Insights</h1>
        <p class="header-subtitle">Analyze, evaluate, and predict placement suitability across different degrees, tech, and non-tech corporate sectors using Advanced Machine Learning.</p>
    </div>
""", unsafe_allow_html=True)

# Layout: Split page into Inputs (Left) and Results (Right)
col_input, col_result = st.columns([1, 1.25], gap="large")

with col_input:
    st.markdown('<div class="result-section-header">👤 Candidate Profile</div>', unsafe_allow_html=True)
    
    with st.form("placement_form"):
        # Categorized tabs for inputs to prevent vertical scrolling
        tab_edu, tab_ach, tab_ski = st.tabs(["🎓 Education & Info", "💡 Achievements", "🛠️ Core Skills"])
        
        with tab_edu:
            st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                gender_sel = st.selectbox("Gender", encoder_classes['gender'])
                degree_sel = st.selectbox("Degree Pursuing", list(streams_map.keys()))
            with c2:
                stream_options = streams_map[degree_sel]
                stream_sel = st.selectbox("Specialization / Stream", stream_options)
                workex_sel = st.selectbox("Prior Work Experience", encoder_classes['workex'])

            st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            c3, c4, c5 = st.columns(3)
            with c3:
                ssc_p = st.number_input("SSC (10th) %", min_value=40.0, max_value=100.0, value=75.0, step=1.0)
            with c4:
                hsc_p = st.number_input("HSC (12th) %", min_value=40.0, max_value=100.0, value=72.0, step=1.0)
            with c5:
                cgpa = st.number_input("Degree CGPA", min_value=4.0, max_value=10.0, value=7.8, step=0.1)

        with tab_ach:
            st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            c6, c7, c8 = st.columns(3)
            with c6:
                projects_val = st.slider("Completed Projects", 0, 5, 2)
            with c7:
                intern_val = st.slider("Completed Internships", 0, 3, 1)
            with c8:
                cert_val = st.slider("Certifications Earned", 0, 5, 1)

        with tab_ski:
            st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            # Guide defaults based on degree selection
            default_coding = 7.0 if degree_sel in ['B.Tech', 'MCA', 'BCA'] else 3.0
            default_domain = 7.5 if degree_sel in ['MBA', 'B.Com', 'BBA'] else 6.0
            
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                coding_val = st.slider("Coding & Scripting Skills", 1.0, 10.0, default_coding, 0.5)
                comm_val = st.slider("Communication & Soft Skills", 1.0, 10.0, 7.0, 0.5)
            with c_s2:
                anl_val = st.slider("Analytical & Problem Solving", 1.0, 10.0, 6.5, 0.5)
                dom_val = st.slider("Domain Knowledge", 1.0, 10.0, default_domain, 0.5)
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        submit_btn = st.form_submit_button("⚡ Run Placement Analytics")

with col_result:
    st.markdown('<div class="result-section-header">📊 Placement Analytics Dashboard</div>', unsafe_allow_html=True)
    
    if submit_btn:
        # ----------------- Preprocessing & Prediction -----------------
        # Prepare input data matching feature columns exactly
        encoded_gender = encoders['gender'].transform([gender_sel])[0]
        encoded_degree = encoders['degree'].transform([degree_sel])[0]
        encoded_stream = encoders['stream'].transform([stream_sel])[0]
        encoded_workex = encoders['workex'].transform([workex_sel])[0]
        
        input_row = {
            'gender': encoded_gender,
            'degree': encoded_degree,
            'stream': encoded_stream,
            'ssc_p': ssc_p,
            'hsc_p': hsc_p,
            'cgpa': cgpa,
            'workex': encoded_workex,
            'coding_skills': coding_val,
            'communication_skills': comm_val,
            'analytical_skills': anl_val,
            'domain_knowledge': dom_val,
            'projects': projects_val,
            'internships': intern_val,
            'certifications': cert_val
        }
        
        input_df = pd.DataFrame([input_row])[feature_cols]
        
        # 1. Predict placement probability
        placement_prob = status_model.predict_proba(input_df)[0][1]
        prob_percent = int(placement_prob * 100)
        
        # Determine probability rating classes for display
        if prob_percent >= 75:
            val_class = "val-high"
        elif prob_percent >= 50:
            val_class = "val-med"
        else:
            val_class = "val-low"
            
        # 2. Predict best fit sectors and probabilities
        sector_probs = sector_model.predict_proba(input_df)[0]
        sector_classes = encoders['placed_sector'].classes_
        sector_fit = sorted(zip(sector_classes, sector_probs), key=lambda x: x[1], reverse=True)
        best_sector = sector_fit[0][0]
        
        # 3. Predict salary
        predicted_salary = salary_model.predict(input_df)[0]
        sal_display = f"{predicted_salary:.1f} LPA" if placement_prob >= 0.40 else "N/A"
        
        # Display Premium Metrics Dashboard Row
        st.markdown(f"""
            <div class="dashboard-metrics">
                <div class="result-card card-prob">
                    <div class="res-val {val_class}">{prob_percent}%</div>
                    <div class="res-label">Placement Likelihood</div>
                </div>
                <div class="result-card card-sector">
                    <div class="res-val val-blue" style="font-size: 1.8rem; padding: 0.45rem 0;">{best_sector}</div>
                    <div class="res-label">Top Sector Fit</div>
                </div>
                <div class="result-card card-sal">
                    <div class="res-val val-purple">{sal_display}</div>
                    <div class="res-label">Est. Salary Package</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display Bespoke Progress Bar Chart
        st.markdown('<div class="custom-chart-card"><div class="chart-header">🎯 Corporate Sector Suitability Match</div>', unsafe_allow_html=True)
        
        # Mapping colors & classes to sectors
        fill_class_map = {
            'Tech': 'fill-tech',
            'Finance': 'fill-finance',
            'Consulting': 'fill-consulting',
            'Core Engineering': 'fill-core',
            'Marketing & HR': 'fill-mkt'
        }
        
        for sector, prob in sector_fit:
            pct = int(prob * 100)
            fill_class = fill_class_map.get(sector, 'fill-tech')
            st.markdown(f"""
                <div class="custom-bar-row">
                    <div class="custom-bar-label">{sector}</div>
                    <div class="custom-bar-track">
                        <div class="custom-bar-fill {fill_class}" style="width: {pct}%;"></div>
                    </div>
                    <div class="custom-bar-value">{pct}%</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ----------------- Strengths & Weaknesses Logic -----------------
        strengths = []
        weaknesses = []
        
        # Academic Checks
        if cgpa >= 8.5:
            strengths.append(("🌟 Excellent CGPA", f"Your CGPA of {cgpa} is outstanding and puts you in the top tier of applicants."))
        elif cgpa >= 7.8:
            strengths.append(("📈 Solid Academic Standing", f"Your CGPA of {cgpa} is good, meeting key eligibility cutoffs for most corporates."))
        elif cgpa < 7.0:
            weaknesses.append(("⚠️ Below Average CGPA", f"A CGPA of {cgpa} is on the lower side. Some recruiters have strict filters at 7.0 or 7.5."))
            
        if ssc_p >= 85.0 and hsc_p >= 85.0:
            strengths.append(("🏫 Strong Schooling Foundation", f"Excellent high-school scores (10th: {ssc_p}%, 12th: {hsc_p}%) reflect consistent learning ability."))
        elif ssc_p < 60.0 or hsc_p < 60.0:
            weaknesses.append(("📉 Academic Inconsistency", "Low percentages in 10th or 12th standards might trigger filters in premium consulting or finance firms."))

        # Work Experience
        if workex_sel == 'Yes':
            strengths.append(("💼 Industry Experience", "Prior work experience gives you a substantial advantage in roles requiring business logic or leadership."))
        
        # Experience & Accomplishments
        if intern_val >= 2:
            strengths.append(("🧑‍💼 Hands-on Internship Profile", f"You have completed {intern_val} internships, showing strong industry readiness and practical exposure."))
        elif intern_val == 0:
            weaknesses.append(("🚫 Zero Internships", "No prior internships could raise questions about your ability to apply concepts in professional settings."))
            
        if projects_val >= 3:
            strengths.append(("🚀 Project-Rich Resume", f"With {projects_val} projects, you have a solid showcase of portfolio works to present in interviews."))
        elif projects_val < 1:
            weaknesses.append(("📂 Lack of Projects", "Having no projects makes your profile less competitive compared to candidates with active githubs/portfolios."))

        # Certifications
        if cert_val >= 2:
            strengths.append(("📜 Verified Certifications", f"Having {cert_val} professional certifications displays self-driven learning and domain expertise."))
        elif cert_val == 0:
            weaknesses.append(("📭 No Domain Certifications", "Lack of professional certifications means missing out on secondary validation badges on your resume."))

        # Skills matrix checks
        if coding_val >= 8.0:
            strengths.append(("💻 Top-Tier Coding Ability", f"Your programming competence ({coding_val}/10) is excellent, suited for competitive developer roles."))
        elif coding_val < 5.0 and degree_sel in ['B.Tech', 'MCA', 'BCA']:
            weaknesses.append(("💻 Weak Programming Skills", f"A coding rating of {coding_val}/10 is weak for a technology graduate seeking core developer profiles."))
            
        if comm_val >= 8.0:
            strengths.append(("🗣️ Executive Communication", f"Excellent soft skills ({comm_val}/10), a critical selector for management, consulting, and client-facing roles."))
        elif comm_val < 5.5:
            weaknesses.append(("🔇 Communication Barrier", f"Low communication rating ({comm_val}/10) can hinder your performance in group discussions and HR interviews."))
            
        if anl_val >= 8.0:
            strengths.append(("🧠 Analytical Strength", f"Outstanding problem-solving capability ({anl_val}/10), crucial for Finance, Consulting, and Analyst roles."))
        elif anl_val < 5.5:
            weaknesses.append(("🧩 Moderate Problem Solving", "Analytical rating is low, indicating you may need practice with quantitative tests and case studies."))
            
        if dom_val >= 8.0:
            strengths.append(("🎯 Comprehensive Domain Knowledge", f"Strong grasp of core subjects ({dom_val}/10), which will help clear technical rounds."))
        elif dom_val < 5.5:
            weaknesses.append(("📖 Insufficient Subject Knowledge", f"A rating of {dom_val}/10 in core domain subjects indicates potential gaps in fundamental concepts."))

        # Fallbacks if list is empty
        if not strengths:
            strengths.append(("⚖️ Balanced Competency", "Your profile shows balanced indicators without extreme outliers. Consistent performance across parameters."))
        if not weaknesses:
            weaknesses.append(("🛡️ No Critical Weaknesses", "Your scores meet or exceed baseline criteria across all measured parameters. Excellent balance!"))

        # Display Strengths and Weaknesses Side-By-Side in beautiful Custom CSS grid
        st.markdown('<div class="chart-header">⚖️ Profile Diagnostic: Strengths vs Areas to Improve</div>', unsafe_allow_html=True)
        
        col_str, col_wk = st.columns(2)
        
        with col_str:
            st.markdown("""
                <div style="background: rgba(16, 185, 129, 0.04); border-left: 4px solid #10b981; border-top: 1px solid rgba(255, 255, 255, 0.05); border-right: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 0 16px 16px 0; height: 100%;">
                    <div style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #34d399; font-size: 1.15rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.5rem;">
                        <span>✅</span> STRENGTHS & ASSETS
                    </div>
            """, unsafe_allow_html=True)
            for title, desc in strengths:
                st.markdown(f"""
                    <div style="margin-bottom: 1rem;">
                        <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem; margin-bottom: 0.2rem;">{title}</div>
                        <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.4;">{desc}</div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_wk:
            st.markdown("""
                <div style="background: rgba(239, 68, 68, 0.04); border-left: 4px solid #ef4444; border-top: 1px solid rgba(255, 255, 255, 0.05); border-right: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 0 16px 16px 0; height: 100%;">
                    <div style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #f87171; font-size: 1.15rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.5rem;">
                        <span>⚠️</span> AREAS TO IMPROVE
                    </div>
            """, unsafe_allow_html=True)
            for title, desc in weaknesses:
                st.markdown(f"""
                    <div style="margin-bottom: 1rem;">
                        <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem; margin-bottom: 0.2rem;">{title}</div>
                        <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.4;">{desc}</div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Display Actionable Career Roadmap
        st.markdown('<div class="rec-card-container"><div class="chart-header">🛣️ Strategic Career Roadmap Suggestions</div>', unsafe_allow_html=True)
        
        recommendations = []
        
        # Checking conditions for actionable items
        if degree_sel in ['B.Tech', 'MCA', 'BCA', 'B.Sc'] and stream_sel in ['Computer Science', 'Information Technology', 'Computer Applications']:
            if coding_val < 7.0:
                recommendations.append({
                    "icon": "💻",
                    "title": "Boost Programming Competency",
                    "desc": f"Your coding rating of {coding_val}/10 is below the benchmark for elite Tech companies. Learn Data Structures & Algorithms, practice on LeetCode/HackerRank, or build projects in Python/Java.",
                    "priority": "high"
                })
        elif coding_val < 5.0 and best_sector == 'Tech':
            recommendations.append({
                "icon": "💻",
                "title": "General Technology Literacy",
                "desc": "Tech roles are highly compatible with your profile, but your coding score is moderate. Learn databases (SQL) and basic Python programming.",
                "priority": "med"
            })
            
        if projects_val < 2:
            recommendations.append({
                "icon": "🚀",
                "title": "Build Hands-on Projects",
                "desc": f"You have completed only {projects_val} projects. Build at least 2 comprehensive, end-to-end projects related to your sector (e.g. Full-Stack Web App, Financial Analysis/Valuation Reports).",
                "priority": "high"
            })
            
        if intern_val == 0:
            recommendations.append({
                "icon": "💼",
                "title": "Secure a Practical Internship",
                "desc": "Prior internships boost placement rates by up to 15%. Focus on securing a 2-3 month internship in your target sector to show real-world experience.",
                "priority": "high"
            })
            
        if cgpa < 7.5:
            recommendations.append({
                "icon": "📚",
                "title": "Academic GPA Improvement",
                "desc": f"A CGPA of {cgpa} might restrict you from passing automated recruiter cutoffs. Focus on maintaining or raising your CGPA above 7.5.",
                "priority": "med"
            })
            
        if cert_val < 2:
            if best_sector == 'Finance':
                cert_desc = "Consider pursuing CFA Level 1, NCFM modules, or Financial Modeling certifications to establish credentials."
            elif best_sector == 'Tech':
                cert_desc = "Consider AWS Cloud Practitioner, Google Cloud Associate, or Microsoft Azure certifications depending on your domain interest."
            else:
                cert_desc = "Earn certifications related to your stream (e.g. Agile Scrum, HubSpot Marketing, or SHRM for HR)."
            
            recommendations.append({
                "icon": "📜",
                "title": "Earn Industry Certifications",
                "desc": f"Professional certifications show initiative. {cert_desc}",
                "priority": "low"
            })
            
        if comm_val < 7.0:
            recommendations.append({
                "icon": "🗣️",
                "title": "Polish Business Communication",
                "desc": "Recruiters rank communication highly for consulting and business analyst roles. Focus on practicing mock interviews and public speaking.",
                "priority": "med"
            })
            
        if not recommendations:
            recommendations.append({
                "icon": "🌟",
                "title": "Exemplary Candidate Profile!",
                "desc": "Your profile meets all standard benchmarks. Keep revising core subjects, practice mock case interviews, and prepare well for HR rounds.",
                "priority": "low"
            })
            
        # Render recommendations using premium styling
        for rec in recommendations:
            badge_class = f"badge-{rec['priority']}"
            priority_lbl = f"{rec['priority']} priority"
            st.markdown(f"""
                <div class="rec-item">
                    <div class="rec-icon">{rec['icon']}</div>
                    <div class="rec-content">
                        <div class="rec-title">{rec['title']}</div>
                        <div class="rec-desc">{rec['desc']}</div>
                    </div>
                    <div class="rec-badge {badge_class}">{priority_lbl}</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        # Prompt candidate to run prediction with custom glassmorphic styling
        st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.35); border: 1px solid rgba(255, 255, 255, 0.05); padding: 2.5rem; border-radius: 20px; text-align: center; color: #94a3b8;">
                <span style="font-size: 2.5rem; display: block; margin-bottom: 1rem;">👈</span>
                <h4 style="color: #f8fafc; margin-bottom: 0.5rem; font-size: 1.25rem;">Ready to Analyze Your Career Path?</h4>
                Fill out the profile details on the left and click the <b>⚡ Run Placement Analytics</b> button to generate your compatibility report.
            </div>
        """, unsafe_allow_html=True)