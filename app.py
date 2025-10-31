"""
TELECOM CHURN PREDICTION - STREAMLIT WEB APPLICATION
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
from datetime import datetime
import io
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Telecom Churn Analytics",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# Custom CSS - THEME-AWARE
st.markdown("""
    <style>
    /* =================================================================================
       BASE STYLING & THEME VARIABLES
    ================================================================================= */
    :root {
        /* Color Palette for Light Theme */
        --success-bg-light: #d4edda;
        --success-border-light: #28a745;
        --success-text-light: #155724;

        --risk-low-bg-light: #d1e7dd;
        --risk-low-border-light: #0f5132;
        --risk-low-text-light: #0a3622;

        --risk-medium-bg-light: #fff3cd;
        --risk-medium-border-light: #ffc107;
        --risk-medium-text-light: #664d03;

        --risk-high-bg-light: #f8d7da;
        --risk-high-border-light: #dc3545;
        --risk-high-text-light: #58151c;
    }

    [data-theme="dark"] {
        /* Color Palette for Dark Theme */
        --success-bg-dark: #0a3622;
        --success-border-dark: #28a745;
        --success-text-dark: #75b798;

        --risk-low-bg-dark: #0a3622;
        --risk-low-border-dark: #0f5132;
        --risk-low-text-dark: #75b798;

        --risk-medium-bg-dark: #332701;
        --risk-medium-border-dark: #ffc107;
        --risk-medium-text-dark: #ffda6a;

        --risk-high-bg-dark: #3a1a1f;
        --risk-high-border-dark: #dc3545;
        --risk-high-text-dark: #f1aeb5;
    }

    /* General page padding */
    .main .block-container {
        padding: 1rem 2rem 2rem;
    }

    /* =================================================================================
       COMPONENT STYLING
    ================================================================================= */

    /* Headers */
    h1 {
        color: var(--primary-color);
        padding-bottom: 10px;
        border-bottom: 3px solid var(--primary-color);
        margin-bottom: 20px;
    }

    h3 {
        color: var(--text-color);
        margin-top: 25px;
        font-weight: 600;
    }

    /* st.metric styling to look like cards */
    .stMetric {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--gray-300);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    
    /* File Uploader styling */
    .stFileUploader {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 10px;
        border: 2px dashed var(--gray-400);
        margin-bottom: 15px;
    }
    .stFileUploader:hover {
        border-color: var(--primary-color);
        background-color: var(--background-color);
    }
    div[data-testid="stFileUploader"] label {
        font-size: 16px;
        font-weight: 600;
        color: var(--primary-color);
    }

    /* =================================================================================
       CUSTOM BOXES (Success, Risk, etc.)
    ================================================================================= */
    
    .custom-box {
        padding: 20px;
        border-radius: 10px;
        border-left-width: 5px;
        border-left-style: solid;
        margin: 20px 0;
    }
    
    /* Light Theme Box Styles */
    .success-box {
        background-color: var(--success-bg-light);
        border-left-color: var(--success-border-light);
        color: var(--success-text-light);
    }
    .risk-low {
        background-color: var(--risk-low-bg-light);
        border-left-color: var(--risk-low-border-light);
        color: var(--risk-low-text-light);
    }
    .risk-medium {
        background-color: var(--risk-medium-bg-light);
        border-left-color: var(--risk-medium-border-light);
        color: var(--risk-medium-text-light);
    }
    .risk-high {
        background-color: var(--risk-high-bg-light);
        border-left-color: var(--risk-high-border-light);
        color: var(--risk-high-text-light);
    }

    /* Dark Theme Box Styles */
    [data-theme="dark"] .success-box {
        background-color: var(--success-bg-dark);
        border-left-color: var(--success-border-dark);
        color: var(--success-text-dark);
    }
    [data-theme="dark"] .risk-low {
        background-color: var(--risk-low-bg-dark);
        border-left-color: var(--risk-low-border-dark);
        color: var(--risk-low-text-dark);
    }
    [data-theme="dark"] .risk-medium {
        background-color: var(--risk-medium-bg-dark);
        border-left-color: var(--risk-medium-border-dark);
        color: var(--risk-medium-text-dark);
    }
    [data-theme="dark"] .risk-high {
        background-color: var(--risk-high-bg-dark);
        border-left-color: var(--risk-high-border-dark);
        color: var(--risk-high-text-dark);
    }

    /* Padding for risk boxes in st.metric */
    .risk-low, .risk-medium, .risk-high {
        padding: 10px;
        border-radius: 5px;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        color: var(--gray-600);
        padding: 20px;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'master_data' not in st.session_state:
    st.session_state.master_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'predictions_data' not in st.session_state:
    st.session_state.predictions_data = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def merge_datasets(customers, billing, usage, complaints):
    """Merge all datasets into master dataset"""
    try:
        # Start with customers
        master = customers.copy()
        
        # Merge billing
        master = master.merge(billing, on='customer_id', how='left')
        
        # Merge usage
        master = master.merge(usage, on='customer_id', how='left')
        
        # Aggregate complaints - Check if complaints dataframe is not empty
        if len(complaints) > 0 and 'customer_id' in complaints.columns:
            # Count total complaints per customer
            complaints_total = complaints.groupby('customer_id').size().reset_index(name='complaints_total')
            
            # Count open complaints per customer
            if 'status' in complaints.columns:
                complaints_open = complaints[complaints['status'] == 'Open'].groupby('customer_id').size().reset_index(name='complaints_open')
            else:
                complaints_open = pd.DataFrame(columns=['customer_id', 'complaints_open'])
            
            # Merge total complaints
            master = master.merge(complaints_total, on='customer_id', how='left')
            
            # Merge open complaints
            master = master.merge(complaints_open, on='customer_id', how='left')
            
            # Fill NaN values with 0
            master['complaints_total'] = master['complaints_total'].fillna(0).astype(int)
            master['complaints_open'] = master['complaints_open'].fillna(0).astype(int)
        else:
            # If no complaints data, set to 0
            master['complaints_total'] = 0
            master['complaints_open'] = 0
        
        # Convert churn to binary
        master['churn_flag'] = master['churn'].map({'Yes': 1, 'No': 0})
        
        return master
    except Exception as e:
        st.error(f"Error merging datasets: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def train_models(data):
    """Train both Logistic Regression and Decision Tree models"""
    try:
        # Prepare features
        feature_cols = ['plan_type', 'region', 'contract_type', 'tenure', 'monthly_charges',
                       'data_used_gb', 'calls_made', 'revenue_inr', 'complaints_total', 'complaints_open']
        
        X = data[feature_cols]
        y = data['churn_flag']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Create preprocessor
        numeric_features = ['tenure', 'monthly_charges', 'data_used_gb', 'calls_made', 
                          'revenue_inr', 'complaints_total', 'complaints_open']
        categorical_features = ['plan_type', 'region', 'contract_type']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
            ])
        
        # Train Logistic Regression
        lr_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        lr_pipeline.fit(X_train, y_train)
        lr_pred = lr_pipeline.predict(X_test)
        
        lr_metrics = {
            'accuracy': accuracy_score(y_test, lr_pred),
            'precision': precision_score(y_test, lr_pred),
            'recall': recall_score(y_test, lr_pred),
            'f1_score': f1_score(y_test, lr_pred)
        }
        
        # Train Decision Tree
        dt_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42, max_depth=10))
        ])
        dt_pipeline.fit(X_train, y_train)
        dt_pred = dt_pipeline.predict(X_test)
        
        dt_metrics = {
            'accuracy': accuracy_score(y_test, dt_pred),
            'precision': precision_score(y_test, dt_pred),
            'recall': recall_score(y_test, dt_pred),
            'f1_score': f1_score(y_test, dt_pred)
        }
        
        # Select best model based on F1 score
        if lr_metrics['f1_score'] >= dt_metrics['f1_score']:
            best_model = lr_pipeline
            best_model_name = "Logistic Regression"
            best_metrics = lr_metrics
        else:
            best_model = dt_pipeline
            best_model_name = "Decision Tree"
            best_metrics = dt_metrics
        
        model_info = {
            'best_model': best_model_name,
            'logistic_regression': lr_metrics,
            'decision_tree': dt_metrics,
            'evaluation_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        return best_model, model_info, preprocessor
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return None, None, None

def generate_predictions(data, model):
    """Generate predictions for all customers"""
    try:
        feature_cols = ['plan_type', 'region', 'contract_type', 'tenure', 'monthly_charges',
                       'data_used_gb', 'calls_made', 'revenue_inr', 'complaints_total', 'complaints_open']
        
        X = data[feature_cols]
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        data['churn_prob'] = probabilities
        data['predicted_churn'] = predictions
        
        # Risk segmentation
        data['risk_segment'] = pd.cut(probabilities, 
                                      bins=[0, 0.3, 0.7, 1.0],
                                      labels=['Low Risk', 'Medium Risk', 'High Risk'])
        
        # Offer recommendations - use list comprehension instead of apply
        def recommend_offer(prob):
            if prob > 0.8:
                return "Upgrade Plan / Retention Call"
            elif prob > 0.6:
                return "Offer Discount"
            else:
                return "No Action Needed"
        
        data['offer_recommendation'] = [recommend_offer(prob) for prob in probabilities]
        
        return data
        
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/phone.png", width=100)
    st.title("📱 Navigation")
    
    page = st.radio(
        "Select Page:",
        ["🏠 Home & Upload", "📊 Dashboard", "📈 Analytics", "🔮 Single Prediction", 
         "📁 Bulk Prediction", "⚙️ Model Info"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📅 App Info")
    st.info(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    
    if st.session_state.master_data is not None:
        st.success(f"**Customers Loaded:** {len(st.session_state.master_data):,}")
    
    if st.session_state.predictions_data is not None:
        high_risk = (st.session_state.predictions_data['risk_segment'] == 'High Risk').sum()
        st.warning(f"**High Risk:** {high_risk:,}")

# ============================================================================
# PAGE 1: HOME & FILE UPLOAD
# ============================================================================

if page == "🏠 Home & Upload":
    st.title("📱 Telecom Churn Prediction System")
    st.markdown("### Upload Your Data Files to Get Started")
    
    # Instructions box
    st.info("""
    **📋 Instructions:**
    1. Upload all 4 CSV files below
    2. Click "Process Data & Train Model" button
    3. Explore insights in other pages
    """)
    
    # Required files information
    with st.expander("📂 Required Files - Click to view details", expanded=True):
        st.markdown("""
        #### Please upload the following CSV files:
        
        **1. Customers.csv**
        - Columns: `customer_id`, `name`, `plan_type`, `region`
        - Customer information and plan details
        
        **2. Billing.csv**
        - Columns: `customer_id`, `tenure`, `contract_type`, `monthly_charges`, `churn`
        - Billing details and churn status
        
        **3. Usage Data.csv**
        - Columns: `customer_id`, `data_used_gb`, `calls_made`, `revenue_inr`
        - Usage statistics per customer
        
        **4. Complaints.csv**
        - Columns: `customer_id`, `category`, `created_at`, `status`
        - Customer complaint records
        """)
    
    st.markdown("---")
    st.markdown("### 📤 Upload Files")
    
    # File uploaders in a better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1️⃣ Customers CSV**")
        customers_file = st.file_uploader("Choose Customers file", type=['csv'], key='customers', 
                                          help="Upload customers.csv with customer information")
        
        st.markdown("**2️⃣ Billing CSV**")
        billing_file = st.file_uploader("Choose Billing file", type=['csv'], key='billing',
                                       help="Upload billing.csv with billing and churn data")
    
    with col2:
        st.markdown("**3️⃣ Usage Data CSV**")
        usage_file = st.file_uploader("Choose Usage Data file", type=['csv'], key='usage',
                                     help="Upload usage_data.csv with usage statistics")
        
        st.markdown("**4️⃣ Complaints CSV**")
        complaints_file = st.file_uploader("Choose Complaints file", type=['csv'], key='complaints',
                                          help="Upload complaints.csv with complaint records")
    
    # Show upload status
    st.markdown("---")
    st.markdown("### 📊 Upload Status")
    
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        if customers_file:
            st.success("✅ Customers")
        else:
            st.error("❌ Customers")
    
    with status_col2:
        if billing_file:
            st.success("✅ Billing")
        else:
            st.error("❌ Billing")
    
    with status_col3:
        if usage_file:
            st.success("✅ Usage Data")
        else:
            st.error("❌ Usage Data")
    
    with status_col4:
        if complaints_file:
            st.success("✅ Complaints")
        else:
            st.error("❌ Complaints")
    
    st.markdown("---")
    
    # Process button - only show if all files uploaded
    if all([customers_file, billing_file, usage_file, complaints_file]):
        st.success("✅ All files uploaded! Ready to process.")
        
        if st.button("🚀 Process Data & Train Model", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing data and training models..."):
                try:
                    # Load all files
                    customers = pd.read_csv(customers_file)
                    billing = pd.read_csv(billing_file)
                    usage = pd.read_csv(usage_file)
                    complaints = pd.read_csv(complaints_file)
                    
                    # Show preview
                    st.markdown("### ✅ Files Loaded Successfully!")
                    
                    with st.expander("📊 View Data Previews"):
                        tab1, tab2, tab3, tab4 = st.tabs(["Customers", "Billing", "Usage", "Complaints"])
                        
                        with tab1:
                            st.write(f"**Rows:** {len(customers)}")
                            st.dataframe(customers.head(10), use_container_width=True)
                        with tab2:
                            st.write(f"**Rows:** {len(billing)}")
                            st.dataframe(billing.head(10), use_container_width=True)
                        with tab3:
                            st.write(f"**Rows:** {len(usage)}")
                            st.dataframe(usage.head(10), use_container_width=True)
                        with tab4:
                            st.write(f"**Rows:** {len(complaints)}")
                            st.dataframe(complaints.head(10), use_container_width=True)
                    
                    # Merge datasets
                    st.info("🔗 Merging datasets...")
                    master_data = merge_datasets(customers, billing, usage, complaints)
                    
                    if master_data is not None:
                        st.success(f"✅ Master dataset created with {len(master_data)} customers!")
                        
                        # Train models
                        st.info("🤖 Training machine learning models...")
                        model, metrics, preprocessor = train_models(master_data)
                        
                        if model is not None:
                            st.success(f"✅ Model trained successfully! Best model: **{metrics['best_model']}**")
                            
                            # Generate predictions
                            st.info("🔮 Generating predictions...")
                            predictions_data = generate_predictions(master_data, model)
                            
                            if predictions_data is not None:
                                # Save to session state
                                st.session_state.master_data = master_data
                                st.session_state.model = model
                                st.session_state.model_metrics = metrics
                                st.session_state.predictions_data = predictions_data
                                st.session_state.preprocessor = preprocessor
                                
                                st.balloons()
                                
                                st.markdown("""
                                <div class='success-box'>
                                <h3>🎉 Success! All steps completed!</h3>
                                <p>✅ Data merged successfully</p>
                                <p>✅ Models trained and evaluated</p>
                                <p>✅ Predictions generated for all customers</p>
                                <p><b>👉 Navigate to other pages using the sidebar to explore insights!</b></p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Quick stats
                                st.markdown("### 📊 Quick Summary")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Customers", f"{len(predictions_data):,}")
                                with col2:
                                    churn_rate = (predictions_data['churn_flag'].sum() / len(predictions_data)) * 100
                                    st.metric("Churn Rate", f"{churn_rate:.1f}%")
                                with col3:
                                    high_risk = (predictions_data['risk_segment'] == 'High Risk').sum()
                                    st.metric("High Risk", f"{high_risk:,}")
                                with col4:
                                    accuracy = metrics[metrics['best_model'].lower().replace(' ', '_')]['accuracy']
                                    st.metric("Model Accuracy", f"{accuracy:.1%}")
                
                except Exception as e:
                    st.error(f"❌ Error processing files: {str(e)}")
                    import traceback
                    with st.expander("🔍 View Error Details"):
                        st.code(traceback.format_exc())
                    st.info("💡 Please check that your CSV files have the correct format and column names.")
    else:
        st.warning("⚠️ Please upload all 4 required CSV files before processing.")
        
        missing = []
        if not customers_file:
            missing.append("Customers.csv")
        if not billing_file:
            missing.append("Billing.csv")
        if not usage_file:
            missing.append("Usage Data.csv")
        if not complaints_file:
            missing.append("Complaints.csv")
        
        if missing:
            st.error(f"Missing files: **{', '.join(missing)}**")
    
    # Show sample data format
    st.markdown("---")
    st.markdown("### 📝 Sample Data Format")
    
    with st.expander("Click to view expected CSV formats"):
        st.markdown("#### Customers.csv")
        st.code("""customer_id,name,plan_type,region
1001,Asha Mehta,Prepaid,Delhi
1002,Ravi Kumar,Postpaid,Mumbai""")
        
        st.markdown("#### Billing.csv")
        st.code("""customer_id,tenure,contract_type,monthly_charges,churn
1001,12,Month-to-Month,180,Yes
1002,24,One Year,280,No""")
        
        st.markdown("#### Usage Data.csv")
        st.code("""customer_id,data_used_gb,calls_made,revenue_inr
1001,5.2,25,180
1002,12.5,40,280""")
        
        st.markdown("#### Complaints.csv")
        st.code("""customer_id,category,created_at,status
1002,Billing,2025-09-25 10:45,Open
1004,Network,2025-09-25 09:30,Closed""")

# ============================================================================
# PAGE 2: DASHBOARD
# ============================================================================

elif page == "📊 Dashboard":
    if st.session_state.predictions_data is None:
        st.warning("⚠️ Please upload and process data files first from the Home page!")
        st.stop()
    
    predictions_data = st.session_state.predictions_data
    
    st.title("📱 Telecom Churn Analytics Dashboard")
    st.markdown("### Real-time Customer Churn Monitoring & Insights")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(predictions_data)
    churned = predictions_data['churn_flag'].sum()
    churn_rate = (churned / total_customers) * 100
    high_risk = (predictions_data['risk_segment'] == 'High Risk').sum()
    avg_revenue = predictions_data['revenue_inr'].mean()
    
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("Churn Rate", f"{churn_rate:.2f}%", delta=f"-{churned} customers", delta_color="inverse")
    with col3:
        st.metric("High Risk Customers", f"{high_risk:,}")
    with col4:
        st.metric("Avg Revenue", f"₹{avg_revenue:,.2f}")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Risk Segment Distribution")
        risk_counts = predictions_data['risk_segment'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'},
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📞 Churn by Plan Type")
        plan_churn = predictions_data.groupby('plan_type')['churn_flag'].agg(['sum', 'count'])
        plan_churn['rate'] = (plan_churn['sum'] / plan_churn['count'] * 100).round(2)
        
        fig = px.bar(
            x=plan_churn.index,
            y=plan_churn['rate'],
            text=plan_churn['rate'],
            color=plan_churn['rate'],
            color_continuous_scale='Reds'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(xaxis_title="Plan Type", yaxis_title="Churn Rate (%)", showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌍 Regional Churn Distribution")
        region_data = predictions_data.groupby(['region', 'risk_segment']).size().reset_index(name='count')
        fig = px.bar(
            region_data,
            x='region',
            y='count',
            color='risk_segment',
            color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'},
            barmode='stack'
        )
        fig.update_layout(height=350, xaxis_title="Region", yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 💰 Revenue vs Churn Probability")
        sample_data = predictions_data.sample(min(500, len(predictions_data)))
        fig = px.scatter(
            sample_data,
            x='monthly_charges',
            y='churn_prob',
            color='risk_segment',
            size='revenue_inr',
            color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'},
            hover_data=['customer_id', 'plan_type']
        )
        fig.update_layout(height=350, xaxis_title="Monthly Charges (₹)", yaxis_title="Churn Probability")
        st.plotly_chart(fig, use_container_width=True)
    
    # High Risk Customers
    st.markdown("---")
    st.markdown("### 🚨 Top 20 High-Risk Customers")
    
    high_risk_customers = predictions_data.nlargest(20, 'churn_prob')[[
        'customer_id', 'name', 'plan_type', 'region', 'monthly_charges', 'tenure',
        'complaints_total', 'churn_prob', 'risk_segment', 'offer_recommendation'
    ]].copy()
    
    high_risk_customers['churn_prob'] = high_risk_customers['churn_prob'].apply(lambda x: f"{x:.2%}")
    high_risk_customers['monthly_charges'] = high_risk_customers['monthly_charges'].apply(lambda x: f"₹{x:.2f}")
    
    st.dataframe(high_risk_customers, use_container_width=True, height=400)
    
    # Download button
    csv = high_risk_customers.to_csv(index=False)
    st.download_button(
        "📥 Download High-Risk Customers",
        csv,
        f"high_risk_customers_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        use_container_width=True
    )

# ============================================================================
# PAGE 3: ANALYTICS
# ============================================================================

elif page == "📈 Analytics":
    if st.session_state.predictions_data is None:
        st.warning("⚠️ Please upload and process data files first!")
        st.stop()
    
    predictions_data = st.session_state.predictions_data
    
    st.title("📈 Deep Dive Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Trends", "🔍 Feature Analysis", "💡 Insights"])
    
    with tab1:
        st.markdown("### Temporal and Usage Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Tenure Distribution")
            fig = px.histogram(
                predictions_data,
                x='tenure',
                color='churn_flag',
                nbins=30,
                labels={'churn_flag': 'Churned'},
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Data Usage by Churn")
            fig = px.box(
                predictions_data,
                x='churn_flag',
                y='data_used_gb',
                color='churn_flag',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Contract Type Analysis")
        contract_stats = predictions_data.groupby('contract_type').agg({
            'churn_flag': ['sum', 'count', 'mean'],
            'monthly_charges': 'mean',
            'tenure': 'mean'
        }).round(2)
        contract_stats.columns = ['Churned', 'Total', 'Churn Rate', 'Avg Charges', 'Avg Tenure']
        contract_stats['Churn Rate'] = (contract_stats['Churn Rate'] * 100).round(2)
        st.dataframe(contract_stats, use_container_width=True)
    
    with tab2:
        st.markdown("### Feature Correlations")
        
        numeric_cols = ['tenure', 'monthly_charges', 'data_used_gb', 'calls_made',
                       'revenue_inr', 'complaints_total', 'complaints_open']
        
        correlations = predictions_data[numeric_cols + ['churn_flag']].corr()['churn_flag'].drop('churn_flag').sort_values()
        
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            color=correlations.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=500, showlegend=False, xaxis_title="Correlation with Churn")
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Non-Churned Stats**")
            st.dataframe(predictions_data[predictions_data['churn_flag']==0][numeric_cols].describe())
        
        with col2:
            st.markdown("**Churned Stats**")
            st.dataframe(predictions_data[predictions_data['churn_flag']==1][numeric_cols].describe())
    
    with tab3:
        st.markdown("### 💡 Key Insights")
        
        avg_tenure_churned = predictions_data[predictions_data['churn_flag']==1]['tenure'].mean()
        avg_tenure_retained = predictions_data[predictions_data['churn_flag']==0]['tenure'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Tenure Impact:**
            - Churned: {avg_tenure_churned:.1f} months
            - Retained: {avg_tenure_retained:.1f} months
            - Insight: Early churn risk in first year
            """)
        
        with col2:
            st.success("""
            **Recommended Actions:**
            1. Early engagement programs
            2. Loyalty rewards after 12 months
            3. Flexible pricing plans
            4. Enhanced customer support
            """)

# ============================================================================
# PAGE 4: SINGLE PREDICTION
# ============================================================================

elif page == "🔮 Single Prediction":
    if st.session_state.model is None:
        st.warning("⚠️ Please train the model first!")
        st.stop()
    
    model = st.session_state.model
    predictions_data = st.session_state.predictions_data
    
    st.title("🔮 Individual Churn Prediction")
    st.markdown("### Enter customer details for prediction")
    
    col1, col2, col3 = st.columns(3)
    
    # Get unique regions from data if available
    if predictions_data is not None:
        available_regions = sorted(predictions_data['region'].unique().tolist())
    else:
        available_regions = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Pune", "Bengaluru", "Hyderabad", "Jaipur", "Chandigarh"]
    
    with col1:
        plan_type = st.selectbox("Plan Type", ["Prepaid", "Postpaid"])
        region = st.selectbox("Region", available_regions)
        contract_type = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
    
    with col2:
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        monthly_charges = st.number_input("Monthly Charges (₹)", 100, 5000, 200, 50)
    
    with col3:
        data_used_gb = st.number_input("Data Used (GB)", 0.0, 100.0, 5.0, 1.0)
        calls_made = st.number_input("Calls Made", 0, 500, 25, 5)
        revenue_inr = st.number_input("Revenue (₹)", 100.0, 5000.0, 200.0, 50.0)
    
    col1, col2 = st.columns(2)
    with col1:
        complaints_total = st.number_input("Total Complaints", 0, 20, 0)
    with col2:
        complaints_open = st.number_input("Open Complaints", 0, 10, 0)
    
    if st.button("🎯 Predict", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            'plan_type': plan_type,
            'region': region,
            'contract_type': contract_type,
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'data_used_gb': data_used_gb,
            'calls_made': calls_made,
            'revenue_inr': revenue_inr,
            'complaints_total': complaints_total,
            'complaints_open': complaints_open
        }])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        if probability < 0.3:
            risk = "Low Risk"
        elif probability < 0.7:
            risk = "Medium Risk"
        else:
            risk = "High Risk"
        
        if probability > 0.8:
            offer = "Upgrade Plan / Retention Call"
        elif probability > 0.6:
            offer = "Offer Discount"
        else:
            offer = "No Action Needed"
        
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("### ⚠️ WILL CHURN")
            else:
                st.success("### ✅ WILL STAY")
        
        with col2:
            st.metric("Churn Probability", f"{probability:.1%}")
        
        with col3:
            if risk == "High Risk":
                st.markdown(f'<div class="risk-high"><b>Risk:</b> {risk}</div>', unsafe_allow_html=True)
            elif risk == "Medium Risk":
                st.markdown(f'<div class="risk-medium"><b>Risk:</b> {risk}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low"><b>Risk:</b> {risk}</div>', unsafe_allow_html=True)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Churn Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if probability > 0.7 else "orange" if probability > 0.3 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 70}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 💡 Recommended Action")
        if offer == "Upgrade Plan / Retention Call":
            st.error(f"**{offer}**")
            st.markdown("""
            - Schedule immediate retention call
            - Offer exclusive upgrade with discount
            - Assign dedicated account manager
            """)
        elif offer == "Offer Discount":
            st.warning(f"**{offer}**")
            st.markdown("""
            - Send personalized discount offer (15-20%)
            - Survey customer satisfaction
            """)
        else:
            st.success(f"**{offer}**")
            st.markdown("""
            - Customer is in good standing
            - Continue regular engagement
            """)

# ============================================================================
# PAGE 5: BULK PREDICTION
# ============================================================================

elif page == "📁 Bulk Prediction":
    if st.session_state.model is None:
        st.warning("⚠️ Please train the model first!")
        st.stop()
    
    model = st.session_state.model
    
    st.title("📁 Bulk Churn Prediction")
    st.markdown("### Upload CSV for batch predictions")
    
    st.info("""
    **Required columns:** plan_type, region, contract_type, tenure, monthly_charges,
    data_used_gb, calls_made, revenue_inr, complaints_total, complaints_open
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.markdown("#### 📊 Uploaded Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🎯 Generate Predictions", type="primary", use_container_width=True):
                with st.spinner("Generating predictions..."):
                    required_cols = ['plan_type', 'region', 'contract_type', 'tenure', 'monthly_charges',
                                   'data_used_gb', 'calls_made', 'revenue_inr', 'complaints_total', 'complaints_open']
                    
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        st.error(f"Missing columns: {', '.join(missing_cols)}")
                        st.stop()
                    
                    X = df[required_cols]
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)[:, 1]
                    
                    df['churn_prediction'] = predictions
                    df['churn_probability'] = probabilities
                    df['risk_segment'] = pd.cut(probabilities, bins=[0, 0.3, 0.7, 1.0],
                                                labels=['Low Risk', 'Medium Risk', 'High Risk'])
                    
                    def rec_offer(prob):
                        if prob > 0.8:
                            return "Upgrade Plan / Retention Call"
                        elif prob > 0.6:
                            return "Offer Discount"
                        return "No Action Needed"
                    
                    df['offer_recommendation'] = probabilities.apply(rec_offer)
                    
                    st.success(f"✅ Predictions generated for {len(df)} customers!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", len(df))
                    with col2:
                        st.metric("Predicted Churners", (predictions == 1).sum())
                    with col3:
                        st.metric("High Risk", (df['risk_segment'] == 'High Risk').sum())
                    
                    st.markdown("#### 📊 Results")
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Predictions",
                        csv,
                        f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        risk_dist = df['risk_segment'].value_counts()
                        fig = px.pie(values=risk_dist.values, names=risk_dist.index,
                                    title="Risk Distribution",
                                    color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.histogram(df, x='churn_probability', nbins=30,
                                         title="Probability Distribution")
                        st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE 6: MODEL INFO
# ============================================================================

elif page == "⚙️ Model Info":
    if st.session_state.model_metrics is None:
        st.warning("⚠️ Please train the model first!")
        st.stop()
    
    metrics = st.session_state.model_metrics
    
    st.title("⚙️ Model Information")
    
    st.markdown("### 🏆 Best Model")
    st.success(f"**Selected Model:** {metrics['best_model']}")
    
    st.markdown("### 📊 Performance Comparison")
    
    lr_metrics = metrics['logistic_regression']
    dt_metrics = metrics['decision_tree']
    
    comparison_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Logistic Regression': [
            f"{lr_metrics['accuracy']:.4f}",
            f"{lr_metrics['precision']:.4f}",
            f"{lr_metrics['recall']:.4f}",
            f"{lr_metrics['f1_score']:.4f}"
        ],
        'Decision Tree': [
            f"{dt_metrics['accuracy']:.4f}",
            f"{dt_metrics['precision']:.4f}",
            f"{dt_metrics['recall']:.4f}",
            f"{dt_metrics['f1_score']:.4f}"
        ]
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        metrics_viz = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Logistic Regression': [lr_metrics['accuracy'], lr_metrics['precision'],
                                   lr_metrics['recall'], lr_metrics['f1_score']],
            'Decision Tree': [dt_metrics['accuracy'], dt_metrics['precision'],
                            dt_metrics['recall'], dt_metrics['f1_score']]
        })
        
        fig = px.bar(metrics_viz.melt(id_vars='Metric', var_name='Model', value_name='Score'),
                    x='Metric', y='Score', color='Model', barmode='group',
                    title="Performance Comparison")
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[lr_metrics['accuracy'], lr_metrics['precision'],
               lr_metrics['recall'], lr_metrics['f1_score']],
            theta=categories, fill='toself', name='Logistic Regression'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[dt_metrics['accuracy'], dt_metrics['precision'],
               dt_metrics['recall'], dt_metrics['f1_score']],
            theta=categories, fill='toself', name='Decision Tree'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                         title="Radar Chart")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📚 Metric Explanations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Accuracy:** Overall correct predictions")
        st.info("**Precision:** Accuracy of churn predictions")
    
    with col2:
        st.info("**Recall:** Percentage of churners caught")
        st.info("**F1 Score:** Balance of precision and recall")
    
    st.markdown("---")
    st.markdown("### 🔧 Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Features Used")
        st.code("""
Numeric:
• tenure, monthly_charges
• data_used_gb, calls_made
• revenue_inr
• complaints_total, complaints_open

Categorical:
• contract_type
• plan_type, region
        """)
    
    with col2:
        st.markdown("#### Pipeline")
        st.code("""
1. StandardScaler (numeric)
2. OneHotEncoder (categorical)
3. ColumnTransformer

Split: 80% train, 20% test
        """)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Date", metrics['evaluation_date'])
    
    if st.session_state.predictions_data is not None:
        with col2:
            st.metric("Training Samples", f"{int(len(st.session_state.predictions_data) * 0.8):,}")
        with col3:
            st.metric("Test Samples", f"{int(len(st.session_state.predictions_data) * 0.2):,}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p><b>Telecom Churn Prediction System</b> | Built with Streamlit</p>
    <p>© 2025 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)