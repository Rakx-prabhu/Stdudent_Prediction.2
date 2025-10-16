import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# --- Model Training (cached to run only once) ---
@st.cache_data
def train_model():
    """
    Loads data, trains an improved model, and returns the fitted pipeline and label encoder.
    """
    try:
        df = pd.read_csv('root_data.csv')
    except FileNotFoundError:
        st.error("Error: 'root_data.csv' not found. Please ensure the file is in the same directory as the app.")
        st.stop()
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Standardize yes/no columns to lowercase
    yes_no_columns = ['can work long time before system?', 'self-learning capability?']
    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip()
    
    # Remove rows with missing Suggested Job Role
    df = df.dropna(subset=['Suggested Job Role'])
    
    # Filter out job roles with very few samples (less than 20) for better model performance
    job_counts = df['Suggested Job Role'].value_counts()
    valid_jobs = job_counts[job_counts >= 20].index
    df_filtered = df[df['Suggested Job Role'].isin(valid_jobs)].copy()
    
    # If we still have more than 15 job roles, keep top 15 by frequency
    if len(valid_jobs) > 15:
        top_15_jobs = job_counts.head(15).index
        df_filtered = df_filtered[df_filtered['Suggested Job Role'].isin(top_15_jobs)].copy()
    
    # Split features and target
    X = df_filtered.drop('Suggested Job Role', axis=1)
    y = df_filtered['Suggested Job Role']
    
    # Label Encoding for target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Identify feature types
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Use Gradient Boosting for better performance on multi-class classification
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ])
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, label_encoder, X.columns, accuracy, len(label_encoder.classes_)

# --- Web App Interface ---
st.set_page_config(page_title="AI Career Predictor", page_icon="🎯", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- FIX START ---
# Initialize variables to prevent NameError if train_model() fails before assignment completes
model, label_encoder, feature_columns, accuracy, num_jobs = None, None, None, 0.0, 0 

# Train the model
try:
    model, label_encoder, feature_columns, accuracy, num_jobs = train_model()
except Exception as e:
    st.error(f"Error training model: {str(e)}")
    st.error("Please make sure 'root_data.csv' exists and has the correct format.")
    st.stop()
# --- FIX END ---

st.markdown('<p class="main-header">🎯 AI-Powered Career Predictor</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">Get intelligent career recommendations from {num_jobs} job roles | Model Accuracy: {accuracy:.1%}</p>', unsafe_allow_html=True)

# Display model info in columns
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Job Roles Available", num_jobs)
with col_info2:
    st.metric("Model Accuracy", f"{accuracy:.1%}")
with col_info3:
    st.metric("Features Analyzed", len(feature_columns) if feature_columns is not None else 0) # Added check for feature_columns

st.markdown("---")

# Create tabs for better organization
tab1, tab2 = st.tabs(["📝 Make Prediction", "ℹ️ About"])

# Ensure we have a model before proceeding with prediction logic
if model is None or label_encoder is None:
    st.error("Model could not be loaded or trained. Please check the data file.")
else:
    with tab1:
        with st.form("prediction_form"):
            st.subheader("Enter Your Academic & Skills Profile")
            
            # Academic Performance Section
            st.markdown("### 📚 Academic Performance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                os = st.number_input("Operating Systems (%)", 0, 100, 75)
                algo = st.number_input("Algorithms (%)", 0, 100, 75)
                prog = st.number_input("Programming Concepts (%)", 0, 100, 75)
            
            with col2:
                se = st.number_input("Software Engineering (%)", 0, 100, 75)
                cn = st.number_input("Computer Networks (%)", 0, 100, 75)
                elec = st.number_input("Electronics Subjects (%)", 0, 100, 75)
            
            with col3:
                ca = st.number_input("Computer Architecture (%)", 0, 100, 75)
                math = st.number_input("Mathematics (%)", 0, 100, 75)
                comm = st.number_input("Communication Skills (%)", 0, 100, 75)
            
            st.markdown("---")
            
            # Skills & Experience Section
            st.markdown("### 💡 Skills & Experience")
            col4, col5 = st.columns(2)
            
            with col4:
                hours = st.slider("Hours working per day", 1, 15, 8, 
                                help="Average hours you can dedicate to work daily")
                logical = st.slider("Logical quotient rating", 1, 10, 7,
                                  help="Rate your logical thinking ability")
                hackathons = st.slider("Hackathons attended", 0, 10, 1,
                                     help="Number of hackathons you've participated in")
            
            with col5:
                coding = st.slider("Coding skills rating", 1, 10, 7,
                                 help="Rate your programming proficiency")
                public_speaking = st.slider("Public speaking points", 1, 10, 5,
                                           help="Rate your public speaking confidence")
            
            st.markdown("---")
            
            # Work Preferences Section
            st.markdown("### ⚙️ Work Preferences")
            col6, col7 = st.columns(2)
            
            with col6:
                long_time = st.selectbox("Can work long time before system?", 
                                        ('yes', 'no'),
                                        help="Can you work extended hours on a computer?")
            
            with col7:
                self_learning = st.selectbox("Self-learning capability?", 
                                            ('yes', 'no'),
                                            help="Do you enjoy learning new things independently?")
            
            st.markdown("---")
            submitted = st.form_submit_button("🔮 Predict My Career Path", use_container_width=True)

        # Prediction Logic
        if submitted:
            with st.spinner("🤖 Analyzing your profile..."):
                # Create DataFrame with exact column names from training
                user_data = pd.DataFrame([[
                    os, algo, prog, se, cn, elec, ca, math, comm, 
                    hours, logical, hackathons, coding, public_speaking, 
                    long_time, self_learning
                ]], columns=feature_columns)
                
                # Make prediction
                prediction_encoded = model.predict(user_data)
                prediction_proba = model.predict_proba(user_data)[0]
                prediction = label_encoder.inverse_transform(prediction_encoded)[0]
                
                # Get top 3 predictions
                top_3_indices = prediction_proba.argsort()[-3:][::-1]
                top_3_jobs = label_encoder.inverse_transform(top_3_indices)
                top_3_probs = prediction_proba[top_3_indices]
                
                # Display results
                st.balloons()
                st.success("✅ Analysis Complete!")
                
                st.markdown("### 🎯 Your Best Career Match")
                st.markdown(f"## **{prediction}**")
                st.progress(top_3_probs[0])
                st.caption(f"Confidence: {top_3_probs[0]:.1%}")
                
                st.markdown("---")
                st.markdown("### 📊 Alternative Career Paths")
                
                col_alt1, col_alt2 = st.columns(2)
                with col_alt1:
                    st.info(f"**{top_3_jobs[1]}**")
                    st.progress(top_3_probs[1])
                    st.caption(f"Match: {top_3_probs[1]:.1%}")
                
                with col_alt2:
                    st.info(f"**{top_3_jobs[2]}**")
                    st.progress(top_3_probs[2])
                    st.caption(f"Match: {top_3_probs[2]:.1%}")

    with tab2:
        st.markdown("""
        ### 🎓 About This Career Predictor
        
        This intelligent career recommendation system uses machine learning to analyze your:
        - **Academic Performance** across 9 key subjects
        - **Skills & Experience** in coding, hackathons, and logical thinking
        - **Work Preferences** and capabilities
        
        #### 🤖 Model Details
        - **Algorithm**: Gradient Boosting Classifier
        - **Features**: 16 input parameters
        - **Training Data**: 2000+ student profiles
        - **Job Roles**: Predicting from multiple career paths
        
        #### 📈 How It Works
        1. Enter your academic scores and skill ratings
        2. The model analyzes patterns from thousands of successful professionals
        3. Get your top career match with confidence score
        4. Explore alternative career paths that suit your profile
        
        #### 💡 Tips for Best Results
        - Be honest with your ratings
        - Consider your genuine interests and strengths
        - Use the alternative suggestions to explore related fields
        """)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-learn")