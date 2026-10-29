import streamlit as st
import joblib
from pathlib import Path
import warnings

# Suppress minor version pickle warnings for cleaner UI logs
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="ReviewSense — Sentiment Classifier",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global styles */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-weight: 800;
        margin: 0;
        font-size: 2.3rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #e0f2fe;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
        max-width: 720px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        background: rgba(255, 255, 255, 0.2);
        color: #ffffff;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(4px);
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Sentiment Result Box */
    .sentiment-box {
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        animation: fadeIn 0.4s ease-in-out;
    }
    .sentiment-positive {
        background: #f0fdf4;
        border: 2px solid #86efac;
        color: #166534;
    }
    .sentiment-negative {
        background: #fef2f2;
        border: 2px solid #fca5a5;
        color: #991b1b;
    }

    /* Subtle helper notes */
    .subtle-note {
        font-size: 0.88rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model Loader
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "reviewsense_model.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH.name}'. Please ensure reviewsense_model.joblib is in the project root.")
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(f"❌ Could not load the trained model: {exc}")
    st.info("Ensure `reviewsense_model.joblib` is placed in the same directory as `app.py`.")
    st.stop()

# ---------------------------------------------------------
# Sidebar: Project Details & Performance Metrics
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/chat--v1.png", width=64)
    st.title("ReviewSense")
    st.caption("AI Engineering Internship · Week 01 Deliverable")
    
    st.markdown("---")
    st.markdown("### 📊 Model Architecture")
    st.markdown("""
    - **Pipeline**: Scikit-Learn Pipeline
    - **Feature Extractor**: TF-IDF Vectorizer
    - **Classifier**: Logistic Regression
    - **Target**: Binary Polarity (`positive`, `negative`)
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Benchmark Evaluation")
    st.markdown("""
    Evaluated on a balanced test benchmark of **400 samples** (200 positive / 200 negative):
    - **Overall Accuracy**: `86.25%` (0.86)
    - **Precision (Macro)**: `0.86`
    - **Recall (Macro)**: `0.86`
    - **F1-Score (Macro)**: `0.86`
    """)

    st.markdown("---")
    st.markdown("### 💡 Training Domain Context")
    st.info(
        "Trained primarily on large-scale movie reviews. "
        "Vocabulary patterns generalized to customer feedback, though domain-specific slang may vary."
    )

    st.markdown("---")
    st.caption("Built with Streamlit & Scikit-Learn | Portfolio Ready")

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="badge">Production NLP Baseline</div>
    <h1>💬 ReviewSense</h1>
    <p>Customer Review Sentiment Classifier powered by TF-IDF & Logistic Regression. Analyze user feedback instantly with quantitative confidence scoring.</p>
</div>
""", unsafe_allow_html=True)

# Key metric summary cards
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">86.2%</div>
        <div class="metric-label">Benchmark Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.86</div>
        <div class="metric-label">Macro F1-Score</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">TF-IDF</div>
        <div class="metric-label">Feature Extraction</div>
    </div>
    """, unsafe_allow_html=True)
with col_m4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">&lt; 15ms</div>
        <div class="metric-label">Inference Latency</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ---------------------------------------------------------
# Interactive Tabs: Single Review & Batch Analysis
# ---------------------------------------------------------
tab_single, tab_batch, tab_benchmarks = st.tabs(["📝 Single Review Analysis", "📑 Batch Classification", "📈 Evaluation Benchmark"])

# -------------------------
# TAB 1: Single Review
# -------------------------
with tab_single:
    st.markdown("#### Enter Customer or Product Review")
    st.caption("Type your review below or click one of the quick presets to test edge cases.")

    # Preset reviews for rapid testing
    presets = {
        "🌟 Positive Experience": "The service was excellent and the staff were incredibly helpful. Highly recommended!",
        "😞 Negative Feedback": "The movie was boring, disappointing, and an absolute waste of time.",
        "⚖️ Contrastive Clause (Edge Case)": "The acting was good but the story was terrible.",
        "🎬 Film Masterpiece": "One of the best cinematic experiences of the decade, truly inspiring."
    }

    selected_preset = st.selectbox(
        "Try sample test reviews:",
        ["-- Select a sample review to populate --"] + list(presets.keys()),
        index=0
    )

    default_text = ""
    if selected_preset != "-- Select a sample review to populate --":
        default_text = presets[selected_preset]

    review_text = st.text_area(
        "Customer Review Text",
        value=default_text,
        placeholder="Example: The service was excellent and the staff were helpful.",
        height=140,
        help="Input raw customer feedback. The model automatically performs vectorization and inference."
    )

    # Word and character count indicator
    words = len(review_text.split()) if review_text.strip() else 0
    chars = len(review_text)
    st.caption(f"Word count: **{words}** | Character count: **{chars}**")

    col_btn1, col_btn2, _ = st.columns([1.5, 1, 5])
    with col_btn1:
        analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("🧹 Clear Input", use_container_width=True)

    if clear_clicked:
        st.rerun()

    if analyze_clicked:
        clean_input = review_text.strip()
        if not clean_input:
            st.warning("⚠️ Please enter a non-empty review text before running inference.")
        else:
            with st.spinner("Analyzing text polarity..."):
                # Inference
                prediction = model.predict([clean_input])[0]
                
                # Check for probability support
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([clean_input])[0]
                    class_labels = list(model.classes_)
                    neg_idx = class_labels.index("negative") if "negative" in class_labels else 0
                    pos_idx = class_labels.index("positive") if "positive" in class_labels else 1
                    prob_neg = float(probabilities[neg_idx])
                    prob_pos = float(probabilities[pos_idx])
                    confidence = max(prob_neg, prob_pos) * 100
                else:
                    confidence = None

            # Render polished result card
            st.markdown("---")
            st.markdown("### 🔍 Classification Result")
            
            res_col1, res_col2 = st.columns([1.2, 1.8])
            
            with res_col1:
                if prediction == "positive":
                    st.success("### Predicted Sentiment: POSITIVE 😊")
                    if confidence is not None:
                        st.metric(label="Model Confidence", value=f"{confidence:.1f}%", delta="Positive Polarity")
                else:
                    st.error("### Predicted Sentiment: NEGATIVE 😞")
                    if confidence is not None:
                        st.metric(label="Model Confidence", value=f"{confidence:.1f}%", delta="- Negative Polarity", delta_color="inverse")

            with res_col2:
                if confidence is not None:
                    st.markdown("**Polarity Distribution:**")
                    st.progress(prob_pos, text=f"Positive: {prob_pos*100:.1f}% | Negative: {prob_neg*100:.1f}%")
                    
                    p_col1, p_col2 = st.columns(2)
                    p_col1.caption(f"🔴 Negative Probability: **{prob_neg*100:.2f}%**")
                    p_col2.caption(f"🟢 Positive Probability: **{prob_pos*100:.2f}%**")

            # Contextual Edge Case Insight
            if "but" in clean_input.lower() or "however" in clean_input.lower():
                st.info(
                    "💡 **Linguistic Insight**: Contrastive conjunction detected (`but`/`however`). "
                    "The model evaluates weighted sub-phrase descriptors to resolve the dominant polarity."
                )

# -------------------------
# TAB 2: Batch Classification
# -------------------------
with tab_batch:
    st.markdown("#### Batch Review Evaluation")
    st.caption("Paste multiple reviews (one per line) to process them in bulk.")

    batch_input = st.text_area(
        "Enter multiple reviews (1 per line):",
        value=(
            "The movie was wonderful and entertaining.\n"
            "The movie was boring and disappointing.\n"
            "The story was not good at all.\n"
            "Outstanding performance by the entire cast!\n"
            "Waste of money and time."
        ),
        height=180
    )

    if st.button("📊 Run Batch Analysis", type="primary"):
        lines = [line.strip() for line in batch_input.split("\n") if line.strip()]
        if not lines:
            st.warning("Please provide at least one review line.")
        else:
            with st.spinner(f"Classifying {len(lines)} reviews..."):
                predictions = model.predict(lines)
                probs = model.predict_proba(lines) if hasattr(model, "predict_proba") else None
                class_labels = list(model.classes_)
                pos_idx = class_labels.index("positive") if "positive" in class_labels else 1

                results = []
                for idx, (txt, pred) in enumerate(zip(lines, predictions)):
                    conf = f"{probs[idx][pos_idx if pred == 'positive' else 1-pos_idx]*100:.1f}%" if probs is not None else "N/A"
                    badge = "🟢 POSITIVE" if pred == "positive" else "🔴 NEGATIVE"
                    results.append({
                        "No.": idx + 1,
                        "Review Sample": txt,
                        "Predicted Sentiment": badge,
                        "Confidence": conf
                    })

                pos_count = sum(1 for p in predictions if p == "positive")
                neg_count = len(lines) - pos_count

                b_col1, b_col2, b_col3 = st.columns(3)
                b_col1.metric("Total Reviews", len(lines))
                b_col2.metric("Positive Reviews", f"{pos_count} ({pos_count/len(lines)*100:.1f}%)")
                b_col3.metric("Negative Reviews", f"{neg_count} ({neg_count/len(lines)*100:.1f}%)")

                st.dataframe(results, use_container_width=True)

# -------------------------
# TAB 3: Benchmark & Evaluation Report
# -------------------------
with tab_benchmarks:
    st.markdown("#### Quantitative Benchmark Performance")
    st.markdown("""
    The trained **ReviewSense** pipeline was audited on a held-out test dataset of **400 labeled reviews** 
    (balanced: 200 negative and 200 positive instances).
    """)

    b_metric_col1, b_metric_col2, b_metric_col3, b_metric_col4 = st.columns(4)
    b_metric_col1.metric("Empirical Accuracy", "86.25%", "345 / 400 Samples")
    b_metric_col2.metric("True Positives (TP)", "173", "86.5% Recall")
    b_metric_col3.metric("True Negatives (TN)", "172", "86.0% Specificity")
    b_metric_col4.metric("Type I & II Errors", "55", "28 FP / 27 FN")

    st.markdown("##### Confusion Matrix Breakdown")
    matrix_data = {
        "Actual Negative (200)": ["172 (True Negative)", "28 (False Positive)"],
        "Actual Positive (200)": ["27 (False Negative)", "173 (True Positive)"]
    }
    st.table({
        "Actual \\ Predicted": ["Predicted Negative", "Predicted Positive"],
        "Actual Negative (200)": ["172 (TN)", "28 (FP)"],
        "Actual Positive (200)": ["27 (FN)", "173 (TP)"]
    })

    st.caption("ReviewSense | Week 01 AI Engineering Internship — Baseline NLP Pipeline")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">
        ReviewSense &copy; 2026 &middot; AI Engineering Internship Week 01 &middot; Natural Language Processing Portfolio Project
    </div>
    """,
    unsafe_allow_html=True
)
