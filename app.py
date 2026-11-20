import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PhishGuard · MaxEnt Email Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — modern clean look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

      .block-container { padding-top: 2rem; padding-bottom: 3rem; }

      /* Hero */
      .hero {
          background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #2563eb 100%);
          border-radius: 18px;
          padding: 2.2rem 2.4rem;
          color: #fff;
          margin-bottom: 1.5rem;
          box-shadow: 0 12px 32px rgba(37,99,235,.25);
      }
      .hero h1 { margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
      .hero p  { margin: .4rem 0 0; opacity: .85; font-size: .95rem; }

      /* Metric cards */
      .metric-card {
          background: #fff;
          border: 1px solid #e5e7eb;
          border-radius: 14px;
          padding: 1.1rem 1.2rem;
          box-shadow: 0 1px 3px rgba(0,0,0,.04);
      }
      .metric-card .label { font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; color:#6b7280; font-weight:600;}
      .metric-card .value { font-size:1.65rem; font-weight:700; color:#111827; margin-top:.15rem;}
      .metric-card .sub   { font-size:.75rem; color:#6b7280; margin-top:.1rem;}

      /* Prediction result cards */
      .result-card {
          border-radius: 16px;
          padding: 1.5rem 1.7rem;
          margin: 1rem 0;
          color: #fff;
      }
      .result-ham   { background: linear-gradient(135deg,#059669,#10b981); }
      .result-phish { background: linear-gradient(135deg,#b91c1c,#ef4444); }
      .result-spam  { background: linear-gradient(135deg,#b45309,#f59e0b); }
      .result-card h2 { margin:0; font-size:1.6rem; font-weight:700; }
      .result-card p  { margin:.3rem 0 0; opacity:.9; font-size:.9rem; }
      .result-card .big-prob { font-size:2.4rem; font-weight:800; margin-top:.5rem; }

      /* Pills */
      .pill {
          display:inline-block; padding:.25rem .7rem; border-radius:999px;
          font-size:.75rem; font-weight:600; margin-right:.4rem;
      }
      .pill-ham   { background:#d1fae5; color:#065f46; }
      .pill-phish { background:#fee2e2; color:#991b1b; }
      .pill-spam  { background:#fef3c7; color:#92400e; }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] { gap: 6px; }
      .stTabs [data-baseweb="tab"] {
          border-radius: 10px 10px 0 0; padding: 8px 18px;
          font-weight: 600; font-size: .9rem;
      }

      /* Textarea */
      .stTextArea textarea { border-radius: 12px !important; font-size:.92rem !important; }

      /* Reduce default padding in containers */
      div[data-testid="stVerticalBlock"] > div:has(> div.stButton) { padding: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).resolve().parent / "model"


@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load(ARTIFACT_DIR / "maxent_phishing_model.joblib")
    vectorizer = joblib.load(ARTIFACT_DIR / "tfidf_vectorizer.joblib")
    return model, vectorizer


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


if not (ARTIFACT_DIR / "maxent_phishing_model.joblib").exists():
    st.error("⚠️ Model artifacts not found in `model/`.\n\n")
    st.stop()

model, vectorizer = load_model()

LABELS = ["Ham", "Phish", "Spam"] 
COLOR_MAP = {"Ham": "#10b981", "Phish": "#ef4444", "Spam": "#f59e0b"}
EMOJI = {"Ham": "✅", "Phish": "🎣", "Spam": "📬"}
DESCRIPTION = {
    "Ham": "Legitimate message",
    "Phish": "Phishing attempt — high risk",
    "Spam": "Unsolicited / unwanted message",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ PhishGuard")
    st.caption("MaxEnt email classifier")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🔍 Single Email", "📊 Batch Analysis", "📈 Model Insights", "📖 About"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Model info**")
    st.markdown(
        f"""
        <div style="font-size:.82rem; line-height:1.9; color:#4b5563;">
        <b>Algorithm:</b> Multinomial Logistic Regression (MaxEnt)<br>
        <b>Features:</b> TF-IDF (1–2 grams, 75k)<br>
        <b>Classes:</b> Ham · Phish · Spam<br>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Educational prototype — not a production email security system.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def predict(message: str):
    cleaned = clean_text(message)
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    proba_map = dict(zip(model.classes_, proba))
    return pred, proba_map, cleaned


def prob_bar_chart(proba_map: dict, title: str = "Class Probabilities"):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    classes = LABELS
    values = [proba_map[c] * 100 for c in classes]
    colors = [COLOR_MAP[c] for c in classes]
    bars = ax.barh(classes, values, color=colors, height=0.55)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=11, fontweight="bold", color="#111827",
        )
    ax.set_xlim(0, 110)
    ax.set_xlabel("Probability (%)")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    plt.tight_layout()
    return fig


def result_card(pred: str, proba_map: dict):
    css = {"Ham": "result-ham", "Phish": "result-phish", "Spam": "result-spam"}[pred]
    pct = proba_map[pred] * 100
    st.markdown(
        f"""
        <div class="result-card {css}">
          <p style="opacity:.85; margin:0; font-size:.8rem; letter-spacing:.08em; text-transform:uppercase; font-weight:600;">
             Prediction
          </p>
          <h2>{EMOJI[pred]} {pred.upper()}</h2>
          <p>{DESCRIPTION[pred]}</p>
          <div class="big-prob">{pct:.1f}%</div>
          <p style="opacity:.85; margin:0;">model confidence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🛡️ PhishGuard — Email Threat Detector</h1>
      <p>Maximum Entropy classifier · distinguishes Ham, Phishing, and Spam messages in real time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# PAGE 1 — Single Email
# ===========================================================================
if page == "🔍 Single Email":
    col_in, col_out = st.columns([1.15, 1], gap="large")

    with col_in:
        st.markdown("#### ✍️ Enter an email / message")
        text = st.text_area(
            "Message",
            height=240,
            placeholder=(
                "Paste an email body or subject here…\n\n"
                "Example: 'Dear customer, your account has been suspended. "
                "Click http://secure-verify-login.com to reactivate immediately.'"
            ),
            label_visibility="collapsed",
        )

        c1, c2 = st.columns([1, 1])
        classify = c1.button("🔍 Classify", type="primary", use_container_width=True)
        clear = c2.button("🧹 Clear", use_container_width=True)

        if clear:
            st.rerun()


    # Prefill example if clicked
    if "example_text" in st.session_state and not text:
        text = st.session_state.pop("example_text")

    with col_out:
        st.markdown("#### 📊 Analysis")
        if classify and text.strip():
            with st.spinner("Analyzing message…"):
                pred, proba_map, cleaned = predict(text)

            result_card(pred, proba_map)

            st.markdown("##### Class probability breakdown")
            st.pyplot(prob_bar_chart(proba_map), use_container_width=True)

            # Small detail metrics
            m1, m2, m3 = st.columns(3)
            for col, cls in zip([m1, m2, m3], LABELS):
                col.metric(
                    label=f"{EMOJI[cls]} {cls}",
                    value=f"{proba_map[cls]*100:.1f}%",
                )

            # Risk indicator
            phish_p = proba_map["Phish"]
            if phish_p >= 0.5:
                st.error(f"🚨 **High phishing risk** — {phish_p*100:.1f}% probability of being phishing.")
            elif phish_p >= 0.2:
                st.warning(f"⚠️ **Elevated phishing risk** — {phish_p*100:.1f}% probability of being phishing.")
            else:
                st.success("✅ Low phishing risk detected.")

            with st.expander("🔍 View processed text"):
                st.code(cleaned[:1500] + ("…" if len(cleaned) > 1500 else ""), language="text")

        elif classify and not text.strip():
            st.warning("Please enter some text to classify.")
        else:
            st.info("👈 Enter a message and click **Classify** to see results.")

            # Placeholder illustration
            st.markdown(
                f"""
                <div style="border:2px dashed #d1d5db; border-radius:14px;
                            padding:2.5rem 1.5rem; text-align:center; color:#9ca3af; margin-top:1rem;">
                  <div style="font-size:2.5rem;">🛡️</div>
                  <div style="font-weight:600; color:#6b7280; margin-top:.5rem;">
                    Awaiting input
                  </div>
                  <div style="font-size:.85rem;">
                    Predictions, probabilities and risk level will appear here.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ===========================================================================
# PAGE 2 — Batch Analysis
# ===========================================================================
elif page == "📊 Batch Analysis":
    st.markdown("#### 📊 Classify multiple emails at once")
    st.caption("Upload a CSV with a `text` column, or paste one message per line.")

    tab1, tab2 = st.tabs(["📁 Upload CSV", "📝 Paste lines"])

    batch_texts = []

    with tab1:
        uploaded = st.file_uploader("CSV file with a `text` column", type=["csv"])
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                if "text" not in df_up.columns:
                    st.error("CSV must contain a `text` column.")
                else:
                    batch_texts = df_up["text"].astype(str).tolist()
                    st.success(f"Loaded {len(batch_texts)} messages.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    with tab2:
        raw = st.text_area(
            "One message per line",
            height=180,
            placeholder="Hello, are we still meeting tomorrow?\nClick here to claim your prize!!!",
        )
        if raw.strip():
            batch_texts = [line for line in raw.split("\n") if line.strip()]
            st.info(f"{len(batch_texts)} messages ready.")

    if batch_texts and st.button("🚀 Run batch prediction", type="primary"):
        with st.spinner(f"Classifying {len(batch_texts)} messages…"):
            X = vectorizer.transform([clean_text(t) for t in batch_texts])
            preds = model.predict(X)
            probas = model.predict_proba(X)

        results = []
        for i, (txt, pred, proba) in enumerate(zip(batch_texts, preds, probas)):
            row = {"#": i + 1, "text": txt[:180] + ("…" if len(txt) > 180 else ""),
                   "prediction": pred, "confidence": float(proba.max())}
            for j, cls in enumerate(model.classes_):
                row[f"P({cls})"] = float(proba[j])
            results.append(row)
        res_df = pd.DataFrame(results)

        # Summary metrics
        st.markdown("##### Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total", len(res_df))
        s2.metric("✅ Ham", int((res_df["prediction"] == "Ham").sum()))
        s3.metric("🎣 Phish", int((res_df["prediction"] == "Phish").sum()))
        s4.metric("📬 Spam", int((res_df["prediction"] == "Spam").sum()))

        # Donut
        c1, c2 = st.columns([1, 1.4])
        with c1:
            counts = res_df["prediction"].value_counts().reindex(LABELS).fillna(0)
            fig, ax = plt.subplots(figsize=(4, 4))
            wedges, texts, autotexts = ax.pie(
                counts.values,
                labels=[f"{c}" for c in LABELS],
                colors=[COLOR_MAP[c] for c in LABELS],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops=dict(width=0.42, edgecolor="white"),
                textprops={"fontsize": 10, "fontweight": "600"},
            )
            ax.set_title("Prediction distribution", fontweight="bold")
            st.pyplot(fig, use_container_width=True)

        with c2:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            sns.histplot(res_df["confidence"], bins=12, color="#2563eb", ax=ax, kde=True)
            ax.set_title("Confidence distribution", loc="left", fontweight="bold")
            ax.set_xlabel("Max predicted probability")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # Table with colored prediction
        st.markdown("##### Results")
        styled = res_df.style.map(
            lambda v: f"color:{COLOR_MAP[v]}; font-weight:700",
            subset=["prediction"],
        ).format({c: "{:.4f}" for c in ["confidence"] + [f"P({c})" for c in LABELS]})
        st.dataframe(styled, use_container_width=True, height=340)

        st.download_button(
            "⬇️ Download results as CSV",
            res_df.to_csv(index=False).encode(),
            file_name="phishguard_batch_results.csv",
            mime="text/csv",
        )

# ===========================================================================
# PAGE 3 — Model Insights
# ===========================================================================
elif page == "📈 Model Insights":
    st.markdown("#### 📈 Model performance & interpretability")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        f"""<div class="metric-card">
        <div class="label">Test Accuracy</div>""",
        unsafe_allow_html=True,
    )
    k2.markdown(
        f"""<div class="metric-card">
        <div class="label">Macro-F1</div>""",
        unsafe_allow_html=True,
    )
    k3.markdown(
        f"""<div class="metric-card">
        <div class="label">Phishing Recall</div>
        <div class="sub">Detection rate for phishing</div></div>""",
        unsafe_allow_html=True,
    )
    k4.markdown(
        f"""<div class="metric-card">
        <div class="label">Training Set</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Per-class metrics", "🔀 Confusion matrix", "⚙️ Tuning", "🔑 Top features"]
    )

    # Per-class metrics
    with tab1:
        pc = pd.DataFrame(metrics["per_class"]).T.reset_index().rename(columns={"index": "class"})
        fig, ax = plt.subplots(figsize=(8, 3.6))
        pc_melt = pc.melt(id_vars="class", value_vars=["precision", "recall", "f1"],
                          var_name="metric", value_name="score")
        sns.barplot(data=pc_melt, x="class", y="score", hue="metric",
                    palette=["#2563eb", "#f59e0b", "#10b981"], ax=ax)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title("Per-class performance on test set", loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.dataframe(
            pc.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                             "f1": "{:.4f}", "support": "{:,.0f}"}),
            use_container_width=True, hide_index=True,
        )

    # Confusion matrix
    with tab2:
        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax,
                    cbar_kws={"shrink": .8})
        ax.set_xlabel("Predicted", fontweight="600")
        ax.set_ylabel("Actual", fontweight="600")
        ax.set_title("Confusion Matrix (test set)", loc="left", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

        phish_as_ham = cm[1, 0]
        st.markdown(
            f"""
            - **Phish → Ham** (critical): `{phish_as_ham}` messages — missed phishing.
            - **Ham → Phish** (false alarm): `{cm[0, 1]}` messages.
            - **Phish → Spam** (caught but mislabeled): `{cm[1, 2]}` messages.
            """
        )

    # Tuning
    with tab3:
        tune_df = pd.DataFrame(metrics["tuning"])
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(tune_df["C"], tune_df["macro_f1"], marker="o", label="Macro-F1", linewidth=2)
        ax.plot(tune_df["C"], tune_df["phish_recall"], marker="s", label="Phish Recall", linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("C (regularization)")
        ax.set_ylabel("Validation score")
        ax.set_title("Validation performance vs C", loc="left", fontweight="bold")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.dataframe(
            tune_df.style.format({c: "{:.4f}" for c in ["accuracy", "macro_f1", "phish_recall", "log_loss"]}),
            use_container_width=True, hide_index=True,
        )
        st.success(f"Selected C = **{metrics['best_C']}** (highest validation macro-F1).")

    # Top features
    with tab4:
        cls = st.radio("Class", LABELS, horizontal=True, label_visibility="collapsed")
        feats = pd.DataFrame(metrics["top_features"][cls])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(feats["feature"][::-1], feats["coefficient"][::-1],
                color=COLOR_MAP[cls], height=0.7)
        ax.set_title(f"Top features most associated with '{cls}'", loc="left", fontweight="bold")
        ax.set_xlabel("Coefficient weight")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption(
            "⚠️ These are statistical associations learned from the dataset — not definitive rules. "
        )