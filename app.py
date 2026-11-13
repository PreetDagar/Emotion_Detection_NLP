
import streamlit as st
import numpy as np

from predict import (
    vectorizer,
    model,
    preprocess_text,
    reverse_mapping
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Emotion AI",
    page_icon="◈",
    layout="wide"
)

# ---------------- STYLING ----------------

st.markdown("""
<style>
.stApp {
    background: #09070f;
    color: #eeeeF5;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1050px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    color: #eeeeF5;
}

.main-title {
    font-size: 38px;
    font-weight: 600;
    letter-spacing: -1.5px;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #888397;
    font-size: 14px;
    margin-bottom: 35px;
}

textarea {
    background: #15111f !important;
    color: white !important;
    border: 1px solid #392650 !important;
    border-radius: 12px !important;
}

textarea:focus {
    border: 1px solid #a855f7 !important;
    box-shadow: 0 0 0 1px #a855f7 !important;
}

.stButton > button {
    background: #7627b9;
    color: white;
    border: 1px solid #913bd4;
    border-radius: 10px;
    height: 46px;
    font-size: 15px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #913bd4;
    border-color: #b66aef;
    color: white;
}

.result-card {
    background: linear-gradient(145deg, #1b1029, #100d19);
    border: 1px solid #572b78;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 25px;
}

.result-emoji {
    font-size: 48px;
}

.result-name {
    color: #d8a4ff;
    font-size: 29px;
    margin-top: 8px;
    font-weight: 500;
    text-transform: capitalize;
}

.confidence {
    color: #a78bfa;
    font-size: 16px;
    margin-top: 8px;
}

.score-card {
    background: #15111f;
    border: 1px solid #30223f;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 10px;
}

.score-label {
    color: #ded7e9;
    font-size: 14px;
}

.score-value {
    color: #c084fc;
    font-size: 14px;
    text-align: right;
}

div[data-testid="stProgressBar"] > div {
    background-color: #30213f;
}

div[data-testid="stProgressBar"] > div > div {
    background-color: #a855f7;
}

hr {
    border-color: #30223f;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- EMOTION DETAILS ----------------

emotion_details = {
    "sadness": "😢",
    "anger": "😡",
    "love": "❤️",
    "surprise": "😲",
    "fear": "😨",
    "joy": "😊"
}


# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">◈ Emotion AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">NLP · Machine Learning · Emotion Classification</div>',
    unsafe_allow_html=True
)


# ---------------- INPUT ----------------

st.markdown("### Enter your text")

with st.form("emotion_form"):

    sentence = st.text_area(
        "Text",
        placeholder="Write something here...",
        height=150,
        max_chars=2000,
        label_visibility="collapsed"
    )

    submitted = st.form_submit_button(
        "Analyze Emotion",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------

if submitted:

    if not sentence.strip():
        st.warning("Please enter some text.")

    elif not preprocess_text(sentence).strip():
        st.warning("Please enter meaningful text.")

    else:
        try:
            cleaned_text = preprocess_text(sentence)

            text_vector = vectorizer.transform([cleaned_text])

            prediction = int(model.predict(text_vector)[0])

            probabilities = model.predict_proba(text_vector)[0]

            scores = {
                reverse_mapping[int(class_id)]: float(probability)
                for class_id, probability
                in zip(model.classes_, probabilities)
            }

            predicted_emotion = reverse_mapping[prediction]

            confidence = scores[predicted_emotion] * 100

            st.session_state["emotion_result"] = {
                "emotion": predicted_emotion,
                "confidence": confidence,
                "scores": scores
            }

        except Exception as error:
            st.error(f"Prediction error: {error}")


# ---------------- RESULTS ----------------

if "emotion_result" in st.session_state:

    result = st.session_state["emotion_result"]

    emotion = result["emotion"]
    confidence = result["confidence"]
    scores = result["scores"]

    st.markdown("---")

    st.markdown("### Prediction")

    st.markdown(f"""
    <div class="result-card">
        <div class="result-emoji">
            {emotion_details[emotion]}
        </div>
        <div class="result-name">
            {emotion}
        </div>
        <div class="confidence">
            {confidence:.2f}% confidence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Emotion distribution")

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for name, score in sorted_scores:

        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(
                f"<div class='score-label'>"
                f"{emotion_details[name]} {name.capitalize()}"
                f"</div>",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"<div class='score-value'>{score * 100:.2f}%</div>",
                unsafe_allow_html=True
            )

        st.progress(score)

    st.caption(
        "Scores represent the model's predicted class probabilities, "
        "not guaranteed real-world accuracy."
    )


# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    "<p style='text-align:center;color:#655b72;font-size:12px;'>"
    "EMOTION AI · POWERED BY LOGISTIC REGRESSION"
    "</p>",
    unsafe_allow_html=True
)
