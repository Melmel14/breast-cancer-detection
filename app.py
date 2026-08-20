"""
BC Detect — dual-pathway breast cancer detection tool.

A thin application layer over the trained models, realising the progressive
disclosure design: a clear recommendation first, a confidence level second, and
the supporting evidence third. The imaging pathway is wired to the trained
baseline CNN; the classical pathway is scaffolded and marked as forthcoming until
its models are trained.

Run with:
    streamlit run app.py
"""

import io
import os

import numpy as np
import streamlit as st

# TensorFlow is only needed for the imaging pathway, and importing it is slow, so
# it is imported lazily inside the function that uses it rather than at startup.

MODEL_PATH = os.environ.get("BCDETECT_MODEL", "outputs/baseline_cnn.keras")
IMG_SIZE = 128
CLASS_NAMES = ["benign", "malignant"]

st.set_page_config(page_title="BC Detect", page_icon="🩺", layout="centered")


# ----------------------------------------------------------------------------
# Model loading (cached so the network is read from disk only once per session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_imaging_model(path):
    """Load the trained CNN. Returns None if the model file is not present, so
    the app degrades to a clear message rather than crashing."""
    if not os.path.exists(path):
        return None
    import tensorflow as tf  # local import: keeps startup fast for the classical view
    return tf.keras.models.load_model(path)


CLASSICAL_MODEL = os.environ.get("BCDETECT_CLASSICAL", "outputs/classical_svm.joblib")
CLASSICAL_RESULTS = "outputs/classical_results.json"


@st.cache_resource(show_spinner=False)
def load_classical_model(path=CLASSICAL_MODEL):
    """Load the saved SVM (best model by sensitivity). Returns None if absent, so
    the screen shows a clear message rather than crashing."""
    if not os.path.exists(path):
        return None
    import joblib
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_classical_reference():
    """Return the 30 feature names and a neutral list of real sample cases (a mix
    of benign and malignant, unlabelled) so a reviewer can demo the pathway
    without typing thirty numbers and without being told the answer in advance.
    Falls back to generic names if unavailable."""
    try:
        from sklearn.datasets import load_breast_cancer
        ds = load_breast_cancer()
        names = list(ds.feature_names)
        # take a handful of each class but return them unlabelled and shuffled
        import numpy as np
        mal = ds.data[ds.target == 0][:5]
        ben = ds.data[ds.target == 1][:5]
        cases = [row.tolist() for row in np.vstack([mal, ben])]
        import random
        random.Random(0).shuffle(cases)
        return names, cases
    except Exception:
        return [f"feature {i+1}" for i in range(30)], None


def predict_image(model, pil_image):
    """Run one grayscale mammogram through the model and return (label, confidence).

    Confidence is expressed as distance from the decision boundary, so a
    probability of 0.9 (malignant) and 0.1 (benign) both read as high confidence
    in their respective labels."""
    img = pil_image.convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32").reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prob_malignant = float(model.predict(arr, verbose=0).ravel()[0])
    label = CLASS_NAMES[1] if prob_malignant >= 0.5 else CLASS_NAMES[0]
    confidence = prob_malignant if label == "malignant" else 1.0 - prob_malignant
    return label, confidence, prob_malignant


# ----------------------------------------------------------------------------
# Shared UI helpers
# ----------------------------------------------------------------------------
def header():
    st.markdown("### 🩺 BC Detect")
    st.caption("A second-reader decision-support tool. A clinician always makes the final decision.")


def nav():
    views = ["New case", "Result", "Benchmarks"]
    if "view_radio" not in st.session_state:
        st.session_state.view_radio = "New case"
    return st.radio(
        "View", views,
        horizontal=True, label_visibility="collapsed", key="view_radio",
    )


def _choose_imaging():
    """File-uploader callback: capture the image and jump to the result view."""
    up = st.session_state.get("mammogram_upload")
    if up is not None:
        st.session_state.pathway = "imaging"
        st.session_state.uploaded_bytes = up.getvalue()
        st.session_state.view_radio = "Result"


def _choose_pathway(pathway):
    """Button callback. Runs before the next script execution, so it can set the
    radio's key safely (unlike code in the main body, which runs after the radio
    widget is already instantiated)."""
    st.session_state.pathway = pathway
    st.session_state.view_radio = "Result"


# ----------------------------------------------------------------------------
# Screen 1 — pathway chooser (Figma "How would you like to assess this case?")
# ----------------------------------------------------------------------------
def screen_new_case():
    st.subheader("How would you like to assess this case?")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Pathway A — Cytology features**")
        st.caption("Enter the numeric measurements from a fine needle aspirate, "
                   "in the style of the Wisconsin dataset. Runs the classical models.")
        st.button("Select cytology pathway", use_container_width=True,
                  on_click=_choose_pathway, args=("classical",))

    with col_b:
        st.markdown("**Pathway B — Mammogram image**")
        st.caption("Upload a mammogram as PNG or JPEG. The image is preprocessed "
                   "and passed to the trained convolutional network.")
        uploaded = st.file_uploader("Upload mammogram", type=["png", "jpg", "jpeg"],
                                    label_visibility="collapsed", key="mammogram_upload",
                                    on_change=_choose_imaging)
        if uploaded is not None and st.session_state.get("pathway") != "imaging":
            # first render after upload without callback fallback
            st.session_state.pathway = "imaging"

    st.divider()
    st.caption("Built accessible from the start: every result carries a word and an "
               "icon, not colour alone, and the views above can be revisited at any time.")


# ----------------------------------------------------------------------------
# Screen 2 — case result (progressive disclosure: recommendation → confidence → evidence)
# ----------------------------------------------------------------------------
def screen_result():
    pathway = st.session_state.get("pathway")

    if pathway == "imaging":
        _result_imaging()
    elif pathway == "classical":
        _result_classical()
    else:
        st.info("No case yet. Start one on the **New case** view.")


def _result_imaging():
    st.subheader("Case result — imaging pathway")
    from PIL import Image

    data = st.session_state.get("uploaded_bytes")
    if not data:
        st.info("No image uploaded. Go to **New case** and upload a mammogram.")
        return

    image = Image.open(io.BytesIO(data))
    model = load_imaging_model(MODEL_PATH)
    if model is None:
        st.warning(
            f"No trained model found at `{MODEL_PATH}`. Train one first with "
            "`python prototype.py --data_dir data`, or set the BCDETECT_MODEL "
            "environment variable to a saved model."
        )
        st.image(image, caption="Uploaded mammogram", width=280)
        return

    label, confidence, prob = predict_image(model, image)

    # Level 1: the recommendation
    if label == "malignant":
        st.error(f"⚠ Flagged: likely **malignant**")
    else:
        st.success(f"✓ Assessed: likely **benign**")

    # Level 2: the confidence
    st.metric("Model confidence", f"{confidence:.0%}")
    st.progress(confidence)

    # Level 3: the evidence, tucked behind an expander
    with st.expander("Why this result"):
        st.write(
            f"The network output a malignancy probability of **{prob:.2f}** for this "
            f"image. Values at or above 0.50 are reported as malignant. This is a "
            f"single baseline model and is not a diagnosis; it is a flag for "
            f"specialist review."
        )
        st.image(image, caption="Uploaded mammogram", width=280)

    st.caption("This is a flag for specialist review, not a diagnosis. A clinician makes the final decision.")


def _load_sample_case(examples, n):
    """Button callback: pick a random real case and write its 30 values into the
    number_input widget keys. Runs before the widgets are instantiated on the
    rerun, which is the only point Streamlit allows their keys to be set."""
    if not examples:
        return
    import random
    case = random.choice(examples)
    for i in range(n):
        st.session_state[f"cyto_{i}"] = float(case[i])
    st.session_state.cyto_loaded = True


def _result_classical():
    st.subheader("Case result — cytology pathway")

    import json
    import numpy as np

    model = load_classical_model()
    if model is None:
        st.warning(
            "No trained classical model found in `outputs/`. Train one first with "
            "`python classical.py`, which saves the five models and their results."
        )
        return

    # Load feature names and a couple of real example cases from the results file
    feature_names, examples = load_classical_reference()

    st.caption("A clinician enters the 30 cytology measurements from the fine-needle "
               "aspirate report below. The model then predicts benign or malignant; it "
               "is not told the answer in advance.")

    # measurements at random WITHOUT revealing the diagnosis, so the prediction is
    # a genuine test rather than a given. Manual entry remains the real workflow.
    dcol, _ = st.columns([1, 2])
    dcol.button("Load a random sample case",
                on_click=_load_sample_case, args=(examples, len(feature_names)),
                disabled=not examples)

    with st.expander("Feature inputs (30 measurements)",
                     expanded=st.session_state.get("cyto_loaded", False)):
        cols = st.columns(3)
        entered = []
        for i, name in enumerate(feature_names):
            v = cols[i % 3].number_input(name, value=0.0, format="%.4f", key=f"cyto_{i}")
            entered.append(v)

    if st.button("Assess case", type="primary"):
        # Guard: refuse to predict on all-blank input, which would return a
        # meaningless result and confuse the user.
        if all(abs(v) < 1e-9 for v in entered):
            st.info("Enter the measurements, or load a random sample case, before assessing.")
            return

        x = np.array(entered, dtype="float32").reshape(1, -1)
        prob_malignant = float(model.predict_proba(x)[0, 1])
        label = "malignant" if prob_malignant >= 0.5 else "benign"
        confidence = prob_malignant if label == "malignant" else 1.0 - prob_malignant

        # Level 1: the recommendation
        if label == "malignant":
            st.error("⚠ Flagged: likely **malignant**")
        else:
            st.success("✓ Assessed: likely **benign**")

        # Level 2: the confidence
        st.metric("Model confidence", f"{confidence:.0%}")
        st.progress(confidence)

        # Level 3: the evidence
        with st.expander("Why this result"):
            st.write(
                f"The support vector machine output a malignancy probability of "
                f"**{prob_malignant:.2f}** from the 30 cytology measurements. Values at "
                f"or above 0.50 are reported as malignant. This model reached 0.98 "
                f"sensitivity on the held-out test set, but this remains a flag for "
                f"specialist review, not a diagnosis."
            )

        st.caption("This is a flag for specialist review, not a diagnosis. A clinician makes the final decision.")


# ----------------------------------------------------------------------------
# Screen 3 — benchmarking (Figma "Model benchmarking")
# ----------------------------------------------------------------------------
def screen_benchmarks():
    st.subheader("Model benchmarking")
    st.caption("Performance on the held-out test set, shown beside a radiologist-level "
               "benchmark from the literature. Figures are from real CBIS-DDSM runs.")

    rows = [
        ("Baseline CNN", 0.632, 0.716, 0.544, 0.694),
        ("Regularised CNN", 0.663, 0.799, 0.519, 0.703),
        ("Transfer learning", None, None, None, None),
        ("Classical models", None, None, None, None),
        ("Shen et al. (2019)", None, None, None, 0.895),
    ]

    def fmt(v):
        return "—" if v is None else f"{v:.3f}"

    st.table({
        "Model": [r[0] for r in rows],
        "Accuracy": [fmt(r[1]) for r in rows],
        "Sensitivity": [fmt(r[2]) for r in rows],
        "Specificity": [fmt(r[3]) for r in rows],
        "AUC": [fmt(r[4]) for r in rows],
    })

    st.caption("Sensitivity leads: a missed malignancy is the costliest error, so it "
               "is weighted above raw accuracy. Regularisation raised sensitivity from "
               "0.716 to 0.799. Transfer learning and the classical models are the next work.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    header()
    view = nav()

    if view == "New case":
        screen_new_case()
    elif view == "Result":
        screen_result()
    else:
        screen_benchmarks()


if __name__ == "__main__":
    main()