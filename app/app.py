"""SwarmSort demo: find waste items in a photo and say which bin each one goes in.

Run from the repo root:
    .venv/Scripts/python -m streamlit run app/app.py

The page is built here; bin rules, image handling, inference and drawing live in app/helpers.py.
"""
import random
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find helpers.py however the app is started
import helpers as h  # noqa: E402

st.set_page_config(page_title="SwarmSort", page_icon="♻️", layout="wide")


# ---------------------------------------------------------------------------------------------------------------
# Cached loading: each model is loaded once per server, the test-image list once per session start
# ---------------------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model...")
def get_model(path):
    return h.load_model(path)


@st.cache_data(show_spinner=False)
def get_test_images():
    return [str(p) for p in h.test_images()]


# ---------------------------------------------------------------------------------------------------------------
# Sidebar: model and thresholds
# ---------------------------------------------------------------------------------------------------------------
def sidebar(results):
    st.sidebar.header("Settings")
    labels = list(h.MODELS)
    label = st.sidebar.radio("Model", labels, index=labels.index(h.default_model(results)),
                             format_func=lambda k: h.model_label(k, results))
    tied = h.models_tied(results)
    if tied:
        st.sidebar.caption("On the test set the three models are statistically tied "
                           "(every paired bootstrap 95% CI of their differences contains 0).")
    elif tied is None:
        st.sidebar.caption("Test scores unavailable: results/test_results.json was not found.")
    else:
        st.sidebar.caption("Some test-set differences are significant; see results/test_comparison.md.")

    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05,
                             help="Boxes below this confidence are hidden.")
    iou = st.sidebar.slider("IoU threshold (NMS)", 0.10, 0.95, 0.70, 0.05,
                            help="Overlapping boxes of the same class above this IoU are merged into one.")
    st.sidebar.caption(f"Inference runs at image size {h.IMGSZ} px, the size the models were trained at.")
    return label, conf, iou


# ---------------------------------------------------------------------------------------------------------------
# One image: detect, draw, and give bin advice
# ---------------------------------------------------------------------------------------------------------------
def show_result(data, title, model, conf, iou):
    st.subheader(title)
    try:
        image = h.prepare_image(data)
    except h.ImageError as e:
        st.error(f"{title}: {e}")
        return
    try:
        detections, ms = h.detect(model, image, conf, iou)
    except Exception as e:  # keep the demo alive whatever goes wrong inside the model
        st.error(f"{title}: inference failed ({e}).")
        return

    left, right = st.columns([3, 2])
    left.image(h.draw_detections(image, detections), width="stretch")
    left.caption("Box colour shows the item type; the bin is listed in the table.")
    with right:
        c1, c2 = st.columns(2)
        c1.metric("Inference time", f"{ms:.0f} ms")
        c2.metric("Items found", len(detections))
        if not detections:
            st.warning("No waste items detected. Try a lower confidence threshold in the sidebar, "
                       "or a clearer, closer photo with the item in the middle.")
            return
        summary = h.bin_summary(detections)  # one metric per bin that has items
        if summary:
            for col, (bin_label, count) in zip(st.columns(len(summary)), summary.items()):
                col.metric(bin_label, f"{count} item{'s' if count != 1 else ''}")
    st.dataframe(h.detection_rows(detections), hide_index=True)


# ---------------------------------------------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------------------------------------------
def about(results):
    with st.expander("About this project"):
        n_images = (results or {}).get("images")
        test_split = f"a held-out test split of {n_images} images" if n_images else "a held-out test split"
        st.markdown(
            "SwarmSort finds waste items in a photo and says which bin each one belongs in. "
            "It uses YOLOv8n, a small real-time object detector, trained on a public dataset of labelled waste "
            "photos with six classes: biodegradable, cardboard, glass, metal, paper and plastic. "
            "How well a detector learns depends on training hyperparameters such as the learning rate, momentum "
            "and data augmentation, which are usually left at their defaults. "
            "This project tunes six of them with Particle Swarm Optimization (PSO) and compares three models "
            "trained the same way: Ultralytics' defaults (A), the best of a random search (B) and the best PSO "
            "setting (C), where B and C had the same search budget. "
            f"All three were scored once on {test_split}.")
        st.markdown("**PSO in one line:** a swarm of candidate settings (particles) moves through the search "
                    "space, each pulled towards its own best setting so far and the swarm's best one.")
        rows = h.comparison_rows(results)
        if rows:
            st.table(rows, hide_index=True)
        else:
            st.info("Test results not found (results/test_results.json); run src/evaluate.py to create them.")
        if h.models_tied(results):
            st.markdown("**Conclusion:** the three models are statistically tied: every pairwise difference "
                        "has a 95% CI that contains 0. Ultralytics' default hyperparameters, themselves tuned "
                        "on large datasets, were already near-optimal here, so neither search gave a "
                        "measurable gain.")


def bin_rules():
    with st.expander("Bin rules (India's Solid Waste Management Rules 2016)"):
        st.table([{"Item": name, "Bin": h.BINS[rule["bin"]], "Disposal tip": rule["tip"]}
                  for name, rule in h.BIN_RULES.items()], hide_index=True)


# ---------------------------------------------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------------------------------------------
def main():
    results = h.load_results()
    label, conf, iou = sidebar(results)

    st.title("♻️ SwarmSort")
    st.caption("YOLOv8n waste detector, comparing default, random-search and PSO-tuned training settings: "
               "show it a photo and it tells you which bin each item goes in "
               "(green = wet / compost, blue = dry recyclables).")

    path = h.model_path(label)
    if not path.is_file():
        st.error(f"Model file not found: app/models/{path.name}. It is a copy of the trained model "
                 "(see app/models/README.md for its source).")
        st.stop()
    model = get_model(str(path))

    test_images = get_test_images()
    modes = ["Upload photos", "Camera"] + (["Random test image"] if test_images else [])
    mode = st.radio("Input", modes, horizontal=True)

    if mode == "Upload photos":
        files = st.file_uploader("Photos of waste (JPG, PNG or WebP)", type=["jpg", "jpeg", "png", "webp"],
                                 accept_multiple_files=True)
        if not files:
            st.info("Upload one or more photos to start.")
        for f in files or []:
            show_result(f.getvalue(), f.name, model, conf, iou)
    elif mode == "Camera":
        shot = st.camera_input("Take a photo of the waste")
        if shot is not None:
            show_result(shot.getvalue(), "Camera photo", model, conf, iou)
    else:
        if st.button("Pick another test image") or st.session_state.get("test_image") not in test_images:
            st.session_state["test_image"] = random.choice(test_images)
        image_path = Path(st.session_state["test_image"])
        st.caption(f"Dataset image from the test split (not a live photo): {image_path.name}")
        show_result(image_path.read_bytes(), "Random test image", model, conf, iou)

    about(results)
    bin_rules()


main()
