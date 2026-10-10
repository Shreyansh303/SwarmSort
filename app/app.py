"""SwarmSort demo: find waste items in a photo and say which bin each one goes in.

Run from the repo root:
    .venv/Scripts/python -m streamlit run app/app.py

The page is built here; generations, bin rules, image handling, inference and drawing live in app/helpers.py.
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
def get_model(path, fallback_imgsz):
    return h.load_model(path, fallback_imgsz)


@st.cache_data(show_spinner=False)
def get_test_images():
    return [str(p) for p in h.test_images()]


# ---------------------------------------------------------------------------------------------------------------
# Sidebar: generation, model and thresholds
# ---------------------------------------------------------------------------------------------------------------
def sidebar(results_by_gen):
    st.sidebar.header("Settings")
    gens = list(h.GENERATIONS)
    gen = st.sidebar.radio("Generation", gens, index=gens.index(h.DEFAULT_GENERATION),
                           format_func=lambda g: h.GENERATIONS[g]["label"])
    results = results_by_gen[gen]
    labels = list(h.MODELS)
    # Each generation remembers its own arm. Streamlit forgets the state of a widget that is not drawn, so the
    # choice is also kept in a plain session_state entry and restored when the user switches back.
    saved = st.session_state.get(f"arm_{gen}")
    label = st.sidebar.radio("Model", labels, index=labels.index(saved if saved in labels else
                                                                 h.default_model(results)),
                             format_func=lambda k: h.model_label(k, results), key=f"model_{gen}")
    st.session_state[f"arm_{gen}"] = label
    note = h.pairwise_note(results)
    if note:
        st.sidebar.caption(f"Test set: {note}")
    else:
        st.sidebar.caption(f"Test scores unavailable: {h.GENERATIONS[gen]['results'].relative_to(h.REPO).as_posix()} "
                           "was not found.")

    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05,
                             help="Boxes below this confidence are hidden.")
    iou = st.sidebar.slider("IoU threshold (NMS)", 0.10, 0.95, 0.70, 0.05,
                            help="Overlapping boxes of the same class above this IoU are merged into one.")
    return gen, label, conf, iou


# ---------------------------------------------------------------------------------------------------------------
# One image: detect, draw, and give bin advice
# ---------------------------------------------------------------------------------------------------------------
def show_result(data, title, model, conf, iou, imgsz):
    st.subheader(title)
    try:
        image = h.prepare_image(data)
    except h.ImageError as e:
        st.error(f"{title}: {e}")
        return
    try:
        detections, ms = h.detect(model, image, conf, iou, imgsz)
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
def about(results_by_gen, domain_results):
    with st.expander("About this project"):
        st.markdown(
            "SwarmSort finds waste items in a photo and says which bin each one belongs in. "
            "It uses YOLOv8n, a small real-time object detector. "
            "How well a detector learns depends on training hyperparameters such as the learning rate, momentum "
            "and data augmentation, which are usually left at their defaults. "
            "This project tunes six of them with Particle Swarm Optimization (PSO) and compares three models "
            "trained the same way: Ultralytics' defaults (A), the best of a random search (B) and the best PSO "
            "setting (C), where B and C had the same search budget. Every model was scored once on a held-out "
            "test split.")
        st.markdown("**PSO in one line:** a swarm of candidate settings (particles) moves through the search "
                    "space, each pulled towards its own best setting so far and the swarm's best one.")

        gen1 = results_by_gen["gen1"]
        n_images = (gen1 or {}).get("images")
        st.markdown(f"#### Generation 1: studio photos, 6 classes, image size 416\n"
                    "Trained on a public dataset of studio photos of waste with six classes: biodegradable, "
                    "cardboard, glass, metal, paper and plastic."
                    + (f" Test split: {n_images} images." if n_images else ""))
        rows = h.comparison_rows(gen1)
        if rows:
            st.table(rows, hide_index=True)
        else:
            st.info("Generation 1 test results not found (results/test_results.json).")
        if h.models_tied(gen1):
            st.markdown("The three models are statistically tied: every pairwise difference has a 95% CI that "
                        "contains 0. Ultralytics' defaults, themselves tuned on large datasets, were already "
                        "near-optimal on these clean studio photos.")

        st.markdown("#### Generation 2: studio + real-world photos, 7 classes, image size 640\n"
                    "Generation 2 adds real-world litter photos to the training data and a seventh class, OTHER "
                    "(cigarette butts, unlabelled litter, garbage bags, textiles, hazardous bits), which goes to "
                    "the reject bin. It is tested on four sets: all, studio, real_world, and india (photos of "
                    "Indian waste from a separate dataset that was not used for training). Test mAP@50 [95% CI]:")
        rows = h.domain_rows(domain_results)
        if rows:
            st.table(rows, hide_index=True)
        else:
            st.info("Generation 2 test results not found (results/gen2/test_*_results.json).")
        st.markdown(
            "**What we found:** in Generation 2 tuning helped: on the full test set both random search (B) and "
            "PSO (C) are significantly better than the defaults (A), while PSO and random search are tied. "
            "On real-world photos alone the three arms score about the same. The India test set is very hard "
            "for every arm (mAP@50 about 0.03) because there were no Indian photos in the training data.")


def bin_rules():
    with st.expander("Bin rules (India's Solid Waste Management Rules 2016)"):
        st.table([{"Item": name, "Bin": h.BINS[rule["bin"]], "Disposal tip": rule["tip"]}
                  for name, rule in h.BIN_RULES.items()], hide_index=True)
        st.caption("OTHER is detected by Generation 2 only. The rules ask households to keep wet, dry and "
                   "domestic hazardous waste apart; what the reject bin takes differs between cities.")


# ---------------------------------------------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------------------------------------------
def main():
    results_by_gen = {g: h.load_results(cfg["results"]) for g, cfg in h.GENERATIONS.items()}
    domain_results = h.load_domain_results()
    gen, label, conf, iou = sidebar(results_by_gen)

    st.title("♻️ SwarmSort")
    st.caption("YOLOv8n waste detector, comparing default, random-search and PSO-tuned training settings: "
               "show it a photo and it tells you which bin each item goes in "
               "(green = wet / compost, blue = dry recyclables, reject = non-recyclable).")

    path = h.model_path(gen, label)
    if not path.is_file():
        st.error(h.missing_weights_message(gen, label))
        about(results_by_gen, domain_results)
        bin_rules()
        st.stop()
    model = get_model(str(path), h.GENERATIONS[gen]["imgsz"])
    imgsz = h.model_imgsz(model, h.GENERATIONS[gen]["imgsz"])
    classes = h.model_class_names(model)
    st.sidebar.caption(f"Inference runs at image size {imgsz} px, the size this model was trained at."
                       + (f" Classes: {', '.join(classes)}." if classes else ""))

    test_images = get_test_images()
    modes = ["Upload photos", "Camera"] + (["Random test image"] if test_images else [])
    mode = st.radio("Input", modes, horizontal=True)

    if mode == "Upload photos":
        files = st.file_uploader("Photos of waste (JPG, PNG or WebP)", type=["jpg", "jpeg", "png", "webp"],
                                 accept_multiple_files=True)
        if not files:
            st.info("Upload one or more photos to start.")
        for f in files or []:
            show_result(f.getvalue(), f.name, model, conf, iou, imgsz)
    elif mode == "Camera":
        shot = st.camera_input("Take a photo of the waste")
        if shot is not None:
            show_result(shot.getvalue(), "Camera photo", model, conf, iou, imgsz)
    else:
        if st.button("Pick another test image") or st.session_state.get("test_image") not in test_images:
            st.session_state["test_image"] = random.choice(test_images)
        image_path = Path(st.session_state["test_image"])
        st.caption(f"Dataset image from the studio test split (not a live photo): {image_path.name}")
        show_result(image_path.read_bytes(), "Random test image", model, conf, iou, imgsz)

    about(results_by_gen, domain_results)
    bin_rules()


main()
