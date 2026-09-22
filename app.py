import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "best.pt"

MAIN_CLASSES = {
    "Recyclable": [
        "plastic", "paper", "cardboard", "carton", "glass", "metal", "can",
        "aluminium", "aluminum", "tin", "bottle", "recycl",
    ],
    "Compost": [
        "food", "organic", "biodegrad", "compost", "fruit", "vegetable",
        "leaf", "leaves", "wood", "peel",
    ],
    "Trash": [
        "trash", "landfill", "garbage", "waste", "other", "styrofoam",
        "polystyrene", "diaper", "battery", "e-waste",
    ],
}

CLASS_COLORS = {
    "Recyclable": (79, 125, 47),
    "Compost": (43, 90, 138),
    "Trash": (58, 58, 178),
}
DEFAULT_CLASS="Trash"

def auto_bin(class_name: str) -> str:
    name = class_name.lower()
    for bin_label, keywords in MAIN_CLASSES.items():
        if any(k in name for k in keywords):
            return bin_label
    return DEFAULT_CLASS

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

def annotate(img_bgr, results, conf_thresh):
    counts = {"Recyclable": 0, "Compost": 0, "Trash": 0}
    rows = []
    
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
            
        cls_name = results.names[int(box.cls[0])]
        bin_label = auto_bin(cls_name)
        color = CLASS_COLORS[bin_label]
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{bin_label} ({conf:.0%})"
        
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (x1, max(y1 - th - 10, 0)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img_bgr, label, (x1 + 4, max(y1 - 6, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
        counts[bin_label] += 1
        rows.append({"Category": bin_label, "Confidence": f"{conf:.1%}"})
        
    return img_bgr, counts, rows

st.set_page_config(page_title="SortAI", page_icon="♻️", layout="wide")
st.title("Automated Waste Sorting Assistant")
st.caption("Classify waste into Recyclable, Compost or Landfill Trash.")

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load `{MODEL_PATH}`. Make sure `best.pt` is in the same directory as `app.py`.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)

tab_upload, tab_camera = st.tabs(["Upload Image", "Capture Photo"])

with tab_upload:
    uploaded = st.file_uploader("Select an image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

with tab_camera:
    snapshot = st.camera_input("Capture an image", label_visibility="collapsed")

source = uploaded or snapshot

if source is None:
    st.info("Add an image or take a snapshot to see which bin it belongs in.")
    c1, c2, c3 = st.columns(3)
    c1.success("🟢 **Recyclable:** Plastic, paper, glass, metal")
    c2.warning("🟤 **Compost:** Food waste, organic matter")
    c3.error("🔴 **Trash:** Non-recyclable landfill waste")
    st.stop()

pil = Image.open(source).convert("RGB")
img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

with st.spinner("Processing image..."):
    results = model.predict(img_bgr, conf=conf_thresh, verbose=False)[0]

annotated, counts, rows = annotate(img_bgr.copy(), results, conf_thresh)

left, right = st.columns([3, 2])

with left:
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection Output", use_container_width=True)

with right:
    total = sum(counts.values())
    if total == 0:
        st.warning("No objects detected above the confidence threshold.")
    else:
        st.subheader(f"Detected {total} Object(s)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Recyclable", counts["Recyclable"])
        m2.metric("Compost", counts["Compost"])
        m3.metric("Trash", counts["Trash"])
        st.dataframe(rows, use_container_width=True, hide_index=True)
