import streamlit as st
from PIL import Image
from depth_engine import load_model, get_depth_map
from zone_analyzer import analyze_zones, get_alerts, colorize_depth, alerts_to_speech, draw_zones_on_image
from audio_engine import text_to_audio_file

st.set_page_config(
    page_title="Depth Vision",
    page_icon="./assets/logo.png",
    layout="wide"
)

@st.cache_resource
def get_model():
    return load_model()

model, transform, device = get_model()

st.title("Depth Vision")
st.caption("AI-powered obstacle detection for indoor navigation")
st.divider()

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This tool uses Intel's **MiDaS** model to estimate depth "
        "from a single photo, then analyzes it zone-by-zone to detect "
        "obstacles in indoor spaces — built as an assistive navigation aid."
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. Upload a photo\n"
        "2. MiDaS estimates depth\n"
        "3. Image split into 5 zones + floor\n"
        "4. Alerts generated per zone\n"
        "5. Spoken summary plays automatically"
    )
    st.markdown("---")
    st.markdown("**Built by Raajvir Mehta**")
    st.markdown("[GitHub Repo](https://github.com/raajvir15/Object-detection-using-MiDas)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)

    with st.spinner("Running MiDaS..."):
        depth_map = get_depth_map(image, model, transform, device)

    depth_colored = colorize_depth(depth_map)
    zones = analyze_zones(depth_map)
    alerts, threshold = get_alerts(zones)
    depth_with_zones = draw_zones_on_image(depth_colored, zones, threshold)
    print("draw_zones_on_image called, type:", type(depth_with_zones))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Depth Map")
        st.image(depth_with_zones, use_container_width=True)

    for alert_text, level in alerts:
        if level == "critical":
            st.error(alert_text)
        elif level == "warning":
            st.warning(alert_text)
        else:
            st.success(alert_text)

    speech_text = alerts_to_speech(alerts)
    audio_path = text_to_audio_file(speech_text)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/wav", autoplay=True)

    depth_with_zones = draw_zones_on_image(depth_colored, zones, threshold)
    depth_with_zones.save("Ouput_image.png")