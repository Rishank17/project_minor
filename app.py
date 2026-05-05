"""
Streamlit Web Application for AttriDiffuser
Text-to-Face Generation Interface
"""
import streamlit as st
import torch
from PIL import Image
import os
from model import SimplifiedAttriDiffuser
from utils import extract_dataset, parse_attributes
import time
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="AttriDiffuser - Face Generation",
    page_icon="🎭",
    layout="wide",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
)

st.title("🎭 AttriDiffuser: Text-to-Face Generation")
st.markdown("""
This application implements a simplified version of **AttriDiffuser** from the research paper:
*"AttriDiffuser: Adversarially enhanced diffusion model for text-to-facial attribute image synthesis"*

Generate realistic face images from text descriptions with multiple attributes!
""")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'generated_image_paths' not in st.session_state:
    st.session_state.generated_image_paths = []
if 'last_prompt' not in st.session_state:
    st.session_state.last_prompt = ""

# Sidebar
st.sidebar.header("⚙️ Settings")

with st.sidebar.expander("⚡ Performance Tips"):
    st.markdown("""
    **Running on CPU - Tips for faster generation:**
    - Use 15-20 inference steps (default: 20)
    - Generate 1 image at a time
    - Lower guidance scale (7.0-7.5)
    - Close other applications

    **Estimated times (CPU):**
    - 15 steps: ~2-3 minutes
    - 20 steps: ~3-4 minutes
    - 30 steps: ~5-7 minutes

    💡 For faster generation, use a GPU-enabled system!
    """)

st.sidebar.markdown("---")

dataset_path = st.sidebar.text_input(
    "Dataset ZIP Path",
    value=r"C:\Users\hp\Downloads\drive-download-20260312T133839Z-3-003.zip",
    help="Path to your local dataset ZIP file"
)

if st.sidebar.button("📦 Extract Dataset"):
    with st.spinner("Extracting dataset..."):
        success = extract_dataset(dataset_path)
        if success:
            st.sidebar.success("✅ Dataset extracted successfully!")
        else:
            st.sidebar.error("❌ Failed to extract dataset")

st.sidebar.header("🤖 Model")
if st.sidebar.button("Load Model"):
    with st.spinner("Loading AttriDiffuser model... This may take a few minutes..."):
        try:
            st.session_state.model = SimplifiedAttriDiffuser()
            st.sidebar.success("✅ Model loaded!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {e}")

# ── Main interface ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Input")

    text_description = st.text_area(
        "Describe the face you want to generate:",
        placeholder="Example: A young female with a smiling expression, brown hair, and blue eyes",
        height=100,
        help="Describe facial attributes like age, gender, expression, hair color, etc."
    )

    with st.expander("💡 Example Descriptions"):
        st.markdown("""
        - "A young male with a serious expression and black hair"
        - "An elderly female with a smiling face and gray hair"
        - "A middle-aged male with glasses and brown hair"
        - "A young female with long blonde hair and green eyes"
        - "An Asian male with short black hair and a neutral expression"
        """)

    st.subheader("Generation Parameters")

    fast_mode = st.checkbox(
        "⚡ Fast Mode (lower quality, faster generation)",
        value=False,
        help="Reduces image size to 256x256 and uses fewer steps for 2-3x faster generation"
    )

    if fast_mode:
        num_images = 1
        num_steps = 15
        st.info("⚡ Fast Mode: 1 image, 15 steps, 256x256 resolution (~1-2 min)")
    else:
        num_images = st.slider(
            "Number of images to generate",
            min_value=1, max_value=4, value=1,
            help="Generate multiple diverse faces from the same description"
        )
        num_steps = st.slider(
            "Quality (inference steps)",
            min_value=10, max_value=50, value=20,
            help="More steps = better quality but slower. 15-25 recommended for CPU."
        )

    guidance_scale = st.slider(
        "Guidance scale",
        min_value=5.0, max_value=15.0, value=7.5, step=0.5,
        help="Higher values = closer to text description"
    )

    generate_btn = st.button("🎨 Generate Face(s)", type="primary", use_container_width=True)

with col2:
    st.header("🖼️ Generated Results")

    if generate_btn:
        if st.session_state.model is None:
            st.warning("⚠️ Please load the model first using the sidebar!")
        elif not text_description.strip():
            st.warning("⚠️ Please enter a text description!")
        else:
            st.session_state.generated_image_paths = []
            st.session_state.last_prompt = text_description

            attributes = parse_attributes(text_description)
            st.info(f"Detected attributes: {attributes}")

            st.markdown("""
                <style>
                .progress-container {
                    width: 100%; background-color: #f0f2f6; border-radius: 10px;
                    padding: 3px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0;
                }
                .progress-bar {
                    height: 30px;
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    border-radius: 8px; transition: width 0.3s ease;
                    display: flex; align-items: center; justify-content: center;
                    color: white; font-weight: bold; font-size: 14px;
                }
                .progress-text {
                    text-align: center; margin-top: 10px;
                    font-size: 16px; color: #667eea; font-weight: 600;
                }
                </style>
            """, unsafe_allow_html=True)

            start_time = time.time()
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            timer_placeholder = st.empty()

            progress_placeholder.markdown("""
                <div class="progress-container">
                    <div class="progress-bar" style="width: 0%;">0%</div>
                </div>
                <div class="progress-text">Initializing generation...</div>
            """, unsafe_allow_html=True)

            generated_images = []

            for img_idx in range(num_images):
                label = (
                    "🎨 Generating your face image..."
                    if num_images == 1
                    else f"🎨 Creating face variation {img_idx + 1}/{num_images}..."
                )
                status_placeholder.info(label)

                def update_progress(step, total_steps, _idx=img_idx):
                    elapsed = time.time() - start_time
                    total_steps_all = num_images * total_steps
                    completed_steps = (_idx * total_steps) + step

                    if completed_steps > 1:
                        avg = elapsed / completed_steps
                        estimated_remaining = avg * (total_steps_all - completed_steps)
                    else:
                        estimated_remaining = 0

                    base_progress = (_idx / num_images) * 100
                    current_progress = (step / total_steps) * (100 / num_images)
                    overall_progress = int(base_progress + current_progress)

                    elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
                    remaining_str = (
                        time.strftime("%M:%S", time.gmtime(estimated_remaining))
                        if estimated_remaining > 0 else "calculating..."
                    )

                    step_label = (
                        f"Step {step}/{total_steps}"
                        if num_images == 1
                        else f"Face {_idx+1}/{num_images} - Step {step}/{total_steps}"
                    )

                    progress_placeholder.markdown(f"""
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {overall_progress}%;">
                                {overall_progress}%
                            </div>
                        </div>
                        <div class="progress-text">{step_label}</div>
                    """, unsafe_allow_html=True)

                    timer_placeholder.markdown(f"""
                        <div style="text-align: center; margin: 10px 0; font-size: 18px;">
                            <span style="color: #667eea; font-weight: bold;">⏱️ Elapsed: {elapsed_str}</span>
                            <span style="margin: 0 20px;">|</span>
                            <span style="color: #764ba2; font-weight: bold;">⏳ Remaining: ~{remaining_str}</span>
                        </div>
                    """, unsafe_allow_html=True)

                guidance = guidance_scale + (img_idx * 0.5)
                image = st.session_state.model.generate_face(
                    text_description,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    callback=update_progress,
                    fast_mode=fast_mode
                )
                if image:
                    generated_images.append(image)

            # Show completion
            final_time = time.time() - start_time
            final_time_str = time.strftime("%M:%S", time.gmtime(final_time))

            progress_placeholder.markdown(f"""
                <div class="progress-container">
                    <div class="progress-bar" style="width: 100%;">100%</div>
                </div>
                <div class="progress-text">
                    {"Generation complete!" if num_images == 1 else "All faces generated!"}
                </div>
            """, unsafe_allow_html=True)

            timer_placeholder.markdown(f"""
                <div style="text-align: center; margin: 10px 0; font-size: 18px;">
                    <span style="color: #28a745; font-weight: bold;">✅ Total Time: {final_time_str}</span>
                </div>
            """, unsafe_allow_html=True)

            status_placeholder.empty()

            # Save images to disk
            if generated_images:
                os.makedirs("generated_faces", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                paths = []
                for idx, img in enumerate(generated_images):
                    path = f"generated_faces/face_{timestamp}_{idx+1}.png"
                    img.save(path)
                    paths.append(path)
                st.session_state.generated_image_paths = paths

            st.success(f"✅ Generated {len(generated_images)} face(s) in {final_time:.2f} seconds!")

    # Display images from saved paths
    if st.session_state.generated_image_paths:
        display_images = [
            Image.open(p)
            for p in st.session_state.generated_image_paths
            if os.path.exists(p)
        ]
        if display_images:
            if len(display_images) == 1:
                st.image(display_images[0], caption="Generated Face", use_container_width=True)
            else:
                img_cols = st.columns(2)
                for idx, img in enumerate(display_images):
                    with img_cols[idx % 2]:
                        st.image(img, caption=f"Variation {idx+1}", use_container_width=True)

            if st.button("💾 Save Images"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_files = []
                for idx, img in enumerate(display_images):
                    filename = f"generated_faces/face_{timestamp}_{idx+1}.png"
                    img.save(filename)
                    saved_files.append(filename)
                st.success(f"✅ Saved {len(display_images)} image(s)!")
                for filename in saved_files:
                    st.text(f"📁 {filename}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
### 📚 About AttriDiffuser
This project is based on the research paper published in Pattern Recognition (2025):
**"AttriDiffuser: Adversarially enhanced diffusion model for text-to-facial attribute image synthesis"**

Key Features:
- Multi-attribute facial control (age, gender, expression, etc.)
- High-fidelity face generation
- Diversity in generated faces
- Attribute-gating cross-attention mechanism

**Reference:** [ScienceDirect Article](https://www.sciencedirect.com/science/article/pii/S0031320325001074)
""")