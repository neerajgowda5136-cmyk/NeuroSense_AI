import streamlit as st
from drawing_analysis import analyze_drawing
from voice_analysis import analyze_voice
from fpdf import FPDF

# --- Streamlit UI ---
st.title("🧠 NeuroSense AI – Cognitive Screening")

# Patient details
st.subheader("Patient Information")
patient_name = st.text_input("Enter Patient Name")
patient_age = st.number_input("Enter Patient Age", min_value=1, max_value=120, step=1)

# Drawing upload
st.subheader("Drawing Analysis")
drawing_file = st.file_uploader("Upload Clock Drawing (PNG/JPG)", type=["png", "jpg", "jpeg"])

# Voice upload
st.subheader("Voice Analysis")
voice_file = st.file_uploader("Upload Voice Sample (WAV/MP3)", type=["wav", "mp3"])

# --- Run Analysis ---
if st.button("Run NeuroSense Analysis"):
    if not patient_name or not patient_age:
        st.error("Please enter patient name and age.")
    elif not drawing_file or not voice_file:
        st.error("Please upload both drawing and voice samples.")
    else:
        # Save uploaded files temporarily
        with open("temp_drawing.png", "wb") as f:
            f.write(drawing_file.read())
        with open("temp_voice.wav", "wb") as f:
            f.write(voice_file.read())

        # Run analyses
        drawing_score, drawing_result = analyze_drawing("temp_drawing.png")
        voice_score, voice_result = analyze_voice("temp_voice.wav")

        if drawing_score is None or voice_score is None:
            st.error("Analysis failed. Check model files or inputs.")
        else:
            combined_index = round((drawing_score + voice_score) / 2, 2)
            st.success("✅ Analysis Completed!")
            st.write(drawing_result)
            st.write(voice_result)
            st.write(f"🧩 Combined NeuroSense Cognitive Index: {combined_index}%")

            # --- Generate PDF Report ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.set_auto_page_break(auto=True, margin=15)

            report_text = f"""
NeuroSense AI – Patient Cognitive Report

Patient Name: {patient_name}
Patient Age: {patient_age}

Drawing Analysis Result:
{drawing_result}

Voice Analysis Result:
{voice_result}

Combined NeuroSense Cognitive Index:
{combined_index}%
"""

            # Encode safely to avoid Unicode errors
            pdf.multi_cell(0, 10, report_text.encode('latin-1', 'replace').decode('latin-1'))
            pdf.output("patient_report.pdf")

            st.success("📄 Patient Report Generated Successfully!")
            st.download_button("Download Report", data=open("patient_report.pdf", "rb"), file_name="patient_report.pdf")
