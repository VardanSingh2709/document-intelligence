"""Streamlit UI for the Document Intelligence pipeline. Run with:
    streamlit run frontend/app.py
Requires the FastAPI server to be running separately (uvicorn app.api.main:app).
"""
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Document Intelligence", layout="centered")
st.title("Document Intelligence")
st.caption("Upload a receipt to extract vendor, date, address, and total.")

uploaded_file = st.file_uploader("Choose a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded receipt", width=300)

    if st.button("Extract fields"):
        with st.spinner("Uploading..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            upload_response = requests.post(f"{API_BASE}/documents", files=files)

        if upload_response.status_code != 200:
            st.error(f"Upload failed: {upload_response.text}")
        else:
            document_id = upload_response.json()["document_id"]
            st.session_state["document_id"] = document_id

            with st.spinner("Processing (OCR + model + fallback, may take a few seconds)..."):
                process_response = requests.post(f"{API_BASE}/documents/{document_id}/process")

            if process_response.status_code != 200:
                st.error(f"Processing failed: {process_response.text}")
            else:
                st.session_state["results"] = process_response.json()

if "results" in st.session_state:
    st.divider()
    st.subheader("Extracted fields")

    results = st.session_state["results"]
    source_labels = {
        "model": "🟢 ML model",
        "llm_fallback": "🟡 LLM fallback",
        "model_fallback_failed": "🔴 Needs review",
    }

    for field, data in results["fields"].items():
        source_label = source_labels.get(data["source"], data["source"])
        confidence_text = f"{data['confidence']:.1%}" if data["confidence"] is not None else "N/A"

        col1, col2 = st.columns([3, 2])

        if data["source"] == "model_fallback_failed":
            with col1:
                corrected = st.text_input(
                    f"{field} (needs review)",
                    value=data["value"] or "",
                    placeholder="Enter the correct value",
                    key=f"edit_{field}",
                )
            with col2:
                st.caption(f"🔴 Needs review · model/fallback both failed")
                if st.button("Submit correction", key=f"submit_{field}"):
                    review_response = requests.post(
                        f"{API_BASE}/documents/{st.session_state['document_id']}/review",
                        json={"field": field, "action": "edit", "corrected_value": corrected},
                    )
                    if review_response.status_code == 200:
                        st.success(f"Saved: {field} = {corrected!r}")
                        results["fields"][field]["value"] = corrected
                        results["fields"][field]["source"] = "human_reviewed"
                    else:
                        st.error(f"Failed to save: {review_response.text}")
        else:
            with col1:
                st.text_input(field, value=data["value"] or "(not found)", key=f"display_{field}", disabled=True)
            with col2:
                st.caption(f"{source_label} · confidence: {confidence_text}")