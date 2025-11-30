from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("AIzaSyB4TFSZjDZXwA7q9bb4vHXAYKMukFfrvJ8"))

def upload_to_gemini(file_path, mime_type=None):
    """Uploads the given file to Gemini and returns the file object."""
    file = genai.upload_file(path=file_path, mime_type=mime_type)
    st.write(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def get_gemini_response(file, prompt):
    """Generates a response from the Gemini model using the uploaded image and prompt."""
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_output_tokens": 1024,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )

    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    file,  # Image file part
                    prompt  # Text prompt part
                ],
            },
        ]
    )
    response = chat_session.send_message(prompt)
    return response.text

# Streamlit App
st.set_page_config(page_title="Human Attribute Detector")
st.header("Human Attribute Detector")

input_prompt = """
You are an AI trained to analyze human attributes from images with high accuracy. Analyze the uploaded image and return structured details in this format:

1. **Gender**: Male / Female / Non-binary
2. **Age Estimate**: (e.g., 25 years)
3. **Ethnicity**: (e.g., Asian, Caucasian, African, etc.)
4. **Mood**: (e.g., Happy, Sad, Neutral)
5. **Facial Expression**: (e.g., Smiling, Frowning, Neutral)
6. **Glasses**: Yes / No
7. **Beard**: Yes / No
8. **Hair Color**: (e.g., Black, Blonde, Brown)
9. **Eye Color**: (e.g., Blue, Green, Brown)
10. **Headwear**: Yes / No (Specify type if applicable)
11. **Emotions Detected**: (e.g., Joyful, Focused, Angry)
12. **Confidence Level**: Accuracy of prediction in percentage

Ensure all attributes are provided based on the image without any apologies.
"""

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_container_width=True)

submit = st.button("Analyze Attributes")

# Handle submit button click
if submit:
    if uploaded_file is not None:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file = upload_to_gemini(temp_path, mime_type=uploaded_file.type)
        response = get_gemini_response(file, input_prompt)
        
        st.subheader("Analysis Result")
        st.write(response)
        
        os.remove(temp_path)
    else:
        st.error("Please upload an image first.")
