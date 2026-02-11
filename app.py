import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Page Config
st.set_page_config(page_title="AI Text Generator", page_icon="🤖")

st.title("🤖 Simple AI Text Generator")

# Load model (cached)
@st.cache_resource
def load_model():
    model_name = "gpt2"   # change to "./results" if fine-tuned
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# User Input
prompt = st.text_area(
    "Enter your prompt:",
    placeholder="Example: Once upon a time...",
    height=150
)

# Generate Button
if st.button("Generate"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Generating..."):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            output = model.generate(
                **inputs,
                max_length=150,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            text = tokenizer.decode(output[0], skip_special_tokens=True)

        st.subheader("Output")
        st.write(text)

        # Download button
        st.download_button(
            "Download Text",
            data=text,
            file_name="generated_text.txt"
        )

st.markdown("---")
st.caption("Simple UI | Streamlit + Transformers")
