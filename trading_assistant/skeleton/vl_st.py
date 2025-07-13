import streamlit as st
import tempfile
import os
import io
import time
from PIL import Image
from pdf2image import convert_from_bytes
from vllm import LLM, SamplingParams

# --- Caching the vLLM model so it loads only once ---
@st.cache_resource
def load_model():
    MODEL_PATH = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

    model = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 3},
        mm_processor_kwargs={
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        tensor_parallel_size=2, # two GPUs
        dtype="float16",
        quantization="awq",
        max_model_len=8192,
        max_num_seqs=1,                           # single request at a time
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=2048,
        stop_token_ids=[]
    )

    return model, sampling_params


def time_it(func):
    """Decorator to measure and display execution time in Streamlit."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        st.write(f"Execution time: {elapsed:.2f} seconds")
        return result
    return wrapper


def main():
    st.title("chart-to-code")

    # Load model once
    model, sampling_params = load_model()

    # File uploader and question input
    uploaded_file = st.file_uploader("Upload an image or PDF", type=["png", "jpg", "jpeg", "pdf"] )
    question = st.text_area("Ask a question about the document")

    if st.button("Get Answer"):
        if not uploaded_file or not question.strip():
            st.warning("Please upload a file and enter a question.")
            return

        # Convert upload to list of PIL images
        with tempfile.TemporaryDirectory() as temp_dir:
            images = []
            extension = uploaded_file.name.split('.')[-1].lower()
            data = uploaded_file.read()

            if extension == 'pdf':
                pages = convert_from_bytes(data, dpi=100)
                images = [page.convert('RGB') for page in pages]
            else:
                img = Image.open(io.BytesIO(data)).convert('RGB')
                images = [img]

            # Generate answer
            answer = generate_answer(images, question, model, sampling_params)
            st.markdown("**Answer:**")
            st.write(answer)


@time_it
def generate_answer(images, question, model, sampling_params):
    """Builds the prompt around multiple images and calls vLLM to generate the answer."""
    # 1) System instruction
    system_msg = (
        "You are a TradingView analyst. "
        "Your job is to detect if the following chart represents a trend or sideways action."
    )
    prompt = f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"

    # 2) User part: image placeholders + question
    prompt += "<|im_start|>user\n"
    for _ in images:
        prompt += "<|vision_start|><|image_pad|><|vision_end|>"
    prompt += f"{question}\n<|im_end|>\n<|im_start|>assistant\n"

    # 3) Call the model
    llm_input = {
        "prompt": prompt,
        "multi_modal_data": {"image": images},
    }
    outputs = model.generate([llm_input], sampling_params=sampling_params)
    return outputs[0].outputs[0].text


if __name__ == '__main__':
    main()
