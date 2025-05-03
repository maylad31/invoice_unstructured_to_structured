import gradio as gr
from llama_cpp import Llama
from paddleocr import PaddleOCR
import tempfile
import os

# Initialize models
ocr = PaddleOCR(use_angle_cls=True, lang='en')
llm = Llama(model_path="unsloth.Q8_0.gguf", n_ctx=2048)

# Helper: extract JSON from LLM output
def extract_json_from_llm_response(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except:
        return "❌ Could not parse <answer> block from LLM output."

# Main function for Gradio
def process_invoice(image, schema):
    if schema=="":
        return "No schema defined"
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        image.save(temp.name)
        temp_path = temp.name

    # OCR to extract raw text
    text = ""
    try:
        result = ocr.ocr(temp_path, cls=True)
        for res in result:
            for line in res:
                text += line[-1][0] + "\n"
    except Exception as e:
        os.remove(temp_path)
        return f"❌ OCR failed: {e}"

    # Remove temp image
    os.remove(temp_path)

    # Prepare LLM prompt
    messages = [
    {"role": "system", "content": """Respond in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ```json 
    ```
    </answer>"""},
        {"role": "user", "content": f"{text}"+f"""

    Extract the data in JSON format using the schema: 

    {schema} """},
    ]

    # Run model
    try:
        output = llm.create_chat_completion(messages, max_tokens=1000)
        response = output['choices'][0]['message']['content']
        extracted_json = extract_json_from_llm_response(response)
    except Exception as e:
        extracted_json = f"❌ LLM inference failed: {e}"

    return extracted_json

with gr.Blocks(title="GRPO-based Invoice Extractor") as iface:
    gr.Markdown(
        """
        # 🧾 Invoice Data Extractor (GRPO-based Model)

        **This runs a Qwen2.5-Coder-1.5B model I fine-tuned using Unsloth and GRPO-based reinforcement learning, optimized for structured data extraction.**  
        Upload an invoice, define the schema you want in JSON, and extract structured data.

        🔍 Uses `PaddleOCR` + `llama-cpp`.

        ---
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📤 Upload Invoice Image")

        with gr.Column(scale=1):
            schema_input = gr.Textbox(
                label="🧩 Define Your JSON Schema",
                lines=20,
                placeholder=''
            )

    with gr.Row():
        submit_btn = gr.Button("🚀 Extract Data", variant="primary")

    output_box = gr.Textbox(label="📦 Extracted JSON", lines=15)

    submit_btn.click(fn=process_invoice, inputs=[image_input, schema_input], outputs=output_box)

iface.launch()

# Optional: Close model when done
llm._sampler.close()
llm.close()

"""
{
    "invoice_no":"string",
    "issued_to": {
        "name": "string", 
        "address": "string" // Address of the client
    },
    "pay_to": {
        "bank_name": "string",  // Name of the bank
        "name": "string", // Name 
        "account_no": "number" 
    },
    "items":[
        {
            "description": "string",
            "quantity": "number",
            "unit_price": "number",
            "total":"number"
        }
      ],
    "subtotal":"number",
    "total":"number"
    }
"""
