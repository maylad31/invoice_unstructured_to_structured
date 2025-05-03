# invoice_grpo

## Model Description

Took a small 1.5B model fine-tuned with RL (GRPO on Qwen2.5-Coder) and asked it to extract structured JSON from OCR text based on any user-defined schema. You can find the model and the gguf.(100% local)

## How to Get Started with the Model

Use it in combination with paddleocr. Define any schema and hopefully you get the json. Needs some more work but it still works! Download gguf from https://huggingface.co/MayankLad31/invoice_schema

````
from llama_cpp import Llama
from paddleocr import PaddleOCR
text = ""
ocr = PaddleOCR(use_angle_cls=True, lang='en')
result = ocr.ocr("test_image.jpg", cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        text =  text + line[-1][0]+ "\n"
        
llm = Llama(model_path="inv.Q8_0.gguf",n_ctx=2048)

import re

def extract_largest_json_block(text):
    pattern = r"```json\s*(.*?)\s*```"
    blocks = re.findall(pattern, text, re.DOTALL)
    if not blocks:
        return None
    return max(blocks, key=len)


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return extract_largest_json_block(answer.strip())

messages = [
    {"role": "system", "content": """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
```json 
```
</answer>"""},
    {"role": "user", "content": f"{text}+""""

Extract the data in JSON format using the schema: 

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
} """},
]

output = llm.create_chat_completion(messages,max_tokens=1000)

print(extract_xml_answer(output['choices'][0]['message']['content']))
llm._sampler.close()
llm.close()
````
