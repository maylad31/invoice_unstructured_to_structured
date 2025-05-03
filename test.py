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

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

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