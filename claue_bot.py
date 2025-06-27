from openai import OpenAI
import os
from time import time
import base64
from pathlib import Path

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="<Your Key>",
)


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

model_name = "claude-3.5-sonnet"

image_path = "datas/PQ7/PQ7_Example1.jpg"
base64_image = encode_image_to_base64(image_path)
data_url = f"data:image/jpeg;base64,{base64_image}"

prompt = """You are a helpful document reading assistant. Your job is to read all the text on the given image of a page, and output an XML representation of what you find on the page. There may be <FILENAME> and <PAGE_NUM> attributes above. Those will be useful in following the instructions below.Here's how you should process the page:1. Try to find a unique document identifier somewhere on the first page, or derive one from the page content or the filename.2. Output the page content in a text-based form that most closely resembles the text you found in the image.3. If you find images in the page, output a description of the image instead of the image itself inside of <IMAGE> tags. Don't count the page image itself. Just count images you find inside the page content itself.4. If you find tables in the page, output the table in JSON lines format, inside of <TABLE> tags.<XML_OUTPUT>"""

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  extra_body={},
  model=f"anthropic/{model_name}",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text":  prompt
        },
        {
          "type": "image_url",
          "image_url": {
            "url": data_url
          }
        }
      ]
    }
  ]
)
if __name__ == "__main__":
    start_time = time()
    os.makedirs(f"PQ7_predict/{model_name.replace('.', '_')}", exist_ok=True)
    with open(f"PQ7_predict/{model_name.replace('.', '_')}/ocr_predict.txt", "w", encoding="utf-8") as f:
        f.write(completion.choices[0].message.content)
    end_time = time()
    time_inference = end_time - start_time
    with open(f"PQ7_predict/{model_name.replace('.', '_')}/ocr_infer_time.txt", "w", encoding="utf-8") as f:
        f.write(f"Inference time: {time_inference:.2f} seconds\n")