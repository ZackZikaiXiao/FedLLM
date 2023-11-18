# import os
# import openai
# openai.api_base = "https://api.nextweb.fun/openai/v1"
# openai.api_key = "ak-E2DObBYRRL87VMjIiNkYJrx461XZB0nQxZwERyuzGg7j4FF9"


# completion = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#   ]
# )

# print(completion.choices[0].message)

import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
  '/home/zikaixiao/zikai/flfm/shepherd/alpaca_native',
  device_map='auto',
  load_in_8bit=True,
  max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB')