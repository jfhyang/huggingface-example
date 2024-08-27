import os
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载Qwen-5B的分词器和模型
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name)
# 编码输入
input_text = "3.9和3.11哪个大？"
input_ids = tokenizer.encode(input_text, return_tensors='pt')


# 模型推理
output = model.generate(input_ids, max_length=100)

# 解码输出
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
