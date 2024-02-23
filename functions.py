import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import math



def get_urls(context):
    url_pattern = r'https?://tai.com.np\S+'
    urls = re.findall(url_pattern, context)
    urls = set(urls)
    for url in urls:
        return url


def load_model(model_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    # tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer


def generate_answer(model_path,context, question):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    prompt = context + " " + question
    ids = tokenizer.encode(f'{prompt}', return_tensors="pt")
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=1024,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    answer = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    return answer
   