import requests
from bs4 import BeautifulSoup
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import math




def get_url(website_url: str):
  request = requests.get(website_url)
  soup = BeautifulSoup(request.text, "html.parser")
  urls = []
  for link in soup.find_all("a"):
    urls.append(link.get("href"))

  return urls


def remove_whitespace(text):
  pattern = r"\s+"
  res = re.sub(pattern," ",text)
  return res

def remove_japanese(text):
    japanese_pattern = re.compile(r'[\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\uFF00-\uFFEF]+')
    clean_text = japanese_pattern.sub('', text)
    return clean_text


def filter_url(main_url):
    filter_url = ["https://tai.com.np/lang/jp","https://tai.com.np/privacy-policy"]
    filtered_url = [url for url in main_url if url not in filter_url]
    filtered_url = set(filtered_url)
    filtered_url = list(filtered_url)
    filtered_url

def dataset(filtered_url,data:list):
    for link in filtered_url:
        if link.startswith("https://tai.com.np"):
            try:
                request = requests.get(link)
                soup = BeautifulSoup(request.text, "html.parser")
                text = soup.text
                cleaned_text = remove_whitespace(text)
                cleaned_text = remove_japanese(cleaned_text)
                
                # Initialize variables for chunking
                chunk_size = 2500
                start_index = 0

                while start_index < len(cleaned_text):
                    # Find the end index of the chunk by searching for the nearest end of sentence marker
                    end_index = min(start_index + chunk_size, len(cleaned_text))
                    while end_index < len(cleaned_text) and cleaned_text[end_index] not in ['.', '!', '?']:
                        end_index += 1
                    
                    # Append the chunk along with the source link
                    data.append({
                        "text": cleaned_text[start_index:end_index],
                        "source": link
                    })

                    # Update the start index for the next chunk
                    start_index = end_index

            except Exception as e:
                print("Invalid Url: ", link)

    return data


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
   