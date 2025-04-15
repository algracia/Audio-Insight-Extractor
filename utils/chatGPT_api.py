import os
import time
import re
import openai
import tiktoken
import nltk
from dotenv import load_dotenv
from openai.error import OpenAIError, RateLimitError
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure NLTK
nltk.data.path.append('./nltk_data')
nltk.download('punkt', download_dir='./nltk_data')
nltk.download('stopwords', download_dir='./nltk_data')


class ChatGPTapi:
    def __init__(self):
        try:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables.")
            openai.api_key = api_key
            self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
        except Exception as e:
            print(f"[ERROR] Initialization failed: {e}")
            raise

    def count_tokens_precise(self, messages):
        return sum(len(self.encoder.encode(msg.get("content", ""))) for msg in messages)

    def clean_encoding(self, text):
        text = text.encode("utf-8", "ignore").decode("utf-8")
        return re.sub(r'[^\x00-\x7F]+', ' ', text)

    def remove_textual_noise(self, text):
        text = re.sub(r"(sí|ajá|pues|entonces|bueno|ah|eh)+[,.]*", "", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    def correct_common_errors(self, text):
        corrections = {
            "envigado": "Envigado",
            "más simple que esto": "es más simple que esto"
        }
        for error, correction in corrections.items():
            text = text.replace(error, correction)
        return text

    def preprocess_text(self, text):
        print("[INFO] Preprocessing text...")
        text = self.clean_encoding(text)
        text = self.remove_textual_noise(text)
        text = self.correct_common_errors(text)
        text = re.sub(r'[^a-záéíóúñü\s]', '', text.lower())
        words = word_tokenize(text)
        stop_words = set(stopwords.words('spanish')) - {
            'muy', 'más', 'menos', 'poco', "para", "con", "sobre",
            "cómo", "dónde", "quién", "porque", "entonces",
            "aunque", "implementado", "destacando", "realizar", "mejorar"
        }
        filtered = [w for w in words if w not in stop_words]
        print("[INFO] Preprocessing complete.")
        return ' '.join(filtered)

    def split_by_paragraphs(self, text, max_tokens):
        print("[INFO] Splitting text by paragraphs...")
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            tokens_para = len(self.encoder.encode(para))
            tokens_current = len(self.encoder.encode(current_chunk))

            if tokens_current + tokens_para <= max_tokens:
                current_chunk += f"\n\n{para}"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if tokens_para > max_tokens:
                    words = para.split()
                    sub_chunk = ""
                    for word in words:
                        test_chunk = f"{sub_chunk} {word}".strip()
                        if len(self.encoder.encode(test_chunk)) <= max_tokens:
                            sub_chunk = test_chunk
                        else:
                            chunks.append(sub_chunk)
                            sub_chunk = word
                    if sub_chunk:
                        chunks.append(sub_chunk)
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        print(f"[INFO] Created {len(chunks)} chunks.")
        return chunks

    def analyze_in_parts(self, context, full_text, base_prompt, summary_path):
        try:
            model = "gpt-3.5-turbo-16k"
            max_total_tokens = 16000
            max_response_tokens = 1000
            max_prompt_tokens = len(self.encoder.encode(context)) + len(self.encoder.encode(base_prompt))
            max_input_tokens = max_total_tokens - max_response_tokens - max_prompt_tokens

            clean_text = self.preprocess_text(full_text)
            token_count = len(self.encoder.encode(clean_text))

            chunks = self.split_by_paragraphs(clean_text, max_input_tokens) if token_count > max_input_tokens else [clean_text]
            responses = []

            for i, chunk in enumerate(chunks):
                print(f"[INFO] Processing chunk {i + 1}/{len(chunks)}")
                messages = [{"role": "user", "content": f"{context}\n\n{base_prompt} \n\nText: {chunk}"}]
                retries = 0

                while retries < 6:
                    try:
                        response = openai.ChatCompletion.create(
                            model=model,
                            messages=messages,
                            max_tokens=max_response_tokens
                        )
                        content = response["choices"][0]["message"]["content"].strip()
                        responses.append(content)
                        print(f"[SUCCESS] Chunk {i + 1} processed.")
                        break
                    except RateLimitError:
                        retries += 1
                        print(f"[RateLimit] Retrying {retries}/6 in 10 seconds...")
                        time.sleep(10 + retries * 5)
                    except Exception as retry_error:
                        print(f"[ERROR] Unexpected error: {retry_error}")
                        break

                if retries == 6:
                    print(f"[ERROR] Max retries exceeded for chunk {i + 1}")
                    with open(summary_path, "a", encoding="utf-8") as file:
                        file.write(f"--- Chunk {i + 1} not processed ---\n{chunk}\n\n")

            return "\n\n".join(responses)
        except OpenAIError as api_err:
            print(f"[ERROR] OpenAI API error: {api_err}")
            return None
        except Exception as err:
            print(f"[ERROR] Processing failed: {err}")
            return None
