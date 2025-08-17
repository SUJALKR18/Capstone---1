import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import time

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cpu"
model = model.to(device)

def detect_lang(text):
    if not isinstance(text, str):
        return "eng_Latn"
    if any("\u0980" <= ch <= "\u09FF" for ch in text):
        return "ben_Beng"
    elif any("\u0900" <= ch <= "\u097F" for ch in text):
        return "hin_Deva"
    else:
        return "eng_Latn"

def translate_nllb(text):
    src_lang = detect_lang(text)
    if src_lang == "eng_Latn":
        return text
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    translated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_new_tokens=100
    )
    return tokenizer.decode(translated[0], skip_special_tokens=True)

df_original = pd.read_csv("Hi-En-Ba/train.csv")
df_translated = pd.read_csv("translated.csv")
translated_texts = [""] * len(df_original)

for i in range(len(df_translated)):
    translated_texts[i] = df_translated["text"].iloc[i]

batch_size = 500
total_rows = len(df_original)
start_time = time.time()

for start in range(0, total_rows, batch_size):
    end = min(start + batch_size, total_rows)
    batch = df_original.iloc[start:end, :].copy()
    for i, text in enumerate(tqdm(batch["text"], desc=f"Translating rows {start+1}-{end}", unit="row")):
        idx = start + i
        if isinstance(translated_texts[idx], str) and translated_texts[idx].strip():
            continue
        translated_texts[idx] = translate_nllb(text)
        elapsed = time.time() - start_time
        avg_time_per_row = elapsed / (idx + 1)
        remaining_rows = total_rows - (idx + 1)
        eta = remaining_rows * avg_time_per_row
        if (i+1) % 50 == 0 or (idx + 1) == total_rows:
            print(f"Processed {idx + 1}/{total_rows} rows. ETA: {int(eta//60)} min {int(eta%60)} sec")
    df_original["text"] = translated_texts
    df_original.to_csv("translated.csv", index=False)
    print(f"✅ Saved progress up to row {end}")

df_original["text"] = translated_texts
df_original.to_csv("translated.csv", index=False)
print("✅ Translation complete. Saved as translated.csv")
