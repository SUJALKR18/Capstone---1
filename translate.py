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

df = pd.read_csv("Hi-En-Ba/train.csv")
batch_size = 500
translated_texts = []

total_rows = len(df)
start_time = time.time()

for start in range(0, total_rows, batch_size):
    end = min(start + batch_size, total_rows)
    batch = df.iloc[start:end, :].copy()
    
    print(f"\n--- Translating rows {start+1} to {end} ---")
    
    for i, text in enumerate(tqdm(batch["text"], desc="Translating", unit="row")):
        translated_text = translate_nllb(text)
        translated_texts.append(translated_text)
        elapsed = time.time() - start_time
        avg_time_per_row = elapsed / len(translated_texts)
        remaining_rows = total_rows - len(translated_texts)
        eta = remaining_rows * avg_time_per_row
        if (i+1) % 50 == 0 or (start + i + 1) == total_rows:
            print(f"Processed {len(translated_texts)}/{total_rows} rows. ETA: {int(eta//60)} min {int(eta%60)} sec")
    
    df_temp = df.iloc[:start+batch_size, :].copy()
    df_temp["text"] = translated_texts
    df_temp.to_csv("translated.csv", index=False)
    print(f"✅ Saved progress up to row {start + batch_size}")

df["text"] = translated_texts
df.to_csv("translated.csv", index=False)
print("\n✅ Translation complete. Saved as translated.csv")
