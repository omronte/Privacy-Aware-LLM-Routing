from presidio_analyzer import AnalyzerEngine
analyzer = AnalyzerEngine()

import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM



DATASET_ID = "AY027/MMADE"

TRAIN_SIZE = 300
VALIDATION_SIZE = 100
TEST_SIZE = 100

TOTAL_REQUIRED = (
    TRAIN_SIZE +
    VALIDATION_SIZE +
    TEST_SIZE
)

RANDOM_SEED = 42

SYNTHETIC_PII_EVERY = 4

OUTPUT_DIR = Path("data")

GOLD_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

GOLD_CACHE_FILE = OUTPUT_DIR / "gold_responses.json"



print("=" * 60)
print("LOADING LOCAL GOLD-RESPONSE MODEL")
print("=" * 60)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")
print(f"Model : {GOLD_MODEL}")

tokenizer = AutoTokenizer.from_pretrained(
    GOLD_MODEL
)

model = AutoModelForCausalLM.from_pretrained(
    GOLD_MODEL,
    torch_dtype=(
        torch.float16
        if device == "cuda"
        else torch.float32
    )
)

model.to(device)
model.eval()

print("Gold model loaded.")



def generate_gold_response(text):

    prompt = f"""
You are creating a reference answer for a machine-learning
evaluation dataset.

Read the clinical text below and write a concise factual
summary.

Rules:
- Summarize ONLY information explicitly present in the text.
- Do not invent patient information.
- Do not invent diagnoses.
- Do not invent treatments.
- Do not invent outcomes.
- Do not provide medical advice.
- Keep the answer concise.
- Focus on the main clinical condition, findings,
  intervention/treatment, and outcome when available.
- Return ONLY the summary.

CLINICAL TEXT:
{text}

REFERENCE SUMMARY:
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful clinical text summarization "
                "assistant. Be factual and concise."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return response



print("\n" + "=" * 60)
print("LOADING CLINICAL TEXT DATA")
print("=" * 60)

dataset = load_dataset(
    DATASET_ID,
    split="train"
)

print(
    f"Total source records: {len(dataset)}"
)



records = []

for item in dataset:

    text = item.get("Text")

    if text is None:
        continue

    text = str(text).strip()

    if not text:
        continue

    records.append({
        "text": text
    })


print(
    f"Valid clinical-text records: {len(records)}"
)



random.seed(RANDOM_SEED)

random.shuffle(records)



if len(records) < TOTAL_REQUIRED:

    raise RuntimeError(
        f"Need {TOTAL_REQUIRED} records, "
        f"but only {len(records)} are available."
    )

selected_records = records[:TOTAL_REQUIRED]



OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if GOLD_CACHE_FILE.exists():

    print(
        "\nExisting gold-response cache found."
    )

    with open(
        GOLD_CACHE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        gold_cache = json.load(f)

else:

    gold_cache = {}



print("\n" + "=" * 60)
print("GENERATING LOCAL GOLD RESPONSES")
print("=" * 60)

print(
    f"Records requiring gold responses: "
    f"{len(selected_records)}"
)

for index, item in enumerate(
    selected_records
):

    text = item["text"]

    cache_key = text

    if cache_key in gold_cache:

        print(
            f"[{index + 1}/{TOTAL_REQUIRED}] "
            f"Already generated - skipping"
        )

        continue

    print(
        f"\nGenerating gold response "
        f"[{index + 1}/{TOTAL_REQUIRED}]..."
    )

    try:

        gold_response = generate_gold_response(
            text
        )

        if not gold_response:

            raise RuntimeError(
                "Model returned an empty response."
            )

        gold_cache[cache_key] = gold_response

        with open(
            GOLD_CACHE_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                gold_cache,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"Gold: {gold_response}"
        )

    except Exception as e:

        print(
            f"Gold generation failed: {e}"
        )

        print(
            "Progress has been saved."
        )

        raise



def detect_pii_units(text):
    """
    Detect PII entities in the clinical text.

    Returns the actual text spans that should be kept local.
    """
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=[
            "PERSON",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "DATE_TIME",
            "LOCATION",
            "MEDICAL_LICENSE",
            "US_SSN",
            "IP_ADDRESS",
        ],
    )

    pii_units = []

    for result in results:
        if result.score < 0.85:
            continue

        if result.entity_type not in {
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "MEDICAL_LICENSE",
            "US_SSN",
            "IP_ADDRESS",
        }:
            continue

        entity = text[result.start:result.end].strip()

        if entity and entity not in pii_units:
            pii_units.append(entity)

    return pii_units

def create_record(text, index):

    gold_response = gold_cache.get(text)

    if not gold_response:
        raise RuntimeError(
            "Missing gold response."
        )

    pii_units = detect_pii_units(text)
    query = (
        "Analyze the following clinical text and "
        "provide a concise clinical summary:\n\n"
    )

    if index % SYNTHETIC_PII_EVERY == 0:
        first_name = ("Alex", "Jordan", "Taylor", "Morgan")[index % 4]
        last_name = f"Patient{index:03d}"
        phone_number = f"555-01{index % 100:02d}"
        synthetic_entities = [
            f"{first_name} {last_name}",
            phone_number,
        ]
        query += (
            f"Patient {first_name} {last_name} can be reached at "
            f"{phone_number}.\n\n"
        )
        pii_units.extend(synthetic_entities)

    return {
        "query": query + text,

        "pii_units": pii_units,

        "target_response": gold_response
    }


all_records = [
    create_record(item["text"], index)
    for index, item in enumerate(selected_records)
]

pii_record_count = sum(
    bool(record["pii_units"])
    for record in all_records
)

non_pii_record_count = (
    len(all_records) - pii_record_count
)

print("\n" + "=" * 60)
print("PII DATASET STATISTICS")
print("=" * 60)

print(f"Total records       : {len(all_records)}")
print(f"Records with PII    : {pii_record_count}")
print(f"Records without PII : {non_pii_record_count}")

if all_records:
    print(
        f"PII record ratio    : "
        f"{pii_record_count / len(all_records) * 100:.2f}%"
    )

print("=" * 60)



train_output = all_records[
    :TRAIN_SIZE
]

validation_output = all_records[
    TRAIN_SIZE:
    TRAIN_SIZE + VALIDATION_SIZE
]

test_output = all_records[
    TRAIN_SIZE + VALIDATION_SIZE:
    TOTAL_REQUIRED
]



def save_json(filename, data):

    path = OUTPUT_DIR / filename

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved {len(data)} records -> {path}"
    )


save_json(
    "train.json",
    train_output
)

save_json(
    "validation.json",
    validation_output
)

save_json(
    "test.json",
    test_output
)



print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print(
    f"Training records   : "
    f"{len(train_output)}"
)

print(
    f"Validation records : "
    f"{len(validation_output)}"
)

print(
    f"Test records       : "
    f"{len(test_output)}"
)

print("\nExample training record:")

print(
    json.dumps(
        train_output[0],
        indent=2,
        ensure_ascii=False
    )
)

print("=" * 60)
