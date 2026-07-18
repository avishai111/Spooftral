
from transformers import AutoProcessor

model_id = "mistralai/Voxtral-Mini-3B-2507"  # replace with your configs.model_id if different
tok = AutoProcessor.from_pretrained(model_id).tokenizer

for text in ["bonafide", "spoof"]:
    ids_no = tok(text, add_special_tokens=False)["input_ids"]
    ids_yes = tok(text, add_special_tokens=True)["input_ids"]
    print(text, "no_special:", ids_no, tok.convert_ids_to_tokens(ids_no))
    print(text, "with_special:", ids_yes, tok.convert_ids_to_tokens(ids_yes))