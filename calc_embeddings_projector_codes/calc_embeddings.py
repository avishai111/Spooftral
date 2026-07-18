"""
Compute per-file Voxtral embeddings from ASVspoof audio splits.

The script loads audio files, builds model-ready inputs, extracts embeddings
from the selected model stage, and saves embeddings plus metadata for later
aggregation/training.
"""

import csv
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, VoxtralForConditionalGeneration



import torchaudio
print(f"Available torchaudio backends: {torchaudio.list_audio_backends()}")
if "ffmpeg" in torchaudio.list_audio_backends():
    print("Setting torchaudio backend to 'ffmpeg'")
    torchaudio.set_audio_backend("ffmpeg")


# ========== CHOOSE DATASET TYPE AND SETS TO PROCESS ==========
# פרמטרים מרכזיים לריצה (דאטהסט/סטים/מודל/ביצועים)
DATASET_TYPE = 'ASVspoof2021'  # 'ASVspoof2019', 'ASVspoof5', or 'ASVspoof2021'
ASVSPOOF2021_TASK = "DF"  # 'LA' or 'DF' (only for ASVspoof2021)
SETS = ['eval']  # ['train', 'dev', 'eval'] (ASVspoof2021 uses only 'eval')
NUM_WORKERS = 4
DEVICE = "cuda"
REPO_ID = "mistralai/Voxtral-Mini-3B-2507"
BATCH_SIZE = 8
OUTPUT_STAGE = 'audio_projector'  # 'audio_projector', 'llama', 'lm_head'
EMBEDDING_QUESTION = "Is this speech spoof or bonafide?"
RESUME = False  # אם נפל באמצע, ממשיך מקבצים שכבר נשמרו
N_AUDIO_TOKENS = 375  # Voxtral expects 375 audio tokens for 30s input
SKIP_BAD_AUDIO = False  # If True, skip files that fail to decode instead of raising

# מגדירים תיקיית עבודה קבועה עבור נתיבים יחסיים (אם קיימים)
os.chdir('/gpfs0/bgu-benshimo/users/wavishay/voxtral/')

# Audio preprocessing
SAMPLE_RATE = 16_000
N_SAMPLES = SAMPLE_RATE * 30


def _ensure_channel_first(waveform: torch.Tensor) -> torch.Tensor:
    """Ensure waveform is channel-first [C, T] when 2D."""
    if waveform.dim() == 2 and waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.transpose(0, 1)
    return waveform


def _load_audio_mono(path: str, sample_rate: int):
    """Load audio, enforce mono, and resample to the target sample_rate."""
    try:
        waveform, sr = torchaudio.load_with_torchcodec(path)
    except Exception as exc:
        # TorchCodec fails on some valid FLACs; fall back to torchaudio.load (ffmpeg/sox/soundfile).
        try:
            waveform, sr = torchaudio.load(path)
        except Exception as exc2:
            msg = f"Failed to decode audio: {path} -> torchcodec: {exc}; fallback: {exc2}"
            print(msg)
            raise RuntimeError(msg) from exc2
    waveform = _ensure_channel_first(waveform)
    if waveform.dim() == 2:
        assert waveform.shape[0] <= waveform.shape[1], f"Expected waveform as [C, T], got {tuple(waveform.shape)}"
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # downmix to mono
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
        sr = sample_rate
    return waveform.squeeze().numpy(), sr


def _extract_input_features(processor, waveform, sr):
    """Extract fixed-length log-mel features with the processor's feature_extractor."""
    feats = processor.feature_extractor(
        waveform,
        sampling_rate=sr,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=N_SAMPLES,
    )
    return feats["input_features"][0]


class VoxtralStageWrapper(nn.Module):
    """
    Wrapper around VoxtralForConditionalGeneration that exposes intermediate stages.

    Key attributes:
    - model: the underlying Voxtral model instance.

    Key methods:
    - forward(..., output_stage): returns one of:
        * "audio_projector": audio embeddings before Llama.
        * "llama": last_hidden_state from the language model.
        * "lm_head": logits from the LM head.
    - generate(...): proxy to the base model generate.
    """
    def __init__(self, base_model):
        """Store the base Voxtral model."""
        super().__init__()
        self.model = base_model

    def forward(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        logits_to_keep=0,
        output_stage=None,
        **kwargs
    ):
        """Run Voxtral and optionally return audio_projector / llama / lm_head outputs."""
        # אם לא סופקו embedding-ים של טקסט, מפיקים אותם מה-embedding layer
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if input_features is not None:
            # === שלב 1: הפקת הטבעות אודיו ===
            input_features_nochan = input_features.squeeze(1)
            if output_stage == 'audio_projector':
                # מפיקים הטבעות של audio_projector ישירות (ללא מעבר דרך Llama)
                # נניח ש-input_features_nochan.shape = [batch, 128, frames]
                batch_size = input_features_nochan.shape[0]
                audio_embeds_list = []
                for i in range(batch_size):
                    emb = self.model.get_audio_embeds(input_features_nochan[i].unsqueeze(0))  # [frames, 3072]
                    audio_embeds_list.append(emb)
                audio_embeds = torch.stack(audio_embeds_list, dim=0)  # [batch, frames, 3072]
                return audio_embeds
            else:
                audio_embeds = self.model.get_audio_embeds(input_features)

            # משבצים את הטבעות האודיו במקום טוקני האודיו ברצף הטקסט
            audio_token_mask = input_ids == self.model.config.audio_token_id
            inputs_embeds[audio_token_mask] = audio_embeds

        # מעבר דרך הטרנספורמר הלשוני (Llama)
        llama_outputs = self.model.language_model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        if output_stage == 'llama':
            return llama_outputs.last_hidden_state

        # הפקת לוגיטים אם צריך
        logits = self.model.language_model.lm_head(llama_outputs.last_hidden_state)
        if output_stage == 'lm_head':
            return logits

        return logits

    def generate(self, *args, **kwargs):
        """Proxy to the base model generate."""
        return self.model.generate(*args, **kwargs)

# =================== DATASET CLASS ===================
class ASVspoofAudioDataset(Dataset):
    """
    Dataset that builds Voxtral-compatible inputs for ASVspoof audio files.

    Key attributes:
    - audio_files: list of audio file paths to load.
    - labels_dict: metadata dict keyed by file_id (label/attack/speaker/etc.).
    - processor: HF processor with tokenizer + feature_extractor.
    - n_audio_tokens: fixed count of [AUDIO] tokens to insert.
    - _bos_id/_inst_id/_endinst_id/_begin_aud_id/_aud_frame_id: special token ids.
    - _audio_block: tensor of length n_audio_tokens filled with [AUDIO] id.
    - _prompt_ids: tokenized prompt/question ids.
    - _input_ids: full static template ids (BOS/INST + audio block + prompt + [/INST]).

    Key methods:
    - __getitem__: loads audio, extracts features, and returns input_ids/input_features + metadata.
    - _build_input_ids: builds the static token template once.
    """
    def __init__(self, audio_files, processor, device, labels_dict, embed_question, n_audio_tokens=N_AUDIO_TOKENS):
        """Initialize metadata, tokenizer tokens, and cached prompt ids."""
        self.audio_files = audio_files
        self.processor = processor
        self.device = device
        self.labels_dict = labels_dict
        self.embed_question = embed_question
        self.n_audio_tokens = n_audio_tokens

        tokenizer = self.processor.tokenizer
        vocab = tokenizer.get_vocab()
        # === שלב 1: שליפת טוקנים מיוחדים ===
        self._bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else vocab.get("<s>")
        self._inst_id = vocab.get("[INST]")
        self._endinst_id = vocab.get("[/INST]")
        self._begin_aud_id = vocab.get("[BEGIN_AUDIO]")
        self._aud_frame_id = vocab.get("[AUDIO]")

        missing = []
        if self._bos_id is None:
            missing.append("<s>")
        if self._inst_id is None:
            missing.append("[INST]")
        if self._endinst_id is None:
            missing.append("[/INST]")
        if self._aud_frame_id is None:
            missing.append("[AUDIO]")
        if missing:
            raise ValueError(f"Missing required tokens in tokenizer vocab: {missing}")

        # === שלב 2: בניית בלוק טוקנים קבוע לאודיו ===
        self._bos_inst_tensor = torch.tensor([self._bos_id, self._inst_id], dtype=torch.long)
        self._begin_aud_tensor = torch.tensor([self._begin_aud_id], dtype=torch.long) if self._begin_aud_id is not None else None
        self._audio_block = torch.full((self.n_audio_tokens,), self._aud_frame_id, dtype=torch.long)

        # === שלב 3: טוקניזציה חד־פעמית לשאלה ===
        self._prompt_ids = tokenizer(
            self.embed_question,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]
        self._input_ids = self._build_input_ids(self._prompt_ids)

    def _build_input_ids(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Assemble BOS/INST + [BEGIN_AUDIO] + audio tokens + prompt + [/INST]."""
        pieces = [self._bos_inst_tensor]
        if self._begin_aud_tensor is not None:
            pieces.append(self._begin_aud_tensor)
        pieces.append(self._audio_block)
        pieces.append(prompt_ids)
        pieces.append(torch.tensor([self._endinst_id], dtype=torch.long))
        return torch.cat(pieces, dim=0)

    def __len__(self):
        """Return dataset length."""
        return len(self.audio_files)

    def __getitem__(self, idx):
        """Load audio, extract features, and return model inputs + metadata."""
        attempts = 0
        while True:
            # === שלב 1: משיכת מטא־דאטה ===
            audio_path = self.audio_files[idx]
            file_id = os.path.splitext(os.path.basename(audio_path))[0]
            label_info = self.labels_dict.get(file_id, {})
            label = label_info.get('label', 'unknown')
            if label == 'unknown':
                raise ValueError(f"File {file_id} not found in labels dictionary.")
            attack = label_info.get('attack', '-')
            speaker_id = label_info.get('speaker_id', 'unknown')
            codec = label_info.get('codec', '-')
            gender = label_info.get('gender', '-')
            # === שלב 2: טעינת אודיו ונרמול ===
            try:
                waveform, sr = _load_audio_mono(audio_path, SAMPLE_RATE)
                break
            except Exception:
                if not SKIP_BAD_AUDIO:
                    raise
                attempts += 1
                if attempts >= len(self.audio_files):
                    raise RuntimeError("All audio files failed to decode.")
                idx = (idx + 1) % len(self.audio_files)
                continue

        # === שלב 3: חישוב מאפייני אודיו ===
        input_features = _extract_input_features(self.processor, waveform, sr)
        # === שלב 4: בניית רצף טוקנים תואם Voxtral ===
        input_ids = self._input_ids.clone()
        return {
            "audio_path": audio_path,
            "file_id": file_id,
            "label": label,
            "attack": attack,
            "speaker_id": speaker_id,
            "codec": codec,
            "gender": gender,
            "input_ids": input_ids,
            "input_features": input_features,
        }

def _norm_field(value, default='-'):
    """Normalize NaN-like fields to a default placeholder."""
    if pd.isna(value):
        return default
    value = str(value)
    return value if value.lower() != 'nan' else default


def get_paths_and_labels(set_name, dataset_type, asvspoof2021_task):
    """Return (audio_dir, labels_csv, parse_labels, read_csv_args) for the selected dataset."""
    if dataset_type == 'ASVspoof2019':
        audio_dir = f'/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof2019/LA/LA/ASVspoof2019_LA_{set_name}/flac/'
        if set_name == 'train':
            labels_csv = f'/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof2019/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{set_name}.trn.txt'
        else:
            labels_csv = f'/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof2019/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{set_name}.trl.txt'
        def parse_labels(df):
            return {
                row['file']: {
                    'label': 'genuine' if row['key'] in ['bonafide', 'genuine'] else 'spoof',
                    'attack': row['attack'],
                    'speaker_id': row['speaker_id']
                }
                for _, row in df.iterrows()
            }
        read_csv_args = dict(sep=' ', header=None,
            names=['speaker_id', 'file', 'd1', 'attack', 'key'])
        csv_fields = ['file_id','embedding_path','label','attack','speaker_id','audio_path']
    elif dataset_type == 'ASVspoof5':
        audio_dirs = {
            'train': '/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof5/flac_T',
            'dev':   '/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof5/flac_D',
            'eval':  '/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof5/flac_E/flac_E_eval'
        }
        audio_dir = audio_dirs[set_name]
        if set_name == 'train':
            labels_csv = '/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof5/ASVspoof5_protocols/ASVspoof5.train.tsv'
        else:
            labels_csv = f'/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof5/ASVspoof5_protocols/ASVspoof5.{set_name}.track_1.tsv'
        def parse_labels(df):
            labels = {}
            for _, row in df.iterrows():
                fields = str(row[0]).split()
                file_id = fields[1]
                if fields[-2] in ['bonafide', 'genuine']:
                    label = 'bonafide'
                    attack = '-'
                    codec = fields[3]
                    gender = fields[2]
                else:
                    label = 'spoof'
                    attack = fields[-3]
                    codec = fields[3]
                    gender = fields[2]
                speaker_id = fields[0]
                labels[file_id] = {
                    'label': label,
                    'attack': attack,
                    'speaker_id': speaker_id,
                    'codec': codec,
                    'gender': gender
                }
            return labels
        read_csv_args = dict(sep='\t', header=None)
        csv_fields = ['file_id','embedding_path','label','attack','speaker_id','codec','gender','audio_path']
    elif dataset_type == 'ASVspoof2021':
        task = asvspoof2021_task.upper()
        if task == 'LA':
            audio_dir = '/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof2021/ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/flac'
            labels_csv = '/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof2021/LA-keys-full/keys/LA/CM/trial_metadata.txt'
            def parse_labels(df):
                labels = {}
                for _, row in df.iterrows():
                    label = 'bonafide' if str(row['key']) in ['bonafide', 'genuine'] else 'spoof'
                    attack = _norm_field(row['attack']) if label == 'spoof' else '-'
                    file_id = _norm_field(row['file'])
                    labels[file_id] = {
                        'label': label,
                        'attack': attack,
                        'speaker_id': _norm_field(row['speaker_id']),
                        'codec': _norm_field(row['codec'])
                    }
                return labels
            read_csv_args = dict(sep=r'\s+', header=None, engine='python',
                names=['speaker_id', 'file', 'codec', 'language', 'attack', 'key', 'trim', 'subset'])
            csv_fields = ['file_id','embedding_path','label','attack','speaker_id','codec','audio_path']
        elif task == 'DF':
            audio_dir = '/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof2021/ASVspoof2021_DF_eval/flac'
            labels_csv = '/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof2021/DF-keys-full/keys/DF/CM/trial_metadata.txt'
            def parse_labels(df):
                labels = {}
                for _, row in df.iterrows():
                    label = 'bonafide' if str(row['key']) in ['bonafide', 'genuine'] else 'spoof'
                    attack = _norm_field(row['attack']) if label == 'spoof' else '-'
                    file_id = _norm_field(row['file'])
                    labels[file_id] = {
                        'label': label,
                        'attack': attack,
                        'speaker_id': _norm_field(row['speaker_id']),
                        'codec': _norm_field(row['codec']),
                        'gender': _norm_field(row['gender'])
                    }
                return labels
            read_csv_args = dict(sep=r'\s+', header=None, engine='python',
                names=['speaker_id', 'file', 'codec', 'source', 'attack', 'key', 'trim', 'subset',
                       'vocoder_type', 'vc_task', 'team', 'gender', 'accent'])
            csv_fields = ['file_id','embedding_path','label','attack','speaker_id','codec','gender','audio_path']
        else:
            raise ValueError('Unknown ASVSPOOF2021_TASK (use LA or DF)')
    else:
        raise ValueError('Unknown DATASET_TYPE')
    return audio_dir, labels_csv, parse_labels, read_csv_args, csv_fields


def _resolve_save_dir(dataset_type, asvspoof2021_task, output_stage, set_name):
    """Resolve output directory based on dataset type and task."""
    if dataset_type == 'ASVspoof5':
        save_base = '/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds'
        return os.path.join(save_base, output_stage, set_name)
    if dataset_type == 'ASVspoof2019':
        save_base = '/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds'
        return os.path.join(save_base, output_stage, set_name)
    if dataset_type == 'ASVspoof2021':
        task = asvspoof2021_task.upper()
        save_base = f'/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_{task}_embeds'
        return os.path.join(save_base, output_stage, set_name)
    raise ValueError('Unknown DATASET_TYPE')


def _metadata_path(save_dir, output_stage, set_name):
    """Build metadata CSV path for a given output directory."""
    return os.path.join(os.path.dirname(save_dir), f'{output_stage}_{set_name}_voxtral_all_embeddings_metadata.csv')


def _load_labels_dict(labels_csv, read_csv_args, parse_labels):
    """Read protocol file and return labels dictionary."""
    labels_df = pd.read_csv(labels_csv, **read_csv_args)
    return parse_labels(labels_df)


def _load_existing_ids(metadata_path, resume):
    """Load already-processed file_ids from metadata when resuming."""
    existing_ids = set()
    metadata_exists = os.path.exists(metadata_path)
    if resume and metadata_exists:
        with open(metadata_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames:
                for row in reader:
                    file_id = row.get('file_id')
                    if file_id:
                        existing_ids.add(file_id)
    return existing_ids, metadata_exists


def run_pipeline(
    dataset_type=DATASET_TYPE,
    asvspoof2021_task=ASVSPOOF2021_TASK,
    sets=SETS,
    num_workers=NUM_WORKERS,
    device=DEVICE,
    repo_id=REPO_ID,
    batch_size=BATCH_SIZE,
    output_stage=OUTPUT_STAGE,
    embedding_question=EMBEDDING_QUESTION,
    resume=RESUME,
    n_audio_tokens=N_AUDIO_TOKENS,
):
    """Run embedding extraction for the selected dataset/task/sets."""
    processor = None
    wrapper = None

    for set_name in sets:
        # === שלב 1: לוג וולידציה ===
        print(f"\nProcessing set: {set_name} (type: {dataset_type})")

        if dataset_type == 'ASVspoof2021' and set_name != 'eval':
            print(f"Skipping set {set_name} for ASVspoof2021 (only eval available).")
            continue

        # === שלב 2: נתיבי שמירה ומטא־דאטה ===
        save_dir = _resolve_save_dir(dataset_type, asvspoof2021_task, output_stage, set_name)
        os.makedirs(save_dir, exist_ok=True)
        metadata_path = _metadata_path(save_dir, output_stage, set_name)

        # === שלב 3: קריאת פרוטוקול/לייבלים ===
        audio_dir, labels_csv, parse_labels, read_csv_args, csv_fields = get_paths_and_labels(
            set_name,
            dataset_type,
            asvspoof2021_task,
        )

        labels_dict = _load_labels_dict(labels_csv, read_csv_args, parse_labels)
        existing_ids, metadata_exists = _load_existing_ids(metadata_path, resume)

        # === שלב 4: בניית רשימת קבצי אודיו ===
        if resume and existing_ids:
            audio_files = []
            skipped = 0
            missing_embeds = 0
            for file_id in labels_dict.keys():
                emb_path = os.path.join(save_dir, f"{file_id}.npy")
                if file_id in existing_ids:
                    if os.path.exists(emb_path):
                        skipped += 1
                        continue
                    missing_embeds += 1
                audio_files.append(os.path.join(audio_dir, f"{file_id}.flac"))
            if skipped > 0:
                print(f"Resume enabled: skipping {skipped} already-processed files with existing embeddings.")
            if missing_embeds > 0:
                print(f"Resume enabled: reprocessing {missing_embeds} files missing embeddings (.npy not found).")
        else:
            audio_files = [os.path.join(audio_dir, f"{file_id}.flac") for file_id in labels_dict.keys()]

        if not audio_files:
            print(f"No new files to process for {set_name} (type: {dataset_type}).")
            continue

        # === שלב 5: טעינת processor ===
        if processor is None:
            processor = AutoProcessor.from_pretrained(repo_id)

        # === שלב 6: יצירת Dataset ו־DataLoader ===
        dataset = ASVspoofAudioDataset(
            audio_files,
            processor,
            device,
            labels_dict,
            embedding_question,
            n_audio_tokens=n_audio_tokens,
        )
        dataset[0]
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        # === שלב 7: טעינת מודל ו־wrapper ===
        if wrapper is None:
            model = VoxtralForConditionalGeneration.from_pretrained(
                repo_id,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            wrapper = VoxtralStageWrapper(model).to(device)

        # === שלב 8: פתיחת מטא־דאטה לכתיבה ===
        metadata_nonempty = metadata_exists and os.path.getsize(metadata_path) > 0
        open_mode = 'a' if resume and metadata_nonempty else 'w'
        with open(metadata_path, open_mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            if not metadata_nonempty:
                writer.writeheader()
            # === שלב 9: הרצת מודל ושמירת תוצאות ===
            # Progress bar per file to show ETA.
            with tqdm(
                total=len(dataset),
                desc=f"{dataset_type}:{set_name}",
                unit="file",
                dynamic_ncols=True,
                mininterval=1.0,
            ) as pbar:
                for batch in loader:
                    # --- שלב 9.1: איסוף נתונים מה־batch ---
                    audio_paths = batch["audio_path"]
                    file_ids = batch["file_id"]
                    labels = batch["label"]
                    attacks = batch["attack"]
                    speaker_ids = batch["speaker_id"]
                    codecs = batch.get("codec", ['-']*len(file_ids))
                    genders = batch.get("gender", ['-']*len(file_ids))

                    # --- שלב 9.2: העברה ל־device ---
                    input_ids = batch["input_ids"].to(device, dtype=torch.long)
                    input_features = batch["input_features"].to(device, dtype=torch.bfloat16)

                    # --- שלב 9.3: חישוב הטבעות ---
                    with torch.no_grad():
                        emb = wrapper(
                            input_ids=input_ids,
                            input_features=input_features,
                            output_stage=output_stage
                        )
                        # ממוצע על ציר הזמן לקבלת הטבעה יחידה לכל אודיו
                        # Average over time frames to get one vector per audio.
                        # Keep batch dimension stable even when batch_size == 1.
                        if emb.dim() == 4 and emb.shape[1] == 1:
                            emb = emb.squeeze(1)
                        if emb.dim() != 3:
                            raise ValueError(f"Unexpected emb shape before pooling: {tuple(emb.shape)}")
                        joint_embeddings = emb.mean(dim=1)

                    # --- שלב 9.4: כתיבת קבצי embedding + מטא־דאטה ---
                    for i, file_id in enumerate(file_ids):
                        # שמירת הטבעה לקובץ ורישום מטא-דאטה
                        emb_i = joint_embeddings[i].to(torch.float32).cpu().numpy()
                        emb_path = os.path.join(save_dir, f"{file_id}.npy")
                        np.save(emb_path, emb_i)
                        row_dict = {
                            'file_id': file_id,
                            'embedding_path': emb_path,
                            'label': labels[i],
                            'attack': attacks[i],
                            'speaker_id': speaker_ids[i],
                            'audio_path': audio_paths[i]
                        }
                        if dataset_type == 'ASVspoof5':
                            row_dict['codec'] = codecs[i]
                            row_dict['gender'] = genders[i]
                        elif dataset_type == 'ASVspoof2021':
                            task = asvspoof2021_task.upper()
                            if task == 'LA':
                                row_dict['codec'] = codecs[i]
                            elif task == 'DF':
                                row_dict['codec'] = codecs[i]
                                row_dict['gender'] = genders[i]
                        if not (resume and file_id in existing_ids):
                            writer.writerow(row_dict)

                    pbar.update(len(file_ids))

        # === שלב 10: סיום סט ===
        print(f"Saved embeddings for {set_name} in {save_dir} and metadata in {metadata_path}")

    print("\nDone! Embeddings for all sets and all dataset types calculated.")


def _parse_csv_list(value: str):
    """Split comma-separated string into a list of values."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_tasks(value: str):
    """Parse LA/DF tasks from comma-separated string and validate."""
    tasks = [t.strip().upper() for t in value.split(",") if t.strip()]
    for t in tasks:
        if t not in ("LA", "DF"):
            raise ValueError(f"Unknown ASVspoof2021 task: {t}")
    return tasks


# ===== Main =====
# === שלב 1: בחירת סטים ===
sets_to_run = _parse_csv_list(SETS) if isinstance(SETS, str) else list(SETS)
if DATASET_TYPE == "ASVspoof2021":
    # === שלב 2: בחירת משימות (LA/DF) ===
    tasks = _parse_tasks(ASVSPOOF2021_TASK)
    # === שלב 3: הרצה סדרתית לכל משימה ===
    for task in tasks:
        run_pipeline(
            dataset_type=DATASET_TYPE,
            asvspoof2021_task=task,
            sets=sets_to_run,
            num_workers=NUM_WORKERS,
            device=DEVICE,
            repo_id=REPO_ID,
            batch_size=BATCH_SIZE,
            output_stage=OUTPUT_STAGE,
            embedding_question=EMBEDDING_QUESTION,
            resume=RESUME,
            n_audio_tokens=N_AUDIO_TOKENS,
        )
else:
    # === שלב 2: הרצה לדאטהסט שאינו ASVspoof2021 ===
    run_pipeline(
        dataset_type=DATASET_TYPE,
        asvspoof2021_task=ASVSPOOF2021_TASK,
        sets=sets_to_run,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        repo_id=REPO_ID,
        batch_size=BATCH_SIZE,
        output_stage=OUTPUT_STAGE,
        embedding_question=EMBEDDING_QUESTION,
        resume=RESUME,
        n_audio_tokens=N_AUDIO_TOKENS,
    )
