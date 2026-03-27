import torch
import soundfile as sf

# Set to "vits" for fast CPU inference or "parler" for higher quality (requires GPU)
BACKEND = "vits"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Parler TTS (AI4Bharat indic-parler-tts) ──────────────────────────────────
PARLER_MODEL_ID = "ai4bharat/indic-parler-tts"

PARLER_DESCRIPTIONS = {
    "hi": "A female speaker delivers slightly expressive and animated Hindi speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up. No background noise.",
    "pa": "A female speaker delivers slightly expressive and animated Punjabi speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up. No background noise.",
    "en": "A female speaker with an Indian English accent delivers slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up. No background noise.",
}

_parler_model = None
_parler_prompt_tokenizer = None
_parler_description_tokenizer = None


def _get_parler_model():
    global _parler_model, _parler_prompt_tokenizer, _parler_description_tokenizer
    if _parler_model is None:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        _parler_model = ParlerTTSForConditionalGeneration.from_pretrained(PARLER_MODEL_ID).to(DEVICE)
        _parler_prompt_tokenizer = AutoTokenizer.from_pretrained(PARLER_MODEL_ID)
        _parler_description_tokenizer = AutoTokenizer.from_pretrained(
            _parler_model.config.text_encoder._name_or_path
        )
    return _parler_model, _parler_prompt_tokenizer, _parler_description_tokenizer


def _synthesize_parler(text: str, lang: str, output_path: str) -> str:
    model, prompt_tokenizer, description_tokenizer = _get_parler_model()
    description = PARLER_DESCRIPTIONS.get(lang, PARLER_DESCRIPTIONS["en"])
    description_ids = description_tokenizer(description, return_tensors="pt").to(DEVICE)
    prompt_ids = prompt_tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_ids.input_ids,
            attention_mask=description_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
        )
    audio = generation.cpu().numpy().squeeze()
    sf.write(output_path, audio, model.config.sampling_rate)
    return output_path


# ── VITS TTS (Facebook MMS-TTS) ──────────────────────────────────────────────
VITS_MODELS = {
    "hi": "facebook/mms-tts-hin",
    "pa": "facebook/mms-tts-pan",
    "en": "facebook/mms-tts-eng",
}

_vits_models = {}
_vits_tokenizers = {}


def _get_vits_model(lang: str):
    if lang not in _vits_models:
        from transformers import VitsModel, AutoTokenizer
        model_id = VITS_MODELS[lang]
        _vits_tokenizers[lang] = AutoTokenizer.from_pretrained(model_id)
        _vits_models[lang] = VitsModel.from_pretrained(model_id).to(DEVICE)
    return _vits_tokenizers[lang], _vits_models[lang]


def _synthesize_vits(text: str, lang: str, output_path: str) -> str:
    tokenizer, model = _get_vits_model(lang)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        audio = model(**inputs).waveform
    sf.write(output_path, audio.squeeze().cpu().numpy(), model.config.sampling_rate)
    return output_path


# ── Public API ────────────────────────────────────────────────────────────────
def synthesize(text: str, lang: str, output_path: str) -> str:
    """
    Convert text to speech and save to a WAV file.

    Set BACKEND = "vits"   for fast CPU inference (Facebook MMS-TTS)
    Set BACKEND = "parler" for higher quality (AI4Bharat Indic Parler-TTS, needs GPU)

    Args:
        text:        Input text in the target language
        lang:        Language code — "hi", "pa", or "en"
        output_path: Path to write the output WAV file

    Returns:
        output_path after writing
    """
    if BACKEND == "parler":
        return _synthesize_parler(text, lang, output_path)
    return _synthesize_vits(text, lang, output_path)
