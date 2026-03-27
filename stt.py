import numpy as np
import soundfile as sf
import whisper


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model("medium")
    return _model


def transcribe(audio_path: str) -> tuple[str, str, str]:
    """
    Transcribe audio, auto-detect language, and translate to English using Whisper.

    Args:
        audio_path: Path to audio file (WAV, FLAC, OGG, MP3)

    Returns:
        (text_en, text_native, language)
          - text_en:     English translation (for LLM + cross-encoder)
          - text_native: Original language transcription (for native-language FAISS query)
          - language:    Detected language code ("hi", "pa", or "en")
    """
    wav, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # stereo → mono
    if sr != 16000:
        num_samples = int(len(wav) * 16000 / sr)
        wav = np.interp(
            np.linspace(0, len(wav), num_samples),
            np.arange(len(wav)),
            wav
        ).astype(np.float32)

    model = _get_model()
    result_en = model.transcribe(wav, task="translate")
    text_en = result_en["text"].strip()
    lang = result_en["language"]

    if lang == "en":
        text_native = text_en
    else:
        result_native = model.transcribe(wav, task="transcribe")
        text_native = result_native["text"].strip()

    print(f"Detected language: {lang}")
    print(f"English query: {text_en}")
    print(f"Native query:  {text_native}")
    return text_en, text_native, lang
