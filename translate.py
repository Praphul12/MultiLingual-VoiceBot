from deep_translator import GoogleTranslator


def translate_from_english(text: str, tgt_lang: str) -> str:
    """
    Translate English text to the target language using Google Translate.
    Returns text unchanged if target is English.

    Args:
        text:     Input text in English
        tgt_lang: Target language code — "hi", "pa", or "en"

    Returns:
        Translated text in the target language
    """
    if tgt_lang == "en":
        return text
    return GoogleTranslator(source="en", target=tgt_lang).translate(text)
