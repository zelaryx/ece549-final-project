### script to correct text from OCR ###

from T5Model import T5LargeSpell

# initialize model
_t5_model_instance = None

# # from transformers import file_utils
# print(file_utils.default_cache_path)

def _get_t5_model():
    """
    Just gets the current T5 model being used in this code, so that it only get initialized once.
    """
    global _t5_model_instance
    if _t5_model_instance is None:
        _t5_model_instance = T5LargeSpell()
    return _t5_model_instance

def spellcheck(text):
    """
    Fixes spell errors in given text. Does NOT know how to deal with spacing issues so...

    Parameters
    ----------
    text : str
        A string with the text to be spellchecked.

    Returns
    -------
    str
        A string with the corrected text

    """
    t5_model = _get_t5_model()
    return t5_model.spellcheck(text)
