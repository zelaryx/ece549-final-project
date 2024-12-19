### class to be the T5 spell check model wrapper ###

# python library imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import T5ForConditionalGeneration, AutoTokenizer


class T5LargeSpell:
    """
    Basically a wrapper for the T5 spell check model. Not context aware

    Parameters
    ----------
    None


    Attributes
    ----------
    arr : numpy.ndarray
        The color image as a Numpy array.
    arr_greyscale : numpy.ndarray
        The grayscale version of the image as a Numpy array.
    size : tuple
        The (w, h) of the image.
    """
    def __init__(self):

        model_path = "ai-forever/T5-large-spell"

        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.prefix = "grammar: "

    def spellcheck(self, string):
        """
        call to the model

        Parameters
        ----------
        string : str
            The string to be spellchecked


        Returns
        ----------
        str
            The spellchecked string
        """

        string = string.lower()
        string = self.prefix + string

        encodings = self.tokenizer(string, return_tensors="pt")
        generated_tokens = self.model.generate(**encodings, max_new_tokens=50)
        answer = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return answer