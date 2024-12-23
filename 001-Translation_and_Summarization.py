# Import the Hugging Face pipeline for NLP tasks
from transformers import pipeline 

# Import the class to load sequence-to-sequence pre-trained models
from transformers import AutoModelForSeq2SeqLM

# Import PyTorch for tensor manipulation and computations
import torch

# Import Python's garbage collection module to manage memory
import gc

# Main block of the script to prevent execution during import
if __name__ == "__main__":

    # Load the pre-trained NLLB (No Language Left Behind) model for translation
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    print("Model loaded successfully")  # Confirmation message

    # Create a translation pipeline with the loaded model
    # Use bfloat16 data type to optimize memory usage
    translator = pipeline(task="translation",
                          model="facebook/nllb-200-distilled-600M",
                          torch_dtype=torch.bfloat16) 

    # Define a block of English text to be translated
    text = """\
        The transformer model was first proposed in 2017 for a machine translation task, 
        and since then, numerous models have
        been developed based on the inspiration of the original
        transformer model to address a variety of tasks across different fields"""

    # Perform translation from English (Latin script) to French (Latin script)
    text_translated = translator(text,
                                 src_lang="eng_Latn",
                                 tgt_lang="fra_Latn")

    # Print the translated text
    print(text_translated)

    # Delete the translator object to free memory
    del translator

    # Invoke garbage collection to clean up memory
    gc.collect()

    # Create a summarization pipeline using the BART-Large CNN model
    # Use bfloat16 data type to optimize memory usage
    summarizer = pipeline(task="summarization",
                          model="facebook/bart-large-cnn",
                          torch_dtype=torch.bfloat16)

    # Define a block of text to be summarized
    text = """Transformer is a deep neural network that employs a self-attention mechanism to comprehend the contextual relationships within sequential data. Unlike conventional neural networks or updated versions of
        Recurrent Neural Networks (RNNs) such as Long Short-Term Memory (LSTM), transformer models excel
        in handling long dependencies between input sequence elements and enable parallel processing. As a result,
        transformer-based models have attracted substantial interest among researchers in the field of artificial intelligence. This can be attributed to their immense potential and remarkable achievements, not only in Natural
        Language Processing (NLP) tasks but also in a wide range of domains, including computer vision, audio
        and speech processing, healthcare, and the Internet of Things (IoT)."""

    # Perform text summarization with a specified range for summary length
    summary = summarizer(text,
                         min_length=10,  # Minimum number of tokens in the summary
                         max_length=100)  # Maximum number of tokens in the summary

    # Print the generated summary
    print(summary)
