from datasets import load_dataset
from IPython.display import Audio as IPythonAudio

from transformers import pipeline
import soundfile as sf
import io

# Main block of the script to prevent execution during import
if __name__ == "__main__":
    dataset = load_dataset("librispeech_asr",
                       split="train.clean.100",
                       streaming=True,
                       trust_remote_code=True)
    example = next(iter(dataset))
    dataset_head = dataset.take(5)
    list(dataset_head)
    list(dataset_head)[2]
    print(example)


    asr = pipeline(task="automatic-speech-recognition", 
               model="distil-whisper/distil-small.en")
    asr.feature_extractor.sampling_rate
    example['audio']['sampling_rate']
    asr(example["audio"]["array"])
    print(example["text"])


