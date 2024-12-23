from datasets import load_dataset, load_from_disk
from IPython.display import Audio as IPythonAudio 
from transformers import pipeline
from datasets import Audio


if __name__ == "__main__":
    # Load the ESC-50 dataset from the Hugging Face Hub
    dataset = load_dataset("ashraq/esc50", split="train[0:10]")
    print("Dataset loaded successfully")
    audio_sample = dataset[0]


    print(f"audio_sample, {audio_sample}")

   
    IPythonAudio(audio_sample["audio"]["array"],
             rate=audio_sample["audio"]["sampling_rate"])
    
    zero_shot_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="laion/clap-htsat-unfused")

    zero_shot_classifier.feature_extractor.sampling_rate
    print(audio_sample["audio"]["sampling_rate"])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=48_000))

    audio_sample = dataset[0]
    candidate_labels = ["Sound of a dog",
                    "Sound of vacuum cleaner"]
    zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels=candidate_labels)
    print(zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels=candidate_labels))
