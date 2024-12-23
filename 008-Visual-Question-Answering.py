from transformers import BlipForQuestionAnswering

from transformers import AutoProcessor

from PIL import Image

if __name__ == "__main__":

    # Load the Model and the Processor.
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base")

    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-vqa-base")

    image = Image.open("./sample_image.png")

    question = "how many dogs are in the picture?"
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))