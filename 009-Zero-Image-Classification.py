# CLIP : Contrastive Language - Image Pre-Training
# CLIP is a multi-modal vision and language model. It can be used for zero-shot image classification.

# Importing the CLIPModel class from the Hugging Face transformers library, which provides the CLIP model implementation.
from transformers import CLIPModel

# Importing AutoProcessor, which handles preprocessing for both the text and image inputs to be compatible with the CLIP model.
from transformers import AutoProcessor

# Importing the Image class from the Python Imaging Library (PIL) to handle image loading and manipulation.
from PIL import Image

# Checking if the script is being executed directly (not imported as a module).
if __name__ == "__main__":
    # Loading the pre-trained CLIP model "openai/clip-vit-large-patch14" from Hugging Face's model hub.
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    # Loading the corresponding processor for the same model, which handles tokenizing text and preparing image inputs.
    processor = AutoProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    # Opening an image file named "sample_image.png" using PIL.
    image = Image.open("./sample_image.png")

    # Defining the labels for classification. These are the text descriptions the model will compare against the image.
    labels = ["a photo of a cat", "a photo of a dog"]

    # Preprocessing the image and text labels using the processor to create input tensors for the CLIP model.
    inputs = processor(
        text=labels,  # Text descriptions of the labels.
        images=image,  # The input image to classify.
        return_tensors="pt",  # Returning PyTorch tensors.
        padding=True  # Ensuring inputs are padded if needed.
    )

    # Passing the preprocessed inputs into the CLIP model to get the output logits (raw scores).
    outputs = model(**inputs)

    # Extracting the logits for the image with respect to the text labels.
    # These logits represent the similarity score between the image and each text label.
    probs = outputs.logits_per_image.softmax(dim=1)[0]  # Applying softmax to get probabilities.

    # Iterating over each label to print its probability of matching the input image.
    for i in range(len(labels)):
        print(f"label: {labels[i]} - probability of {probs[i].item():.4f}")
        # Displaying the label and its corresponding probability rounded to 4 decimal places.
