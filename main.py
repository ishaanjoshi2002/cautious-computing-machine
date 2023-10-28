# Import necessary modules from the transformers library and PIL (Python Imaging Library).
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image as img # used for opening the image.
"""
#ViTForImageClassification
The ViTForImageClassification class works by taking an input image, passing it through the pre-trained Vision Transformer layers,
and producing a probability distribution over predefined class labels,
allowing it to classify the image into one of those classes based on the learned features.

#ViTImageProcessor
This class is part of the Hugging Face Transformers library and is used for preprocessing and
handling images before they are input into a Vision Transformer (ViT) model. ( resize, normalize,)
 Define the file name of the image you want to test.
"""
FILE_NAME = 'image_to_test_3.jpg'

# Open the image file using PIL and store it in an image array.
image_array = img.open(FILE_NAME) # converts the image in the NumPy array format for manipulation.

# Create a ViTImageProcessor object from a pre-trained ViT model. This processor is used for image pre-processing.
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Create a ViTForImageClassification model from a pre-trained ViT model. This model is used for image classification.
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Preprocess the input image using the ViTImageProcessor. It converts the image into a format suitable for the model.
# "return_tensors='pt'" specifies that the output should be PyTorch tensors.
inputs = processor(images=image_array, return_tensors="pt")

# Pass the preprocessed input through the ViTForImageClassification model to obtain predictions.
outputs = model(**inputs)
"""
passing the inputs to the ViT model using double asterisks (**),
which is Python's syntax for unpacking dictionary-like arguments. 
This means that you're providing the model with the necessary input data in the format it expects.
"""
# Extract the logits (raw model outputs) from the model's output.
logits = outputs.logits
"""
Logits are raw model outputs representing unnormalized scores for each class in an image classification task.
They are essential as they capture the model's confidence for each class, helping identify the most likely class.
Logits are typically transformed into probabilities (e.g., using softmax) to make the final classification decision, 
aiding accurate image classification.
"""
# Find the index of the class with the highest probability from the logits.
predicted_class_idx = logits.argmax(-1).item()

# Retrieve the label associated with the predicted class index using the model's configuration.
# This assumes that the model's configuration includes a mapping from class indices to labels.
print("Predicted class index:", predicted_class_idx)
print("Predicted class:", model.config.id2label[predicted_class_idx])