import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, CLIPVisionModel
from sklearn.metrics import f1_score
from PIL import Image
from huggingface_hub import hf_hub_download

# Set path to the dataset directory
dataset_dir = ""

# Retrieve image paths and corresponding labels from the subdirectory names
image_paths = []
labels = []

# Go through subdirectories
for label in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, label)
    for image_name in os.listdir(class_dir):
        if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(class_dir, image_name)
            image_paths.append(image_path)
            labels.append(label)

# Create a mapping of class labels to numeric indices
class_names = sorted(os.listdir(dataset_dir))  # Get all class names
label2index = {label: idx for idx, label in enumerate(class_names)}  # Map labels to indices

# Prepare dataset class for preprocessing
class ASLDataset(Dataset):
    def __init__(self, image_paths, labels, processor, label2index):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.label2index = label2index

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # Convert label (text) to numeric index
        text_label = self.labels[idx]
        numeric_label = self.label2index[text_label]

        # Preprocess the image using the processor
        inputs = self.processor(images=image, return_tensors="pt", padding=True)

        # Return pixel values and label
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": numeric_label
        }

# Initialise device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define module with a classifier head that maps image features from CLIPVision to the number of classes
class CLIPVisionClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.model = clip_model
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        image_features = outputs.pooler_output
        logits = self.classifier(image_features)
        return logits

def load_model(model_id="aalof/clipvision-asl-fingerspelling"):
    """
    Load model and processor with trained weights.
    """
    processor = AutoProcessor.from_pretrained(model_id)

    # Load the base CLIP model
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    # Initialise the full model (including classifier)
    model = CLIPVisionClassifier(clip_model=clip_model, num_classes=26).to(device)

    # Download and load the trained weights
    weights_path = hf_hub_download(
        repo_id=model_id,
        filename="pytorch_model.bin"
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    return model, processor, device


def test(model, dataloader, device, index2label):
    """
    Classify images using the trained model and evaluate performance.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_true_labels = []
    per_class_f1 = {}

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            logits = model(pixel_values=pixel_values)

            # Predicted labels
            predicted_indices = torch.argmax(logits, dim=1)

            # Collect predictions and true labels for F1 score calculation
            all_predictions.extend(predicted_indices.cpu().tolist())
            all_true_labels.extend(labels.cpu().tolist())

            # Calculate accuracy
            correct_predictions = (predicted_indices == labels).sum().item()
            total_correct += correct_predictions
            total_samples += len(labels)

    # Compute overall accuracy
    accuracy = total_correct / total_samples

    # Compute overall F1 score
    f1 = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)

    # Compute per-class F1 scores
    class_f1_scores = f1_score(all_true_labels, all_predictions, average=None, labels=list(index2label.keys()), zero_division=0)

    # Map F1 scores to actual class labels
    for i, score in zip(index2label.keys(), class_f1_scores):
        per_class_f1[index2label[i]] = score

    # Print accuracy and F1 scores
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print F1 scores per class
    for class_name, f1 in per_class_f1.items():
        print(f"{class_name} - F1 Score: {f1:.4f}")

    return accuracy, f1, per_class_f1

# Predict class
if __name__ == "__main__":
    # Load the model
    model, processor, device = load_model()

    # Create dataset using your existing lists
    dataset = ASLDataset(
        image_paths=image_paths,
        labels=labels,
        processor=processor,
        label2index=label2index
    )

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Reversee the label2index to get index2label
    index2label = {idx: label for label, idx in label2index.items()}

    # Make predictions
    accuracy, f1, per_class_f1 = test(
    model=model,
    dataloader=dataloader,
    device=device,
    index2label=index2label
    )
