# CLIP ASL-Fingerspelling
The model is a version of OpenAI’s CLIP Vision Model [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) fine-tuned for the task of American Sign Language (ASL) fingerspelling classification. In order to achieve that, a classifier layer was added on top of the base model. The model card and further model details can be found on the Hugging Face Hub: https://huggingface.co/aalof/clipvision-asl-fingerspelling
## Dataset
Training was done on the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) which is available on Kaggle. It comprises of 206,137 images of signs corresponding to 26 letters of the English alphabet (A-Z), and three additional classes that were not used for training for this task. Dataset was split into train (80%), validation (20%), and test (10%) sets.
## Results
The fine-tuned model achieves:
- Test Accuracy: 99.88%
- Weighted Test F1 Score: 99.88%
- Per-class F1 scores varying from 99.61% to 100% (available in the [notebook version](https://colab.research.google.com/drive/1SHz-t2I9DKyxEbC9F7C4nKdhVSZyUXSJ?authuser=3#scrollTo=r3H2wC7jYcCn) of `clip-asl-fingerspelling.py`)
## How to use
The `inference_script_single_image.py` is a ready script which showcases how to load the trained model and processor from Hugging Face, and classify a single image along with a confidence measure. The base model is loaded, and on topd of it a custom classifier head is defined. Example images, which were used for testing the script, can be found in the small ASL dataset containing 125 images of fingerspelling with conditions (hands, background, lighting) different from the training data: [link](https://drive.google.com/drive/folders/1HuzULPybEkL25P6FCuuGKtWoRtf4W6pE?usp=sharing). There are only 25 classes in this dataset, since the signs for 'Q' are missing.
