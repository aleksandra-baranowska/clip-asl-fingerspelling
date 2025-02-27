# CLIP ASL Fingerspelling
The model is a version of OpenAI’s CLIP Vision Model [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) fine-tuned for the task of American Sign Language (ASL) fingerspelling classification. In order to achieve that, a classifier layer was added on top of the base model. The model card and further model details can be found on the Hugging Face Hub: https://huggingface.co/aalof/clipvision-asl-fingerspelling
## Dataset
Training was done on the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) which is available on Kaggle. It comprises of 206,137 images of signs corresponding to 26 letters of the English alphabet (A-Z), and three additional classes that were not used for training for this task. Dataset was split into train (70%), validation (20%), and test (10%) sets.
## Results
The fine-tuned model achieves:
- Test Accuracy: 99.88%
- Weighted Test F1 Score: 99.88%
- Per-class F1 scores available in the notebook: [link](https://colab.research.google.com/drive/1SHz-t2I9DKyxEbC9F7C4nKdhVSZyUXSJ?authuser=3#scrollTo=r3H2wC7jYcCn)
## Inference
The `inference_single_image.py` is a ready script which showcases how to load the trained model and processor from Hugging Face, and classify a single image along with a confidence measure. The base model is loaded, and on topd of it a custom classifier head is defined. Example images, which were used for testing the script, can be found in the small ASL dataset containing 130 images of fingerspelling with conditions (hands, background, lighting) different from the training data: [link](https://drive.google.com/file/d/1-1tCO8RrPP6fbJz6yE13J-0kDt4nj3dd/view?usp=sharing).

The inference script for batch classification (`inference_batch.py`) can be used for testing the model performance on a dataset containing images unseen during training. It also shows how to load the trained model, as well as preprocess data to make class predictions. Evaluation metrics to track performance are provided (accuracy, F1 score). Results from performance on the small dataset mentioned above are significantly worse than those from initial testing (Accuracy: 79.66%).
