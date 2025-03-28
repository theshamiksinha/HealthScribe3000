# NLP Project README

## Project Overview
This project implements and evaluates multiple NLP models (BART, FLAN-T5, and PEGASUS) for text processing tasks. It includes code for training, evaluation, and generating test predictions. The repository is structured with separate directories for each model, containing code, results, and checkpoints.

## Project Structure
```
Baseline_Folder/
│── BART/
│   │── bart_code_results/
│   │   │── BART_test_predictions_from_checkpoint_bart.txt
│   │   │── Loss_Plots.png
│   │   │── bart.ipynb
│   │   │── PerspectiveDataset.py
│   │── bart-model_checkpoint/
│
│── FLAN_T5/
│   │── flan-t5_code_results/
│   │   │── FLAN_T5_test_predictions_from_checkpoint.txt
│   │   │── Loss_Plots.png
│   │   │── flanT5.ipynb
│   │   │── PerspectiveDataset.py
│   │── flan-t5-model_checkpoint/
│
│── PEGASUS/
│   │── pegasus_code_results/
│   │   │── test_predictions_from_checkpoint.txt
│   │   │── Loss_Plots.png
│   │   │── pegasus.ipynb
│   │   │── PerspectiveDataset.py
│   │── pegasus-model_checkpoint/
│
│── puma_dataset/
│── Elementary_Data_Analysis/
│── 86_proposal.pdf
```

## Setup Instructions

### 1. Install Required Libraries
Ensure you have the following packages installed:

```
Python 3.7+
PyTorch
Transformers
pandas
matplotlib
sacrebleu
bert_score
evaluate
nbformat
```

Before running any code, ensure you have the necessary dependencies installed. Use the following command to install them:
```bash
pip install torch transformers pandas matplotlib sacrebleu bert_score evaluate nbformat
```

### 2. Dataset Preparation
The dataset used in this project should be placed inside the `puma_dataset/` directory. Update the dataset paths in the respective notebooks (`bart.ipynb`, `flanT5.ipynb`, `pegasus.ipynb`) if necessary.

### 3. Running the Code

Each model has its corresponding Jupyter notebook for training and evaluation. Follow these steps to run them:

#### Running BART Model
```bash
jupyter notebook bart.ipynb
```
- Ensure that the dataset paths inside `bart.ipynb` are correctly set.
- The results, including predictions and loss plots, will be saved in `bart_code_results/`.

#### Running FLAN-T5 Model
```bash
jupyter notebook flanT5.ipynb
```
- Adjust dataset paths if required.
- The results will be stored in `flan-t5_code_results/`.

#### Running PEGASUS Model
```bash
jupyter notebook pegasus.ipynb
```
- Verify dataset paths.
- The output predictions and plots will be saved in `pegasus_code_results/`.

### 4. Checkpoints
Model checkpoints are stored in their respective directories:
- `bart-model_checkpoint/`
- `flan-t5-model_checkpoint/`
- `pegasus-model_checkpoint/`

To resume training or use pretrained weights, ensure these directories contain the necessary checkpoint files.

### 5. Results & Outputs
- Predictions are stored as `.txt` files inside each model’s `code_results/` folder.
- Loss plots are available as `Loss_Plots.png`.

## Additional Notes
- `PerspectiveDataset.py` contains the dataset loading logic and should be present in all model directories.
- `Elementary_Data_Analysis/` may contain preprocessing scripts or exploratory analysis; review before training.
- If using a GPU, ensure PyTorch is installed with CUDA support: `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118`

## Contact
For any issues, please reach out to the project contributors.

---
This README provides all essential information to set up, run, and analyze the models in the project. Ensure paths and dependencies are correctly configured before execution.


## References

1. **PyTorch Documentation**: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)  
2. **Hugging Face Transformers**: [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)  
3. **Hugging Face Datasets**: [https://huggingface.co/docs/datasets/](https://huggingface.co/docs/datasets/)  
4. **SacreBLEU for Evaluation**: [https://github.com/mjpost/sacreBLEU](https://github.com/mjpost/sacreBLEU)  
5. **BERTScore for Evaluation**: [https://github.com/Tiiiger/bert_score](https://github.com/Tiiiger/bert_score)  
6. **Hugging Face Evaluate Library**: [https://huggingface.co/docs/evaluate/index](https://huggingface.co/docs/evaluate/index)  
7. **BART Model on Hugging Face**: [https://huggingface.co/facebook/bart-large](https://huggingface.co/facebook/bart-large)  
8. **FLAN-T5 Model on Hugging Face**: [https://huggingface.co/google/flan-t5-large](https://huggingface.co/google/flan-t5-large)  
9. **PEGASUS Model on Hugging Face**: [https://huggingface.co/google/pegasus-large](https://huggingface.co/google/pegasus-large)  
10. **Jupyter Notebook Documentation**: [https://jupyter-notebook.readthedocs.io/en/stable/](https://jupyter-notebook.readthedocs.io/en/stable/)  
11. **Matplotlib for Visualization**: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)  
12. **Pandas for Data Handling**: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)  
13. **Tokenization and Preprocessing**: [https://huggingface.co/docs/transformers/main_classes/tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer)  
14. **Original Reference Paper**: [https://arxiv.org/abs/2406.08881](https://arxiv.org/abs/2406.08881)  
