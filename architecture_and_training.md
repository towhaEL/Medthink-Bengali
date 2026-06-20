# MedThink: Architecture, Training, and Evaluation Details

This document provides a comprehensive overview of the MedThink framework, including its architecture, dataset preparation, image feature extraction, training loop, and evaluation processes, based on the codebase.

## 1. Model Architecture
The core model is `T5ForMultimodalGeneration`, which inherits from Hugging Face's `T5ForConditionalGeneration`. It relies on the **T5 (Text-to-Text Transfer Transformer)** architecture but extends it to handle multimodal inputs (text and images).

### 1.1 JointEncoder
The model replaces the standard T5 encoder with a custom `JointEncoder` (a modified `T5Stack`). The `JointEncoder` integrates visual information into the text encoding process:
- **Image Embedding**: Image patch features (pre-extracted) are passed through a linear layer (`self.image_dense`) to match the text hidden size (`config.d_model`).
- **Cross-Modal Attention**: A Multi-Head Attention layer (`self.mha_layer`, 1 head) is used where the text `hidden_states` act as the query, and the `image_embedding` acts as the key and value. This allows the text representation to attend to relevant parts of the image.
- **Gated Fusion Mechanism**: The text hidden states and the image attention output are concatenated and passed through a dense layer followed by a Sigmoid activation. This creates a **gate** that dynamically controls how much visual information is merged back into the text hidden states.

### 1.2 Decoder & Language Modeling Head
The decoder is a standard `T5Stack` decoder. It takes the cross-modal hidden states from the `JointEncoder` and auto-regressively generates the target text (which can be a direct answer or a rationale/explanation). A standard linear layer (`lm_head`) projects the hidden states to the vocabulary size.

### 1.3 Tokenizer
The model uses standard text tokenizers corresponding to the T5 variant loaded via `AutoTokenizer.from_pretrained()`. Text inputs and targets are tokenized with padding and truncation to predefined max lengths (`source_len` and `target_len`).

---

## 2. Image Feature Extraction
Image features are extracted **offline** before the main training loop begins, using the `extract_img_feature.py` script.
- **Visual Model**: Uses a pretrained **DETR** model with a **ResNet-101** backbone (`detr_resnet101_dc5`) loaded from PyTorch Hub (`cooelf_detr_main`).
- **Preprocessing**: Images are resized to 224x224, converted to PyTorch tensors, and normalized using ImageNet stats.
- **Extraction**: The raw images are passed through the DETR model without gradients (`torch.no_grad()`). The final layer's output is extracted as the visual feature for the image.
- **Storage**: All image features are concatenated into a single large tensor and saved as a `.pth` file. A `name_map.json` is generated to map the image ID (e.g., file name) to its row index in the saved tensor.

---

## 3. Dataset Preparation
The benchmark datasets supported are **RAD** (VQA-RAD) and **Slake**.
The dataset loading logic (`dataset.py`) defines two main dataset classes: `ClosedMedVQADataset` and `OpenMedVQADataset`.

### 3.1 Input and Target Formatting
The datasets support various prompting "methods" which control how the problem is formatted to the model:
- **Explanation**: Prompts for the answer and provides the solution as text.
- **Reasoning**: Asks for the solution first, followed by the answer.
- **First-Stage_Reasoning**: Prompts to generate only the reasoning (solution).
- **Second-Stage_Reasoning**: Uses the generated solution as context to prompt for the final answer.
- **without_R**: Directly asks for the answer without any reasoning steps.

### 3.2 Data Collator
For closed-ended VQA, choices (e.g., A, B) are appended to the prompt.
The `__getitem__` method fetches the tokenized text, target labels, and indexes into the pre-extracted `.pth` feature file to grab the corresponding `image_ids`. These are returned as a dictionary containing `input_ids`, `attention_mask`, `image_ids`, and `labels`.

---

## 4. Train Flow and Loop
The training logic (`closed_end_train.py` and `open_end_train.py`) heavily utilizes the Hugging Face `Seq2SeqTrainer`.

### 4.1 Setup
- Random seeds for PyTorch and NumPy are set for reproducibility.
- `T5ForMultimodalGeneration` and the tokenizer are loaded.
- `DataCollatorForSeq2Seq` is used to dynamically pad batches.

### 4.2 Training Arguments
- Uses `Seq2SeqTrainingArguments`.
- Configurable hyperparameters include Learning Rate (`lr`), Batch Size (`bs`), Weight Decay (`wd`), and Epochs.
- `predict_with_generate=True` allows the model to autoregressively generate sequences during evaluation/prediction.

### 4.3 Training Process
- The `Seq2SeqTrainer` manages the optimization loop, gradient calculation, and logging.
- `loss` is calculated natively inside `T5ForMultimodalGeneration.forward` using standard `CrossEntropyLoss` on the vocabulary, ignoring the `-100` padding indices.

---

## 5. Evaluation Metrics
Depending on the task method, the `compute_metrics` function evaluates the generated text differently:

1. **Accuracy (`compute_metrics_acc`)**:
   - Used mainly for final answer prediction (e.g., predicting option A or B in closed-ended questions).
   - A regular expression (e.g., `r'The answer is \(([A-Z])\)'`) extracts the choice from the generated text and compares it to the ground truth.

2. **ROUGE-L (`compute_metrics_rougel`)**:
   - Used when evaluating generated text that includes reasoning or rationales (`First-Stage_Reasoning` or when `--rational` is passed).
   - Utilizes the `evaluate` library and NLTK sentence tokenization to compute the ROUGE-L score between the predicted rationale and the target solution.
