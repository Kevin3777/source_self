
# CS Paper Abstract Generation and Category Prediction Project

A Computer Science paper processing project based on TinyLlama-1.1B model with two core functionalities:

- 📝 **Abstract Generation**: Generate paper abstracts from paper ID and category  
- 🏷️ **Category Prediction**: Predict paper categories from abstracts

## 🎯 Project Overview

| Model | Function | Input Format | Output Format |
|-------|----------|--------------|---------------|
| **begin_url** | Abstract Generation | `<src> cs.xxx.xx.id </src> [Abstract:` | Generated paper abstract |
| **end_url** | Category Prediction | `[Abstract: abstract content] <src>` | `<src> cs.xxx.xx.id </src>` |

- 1. **begin_url model**: Given a paper ID and category, generates a paper abstract
- 2. **end_url model**: Given a paper abstract, predicts the paper category and subcategory

**The files end with _title mean to inject title during training and testing, and the dataset is pretreated for title version.**


### begin_url model (Abstract Generation)

- **Input format**: `<src> cs.xxx.xx.id </src> [Abstract:`
- **Output**: Generated paper abstract
- **Evaluation metrics**: ROUGE-1, ROUGE-2, ROUGE-L

### end_url model (Category Prediction)

- **Input format**: `[Abstract: abstract content] <src>`
- **Output**: `<src> cs.xxx.xx.id </src>` (where cs.xxx is the predicted category and subcategory)
- **Evaluation metrics**: Category accuracy and subcategory accuracy

## 🚀Environment Setup

### Installing Dependencies

```bash
pip install transformers datasets torch rouge numpy scikit-learn
```

Note: Use `scikit-learn` instead of `sklearn`, as the latter is deprecated.

### Hardware Requirements

- GPU recommended for training (minimum 4GB VRAM)
- Evaluation can run on CPU, but will be slower

## Data Format

### Input Data

Input data should be in JSON format, containing multiple paper entries, each with the following fields:
- `title`: Paper title
- `text`: Paper abstract
- `enc-paper-str`: Paper ID (format like "cs▁AI▁2503.10638v1")
- `categories`: List of categories (e.g., ["cs.CV", "cs.AI", "cs.LG"])

Example:
```json
[
  {
    "title": "Studying Classifier(-Free) Guidance From a Classifier-Centric Perspective",
    "text": "Classifier-free guidance has become a staple for conditional generation with denoising diffusion models...",
    "enc-paper-str": "cs▁AI▁2503.10638v1",
    "categories": ["cs.CV", "cs.AI", "cs.LG"]
  }
]
```

### Processed Data Format

The data workflow processes the raw JSON data into two different formats:

1. **begin_url format**:
   ```
   <src> cs.CV.2503.10638v1 </src> [Abstract: Classifier-free guidance has become a staple for conditional generation...]
   ```

2. **end_url format**:
   ```
   [Abstract: Classifier-free guidance has become a staple for conditional generation...] <src> cs.CV.2503.10638v1 </src>
   ```

### 🚦 How to Run the Project

For finer control, you can manually execute each step:

#### 1. Data Preparation

```bash
python data_splite.py --input_file arxiv_results.json --output_dir dataset
```

This will create:
- `dataset/begin_url/train.txt`: Abstract generation training data
- `dataset/begin_url/test.txt`: Abstract generation test data
- `dataset/begin_url/test_data.json`: Original JSON test data for evaluation
- `dataset/end_url/train.txt`: Category prediction training data
- `dataset/end_url/test.txt`: Category prediction test data
- `dataset/end_url/test_data.json`: Original JSON test data for evaluation

Output Structure:
dataset/
├── begin_url/
│   ├── train.txt          # Abstract generation training data
│   ├── test.txt           # Abstract generation test data
│   └── test_data.json     # Original test data for evaluation
└── end_url/
    ├── train.txt          # Category prediction training data
    ├── test.txt           # Category prediction test data
    └── test_data.json     # Original test data for evaluation

#### 2. Training Models

Training the begin_url model (abstract generation):
```bash
python train.py --model_type begin_url --train_file dataset/begin_url/train.txt --output_dir models
```

Training the end_url model (category prediction):
```bash
python train.py --model_type end_url --train_file dataset/end_url/train.txt --output_dir models
```

#### 3. Evaluating Models

Evaluating the begin_url model (abstract generation):
```bash
python eval_begin_optimized.py \
  --model_path models/begin_url/final_model \
  --test_data dataset/begin_url/test_data.json \
  --output_file begin_url_results.json \
  --batch_size 32 \
  --interval 200
```

Evaluating the end_url model (category prediction):
```bash
python eval_end_optimized.py \
  --model_path models/end_url/final_model \
  --test_data dataset/end_url/test_data.json \
  --output_file end_url_results.json \
  --batch_size 32 \
  --interval 200
```


## 🤗Trained models
Standard Models

Abstract Generation: https://huggingface.co/Kevin3777/source_self_begin

Category Prediction: https://huggingface.co/Kevin3777/source_self_end

Title-Enhanced Models

Abstract Generation + Title: https://huggingface.co/Kevin3777/source_self_begin_title

Category Prediction + Title: https://huggingface.co/Kevin3777/source_self_end_title

# 📈 Performance Results
## Abstract Generation (Begin URL Model)

### Basic Information
| Parameter | Value |
|------|------|
| Model Path | models/begin_url/final_model |

### Evaluation Parameters
| Parameter | Value |
|------|------|
| max_new_tokens | 200 |
| temperature | 0.7 |
| top_p | 0.9 |
| batch_size | 32 |
| num_samples | 1604 |

### Average Scores
| Metric | Score |
|------|------|
| ROUGE-1 | 0.17477637665102388 |
| ROUGE-2 | 0.024183252651398656 |
| ROUGE-L | 0.16048199030608845 |

### Evaluation Performance
| Metric | Value |
|------|------|
| Evaluation Time | 234.59040713310242 seconds |
| Samples Per Second | 6.837449235892749 |

### Example Prediction (Sample #0)

**Paper ID**: econ.EM.econ/EM/2411.16978v2

**True Abstract**:
```
We establish normal approximation in the Wasserstein metric and central limit
theorems for both non-degenerate and degenerate U-statistics with
cross-sectionally dependent samples using Stein's method. For the
non-degenerate case, our results extend recent studies on the asymptotic
properties of sums of cross-sectionally dependent random variables. The
degenerate case is more challenging due to the additional dependence induced by
the nonlinearity of the U-statistic kernel. Through a specific implementation
of Stein's method, we derive convergence rates under conditions on the mixing
rate, the sparsity of the cross-sectional dependence structure, and the moments
of the U-statistic kernel. Finally, we demonstrate the application of our
theoretical results with a nonparametric specification test for data with
cross-sectional dependence.
```

**Generated Abstract**:
```
This paper introduces a novel method for identifying the number of
models that best represent a data set. Our approach leverages the empirical
structure of the data and the estimated model parameters to identify the number
of models that best fit the data. This approach is particularly useful when
there is uncertainty in the estimated model parameters, and it allows for
flexibility in the number of models that can be selected from a large pool of
possible models. We show that our method is efficient and provides a reliable
approach for identifying the number of models that best represent a data set.
```

**ROUGE Scores**:
| Metric | Recall(R) | Precision(P) | F1 Score |
|------|------|------|------|
| ROUGE-1 | 0.2208 | 0.3333 | 0.2656 |
| ROUGE-2 | 0.0566 | 0.0714 | 0.0631 |
| ROUGE-L | 0.1818 | 0.2745 | 0.2187 |

## Category Prediction (End URL Model)

### Evaluation Scores

| Metric | Value |
|------|------|
| category_accuracy | 0.7674563591022444 |
| subcategory_accuracy | 0.53428927680798 |
| total_samples | 1604 |
| evaluation_time | 47.433497190475464 seconds |
| samples_per_second | 33.815765123936075 |

### Evaluation Parameters

| Parameter | Value |
|------|------|
| max_new_tokens | 20 |
| temperature | 0.0 |
| batch_size | 32 |
| num_samples | 1604 |

### Example Prediction

**Sample #0:**

- **True Abstract**: "We establish normal approximation in the Wasserstein metric and central limit theorems for both non-degenerate and degenerate U-statistics with cross-sectionally dependent samples using Stein's method..."
- **True Category**: "econ"
- **True Subcategory**: "EM"
- **Predicted Full**: "stat.ME.stat/TH/2502.20332v1"
- **Predicted Category**: "stat"
- **Predicted Subcategory**: "ME"
- **Category Correct**: false
- **Subcategory Correct**: false
