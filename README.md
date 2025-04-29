
# CS Paper Abstract Generation and Category Prediction Project

This project is based on the TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T model and has trained two independent models to process CS (Computer Science) paper data:

1. **begin_url model**: Given a paper ID and category, generates a paper abstract
2. **end_url model**: Given a paper abstract, predicts the paper category and subcategory

## Project Overview

This project demonstrates how to use small language models for two different types of text generation tasks. By using the pre-trained TinyLlama model and fine-tuning with different data formats, we can obtain models specifically optimized for different tasks.

### begin_url model (Abstract Generation)

- **Input format**: `<src> cs.xxx.xx.id </src> [Abstract:`
- **Output**: Generated paper abstract
- **Evaluation metrics**: ROUGE-1, ROUGE-2, ROUGE-L

### end_url model (Category Prediction)

- **Input format**: `[Abstract: abstract content] <src>`
- **Output**: `<src> cs.xxx.xx.id </src>` (where cs.xxx is the predicted category and subcategory)
- **Evaluation metrics**: Category accuracy and subcategory accuracy

## Environment Setup

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

## Usage Instructions

### Complete Workflow

Use the complete workflow script to perform all steps from data preparation to model evaluation in one go:

```bash
python complete_workflow_updated.py --input_file cs_papers.json --output_dir output --epochs 3
```

#### Main Parameters

- `--input_file`: Input JSON file path (**required**)
- `--output_dir`: Output directory (default: output)
- `--test_size`: Test set proportion (default: 0.2)
- `--epochs`: Training epochs (default: 3)
- `--batch_size`: Batch size (default: 4)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--begin_url_max_tokens`: Maximum token count for abstract generation (default: 200)
- `--end_url_max_tokens`: Maximum token count for category prediction (default: 50)
- `--eval_samples`: Number of samples to use for evaluation (default: use entire test set)

#### Workflow Control Parameters

- `--skip_data_prep`: Skip data preparation step
- `--skip_begin_url_training`: Skip begin_url model training
- `--skip_end_url_training`: Skip end_url model training
- `--skip_begin_url_evaluation`: Skip begin_url model evaluation
- `--skip_end_url_evaluation`: Skip end_url model evaluation

### Manual Execution of Each Step

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


## Trained models
https://huggingface.co/Kevin3777/source_self_begin

https://huggingface.co/Kevin3777/source_self_end

## Evaluation Result for Begin

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
| ROUGE-1 | 0.1748 |
| ROUGE-2 | 0.0242 |
| ROUGE-L | 0.1603 |

### Evaluation Performance
| Metric | Value |
|------|------|
| Evaluation Time | 231.3255 seconds |
| Samples Per Second | 6.9340 |

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
| ROUGE-1 | 0.1558 | 0.2264 | 0.1846 |
| ROUGE-2 | 0.0283 | 0.0390 | 0.0328 |
| ROUGE-L | 0.1429 | 0.2075 | 0.1692 |

## Evaluation Result for End

### Evaluation Scores

| Metric | Value |
|------|------|
| category_accuracy | 0.7381546134663342 |
| subcategory_accuracy | 0.4925187032418953 |
| total_samples | 1604 |
| evaluation_time | 76.6562864780426 seconds |
| samples_per_second | 20.92457218703712 |

### Evaluation Parameters

| Parameter | Value |
|------|------|
| max_new_tokens | 50 |
| temperature | 0.7 |
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
