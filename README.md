# Manga Text Extraction with Qwen2.5-VL

## 1. Project Overview

This project extracts dialogue and other visible text from three consecutive manga pages. The model predicts the text, its reading order, and the speaker associated with each text segment.

The task uses sequence-level training, validation, and test data. Each prediction contains a sequence ID and the predicted text segments for its three pages.

## 2. Model Choice

The base model is [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).

I selected this vision-language model because the task requires understanding both manga images and their visible text. It can process images directly without requiring a separate OCR system.

## 3. Pipeline

The repository contains four main Python scripts:

- `prep_data.py` — prepares the data for the task.
- `train.py` — fine-tunes the model using LoRA.
- `infer.py` — generates predictions for validation or test sequences.
- `score.py` — evaluates predictions against reference annotations.

The development data is split at the sequence level into 64 training sequences and 16 validation sequences. The official test set contains 15 sequences.

Training was performed on an NVIDIA A100 GPU in Google Colab. Final test inference was run locally on an NVIDIA RTX 4050 Laptop GPU with 6 GB of VRAM.

## 4. Fine-Tuning

I used Low-Rank Adaptation (LoRA) to fine-tune the model without updating all of its parameters. The training setup used 4-bit NF4 quantization to reduce GPU memory requirements.

The experiments compared LoRA ranks 16 and 32, different image resolutions, training epochs, and inference approaches.

## 5. Experimental Settings

The main settings explored were:

| Setting | Values explored |
|---|---|
| Base model | Qwen2.5-VL-3B-Instruct |
| LoRA rank | 16, 32 |
| Learning rate | 1e-4 |
| Training epochs | Up to 6 |
| Gradient accumulation | 8 |
| Quantization | 4-bit NF4 |
| Maximum image pixels | 401,408; 802,816 |
| Random seed | 0 |

The selected configuration used LoRA rank 32 and the epoch-5 checkpoint.

## 6. Best Validation Results

The best observed configuration used rank-32 LoRA, the epoch-5 checkpoint, page-by-page inference, a maximum image resolution setting of 802,816 pixels, and 900 maximum new tokens.

| Metric | Validation result |
|---|---:|
| Text order score | 0.7550 |
| Balanced joint F1 | 0.2812 |
| Joint F1 | 0.5056 |
| Speaker accuracy on matched text | 0.5772 |
| Matched-token coverage | 0.8303 |

These results are from a validation set of only 16 sequences and should be interpreted accordingly.

In the configurations tested, page-by-page inference performed better than joint inference on the main metrics. Increasing the maximum output length from 900 to 1,400 tokens did not improve the high-resolution configuration. Increasing image resolution improved the text order score and balanced joint F1 in the experiments.

## 7. Limitations

The model can still miss visible text, generate text that is not present in the image, misorder text segments, and assign incorrect speakers.

One validation sequence had no reference text, but the model generated 11 tokens for it. This illustrates the model's tendency to hallucinate text and the need to evaluate empty or nearly empty pages carefully.

The small validation set also limits how confidently the results can be generalized.

## 8. Reproduction

Install the required Python dependencies and make the base model available from Hugging Face. Prepare the task data using `prep_data.py`, then train with `train.py`.

For inference, use `infer.py` with the trained adapter, the desired split, and the same preprocessing and resolution settings as the selected experiment. Use `score.py` to evaluate validation predictions against the reference annotations.

The exact command-line arguments are defined by the scripts. The selected experiment used rank 32, a learning rate of 1e-4, the epoch-5 checkpoint, page-by-page inference, 802,816 maximum pixels, and 900 maximum new tokens.

## 9. Experiment Artifacts

The repository includes the training and inference scripts, experiment logs, validation predictions, validation scores, and task files.

The selected trained LoRA adapter and additional experiment artifacts are stored separately from the main source repository. They can be provided through a separate download link where required.

## 10. Test Predictions

The final test prediction file is named `preds_test_rank32_e5_page_802k.jsonl`.

It contains predictions for all 15 official test sequences. Structural validation passed, and the predicted sequence IDs matched the official test split exactly, with no missing, unexpected, or duplicate IDs.

The test predictions were generated automatically by the inference script without manually editing the output file.
