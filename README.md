# Llama2-based Data Augmentation for Dark Web Text Classification

## Author
- Chiara Manna

## Overview
This project explores the use of Llama2 for data augmentation in a prompt-based instruction learning setup. The research goal is to investigate to what extent Large Language Models (LLMs) can be leveraged to address the challenge of data imbalance in the context of Dark Web text classification. The dataset, provided by CFLW, consists of text entries extracted from the HTML source code of Dark Web pages. Entries are manually annotated into five crime categories: Financial Crime, Cybercrime, Drugs and Narcotics Trade, Weapons Trade, and Violent Crime (listed in descending order of dominance), with a significant imbalance among classes. This characteristic emerges as domain-specific as the most common data sources used for Dark Web content analysis exhibit a similar imbalance.

In such a context, it is difficult for AI models to detect patterns associated with underrepresented classes without being biased towards the majority class. And as these underrepresented classes appear to have a more tangible societal impact (High-Impact Crimes), it becomes crucial to address this challenge.

While common resampling techniques do not excel in the case of such a severe imbalance, data augmentation emerges as an alternative. However, conventional techniques in NLP (random transformations or backtranslation), do not introduce enough variance for modern language models, such as BERT. To this end, this work explores a Llama2-based data augmentation approach, which is evaluated extrinsically, namely on how it enhances the performance of a BERT model, fine-tuned on the augmented sets, with Balanced Accuracy as primary metric.

## How to run
The scripts are listed below in the order they are\should be used. As these are Jupyter notebooks and some parts require manual settings they should be run progressively, in a step-by-step manner. Starting from preprocessing the dataset, performing augmentation and finally fine-tuning BERT. 
The results in my thesis are based on Thesis_result.ipynb, where the model is retrained on the combined training and validation sets, further investigation suggested that the chosen BERT architecture might be very sensitive to the number of epochs. This sensitivity may lead to changes in results when training on the combined set, as the optimal number is determined for the training set (on validation performance). Therefore, I recommend following Finetuning_BERT.ipynb for potential further or similar experiments. This notebook trains the model exclusively on the training set, includes a fixed seed (when loading batches for training), and enforces deterministic behavior.

# Requirements 
Experiments are conducted using Python 3.10. The full requirements are listed in the requirements.txt.

## Table of Contents
1. Dataset_aggregation_preprocessing.ipynb: The dataset is explored and prepared for the multi-class classification task.
2. Manual_inspection.ipynb: Problematic instances, emerging from the inspection of a random sample are identified and aggregated into an inspecting dataset, that is manually inspected to remove entries uncorrectly linked to specific crime categories.
3. LLAMA2_augmentation_classification.ipynb: The training sets are augmented with a zero- and few-shot prompt. Multiple configurations can be explored. At the end of the script the best-performing Llama2 configuration in augmentation is extended to prompt-based zero-shot classification of the test data.
4. Preprocessing_post_augmentation.ipynb: Preprocessing steps are extended to the augmented sets, including duplicate removal and the manual inspection of specific patterns.
5. Finetuning_BERT.ipynb: Hyper-parameter and Temperature tuning is performed. Different BERT models, fine-tuned on the original, undersampled and augmented sets (under various Llama2 configurations) are compared. BERT_B is the baseline, fine-tuned on the orginal imbalanced dataset; BERT_US is the undersampling-based model, BERT_ZS is fine-tuned on training set augmented with the full precision model, BERT_ZS_Q adds quantization. Then BERT_FS_Q explores the impact of the few-shot prompt and BERT_ZS_Q_chat extends the best-performing configuration to Llama2-Chat.
6. Error_Analysis_Semantic_Composition.ipynb: In this file the error analysis focuses on the semantic composition across and within classes, to observe how it evolves with the incorporation of the synthetic examples for each configuration. The idea is that an effective data augmentation strategy should both preserve semantic fidelity and introduce diversity.
7. Thesis_results.ipynb: This file represents the results reported in my thesis. Nonetheless, after careful consideration I advise to follow the process in Finetuning_BERT instead of training on combined train val sets, as the architecture appears to be very sensitive to the number of epochs that might be suboptimal when training on the combined set. Moreover, Finetuning_BERT.ipynb includes a fixed seed in loading the batches for training and deterministic behaviour is enforced.

While the original dataset cannot be shared, the scripts assume that you have  a data folder, where the original data is stored and where all the transformations are saved as well.


## Findings
We observe that undersampling is able to enhance the performance on underrepresented classes, but at a significant cost to the more dominant classes that are undersampled. Comparing the performance of the BERT models, respectively fine-tuned on training sets augmented with the full precision and quantized Llama2, we observe an overall increase with both, outperforming BERT_B (baseline) and BERT_US (undersampling-based), with the highest increase observed with the configuration involving quantization. This challenges the conventional belief that quantization inevitably leads to performance loss, but might even introduce beneficial variance in specific scenarios.  
Further exploration involves extending quantization to a few-shot prompting strategy, which does not outperform the zero-shot prompt. Consequently, the zero-shot prompting strategy on the quantized model is extended to Llama2-Chat, achieving the best overall performance on a fine-tuned BERT model.

All augmentation approaches outperform the set baseline and alternative and surpass the best-performing Llama2 configuration in a prompt-based zero-shot classification setup. This achievement is particularly significant, considering that these improvements are attained by fine-tuning BERT on (augmented) datasets with a synthetic composition reaching an overall 50%. This proportion is even more pronounced for underrepresented classes, where it reaches approximately 90%.