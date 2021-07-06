# DADER

Entity resolution (ER) is a core problem of data integration. The state-of-the-art results on ER are achieved by deep learning (DL) based methods, trained with a lot of labeled matching/non-matching entity pairs. This may not be a problem when using well-prepared benchmark datasets. Nevertheless, for many real-world ER applications, the situation changes dramatically, with a main issue to collect labeled datasets (i.e., data-centric), instead of designing the models for ER (i.e., model-centric). In this paper, we study the following problem: If we have a well labeled source ER dataset, can we train a DL-based ER model for a target dataset from the same or a related domain of the source dataset, without any labels or with a few labels? This is known as domain adaptation (DA), which has achieved great successes in computer vision and natural language processing. Our goal is to systematically explore the benefits and limitations of a wide range of DA methods for ER. To this purpose, we develop the DADER (Domain Adaptation for Deep Entity Resolution) framework that significantly advances the state of the art in applying DA to ER. DADER consists of three modules, namely Feature Extractor, Matcher, and Feature Alignment. Under each module we further categorize commonly used methods: RNN and pre-trained language models (LMs) are common choices for Feature Extractor; neural networks are typically employed by Matcher; and discrepancy-based, adversarial-based, and reconstruction-based methods are widely used for Feature Alignment. The concrete choices for the categories under Feature Extractor and Matcher have been well studied. Therefore, our main focus is to identify methods for the Feature Alignment module, for which we develop six representative methods. We empirically compare different DA methods for ER and report which methods perform best on what kind of scenarios, based on whether the source and the target are from the same or relevant domains, and whether some labeled target entity pairs are absent or present.

## DataSets
Public datasets used in the paper are from DeepMatcher, Magellan and WDC.
The details of datasets are shown in "data/dataset.md"

## Quick Start
Step 1: Requirements
- Before running the code, please make sure your Python version is 3.6.5 and cuda version is 11.1. Then install necessary packages by :
- `pip install -r requirements.txt`
- `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`

Step 2: Run

-    `  cd main`

Run the main function and indicate source data and target data:
-    ` python main_[mmd/k-order/grl/invgan/invgan_kd/ed/noda].py --src [source data] --tgt [target data]`

An example:
-    ` python main_invgan_kd.py --src b2 --tgt fz`

#### This repository contains the implementation code of six representative methods of DADER: MMD, K-order, GRL, InvGAN, InvGAN+KD, ED.
- Folder "modules/" contains the implementation of three component: Feature Extractor, Feature Alignment, Matcher.
- Folder "main/" contains the main function to run the six representative methods.
- Folder "train/" contains the implementation of six representative alignment.
- Folder "metrics/" contains some statistical metrics for discrepancy-based methods.
- Folder "data/" contains the datasets in our paper.