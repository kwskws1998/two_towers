# Gaze Affect CLIP

This project extends the VA+gaze baseline code with CLIP-style soft contrastive
alignment between affect representations and predicted eye-tracking
representations.

Core experiment:

```text
train: English VA data excluding IEMOCAP
validation/test: automatic split from the no-IEMOCAP pool
zero-shot test: IEMOCAP only
```

Main entry point:

```bash
conda env create -f environment.yml
conda activate gaze_affect_clip
python -m pip install -U pip wheel "setuptools<82"
python -m pip install -r requirements-cuda121.txt

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Same setup as a script:

```bash
bash scripts/setup_conda_gpu.sh
conda activate gaze_affect_clip
```

Then prepare data and train:

```bash
python prepare_english_data.py --output-dir data --seed 42

python train_vad_single.py xlmroberta-base mse \
  --data-dir data \
  --split-dir data_vad_noiemocap \
  --holdout-dataset IEMOCAP \
  --batch-size 16 \
  --maxlen 200 \
  --train-epochs 10 \
  --learning-rate 6e-6

python train_clip_style.py xlmroberta-base mse+ccc \
  --data-dir data \
  --split-dir data_clip_noiemocap \
  --holdout-dataset IEMOCAP \
  --init-vad-checkpoint model_vad_single/<timestamp>/final_model \
  --et2-checkpoint ./checkpoints/et_predictor2_seed123 \
  --projection-dim 256 \
  --tau 0.07 \
  --sigma 0.05 \
  --lambda-align 0.1 \
  --batch-size 16 \
  --maxlen 200
```

If the ET2 checkpoint is missing, `train_clip_style.py` tries to download
`et_predictor2_seed123.safetensors` from `skboy/et_prediction_2`.

Use `train_vad_single.py` for the one-model VAD checkpoint that initializes the
CLIP-style affect tower via `--init-vad-checkpoint`. The inherited
`train_model.py` keeps the original 2-fold out-of-fold evaluation behavior and
saves two fold models.

Outputs are written under `PredsClip/<timestamp>_<host>/`:

```text
training_parameters.json
metrics_noiemocap_val.json
metrics_noiemocap_test.json
metrics_iemocap_zeroshot.json
metrics_summary.json
predictions_noiemocap_val.csv
predictions_noiemocap_test.csv
predictions_iemocap_zeroshot.csv
```

For the shuffled-ET negative control:

```bash
python train_clip_style.py xlmroberta-base mse+ccc \
  --data-dir data \
  --split-dir data_clip_noiemocap \
  --shuffle-gaze
```

# Quantifying Valence and Arousal in Text with Multilingual Pre-trained Transformers 
Repository for Quantifying Valence and Arousal in Text with Multilingual Pre-trained Transformers 

## Current repository scope

- This repo is now focused on **VA prediction + gaze-concat training**.
- RLHF/reward-model code paths were removed.
- English data preparation uses a **single Google Drive zip** (downloaded with `gdown`).


## Dataset
The dataset proposed in this paper was built collecting 34 different public datasets of annotated data for the emotional dimensions of Valence and Arousal.
All the datasets used free to use for research purposes, although some require an authorization to use, and/or individual acceptance of the respective terms of use. For this reason, we cannot publicly provide the dataset we used to train our models. 

As detailed in the Paper, our dataset is a .csv file with three columns, namely "text", "valence", and "arousal".
To reproduce our dataset, follow this procedure:
- Retrieve the 34 original datasets from the Dataset Sorces below. The datasets come in various different file formats, such as .csv, .xlsx, .txt, etc.
- Filter the relevant data:
  - **text**: Word or short text content
  - **valence** and **arousal**: We simply used the Valence and Arousal Mean values.
- Normalize the Valence and Arousal scores between **zero** and **one**, using the following formula.
  - $z_i = (x_i - \textrm{min}(x)) / (\textrm{max}(x) - \textrm{min}(x))$
  - $z_i$ denotes the normalized value, $x_i$ the original value, and $\textrm{min}$ and $\textrm{max}$ denote the extremes of the scales in which the original scores were rated on.


### Dataset Sources
#### EmoBank
- **Source:** EmoBank: Studying the Impact of Annotation Perspective and Representation Format on Dimensional Emotion Analysis
  - https://aclanthology.org/E17-2092/
- **Repository:** https://github.com/JULIELab/EmoBank
- **Download directly here:** https://github.com/JULIELab/EmoBank/raw/master/corpus/emobank.csv

#### IEMOCAP
- **Source:** IEMOCAP: Interactive emotional dyadic motion capture database
  - https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf
- **Repository:** https://sail.usc.edu/iemocap/iemocap_release.htm
  - To obtain the IEMOCAP data you need to fill out an electronic release form.
  
#### Facebook Posts
- **Source:** Modelling Valence and Arousal in Facebook posts
  - https://aclanthology.org/W16-0404/https://aclanthology.org/W16-0404/
- **Repository:** https://github.com/wwbp/additional_data_sets/tree/master/valence_arousal
- **Download directly here:** https://github.com/wwbp/additional_data_sets/raw/master/valence_arousal/dataset-fb-valence-arousal-anon.csv

#### EmoTales
- **Source:** EmoTales: creating a corpus of folk tales with emotional annotations
  - https://link.springer.com/article/10.1007/s10579-011-9140-5
- **Repository:** Request the dataset by contacting the author, Virgina Francisco, at virginia@fdi.ucm.es

#### ANET
- **Source:** Affective Norms for English Text (ANET)
  - https://csea.phhp.ufl.edu/media/anetmessage.html
- **Repository:** https://csea.phhp.ufl.edu/media/anetmessage.html
  - To obtain the ANET data you need to fill out an electronic release form.

#### PANIG
- **Source:** When emotions are expressed figuratively: Psycholinguistic and Affective Norms of 619 Idioms for German (PANIG)
  - https://link.springer.com/article/10.3758/s13428-015-0581-4
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-015-0581-4/MediaObjects/13428_2015_581_MOESM1_ESM.xls

#### COMETA sentences
- **Source:** Affective and psycholinguistic norms for German conceptual metaphors (COMETA)
  - https://link.springer.com/article/10.3758/s13428-019-01300-7
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above. See ESM 2.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-019-01300-7/MediaObjects/13428_2019_1300_MOESM2_ESM.xlsx

#### COMETA stories
- **Source:** Affective and psycholinguistic norms for German conceptual metaphors (COMETA)
  - https://link.springer.com/article/10.3758/s13428-019-01300-7
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above. See ESM 2.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-019-01300-7/MediaObjects/13428_2019_1300_MOESM2_ESM.xlsx

#### CVAT
- **Source:** Building Chinese Affective Resources in Valence-Arousal Dimensions
  - https://aclanthology.org/N16-1066/
- **Repository:** http://nlp.innobic.yzu.edu.tw/resources/ChineseEmoBank.html
- **Download directly here:** http://nlp.innobic.yzu.edu.tw/resources/chinese-emobank_download.html

#### CVAI
- **Source:** A Dimensional Valence-Arousal-Irony Dataset for Chinese Sentence and Context
  - https://aclanthology.org/2022.rocling-1.19/
- **Repository:** http://nlp.innobic.yzu.edu.tw/resources/chinese-vai_download.html

#### ANPST
- **Source:** Affective Norms for 718 Polish Short Texts (ANPST): Dataset with Affective Ratings for Valence, Arousal, Dominance, Origin, Subjective Significance and Source Dimensions
  - https://www.frontiersin.org/articles/10.3389/fpsyg.2016.01030/full
- **Repository:** https://figshare.com/s/e4b4e339138f07c63153
- **Download directly here:** https://figshare.com/ndownloader/files/5343997?private_link=e4b4e339138f07c63153

#### MAS
- **Source:** Minho Affective Sentences (MAS): Probing the roles of sex, mood, and empathy in affective ratings of verbal stimuli
  - https://link.springer.com/article/10.3758/s13428-016-0726-0
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above. See ESM 2.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-016-0726-0/MediaObjects/13428_2016_726_MOESM2_ESM.docx

#### Yee
- **Source:** Valence, arousal, familiarity, concreteness, and imageability ratings for 292 two-character Chinese nouns in Cantonese speakers in Hong Kong
  - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0174569#sec014
- **Repository:** https://figshare.com/articles/dataset/Valence_arousal_familiarity_concreteness_and_imageability_ratings_for_292_two-character_Chinese_nouns_in_Cantonese_speakers_in_Hong_Kong/4791586?file=7883134
- **Download directly here:** https://figshare.com/ndownloader/articles/4791586/versions/1

#### Ćoso et al.
- **Source:** Affective and concreteness norms for 3,022 Croatian words
  - https://journals.sagepub.com/doi/full/10.1177/1747021819834226
- **Repository:** https://www.ucace.com/links/
- **Download directly here:** https://www.ucace.com/app/download/30931812/Supplementary+material_%C4%86oso+et+al.xlsx
 
#### Moors et al.
- **Source:** Norms of valence, arousal, dominance, and age of acquisition for 4,300 Dutch words 
  - https://link.springer.com/article/10.3758/s13428-012-0243-8
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0243-8/MediaObjects/13428_2012_243_MOESM1_ESM.xlsx

#### Verheyen et al.
- **Source:** Lexicosemantic, affective, and distributional norms for 1,000 Dutch adjectives
  - https://link.springer.com/article/10.3758/s13428-019-01303-4
- **Repository:** https://osf.io/nyg8v/
- **Download directly here:** https://osf.io/download/6zxej/

#### NRC-VAD
- **Source:** Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words
  - https://aclanthology.org/P18-1017/
- **Repository:** http://saifmohammad.com/WebPages/nrc-vad.html
- **Download directly here:** http://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip

#### Warriner et al.
- **Source:** Norms of valence, arousal, and dominance for 13,915 English lemmas
  - https://link.springer.com/article/10.3758/s13428-012-0314-x
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0314-x/MediaObjects/13428_2012_314_MOESM1_ESM.zip

#### Scott et al.
- **Source:** The Glasgow Norms: Ratings of 5,500 words on nine scales
  - https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0314-x/MediaObjects/13428_2012_314_MOESM1_ESM.zip
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-018-1099-3/MediaObjects/13428_2018_1099_MOESM2_ESM.csv

#### Söderholm et al.
- **Source:** Valence and arousal ratings for 420 Finnish nouns by age and gender
  - https://pubmed.ncbi.nlm.nih.gov/24023650/
- **Repository:** https://figshare.com/articles/dataset/_Valence_and_Arousal_Ratings_for_420_Finnish_Nouns_by_Age_and_Gender_/785492
- **Download directly here:** https://figshare.com/ndownloader/files/1186672

#### Eilola et al.
- **Source:** Affective norms for 210 British English and Finnish nouns
  - https://link.springer.com/article/10.3758/BRM.42.1.134#SecESM1
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2FBRM.42.1.134/MediaObjects/Eilola-BRM-2010.zip

#### FAN
- **Source:** Affective norms for french words (FAN)
  - https://link.springer.com/article/10.3758/s13428-013-0431-1
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-013-0431-1/MediaObjects/13428_2013_431_MOESM2_ESM.xlsx

#### FEEL
- **Source:** Valence, arousal, and imagery ratings for 835 French attributes by young, middle-aged, and older adults: The French Emotional Evaluation List (FEEL)
  - https://www.sciencedirect.com/science/article/abs/pii/S1162908812000278
- **Repository:** https://osf.io/u52dy/
- **Download directly here:** https://osf.io/download/ps7te/

#### BAWL-R
- **Source:** The Berlin Affective Word List Reloaded (BAWL-R)
  - https://link.springer.com/article/10.3758/BRM.41.2.534
- **Repository:** https://osf.io/hx6r8/
- **Download directly here:** https://osf.io/download/cspef/

#### ANGST
- **Source:** ANGST: Affective norms for German sentiment terms, derived from the affective norms for English words
  - https://link.springer.com/article/10.3758/s13428-013-0426-y
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-013-0426-y/MediaObjects/13428_2013_426_MOESM1_ESM.xlsx

#### LANG
- **Source:** Leipzig Affective Norms for German: A reliability study
  - https://link.springer.com/article/10.3758/BRM.42.4.987#SecESM1
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2FBRM.42.4.987/MediaObjects/Kanske-BRM-2010.zip

#### Italian ANEW
- **Source:** Affective Norms for Italian Words in Older Adults: Age Differences in Ratings of Valence, Arousal and Dominance
  - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169472#sec015
- **Repository:** https://figshare.com/articles/dataset/Affective_Norms_for_Italian_Words_in_Older_Adults_Age_Differences_in_Ratings_of_Valence_Arousal_and_Dominance/4512950
- **Download directly here:** https://figshare.com/ndownloader/files/7305791

#### Xu et al.
- **Source:** Valence and arousal ratings for 11,310 simplified Chinese words
  - https://link.springer.com/article/10.3758/s13428-021-01607-4
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-021-01607-4/MediaObjects/13428_2021_1607_MOESM1_ESM.csv

#### CVAW
- **Source:** Building Chinese Affective Resources in Valence-Arousal Dimensions
  - https://aclanthology.org/N16-1066/
- **Repository:** http://nlp.innobic.yzu.edu.tw/resources/ChineseEmoBank.html
- **Download directly here:** http://nlp.innobic.yzu.edu.tw/resources/chinese-emobank_download.html

#### ANPW_R
- **Source:** Affective Norms for 4900 Polish Words Reload (ANPW_R): Assessments for Valence, Arousal, Dominance, Origin, Significance, Concreteness, Imageability and, Age of Acquisition
  - https://www.frontiersin.org/articles/10.3389/fpsyg.2016.01081/full#h10
- **Repository:** https://figshare.com/articles/dataset/DataSheet1_Affective_Norms_for_4900_Polish_Words_Reload_ANPW_R_Assessments_for_Valence_Arousal_Dominance_Origin_Significance_Concreteness_Imageability_and_Age_of_Acquisition_XLSX/16420035?backTo=/collections/Affective_Norms_for_4900_Polish_Words_Reload_ANPW_R_Assessments_for_Valence_Arousal_Dominance_Origin_Significance_Concreteness_Imageability_and_Age_of_Acquisition/5579574
- **Download directly here:** https://figshare.com/ndownloader/files/30426825

#### NAWL
- **Source:** Nencki Affective Word List (NAWL): the cultural adaptation of the Berlin Affective Word List–Reloaded (BAWL-R) for Polish
  - https://link.springer.com/article/10.3758/s13428-014-0552-1#Sec18
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-014-0552-1/MediaObjects/13428_2014_552_MOESM1_ESM.xlsx

#### Portuguese ANEW
- **Source:** The adaptation of the Affective Norms for English Words (ANEW) for European Portuguese
  - https://link.springer.com/article/10.3758/s13428-011-0131-7
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-011-0131-7/MediaObjects/13428_2011_131_MOESM1_ESM.xls

#### Stadthagen-Gonzalez et al.
- **Source:** Norms of valence and arousal for 14,031 Spanish words
  - https://link.springer.com/article/10.3758/s13428-015-0700-2#Sec16
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-015-0700-2/MediaObjects/13428_2015_700_MOESM1_ESM.csv

#### Kapucu et al.
- **Source:** Turkish Emotional Word Norms for Arousal, Valence, and Discrete Emotion Categories
  - https://journals.sagepub.com/doi/10.1177/0033294118814722?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed
- **Repository:** https://osf.io/86a4g/
- **Download directly here:** https://osf.io/download/x5sm8/








## Models
We make available the models we trained, in three different sizes. These three multilingual models support 100 languages.
- DistilBERT
  - 134M parameters.
- XLM-RoBERTa-base
  - 270M parameters.
- XLM-RoBERTa-large
  - 550M parameters.

#### DistilBERT
https://drive.google.com/drive/folders/1a3ToFHaGQxxAPI4dXc_shUrjROj7OrKt?usp=share_link

#### XLM-RoBERTa-base
https://drive.google.com/drive/folders/1CTgIEIDNHhV75qQ7-uovt6oXkiUIAVH8?usp=share_link

#### XLM-RoBERTa-large
https://drive.google.com/drive/folders/1BzdVmN51f33NHrdemJajz67MmlZljB2J?usp=share_link



## Code

### Repository structure

Core code was reorganized under `src/va_gaze/`:

```text
src/va_gaze/
  cli/
    train_model.py
    setup_et_models.py
    compute_overall_metrics.py
  data/
    dataset.py
    prepare_english_data.py
  models/
    regression.py
    et2_wrapper.py
  train/
    custom_trainer.py
    fold_runner.py
    fold1.py
    fold2.py
  eval/
    metrics.py
    oof_reports.py
```

Root scripts (`train_model.py`, `setup_et_models.py`, `prepare_english_data.py`,
`compute_overall_metrics.py`) are thin wrappers for the modules above.

### One-shot setup (new GPU/server)

Run this once on a fresh machine:

```bash
bash install.sh
```

What it does:
- installs python dependencies from `requirements.txt` (includes `importlib_resources==6.5.2`)
- prepares ET2 and auto-downloads checkpoint from `skboy/et_prediction_2` if missing
- builds English dataset files under `data/`

Re-run quickly without reinstalling deps:

```bash
SKIP_DEPS=1 bash install.sh
```

If you also want ET model 1 assets:

```bash
WITH_ET1=1 bash install.sh
```

### ET model setup (includes ET2 HF auto-download)

If you prefer running ET setup manually:

```bash
python setup_et_models.py --skip-install --skip-et1 --et2-checkpoint ./checkpoints/et_predictor2_seed123
```

If `./checkpoints/et_predictor2_seed123(.pt/.safetensors)` is missing, the setup script
automatically downloads `et_predictor2_seed123.safetensors` from:
`skboy/et_prediction_2` on Hugging Face.
(`checkpoints/` is created automatically if needed.)

### Build English-only dataset (Google Drive zip via gdown)

If you want to run quickly with English data only, use:

```bash
python3 prepare_english_data.py --output-dir data --seed 42
```

The script is idempotent: if `full_dataset_fold1.csv`, `full_dataset_fold2.csv`, and
`full_dataset_english_all.csv` already exist, it skips rebuilding unless `--force` is used.

By default it downloads this zip from Google Drive (via `gdown`) and extracts TSV files:
- `https://drive.google.com/file/d/1xXM32nva_4I3EAVAOrQ84L16f-LjsJbj/view?usp=sharing`

This creates:
- `data/full_dataset_fold1.csv`
- `data/full_dataset_fold2.csv`
- `data/full_dataset_english_all.csv`
- `data/external_english/*.tsv` (extracted source files)

If you already have local TSV files and do not want to download again:

```bash
python3 prepare_english_data.py --output-dir data --seed 42 --force --skip-gdrive-download
```

The builder auto-loads every `*.tsv` in `data/external_english/` when columns
`text`, `valence`, and `arousal` are present.
(`data/external_english/` is auto-created if missing.)

If you prefer another dataset folder:

```bash
python3 prepare_english_data.py --output-dir /path/to/my_data --seed 42 --force
```

To exclude one or more source datasets while keeping the existing fold split:

```bash
python filter_datasets.py --input-dir data --output-dir data_no_iemocap --exclude IEMOCAP
```

Use the filtered folder during training/evaluation:

```bash
python train_model.py xlmroberta-large mse --data-dir data_no_iemocap
```

To fine-tune the model please run the file `train_model.py`.
It expects two arguments:
- Model: **distilbert** or **xlmroberta-base** or **xlmroberta-large**
- Loss function: **mse** or **ccc** or **robust** or **mse+ccc** or **robust+ccc**

### GazeConcat / GazeAdd (ET model 2) for VA

You can enable Seeing Eye to AI-style concatenation of ET features with:

```bash
python train_model.py xlmroberta-base mse+ccc \
  --use-gaze-concat \
  --et2-checkpoint ./checkpoints/et_predictor2_seed123 \
  --features-used 0,1,0,1,0 \
  --fp-dropout 0.1,0.3
```

Or add the projected ET embeddings elementwise to the text embeddings with:

```bash
python train_model.py xlmroberta-base mse+ccc \
  --use-gaze-add \
  --et2-checkpoint ./checkpoints/et_predictor2_seed123 \
  --features-used 0,1,0,1,0 \
  --fp-dropout 0.1,0.3 \
  --gaze-add-scale 0.05
```

- `--use-gaze-concat`: enables gaze-text concatenation.
- `--use-gaze-add`: enables gaze-text embedding addition.
- `--et2-checkpoint`: path to your CMCL-RoBERTa ET2 checkpoint (`.pt` or `.safetensors`).
- `--features-used`: feature flags in `nFix,FFD,GPT,TRT,fixProp` order.
- `--fp-dropout`: dropout values for the ET feature projector.
- `--gaze-add-scale`: fixed scale for GazeAdd's gaze residual (default `0.05`).
- `--train-gaze-add-scale`: make the GazeAdd scale learnable.
- `--use-gaze-concat` and `--use-gaze-add` are mutually exclusive.
- with `--use-gaze-concat`, keep `--maxlen <= 255` (concat doubles sequence length).
- checkpoint options:
  - `--save-total-limit` (default `1`) keeps only recent checkpoints to reduce disk usage.
  - `--save-strategy no` disables periodic checkpoint saving (useful on low-storage GPUs).

### Experiment matrix (single README version)

Default VA hyperparameters (unchanged):
- batch size: `16`
- learning rate: `6e-6`
- train epochs: `10`
- weight decay: `0.01`
- warmup ratio: `0.1`
- optimizer: `adamw_torch`
- gradient accumulation: `1`
- seed: `42`
- maxlen: `200`

Full CLI:

```bash
python train_model.py <model> <loss> \
  [--use-gaze-concat] \
  [--use-gaze-add] \
  [--et2-checkpoint <path>] \
  [--features-used <f1,f2,f3,f4,f5>] \
  [--fp-dropout <p1,p2>] \
  [--gaze-add-scale <float>] \
  [--train-gaze-add-scale] \
  [--batch-size <int>] \
  [--learning-rate <float>] \
  [--train-epochs <int>] \
  [--max-steps <int>] \
  [--weight-decay <float>] \
  [--warmup-ratio <float>] \
  [--optim <name>] \
  [--gradient-accumulation-steps <int>] \
  [--seed <int>] \
  [--maxlen <int>] \
  [--data-dir <path>]
```

Feature flag order is always: `nFix,FFD,GPT,TRT,fixProp`.
Examples:
- all features: `1,1,1,1,1`
- fcomb2.2 (FFD+TRT): `0,1,0,1,0`
- TRT only: `0,0,0,1,0`
- FFD only: `0,1,0,0,0`

### Out-of-fold overall metrics

After both folds finish, `train_model.py` merges `predictions_fold1.csv` and
`predictions_fold2.csv` and writes:

- `all_predictions.csv`
- `overall_metrics.csv`
- `overall_metrics.json`
- `dataset_metrics.csv`

To recompute these files for an existing run:

```bash
python compute_overall_metrics.py Preds/<run-directory>
# optional: --data-dir /path/to/data
```

## License

This repository is released under the MIT License.
See `LICENSE`.
