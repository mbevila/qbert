# HOW TO RUN
`cd` to the folder containing this file.
## Installation
### Requirements:
- Own a GPU
- Use `python` 3.7
### Install `pytorch`
Go to <https://pytorch.org/get-started/locally/> for instruction
### Install `fairseq`
To make sure things don't break, please install the library by following the procedure:
- `git clone https://github.com/pytorch/fairseq.git`
- `cd fairseq`
- `git reset --hard bbb4120b00e9ac7b12a52608349b7ad753fc0d19`
- `pip install .`
## Install `qbert`
- `cd` to the folder containing this file
- `cd qbert`
- `pip install -r requirements.txt`
- `pip install .`
- `python -c "import nltk; nltk.download('wordnet')"`
## CLM Train
### Preprocess LM data
- `python preprocess_lm.py --only-source --trainpref <plaintextcorpus-train> --validpref <plaintextcorpus-valid> --testpref <plaintextcorpus-test> --destdir <clm-data>`
### Train on CLM
- `CUDA_VISIBLE_DEVICES=<your-cuda-device> python train.py <clm-data> --task language_modeling --self-target --arch transformer_bilm_wiki103 --adaptive-input --adaptive-input-cutoff 35000,100000,200000 --adaptive-input-factor 4 --criterion adaptive_loss --adaptive-softmax-cutoff 35000,100000,200000 --adaptive-softmax-factor 4 --adaptive-softmax-dropout 0.3 --tie-adaptive-weights --decoder-attention-heads 8 --decoder-embed-dim 512 --decoder-ffn-embed-dim 2048 --decoder-input-dim 512 --decoder-layers 5 --decoder-output-dim 512 --dropout 0.1 --max-lr 1.0 --lr 1e-5 --lr-scheduler cosine --lr-shrink 0.5 --lr-period-updates 2000 --min-lr 1e-10 --optimizer nag --save-interval-updates 2000 --t-mult 1.5 --tokens-per-sample 100 --update-freq 16 --warmup-updates 2000 --decoder-share-directions --ddp-backend no_c10d --max-tokens 5000 --save-dir <clm-checkpoints> --clip-norm 0.1`
- You may need to choose different cutoffs if your vocabulary is smaller than 200k
- Takes ~10 days on a 1080 Ti to converge
## WSD Train
### Preprocess WSD Data
- `python preprocess_wsd.py --dictionary <clm-data>/dict.txt --max-length 100 --keep-oov --synsets --xml <Raganato-framework-root>/Training_Corpora/SemCor/semcor.data.xml --output <output-dir-wsd-data>/train`
- `python preprocess_wsd.py --dictionary <clm-data>/dict.txt --max-length 100 --keep-oov --synsets --xml <Raganato-framework-root>/Evaluation_Datasets/semeval2015/semeval2015.data.xml --output <wsd-data>/dev`
- `cp <clm-data>/dict.txt <wsd-data>/dict.txt`
### Train WSD
-`CUDA_VISIBLE_DEVICES=<your-cuda-device> python train.py <wsd-data> --arch transformer_seq --task sequence_tagging --save-dir <wsd-checkpoints> --criterion weighted_cross_entropy --tokens-per-sample 100 --max-tokens 4000 --max-epoch 70 --optimizer adam --lr 5e-4 --lr-scheduler cosine --max-lr 1e-3 --lr-period-updates 200 --t-mult 2.0 --warmup-init-lr 5e-5 --lr-shrink 0.5 --warmup-updates 200 --min-lr 1e-10 --decoder-embed-dim 512 --update-freq 6 --dropout 0.2 --relu-dropout 0.2 --attention-dropout 0.2 --clip-norm 0.25  --decoder-input-dim 512 --context-embeddings --context-embeddings-use-embeddings --context-embeddings-type qbert --context-embeddings-qbert-checkpoint <clm-checkpoints>/checkpoint_best.pt --log-format simple --decoder-layers 2`
### Eval WSD
- `python eval_wsd --checkpoints <wsd-checkpoints>/*.pt --dictionary <clm-data>/dict.txt --use-synsets --w 100 -W 10 -a 1 --eval-on <Raganato-framework-root>/Evaluation_Datasets/*/*.data.xml`
