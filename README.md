# Dual-view molecule pre-training

This repository contains the code for [**Dual-view Molecule Pre-training**](https://arxiv.org/abs/2106.10234). We implement our method based on the codebase of [fairseq](https://github.com/pytorch/fairseq). 

# Requirements and Installation
* PyTorch version == 1.8.0
* PyTorch Geometric version == 1.6.3
* RDKit version == 2020.09.5

You can build the [Dockerfile](Dockerfile) or use the docker image `teslazhu/pretrainmol36:latest`.

To install the code from source
```
git clone https://github.com/microsoft/DVMP
cd DVMP
pwd=$PWD

git clone git@github.com:pytorch/fairseq.git /tmp/fairseq
cp dvmp.patch /tmp/fairseq
cd /tmp/fairseq
git checkout aa5f0119a383e013e56ae5d88e4a7aff0e67f0f9
git apply dvmp.patch

cd $pwd
cp -r /tmp/fairseq/* ./
```
# Getting Started
## Pre-training
### Data Preprocessing
```shell
DATADIR=/yourdatadir

# Canonicalize all SMILES
python molecule/canonicalize.py $DATADIR/train.smi --workers 30

# Tokenize all SMILES
python molecule/tokenize_re.py $DATADIR/train.smi.can --workers 30 \
  --output-fn $DATADIR/train.bpe 

# You also should canonicalize and tokenize the dev set.

# Binarize the data
fairseq-preprocess \
    --only-source \
    --trainpref $DATADIR/train.bpe \
    --validpref $DATADIR/valid.bpe \
    --destdir /data/pubchem \
    --workers 30 --srcdict molecule/dict.txt \
    --molecule

```
### Train
```shell
DATADIR=/data/pubchem

TOTAL_UPDATES=125000 # Total number of training steps
WARMUP_UPDATES=10000 # Warmup the learning rate over this many updates
PEAK_LR=0.0005       # Peak learning rate, adjust as needed
UPDATE_FREQ=16       # Increase the batch size 16x
MAXTOKENS=12288
DATATYPE=tg
arch=dmp

fairseq-train --fp16 $DATADIR \
    --task dmp --criterion dmp \
    --arch $arch --max-tokens $MAXTOKENS --update-freq $UPDATE_FREQ \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --validate-interval-updates 1000 --save-interval-updates 1000 \
    --datatype $DATATYPE --required-batch-size-multiple 1 \
    --use-bottleneck-head --bottleneck-ratio 4  --use-mlm | tee -a ${SAVE_DIR}/training.log
```
## Finetuning
Finetune with the pre-trained model {checkpoint}.
```shell
bash finetune_alltasks.sh -d clintox --lr 0.0001 -u 1 \
  --wd 0.01 --seed 1 \
  --datatype tg --gradmultiply --use-bottleneck-head --bottleneck-ratio 4 \
   -m {checkpoint} --coeff1 0 --bd --dict /yourdict
```
### Inference
```shell
python molecule/inference.py molnet-bin/$DATASET/$taskname {checkpoint}
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
