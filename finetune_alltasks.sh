#!/usr/bin/env bash
set -x
set -e

DATASET=clintox
DICT=/youtdict
POSITIONAL=()
BINARIZE_DATA=false
UPDATEFREQ=1
ARCH=dmp
MODEL=/yourmodel
LR=1e-4
DROPOUT=0.1
WEIGHTDECAY=0.1
WARMUP=6
MP=512

while [[ $# -gt 0 ]]; do
    key=$1
    case $key in
    -d | --dataset)
        DATASET=$2
        shift 2
        ;;
    --dict)
        DICT=$2
        shift 2
        ;;
    --bd)
        BINARIZE_DATA=true
        shift
        ;;
    -u | --updatefreq)
        UPDATEFREQ=$2
        shift 2
        ;;
    -a | --arch)
        ARCH=$2
        shift 2
        ;;
    -m | --model)
        MODEL=$2
        shift 2
        ;;
    --lr)
        LR=$2
        shift 2
        ;;
    --dropout)
        DROPOUT=$2
        shift 2
        ;;
    --wd)
        WEIGHTDECAY=$2
        shift 2
        ;;
    --wu)
        WARMUP=$2
        shift 2
        ;;
    --mp)
        MP=$2
        shift 2
        ;;
    *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done

if [ ! -f $MODEL ]; then
    echo "$MODEL does not exist!"
    exit
fi

if [[ "$BINARIZE_DATA" = true ]]; then
    cd /tmp/pretrainmol
    pip install -e . --user
    
    if [ ! -d ogb ]; then
        git clone https://github.com/snap-stanford/ogb
        cd ogb
        pip install -e . --user
        cd ..
    fi
    python molecule/formatdata.py $DATASET
    for file in molecule/$DATASET/*; do
        if [ -d "$file" ]; then
            for SPLIT in train dev test; do
                python molecule/canonicalize.py \
                    $file/${SPLIT}.input0
                python molecule/tokenize_re.py \
                    $file/${SPLIT}.input0.can \
                    --output-fn $file/${SPLIT}.input0.bpe
            done
            taskname=$(basename $file)
            rm -rf molnet-bin/$DATASET/$taskname
            fairseq-preprocess \
                --only-source \
                --trainpref $file/train.input0.bpe \
                --validpref $file/dev.input0.bpe \
                --testpref $file/test.input0.bpe \
                --destdir molnet-bin/$DATASET/$taskname/input0 \
                --workers 30 --srcdict molecule/dict.txt \
                --molecule

            fairseq-preprocess \
                --only-source \
                --trainpref $file/train.label \
                --validpref $file/dev.label \
                --testpref $file/test.label \
                --destdir molnet-bin/$DATASET/$taskname/label \
                --workers 30 \
                --srcdict molecule/dict_label.txt

        fi
    done
fi

MAX_SENTENCES=8 # Batch size.
cd /tmp/$CODEDIR
case $DATASET in
bbbp | kfold_s*_bbbp | kfold_s1_bbbp | kfold_s2_bbbp)
    total=2040
    ;;
clintox | kfold_s*_clintox | kfold_s1_clintox | kfold_s2_clintox)
    total=1480
    ;;
hiv | ogbg-molhiv)
    total=41130
    MAX_SENTENCES=32
    UPDATEFREQ=$((UPDATEFREQ))
    ;;
ogbg-molpcba)
    total=438000
    MAX_SENTENCES=32
    UPDATEFREQ=$((4 * UPDATEFREQ))
    ;;
muv)
    total=93100
    MAX_SENTENCES=32
    UPDATEFREQ=$((4 * UPDATEFREQ))
    ;;
tox21 | kfold_s*_tox21 | kfold_s1_tox21 | kfold_s2_tox21)
    total=7840
    MAX_SENTENCES=16
    ;;
bace | kfold_s*_bace | kfold_s1_bace | kfold_s2_bace)
    total=1520
    ;;
sider | kfold_s*_sider | kfold_s1_sider | kfold_s2_sider)
    total=1480
    ;;
*)
    echo error
    exit
    ;;
esac

epoch=10
deno=$((MAX_SENTENCES * 10 * UPDATEFREQ))
TOTAL_NUM_UPDATES=$(((total * 8 * epoch + deno - 1) / deno))
deno=$((deno * 100))
WARMUP_UPDATES=$(((total * 8 * epoch * WARMUP + deno - 1) / deno))
HEAD_NAME=molecule_head # Custom name for the classification head.
NUM_CLASSES=2           # Number of classes for the classification task.

cudacap=$(python -c "import torch;print(torch.cuda.get_device_capability(0)[0] >= 7)")
if [ "$cudacap" == 'True' ]; then
    FP16="--fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128"
else
    FP16=''
fi

for file in molnet-bin/$DATASET/*; do
    if [ -d "$file" ]; then
        taskname=$(basename $file)
        SAVE_DIR=checkpoints/$DATASET/$taskname

        CUDA_VISIBLE_DEVICES=0 fairseq-train molnet-bin/$DATASET/$taskname \
            --restore-file $MODEL \
            --max-positions $MP \
            --batch-size $MAX_SENTENCES \
            --task graph_sp \
            --reset-optimizer --reset-dataloader --reset-meters \
            --required-batch-size-multiple 1 \
            --arch $ARCH \
            --criterion graph_sp \
            --classification-head-name $HEAD_NAME \
            --num-classes $NUM_CLASSES \
            --dropout $DROPOUT --attention-dropout 0.1 \
            --weight-decay $WEIGHTDECAY --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
            --clip-norm 0.0 \
            --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
            $FP16 --max-epoch $epoch \
            --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
            --shorten-method "truncate" \
            --find-unused-parameters \
            --update-freq $UPDATEFREQ \
            --save-dir $SAVE_DIR "${POSITIONAL[@]}"
    fi
done


