export PS4='+{$LINENO($(date +"%F %T")):${FUNCNAME[0]}} '
full_path=$(readlink -f $0)
cd $(dirname $full_path)
exec 1>${full_path%.*}.log
exec 2>${full_path%.*}.err
set -x

ALL_DATA="../../data/short_text_classification/data.txt"
ALL_DATA_SEG="../../data/short_text_classification/data_seg.txt"
ALL_DATA_TO_ID="../../data/short_text_classification/data_to_id.h5"
VOCABULARY_DICT="../../data/short_text_classification/vocabulary_dict.pkl"

THREAD_NUM=24

function preprocess() {
    python parallelizer.py -s $THREAD_NUM split_with_words.py $ALL_DATA $ALL_DATA_SEG
    python preprocess.py $ALL_DATA_SEG $ALL_DATA_TO_ID $VOCABULARY_DICT
}

function run_fasttext() {
    python fasttext.py
}

#preprocess
run_fasttext

set +x
