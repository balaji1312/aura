#!/usr/bin/env bash

# MyST (child speech): zero-shot decoding, fine-tuning, and decoding of fine-tuned models.
# Expects Kaldi-style data/{train,dev,test}/{wav.scp,text}; test is the unfiltered test set.

export rootdir=$(realpath ../../)
export PATH=$PATH:$rootdir/src/bin

stage=1
end_stage=3

methods="aura"                              # lora bitfit red loreft jola aura
model_sizes="tiny base small medium largev3"
dev_set=dev
test_sets="test"

declare -A pretrained=(
  [tiny]=openai/whisper-tiny.en [base]=openai/whisper-base.en [small]=openai/whisper-small.en
  [medium]=openai/whisper-medium.en [largev3]=openai/whisper-large-v3
)

decode() {  # <model> <data split> <output dir>
  mkdir -p $3
  decode_asr.py \
    --wav_scp data/$2/wav.scp \
    --trn_scp data/$2/text \
    --model $1 \
    --processor $1 \
    --result_ref_file $3/ref.txt \
    --result_hyp_file $3/hyp.txt \
    --chunk_length 30 \
    --batch_size 8 > $3/decode.log 2>&1
}

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  # decode the test sets with the pretrained Whisper models
  for size in $model_sizes; do
    for x in $test_sets; do
      decode ${pretrained[$size]} $x exp/whisper_zero_shot/$size/$x
    done
  done
  echo "[Stage 1] Zero-shot decoding finished."
fi

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  # fine-tune every method and model size
  for method in $methods; do
    for size in $model_sizes; do
      exp_dir=exp/whisper_${size}_${method}
      mkdir -p $exp_dir
      torchrun --rdzv-endpoint=localhost:12945 --nproc_per_node=1 \
        $rootdir/src/bin/train_asr.py conf/$method/whisper_${size}_train.yaml > $exp_dir/train.log 2>&1
    done
  done
  echo "[Stage 2] Fine-tuning finished."
fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  # decode every checkpoint on the dev set, then the best dev checkpoint on the test sets
  for method in $methods; do
    for size in $model_sizes; do
      exp_dir=exp/whisper_${size}_${method}
      for ckpt in $exp_dir/checkpoint-*; do
        decode $ckpt $dev_set $ckpt/$dev_set
      done
      best=$(for f in $exp_dir/checkpoint-*/$dev_set/decode.log; do
               echo "$(grep -m1 'Total WER:' $f | awk '{print $3}') $(dirname $(dirname $f))"
             done | sort -g | head -1 | awk '{print $2}')
      for x in $test_sets; do
        decode $best $x $best/$x
      done
      echo "$method $size: best checkpoint on $dev_set is $best"
    done
  done
  echo "[Stage 3] Decoding finished."
fi
