#!/usr/bin/env bash

# Non-speech hallucination (Whisper-large-v3): data preparation, zero-shot decoding, fine-tuning on
# non-speech audio with empty transcripts, and evaluation. Hallucination rate is measured on UrbanSound8K
# (empty references); LibriSpeech test-clean/test-other WER checks that speech recognition is preserved.

export rootdir=$(realpath ../../)
export PATH=$PATH:$rootdir/src/bin

stage=0
end_stage=3

musan_root=/path/to/musan
librispeech_root=/path/to/LibriSpeech
methods="aura"                              # aura jola
eval_sets="urbansound8k libri_test_clean libri_test_other"

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

if [ $stage -le 0 ] && [ $end_stage -ge 0 ]; then
  # non-speech training pool (AudioSet without speech, DEMAND, MUSAN music+noise) and evaluation sets
  bash local/download_demand.sh
  python local/prep_audioset.py
  python local/prep_demand.py
  python local/prep_musan.py --musan_root $musan_root
  python local/prep_nonspeech_train.py
  python local/prep_urbansound8k.py
  python local/prep_librispeech.py --librispeech_root $librispeech_root --split test-clean --out_dir data/libri_test_clean
  python local/prep_librispeech.py --librispeech_root $librispeech_root --split test-other --out_dir data/libri_test_other
  echo "[Stage 0] Data preparation finished."
fi

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  # decode the evaluation sets with the pretrained model
  for x in $eval_sets; do
    decode openai/whisper-large-v3 $x exp/whisper_zero_shot/largev3/$x
  done
  echo "[Stage 1] Zero-shot decoding finished."
fi

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  for method in $methods; do
    exp_dir=exp/whisper_largev3_${method}
    mkdir -p $exp_dir
    torchrun --rdzv-endpoint=localhost:12945 --nproc_per_node=1 \
      $rootdir/src/bin/train_asr.py conf/$method/whisper_largev3_train.yaml > $exp_dir/train.log 2>&1
  done
  echo "[Stage 2] Fine-tuning finished."
fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  # decode every saved checkpoint (one every 5 epochs) on the evaluation sets
  for method in $methods; do
    exp_dir=exp/whisper_largev3_${method}
    for ckpt in $exp_dir/checkpoint-*; do
      for x in $eval_sets; do
        decode $ckpt $x $ckpt/$x
      done
    done
  done
  echo "[Stage 3] Decoding finished."
fi
