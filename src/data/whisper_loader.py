#!/usr/bin/env python3


import os
import numpy as np

from datasets import Dataset, Audio, concatenate_datasets


class _SoundfileAudioDataset:
    """Decode-time dataset: reads audio with soundfile on access instead of the
    `datasets` Audio() feature, downmixes to mono and resamples to 16 kHz."""

    _TARGET_SR = 16000

    def __init__(self, records):
        self._records = records
        self.column_names = ["utt_id", "audio", "sentence"]

    def __len__(self):
        return len(self._records)

    def _decode(self, rec):
        import soundfile as sf
        path = rec["audio"]
        array, sr = sf.read(path, always_2d=False)
        if array.ndim > 1:
            array = array.mean(axis=-1)
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        if int(sr) != self._TARGET_SR:
            import librosa
            array = librosa.resample(array, orig_sr=int(sr), target_sr=self._TARGET_SR)
            sr = self._TARGET_SR
        out = dict(rec)
        out["audio"] = {"array": array, "sampling_rate": int(sr), "path": path}
        return out

    def __iter__(self):
        for rec in self._records:
            yield self._decode(rec)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._decode(r) for r in self._records[idx]]
        return self._decode(self._records[idx])


class WhisperDataset():
    def __init__(self, data_paths, data_args=None, for_decode=False):
        self.data_paths = data_paths
        self.for_decode = for_decode
        print("Loading data from: ", data_paths)

        if for_decode:
            records = []
            for split in data_paths.keys():
                records.extend(self._load_records(data_paths[split]))
            self.data = _SoundfileAudioDataset(records)
            return

        datasets = []
        for split in data_paths.keys():
            datasets.append(self.create_dataset(data_paths[split]))

        self.data = concatenate_datasets(datasets)

    def create_dataset(self, data_path):

        audio_dict = self._load_audio_path(data_path['scp_path'])
        label_dict = self._load_label(data_path['text_label'])
        assert len(audio_dict) == len(label_dict), "label and sample size mismatch"

        paired_dict = {"utt_id": [], "audio": [], "sentence": []}

        # prepare dictionary for huggingface dataset
        for i in range(len(audio_dict)):
            utt, audio_path = audio_dict[i]
            label = label_dict[utt]
            paired_dict["utt_id"].append(utt)
            paired_dict["audio"].append(audio_path)
            paired_dict["sentence"].append(label)

        dataset = Dataset.from_dict(paired_dict).cast_column("audio", Audio())

        return dataset

    def _load_records(self, data_path):
        audio_dict = self._load_audio_path(data_path['scp_path'])
        label_dict = self._load_label(data_path['text_label'])
        assert len(audio_dict) == len(label_dict), "label and sample size mismatch"
        records = []
        for utt, audio_path in audio_dict:
            records.append({
                "utt_id": utt,
                "audio": audio_path,
                "sentence": label_dict[utt],
            })
        return records

    def _load_audio_path(self, wav_path):

        audio_dict = []

        with open(wav_path, 'r') as fin:
            line = fin.readline()
            while line:
                if '\t' in line:
                    line = line.strip().split('\t')
                else:
                    line = line.strip().split(' ')
                utt = line[0]
                i = 1

                try:
                    while not os.path.exists(line[i]):
                        i += 1
                except:
                    print(line)
                    print(utt)
                    print(i)
                    print("SCP FILE Not valid!")
                    break

                audio_dict.append((utt, line[i]))
                line = fin.readline()

        print("Reading {} lines from {}".format(len(audio_dict), wav_path))

        return audio_dict

    def _load_label(self, lab_path):
        label_dict = dict()
        with open(lab_path, 'r') as fin:
            line = fin.readline()

            while line:
                if '\t' in line:
                    line= line.strip().split('\t')
                else:
                    line= line.strip().split(' ')
                label_dict[line[0]] = ' '.join(line[1:])
                line = fin.readline()

        print("Reading {} lines from {}".format(len(label_dict), lab_path))
        return label_dict

