import os
import random
import numpy as np
import torch.utils.data
import torchaudio

from typing import Dict, NoReturn
import datetime
import logging
import os
import pickle
import librosa
import numpy as np
import yaml

def int16_to_float32(x: np.int16) -> np.float32:

    return (x / 32767.0).astype(np.float32)
def read_data(dataset_dir, n_fft, data_len, sampling_rate):
    """Load and preprocess MUSDB18 dataset.
    The Songs in train dataset are split to fixed length segments.
    Args:
        dataset_dir (str): Path of MUSDB18 which was converted to .wav format.
        n_fft (int):  Size of Fourier Transform.
        data_len (int): Number of time frames of a training data.
        sampling_rate (int): Sampling rate.
    Returns:
        Tuple[List[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]:
            A tuple contains train data and test data.
            Train data is a list of tuple which contains magnitude spectrogram
            of input signal and ground truth separation mask. Test data is a
            list of tensors which contain mixture signal and separated signal.
    """
    window = torch.hann_window(n_fft)

    with torch.no_grad():
        # Train data
        train_dir = os.path.join(dataset_dir, 'train')
        wav_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)
                     if os.path.splitext(f)[1] == '.wav']
        train = []
        for path in wav_files:
            sound, sr = torchaudio.load(path)
            assert sr == sampling_rate
            sound_spec = torch.stft(sound, n_fft, window=window)
            sound_spec = sound_spec.pow(2).sum(-1).sqrt()
            x = sound_spec[0]
            t = sound_spec[1:]

            hop = data_len // 4
            # Split to fixed length segments
            for n in range((x.size(1) - data_len) // hop + 1):
                start = n * hop
                train.append((x[:, start:start + data_len],
                              t[:, :, start:start + data_len]))

        # Test data
        test_dir = os.path.join(dataset_dir, 'test')
        wav_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                     if os.path.splitext(f)[1] == '.wav']
        test = []
        for path in wav_files:
            sound, sr = torchaudio.load(path)
            assert sr == sampling_rate
            # Split data into two parts to save memory
            split_sound = torch.split(sound, sound.size(1) // 2, dim=1)
            test.extend(split_sound)

    return train, test

def create_logging(log_dir: str, filemode: str) -> logging:
    r"""Create logging to write out log files.
    Args:
        logs_dir, str, directory to write out logs
        filemode: str, e.g., "w"
    Returns:
        logging
    """
    os.makedirs(log_dir, exist_ok=True)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, "{:04d}.log".format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, "{:04d}.log".format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=log_path,
        filemode=filemode,
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    return logging

def check_configs_gramma(configs: Dict) -> NoReturn:
    r"""Check if the gramma of the config dictionary for training is legal."""

    paired_input_target_data = configs['train']['paired_input_target_data']

    if paired_input_target_data is False:

        input_source_types = configs['train']['input_source_types']
        augmentation_types = configs['train']['augmentations'].keys()

        for augmentation_type in list(
            set(augmentation_types)
            & set(
                [
                    'mixaudio',
                    'pitch_shift',
                    'magnitude_scale',
                    'swap_channel',
                    'flip_axis',
                ]
            )
        ):

            augmentation_dict = configs['train']['augmentations'][augmentation_type]

            for source_type in augmentation_dict.keys():
                if source_type not in input_source_types:
                    error_msg = (
                        "The source type '{}'' in configs['train']['augmentations']['{}'] "
                        "must be one of input_source_types {}".format(
                            source_type, augmentation_type, input_source_types
                        )
                    )
                    raise Exception(error_msg)

def get_pitch_shift_factor(shift_pitch: float) -> float:
    r"""The factor of the audio length to be scaled."""
    return 2 ** (shift_pitch / 12)

def read_yaml(config_yaml: str) -> Dict:
    """Read config file to dictionary.
    Args:
        config_yaml: str
    Returns:
        configs: Dict
    """
    with open(config_yaml, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)

    return 