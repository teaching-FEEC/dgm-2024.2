# %% [markdown]
# # Extract speaker embeddings

# %%
# Para treinamento multilíngue (neste caso, multisotaque), são requeridos datasets distintos,
# sendo cada um deles um objeto da classe BaseDatasetConfig
# A estruturação do dataset é condicionada pelo formatter usado

from TTS.config.shared_configs import BaseDatasetConfig

RE_config = BaseDatasetConfig(
    formatter="vctk_old",
    dataset_name="RE",
    meta_file_train="",
    meta_file_val="",
    path="/workspace/datasets/CORAA/dataset_RE_02",
    language="re",
)

MG_config = BaseDatasetConfig(
    formatter="vctk_old",
    dataset_name="MG",
    meta_file_train="",
    meta_file_val="",
    path="/workspace/datasets/CORAA/dataset_MG_02",
    language="mg",
)

# %%
import os
# Esta célula carrega o modelo que desemepenha o papel de encoder de falante e
# processa o(s) dataset(s) usando a função (compute_embeddings)

from TTS.bin.compute_embeddings import compute_embeddings

SPEAKER_ENCODER_CHECKPOINT_PATH = "/workspace/yourTTS_checkpoints/tts_models--multilingual--multi-dataset--your_tts/model_se.pth.tar"

SPEAKER_ENCODER_CONFIG_PATH = "/workspace/yourTTS_checkpoints/tts_models--multilingual--multi-dataset--your_tts/config_se.json"

D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training

DATASETS_CONFIG_LIST = [RE_config, MG_config]

for dataset_conf in DATASETS_CONFIG_LIST:
    # Check if the embeddings weren't already computed, if not compute it
    embeddings_file = os.path.join(dataset_conf.path, "speakers.pth")
    if not os.path.isfile(embeddings_file):
        print(f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
        compute_embeddings(
            SPEAKER_ENCODER_CHECKPOINT_PATH,
            SPEAKER_ENCODER_CONFIG_PATH,
            embeddings_file,
            old_speakers_file=None,
            config_dataset_path=None,
            formatter_name=dataset_conf.formatter,
            dataset_name=dataset_conf.dataset_name,
            dataset_path=dataset_conf.path,
            meta_file_train=dataset_conf.meta_file_train,
            meta_file_val=dataset_conf.meta_file_val,
            disable_cuda=False,
            no_eval=False,
        )
    D_VECTOR_FILES.append(embeddings_file)

# %%
D_VECTOR_FILES

# %% [markdown]
# # Configs

# %% [markdown]
# * Audio

# %%
SAMPLE_RATE = 16000
MAX_AUDIO_LEN_IN_SECONDS = 10

# %%
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig


audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80
)

# %% [markdown]
# * Model & Training args

# %%
# Init VITSArgs setting the arguments that are needed for the YourTTS model

model_args = VitsArgs(
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    num_layers_text_encoder=10,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="2",
    # Useful parameters to enable the Speaker Consistency Loss (SCL) described in the paper
    # Experiments showed better results for PTBR using SCL
    use_speaker_encoder_as_loss=False,

    # Useful parameters to ENABLE MULTILINGUAL TRAINING
    use_language_embedding=True,
    embedded_language_dim=4
)

# %%
RUN_NAME = "run_02"

OUT_PATH = "/workspace/outputs"

BATCH_SIZE = 128

# %%
config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="TTSotaques",
    run_description="""
            - run_02
        """,
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=1000,
    save_step=5000,
    save_n_checkpoints=2,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=False,
    use_phonemes=False, ### Uses raw characters
    phonemizer="espeak",
    phoneme_language="en",
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="portuguese_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u00af\u00b7\u00df\u00e0\u00e1\u00e2\u00e3\u00e4\u00e6\u00e7\u00e8\u00e9\u00ea\u00eb\u00ec\u00ed\u00ee\u00ef\u00f1\u00f2\u00f3\u00f4\u00f5\u00f6\u00f9\u00fa\u00fb\u0169\u00fc\u00ff\u0101\u0105\u0107\u0113\u0119\u011b\u012b\u0131\u0142\u0144\u014d\u0151\u0153\u015b\u016b\u0171\u017a\u017c\u01ce\u01d0\u01d2\u01d4\u0430\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u0439\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f\u0451\u0454\u0456\u0457\u0491\u2013!'(),-.:;? ",
        punctuations="!'(),-.:;? ",
        phonemes="",
        is_unique=True,
        is_sorted=True,
    ),
    phoneme_cache_path=None,
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    test_sentences=[
        [
            "Voc\u00ea ter\u00e1 a vista do topo da montanha que voc\u00ea escalar.",
            "NURC_RE_EF_273",
            None,
            "re"
        ],

        [
            "Voc\u00ea ter\u00e1 a vista do topo da montanha que voc\u00ea escalar.",
            "NURC_RE_EF_346",
            None,
            "re"
        ],

        [
            "Quando voc\u00ea n\u00e3o corre nenhum risco, voc\u00ea arrisca tudo.",
            "bfammn01",
            None,
            "mg"
        ],

        [
            "Quando voc\u00ea n\u00e3o corre nenhum risco, voc\u00ea arrisca tudo.",
            "bfammn16",
            None,
            "mg"
        ]
    ],
    # Enable the weighted sampler
    use_weighted_sampler=False,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    weighted_sampler_attrs={"speaker_name": 1.0},
    weighted_sampler_multipliers={},
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
    speaker_encoder_loss_alpha=9.0,
)


# %% [markdown]
# * Train/Eval split

# %%
from TTS.tts.datasets import load_tts_samples

train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# %% [markdown]
# # Init model and trainer

# %%
model = Vits.init_from_config(config)

# %%
from trainer import Trainer, TrainerArgs

#VCTK checkpoints
RESTORE_PATH = "/workspace/yourTTS_checkpoints/tts_models--multilingual--multi-dataset--your_tts/model_file.pth.tar"

SKIP_TRAIN_EPOCH = False

trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# %% [markdown]
# # Train

# %%
trainer.fit()

# %% [markdown]
# # Synthesize
# 
# Via terminal:
# 
# ```tts --text <text> --model_path <path to model.pth> --config_path <path to config.json> --out_path <folfer/synth_file.wav> --speaker_idx <'speaker id'> --language_idx <language id ['re', 'mg']>```


