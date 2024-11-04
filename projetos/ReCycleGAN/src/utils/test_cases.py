from pathlib import Path

NEXET = '/content/nexet'
OUT_FOLDER = '/content/out'

# Base Parameters Dictionary (used across all test cases)
BASE = {
    'restart_path': None,
    'parameters_path': None,
    'run_wandb': True,

    'data_folder': Path(NEXET),
    'csv_type': '_filtered',
    'out_folder': Path(OUT_FOLDER),
    'use_cuda': True,

    'print_memory': True,
    "num_epochs": 50,
    "checkpoint_interval": 2,
    "n_samples": None,

    'batch_size': 16,
    'img_height': 256,
    'img_width': 256,

    'channels': 3,
    'n_features': 32,
    'n_residual_blocks': 5,
    'n_downsampling': 2,
    'norm_type': 'instance',
    'add_skip': False,
    'add_attention': None,
    'add_lora': False,
    'lora_rank': 4,

    'use_replay_buffer': True,
    'replay_buffer_size': 50,

    'vanilla_loss': True,
    'cycle_loss_weight': 10,
    'id_loss_weight': 5,
    'plp_loss_weight': 0,
    'plp_step': 16,
    'plp_beta': 0.99,

    'lr': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,

    'step_size': 1000,
    'gamma': 0.5,
    'amp': True
}

# Test Cases
TEST_CASES = {
    "1": BASE | {
        'amp': False,
        'experiment_name': "TEST_CASE_1",
        'experiment_description': "Basic CycleGAN with vanilla setup"
    },

    "2": BASE | {
        'experiment_name': "TEST_CASE_2",
        'experiment_description': "Vanilla CycleGAN with AMP enabled"
    },

    "3": BASE | {
        'add_skip': True,
        'experiment_name': "TEST_CASE_3'",
        'experiment_description': "CycleGAN with skip connections"
    },

    "4": BASE | {
        'vanilla_loss': False,
        'experiment_name': "'TEST_CASE_4",
        'experiment_description': "CycleGAN without MSEloss"
    },

    "5": BASE | {
        'add_skip': True,
        'vanilla_loss': False,
        'experiment_name': "TEST_CASE_5",
        'experiment_description': "CycleGAN with skip connections and MSEloss"
    },

    "6": BASE | {
        'add_attention': 'gen',
        'experiment_name': "TEST_CASE_6",
        'experiment_description': "CycleGAN with self-attention in generator"
    },

    "7": BASE | {
        'add_attention': 'disc',
        'experiment_name': "TEST_CASE_7",
        'experiment_description': "CycleGAN with self-attention in discriminator"
    },

    "8": BASE | {
        'plp_loss_weight': 5,
        'experiment_name': "TEST_CASE_8",
        'experiment_description': "CycleGAN with perceptual loss weight set to 5"
    }
}