# %%
import os
import librosa
import pandas as pd
import shutil
import shlex, subprocess
import argparse
from subprocess import PIPE

# %%
metadata_test = pd.read_csv('/workspace/datasets/CORAA/metadata_test_final.csv')
metadata_dev = pd.read_csv('/workspace/datasets/CORAA/metadata_dev_final.csv')
metadata_train = pd.read_csv('/workspace/datasets/CORAA/metadata_train_final.csv')

metadata = pd.concat([metadata_test, metadata_dev, metadata_train], ignore_index = True)
metadata['file_path'] = metadata['file_path'].str.replace('test/', 'train/')
metadata['file_path'] = metadata['file_path'].str.replace('dev/', 'train/')

metadata_filtered = metadata[(metadata['down_votes'] == 0) &                     # Datasets 01, 02
                             (metadata['votes_for_filled_pause'] == 0) &         #Dataset 02
                             #(metadata['votes_for_second_voice'] == 0) &
                             (metadata['votes_for_hesitation'] == 0)]            #Dataset 02
                             #(metadata['votes_for_no_identified_problem'] >= 1)
                             #(metadata['votes_for_noise_or_low_voice'] == 0)

import re

metadata_RE_273 = metadata_filtered[metadata_filtered['file_path'].str.contains(r"RE_EF_273")]
metadata_MG_20 = metadata_filtered[metadata_filtered['file_path'].str.contains(r"bfammn20")]

parser = argparse.ArgumentParser(description='Synthesize speech samples specifying model version')

parser.add_argument('-r', '--run', help='Run ID (Model version)')
parser.add_argument('-acc', '--accent', help='Accent ID')

arg_run = parser.parse_args()

if arg_run.run == 'run_01':
    run_name = 'run_01-November-14-2024_08+56PM-0000000'

elif arg_run.run == 'run_02':
    run_name = 'run_02-November-15-2024_08+17PM-0000000'

elif arg_run.run == 'run_03':
    run_name = 'run_03-November-23-2024_03+57AM-0000000'

elif arg_run.run == 'run_04':
    run_name = 'run_04-November-23-2024_08+15PM-0000000'

# %%
#RE

if arg_run.accent == 're':
  sample_id = 0

  for idx, row in metadata_RE_273.iterrows():
    if sample_id == 100:
      break
    sample_id_formatted = f"{sample_id:02d}"
    spkr_id, audio_filename = row['file_path'].split('/')[-2:]

    audio_path = os.path.join("/workspace/datasets/CORAA","dataset_RE_02", "wav48", spkr_id, audio_filename)
    if os.path.exists(audio_path):

      duration = librosa.get_duration(path=audio_path)

      if duration > 3 and duration < 10:
        ground_truth_path = f'/workspace/outputs/{run_name}/samples/{spkr_id}/Ground_truth'
        synthesized_path = f'/workspace/outputs/{run_name}/samples/{spkr_id}/Synthesized'
        os.makedirs(ground_truth_path, exist_ok=True)
        os.makedirs(synthesized_path, exist_ok=True)

        ground_truth_file_path = f'{ground_truth_path}/{sample_id_formatted}.wav'      
        
        shutil.copy(audio_path, ground_truth_file_path)

        command = f"sh /workspace/generate_samples.sh -t \"{row['text']}\" -s {spkr_id} -c {sample_id_formatted} -l 're'"
        args = shlex.split(command)
        args.extend(['-r', arg_run.run])
        print(args)
        p = subprocess.Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        
        sample_id += 1

      else:
        continue

    else: 
      continue

# %%
if arg_run.accent == 'mg':

  sample_id = 0

  for _, row in metadata_MG_20.iterrows():
    if sample_id == 100:
      break
    sample_id_formatted = f"{sample_id:02d}"
    spkr_id = row['file_path'].split('_')[-1][:-4]
    audio_filename = row['file_path'].split('/')[-1]

    audio_path = os.path.join("/workspace/datasets/CORAA","dataset_MG_02", "wav48", spkr_id, audio_filename)

    if os.path.exists(audio_path):
      print('exists')

      duration = librosa.get_duration(path=audio_path)

      if duration >= 2 and duration < 10:
        ground_truth_path = f'/workspace/outputs/{run_name}/samples/{spkr_id}/Ground_truth'
        synthesized_path = f'/workspace/outputs/{run_name}/samples/{spkr_id}/Synthesized'

        os.makedirs(ground_truth_path, exist_ok=True)
        os.makedirs(synthesized_path, exist_ok=True)

        ground_truth_file_path = f'{ground_truth_path}/{sample_id_formatted}.wav'

        shutil.copy(audio_path, ground_truth_file_path)

        command = f"sh /workspace/generate_samples.sh -t \"{row['text']}\" -s {spkr_id} -c {sample_id_formatted} -l 'mg'"
        args = shlex.split(command)
        args.extend(['-r', arg_run.run])
        p = subprocess.Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        print(stderr)
        
        sample_id += 1
      else:
        continue
      
    else:
      continue


