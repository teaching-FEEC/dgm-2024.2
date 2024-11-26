## How To Run

As an additional note, we have included a description of how to perform the functions proposed in this project.

**Data processing:**

`1.` Download ATM'22 database from internet.

`2.` Read and process data using class `rawCTData` in datasets.py and process_raw_data.ipynb


**Training GAN:**

1. Training configurations should be described in a YAML file composed by the following keys:

Keys related to model definition:
```yaml

model:
  name_model: "string with model name"
  dir_save_results: "path/name_of_folder_to_be_created/"  # default: ./name_model/
  new_model: True/False # if False, will try to resume a prior training session and restore models,  optimizers and lr_schedulers (default: True)
  use_pretrained_model: True/False # if True, will fine tune a given model (if that's the case new_model should be True) (default: False)
  generate_airway_segmentation: True/False # if False: PulmoNet generates only CT images, if True: tries to genrate both CT images and airway segmentation (default: False)
  path_to_saved_model_gen: "path to generator model to be restored if one wants to resume to a prior training session or one wants to fine tune" # or "" if new model
  path_to_saved_model_disc: "path to discriminator model to be restored if one wants to resume to a prior training session or one wants to fine tune" # or "" if new model
```

Keys related to data:
```yaml
data:
  processed_data_folder: "path to folder with data"  # assumes folder has two folders: gan_train and gan_val, each one containing three folders: images, lungs and labels (or at least images and lungs if you are only interested in generating CT images)
  dataset: "lungCTData" # if one wants to generate only CT images or "processedCTData" if one also wants to generate airway segmentation (or other, but must be  declared in constants.py)
  start_point_train_data: integer 
  end_point_train_data: integer
  start_point_validation_data: integer
  end_point_validation_data: integer
```
If one wants to use all elements in images, lungs and labels for training, just delete these four last keys.  
If one wants to use only part of the data, the indexes will indicate how many .npy files to consider: files are sorted in gan_train/images in alphabethical order and only consider files from start_point_train_data to end_point_train_data (not inclusive). The same goes for other folders and for validation data.

Keys related to training:
```yaml
training:
  batch_size_train: integer
  batch_size_validation: integer
  n_epochs: integer
  steps_to_complete_bfr_upd_disc: integer # the amount of steps the GENERATOR takes BEFORE the disc gets updated (default: 1)
  steps_to_complete_bfr_upd_gen: integer # the amount of steps the DISCRIMINATOR takes BEFORE the gen gets updated (default: 1)
  transformations: 
    transform: "AddGaussianNoise" # or "AddUniformNoise" if one wants to add noise to masks (generator input), if not, just delete the related keys
    info:
      lung_area: True/False # if True only adds noise to the lung area, otherwise adds noise to the entire image (default: False)
      intensity: float between 0 and 1 # modulate noise intensity (default: 1)
      mean: float # only used for Gaussian noise, can be removed if uniform 
      std: float  # only used for Gaussian noise, can be removed if uniform

loss: 
  criterion: 
    name: "MSELoss" # or "BCELoss" (or other, but must be  declared in constants.py)
  regularizer:
    type:
      - "MAE" # or "MSE"/"MAE_mask"/"MSE_mask"/"MAE_outside_mask"/"MSE_outside_mask" (or other, but must be  defined in losses.py) - a list can be made if more than one regularization must be applied
    regularization:
      - integer # level of regularization - a list can be made if more than one regularization must be applied
    
optimizer:
  type: "Adam" #or "SGD" (or other, but must be  declared in constants.py)
  lr: float # initial learning rate
  info:
    (kwargs of optimizer)
  path_to_saved_gen_optimizer: "path to generator optimizer to be restored if one wants to resume to a prior training session" # or "" if new model
  path_to_saved_disc_optimizer: "path to discriminator optimizer to be restored if one wants to resume to a prior training session" # or "" if new model

lr_scheduler:
  activate: True/False # if False don't consider learning rate scheduler
  scheduler_type: "LinearLR" # or "StepLR" (or others, check lr_scheduler.py)
  epoch_to_switch_to_lr_scheduler: integer # epoch to start considering lr scheduler
  info:
    (kwargs of lr scheduler)
  path_to_saved_gen_scheduler: "path to generator lr scheduler to be restored if one wants to resume to a prior training session" # or "" if new model
  path_to_saved_disc_scheduler: "path to discriminator lr scheduler to be restored if one wants to resume to a prior training session" # or "" if new model

save_models_and_results:
  step_to_safe_save_models: integer # save models/optimizer/lr schedulers every step_to_safe_save_models epochs in case something unexpected interrupts training
  save_best_model: True/False # if True saves model that gave the smallest generator's loss for validation data

wandb:
  activate: True/False # if True, integrates with wandb to save versions and losses
```

2. Run in terminal:

```
py training_pipeline.py
```

3. You will be prompted to give the path to YAML file:

```
 config.yaml
```

`4.` This execution returns a folder with the name of your model. Inside this directory, there will be two files showing the evolution of the *losses* (an image and a CSV file with the values) and two directories: *examples*, containing some images generated during the model's training, and *models*, containing the trained models (trained and best).

**Training segmentation network (similar to U-Net):**

1. Training configurations should be described in a YAML file composed by the following keys:

Keys related to model definition:
```yaml

model:
  name_model: "string with model name"
  dir_save_results: "path/name_of_folder_to_be_created/"  # default: ./name_model/
  new_model: True/False # if False, will try to resume a prior training session and restore model and optimizer (default: True)
  fine_tunning: True/False # if True, will fine tune a given model (if that's the case new_model should be True) (default: True)
  freeze_layers: True/False # if True: freeze encoder layers, so only decoder parameters get updated (default: True)
  path_to_saved_model: "path to U-Net model to be restored if one wants to resume to a prior training session or if one wants to fine tune" # or "" if new model

data:
  processed_data_folder: "path to folder with data"  # assumes folder has two folders: seg_train and seg_val, each one containing three folders: images, lungs and labels (or at least images and labels)
  dataset: "processedCTData" # or other if declared in constants.py and datasets.py
  start_point_train_data: integer
  end_point_train_data: integer
  start_point_validation_data: integer
  end_point_validation_data: integer
```
If one wants to use all elements in images, lungs and labels for training, just delete these four last keys.  
If one wants to use only part of the data, the indexes will indicate how many .npy files to consider: files are sorted in gan_train/images in alphabethical order and only consider files from start_point_train_data to end_point_train_data (not inclusive). The same goes for other folders and for validation data.

```yaml
training:
  batch_size_train: integer
  batch_size_validation: integer
  n_epochs: integer
  early_stopping : True/False # if True implements early stopping startegy (monitors validation loss, if it doesn't decrease 'delta' in 'patience' epochs, stops training
  patience : integer
  delta : float

loss: 
  criterion: 
    name: "DiceLoss" # or other if declared in constants.py

optimizer:
  type: "Adam" #or "SGD" (or other, but must be  declared in constants.py)
  lr: float # initial learning rate
  info:
    (kwargs of optimizer)
  path_to_saved_gen_optimizer: "path to optimizer to be restored if one wants to resume to a prior training session" # or "" if new model

save_models_and_results:
  step_to_safe_save_models: integer # save models/optimizer every step_to_safe_save_models epochs in case something unexpected interrupts training
  save_best_model: True/False # if True saves model that gave the smallest loss for validation data
```

2. Run in terminal:

```
py segmentation_pipeline.py
```

3. You will be prompted to give the path to YAML file:

```
 config_segm.yaml
```

`4.` This execution returns a folder with the name of your model. Inside this directory, there will be two files showing the evolution of the *losses* (an image and a CSV file with the values) and two directories: *examples*, containing some images generated during the model's training, and *models*, containing the trained models (trained and best).

**Evaluating GAN (only synthesis of CT images) or Segmentation network:**

`1.` Evaluation configurations should be described in a YAML file composed by the following keys:

```yaml
model:
  name_model: "string with model name"
  dir_trained_model: "path for the folder with model"  # assumes this folder has an inner folder 'models' with file 'name_model_gen_trained.pt', 'name_model_gen_best.pt', 'name_model_unet_trained.pt' or 'name_model_unet_best.pt' (default: ./name_model/)
  use_best_version: True/False # if True, selects '_best.pt', else '_trained.pt'

data:
  processed_data_folder: "path to folder with data"  # assumes folder has 'all_test' folder three folders: images, lungs and labels (or at least images and lungs for GAN or images and labels for U-Net)
  dataset: "lungCTData" # if evaluating GAN or "processedCTData" if evaluating U-Net (or other, but must be  declared in constants.py)
  batch_size: integer
  transformations: # ONLY RELEVANT TO GAN (ignored for U-Net)
    transform: "AddGaussianNoise" # or "AddUniformNoise" if one wants to add noise to masks (generator input), if not, just delete the related keys
    info:
      lung_area: True/False # if True only adds noise to the lung area, otherwise adds noise to the entire image (default: False)
      intensity: float between 0 and 1 # modulate noise intensity (default: 1)
      mean: float # only used for Gaussian noise, can be removed if uniform 
      std: float  # only used for Gaussian noise, can be removed if uniform

evaluation:
  bQualitativa: True/False # if True: generates output samples from test data
  bFID: True/False # if True: calculates FID for test data (only for GAN)
  bSSIM: True/False # if True: calculates SSIM for test data (only for GAN)
  bDice: True/False # if True: calculates DICE for test data (only for U-Net) - if True assumes we are evaluating an U-Net, else a GAN 
```

2. Run in terminal:

```
py evaluate.py
```

3. You will be prompted to give the path to YAML file:

```
 config_eval.yaml
```

`4.` This execution returns a folder *generated_images* within the model folder with samples of model's outputs (either segmented images if U-Net or synthetic CT images if GAN). Also in this directory, one will also find a JSON file with the metrics evaluated on the test data. 

This code, currently, does not support evaluating the GAN version that generates both the CT images and airway segmentation.


