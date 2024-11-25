## How To Run

As an additional note, we have included a description of how to perform the functions proposed in this project.

**Data processing:**

`1.` Download ATM'22 database from internet.

`2.` Read and process data using class `rawCTData` in datasets.py and process_raw_data.ipynb


**Training GAN:**

1. Training configurations should be described in a YAML file composed by the following keys:

model:
  name_model: "string with model name"
  dir_save_results: "path/name_of_folder_to_be_created/"  # if not provided: ./name_model/
  new_model: True/False  # if False, will try to resume a prior training session and restore models, optimizers, and lr_schedulers (default: True)
  use_pretrained_model: True/False  # if True, will fine-tune a given model (if that's the case new_model should be True) (default: False)
  generate_airway_segmentation: True/False  # if False: PulmoNet generates only CT images, if True: tries to generate both CT images and airway segmentation (default: False)
  path_to_saved_model_gen: "path to generator model to be restored if you want to resume to a prior training session or you want to fine-tune"  # or "" if new model
  path_to_saved_model_disc: "path to discriminator model to be restored if you want to resume to a prior training session or you want to fine-tune"  # or "" if new model


yaml
model:
  name_model: "string with model name"
  dir_save_results: "path/name_of_folder_to_be_created/"  # default: ./name_model/
  new_model: True/False # if False, will try to resume a prior training session and restore models,  optimizers and lr_schedulers (default: True)
  use_pretrained_model: True/False # if True, will fine tune a given model (if that's the case new_model should be True) (default: False)
  generate_airway_segmentation: True/False # if False: PulmoNet generates only CT images, if True: tries to genrate both CT images and airway segmentation (default: False)
  path_to_saved_model_gen: "path to generator model to be restored if you want to resume to a prior training session or you want to fine tune" # or "" if new model
  path_to_saved_model_disc: "path to discriminator model to be restored if you want to resume to a prior training session or you want to fine tune" # or "" if new model

data:
  processed_data_folder: "path to folder with data (assumes folder has two folders: gan_train and gan_val, each one containing three folders: images, lungs and labels (or at least images and lungs if you are only interested in generating CT images))"
  dataset: "lungCTData" if you want to generate only CT images or "processedCTData" if you also wants to generate airway segmentation (or other, but must be  declared in constants.py)
  start_point_train_data: 0 (if you want to use all elements in images, lungs and labels for training, just delete these next keys, if you want to use only part of the data, the index in here will indicate how many .npy files to consider - we will sort data in gan_train/images in alphabethical order and only consider files from start_point_train_data to end_point_train_data (not inclusive) - the same goes for other folders and for validation data) 
  end_point_train_data: 20000
  start_point_validation_data: 0
  end_point_validation_data: 800

training:
  batch_size_train: integer
  batch_size_validation: integer
  n_epochs: integer
  steps_to_complete_bfr_upd_disc: the amount of steps the GENERATOR takes BEFORE the disc gets updated (default: 1)
  steps_to_complete_bfr_upd_gen: the amount of steps the DISCRIMINATOR takes BEFORE the gen gets updated (default: 1)
  transformations: 
    transform: "AddGaussianNoise"/"AddUniformNoise" - if one wants to add noise to masks (generator input), if not, just delete the related keys
    info:
      lung_area: True/False - if True only adds noise to the lung area, otherwise adds noise to the entire image (default: False)
      intensity: float between 0 and 1 - modulate noise intensity (default: 1)
      mean: float - only used for Gaussian noise, can be removed if uniform 
      std: float - only used for Gaussian noise, can be removed if uniform

loss: 
  criterion: 
    name: "MSELoss"/"BCELoss" (or other, but must be  declared in constants.py)
  regularizer:
    type:
      - "MAE"/"MSE"/"MAE_mask"/"MSE_mask"/"MAE_outside_mask"/"MSE_outside_mask" (or other, but must be  defined in losses.py) - can be made a list if more than one must be applied
    regularization:
      - integer (level of regularization) - can be made a list if more than one must be applied
    
optimizer:
  type: "Adam"/"SGD" (or other, but must be  declared in constants.py)
  lr: float (initial learning rate)
  info:
    (kwargs of optimizer)
  path_to_saved_gen_optimizer: "path to generator optimizer to be restored if you want to resume to a prior training session" or "" if new model
  path_to_saved_disc_optimizer: "path to discriminator optimizer to be restored if you want to resume to a prior training session" or "" if new model

lr_scheduler:
  activate: True/False - if False don't consider learning rate
  scheduler_type: "LinearLR"/"StepLR" (or others, check lr_scheduler.py)
  epoch_to_switch_to_lr_scheduler: integer (epoch to start considering lr scheduler)
  info:
    (kwargs of lr scheduler)
  path_to_saved_gen_scheduler: "path to generator lr scheduler to be restored if you want to resume to a prior training session" or "" if new model
  path_to_saved_disc_scheduler: "path to discriminator lr scheduler to be restored if you want to resume to a prior training session" or "" if new model

save_models_and_results:
  step_to_safe_save_models: integer (save models/optimizer/lr schedulers every step_to_safe_save_models epochs in case something unexpected interrupts training)
  save_best_model: True/False (if True saves model that gave the smallest generator's loss for validation data) 

wandb:
  activate: True/False (if True, integrates with wandb to save versions and losses)

2. Executar comando em seu terminal:

```
py training_pipeline.py
```

3. Selecionar o arquivo YAML de configuração desejado:

```
 config.yaml
```

**Obtenção das métricas da GAN:**

`1.` Configurar parâmetros do modelo no arquivo `config_eval.yaml` e a localização da pasta com os dados processados.

`2.` Executar comando em seu terminal:

```
test_pipeline.py config_eval.yaml
```

**Treinamento da rede de segmentação:**

`1.` Configurar parâmetros do modelo no arquivo `config_segmentation.yaml` e a localização da pasta com os dados processados.
- Caso queira continuar com um treinamento, deve-se atribuir `False` para o parâmetro `new_model`. Caso seja a primeira vez que está realizando este treinamento, deve atribuir `True` para este parâmetro.
- Caso queira treinar um modelo de segmentação a partir do zero deve-se atribuir `False` para o parâmetro `fine_tunning`.
- Caso queira reaproveitar os pesos iniciais de um gerador pré-treinado e queira treinar todas as camadas do modelo (isto é, não congelar os pesos do codificador), deve-se atribuir `False` para o parâmetro `freeze_layers`.
- Na seção de configuração `data` é possível selecionar a quantidade de dados de treinamento e validação que deseja utilizar.
- Deve-se direcionar o parâmetro `processed_data_folder` para o diretório de dados do projeto. Dentro deste diretório, deverá ter duas pastas (`seg_train` e `seg_val`), contendo os dados de treinamento e validação para a tarefa de segmentação.
- Não mexer no parâmetro `dataset` deste arquivo.
- É possível ajustar parâmetro de treinamento, critério da função de *loss* e parâmetros do otimizador neste arquivo.
- Também é possível configurar a frequência de "check-points" de salvamentos intermediários do modelo durante o treinamento, com o parâmetro `step_to_safe_save_models`. Esse recurso ajuda a salvar o estado do modelo durante o treinamento, o qual poderá ser recuperado caso este processo seja interrompido.

`2.` Executar comando em seu terminal:

```
py segmentation_pipeline.py
```

`3.` Selecionar o arquivo YAML de configuração:

```
config_segmentation.yaml
```

`4.` Esta execução retorna uma pasta com o nome do seu modelo. Dentro deste diretório, haverá 2 arquivos com a evolução das *losses* (uma imagem e um JSON com os valores) e dois diretórios: *examples*, com algumas imagens geradas durante o treinamento do modelo, e *models*, com os modelos treinados (safesave, trained e best).

**Teste da rede de segmentação:**

`1.` Configurar parâmetros do modelo no arquivo `config_eval_seg.yaml` e a localização da pasta com os dados processados.
- O nome do modelo (parâmetro `name_model`) deve ser o nome da pasta onde está o modelo que deseja avaliar. Este modelo deve estar no diretório *models* dentro desta pasta.
- Caso o parâmetro `use_best_version` seja `True`, será avaliado o modelo `<name_model>_unet_best.pt`. Caso seja `False`, será avaliado o modelo `<name_model>_unet_trained.pt`.
- Direcionar o parâmetro `processed_data_folder` para o diretório de dados do projeto. Dentro deste diretório, deverá ter uma pasta chamada `all_test`, contendo os dados de testes.
- Não alterar os valores em `evaluation` e nem o parâmetro `dataset`.

`2.` Executar comando em seu terminal:

```
py evaluate.py
```

`3.` Selecionar o arquivo YAML de configuração:

```
config_eval_seg.yaml
```

`4.` Esta execução retorna uma pasta *generated_images* dentro da pasta do modelo treinado, com algumas imagens de segmentação geradas com dados do conjunto de testes. Além disso, haverá um arquivo JSON com o resultado da métrica DICE.
