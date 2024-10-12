# TO-DO

* Training
    * Report all losses individually.
        * Bundle losses in a single variable => @dataclass
    * Add validation to end of every epoch.

* Output
    * Test sending data to WandB.
    * Rewrite plot_losses and save_losses
    * Add option to build TensorBoard results.
    * Plot examples every epoch
        * Keep same images in all examples?
    * Keep track of GPU memory with Tensorboard.

* Networks
    * Add LoRA + skip connection options.

* Data-base
    * Remove bad examples:
        * Tunnels (manually?)
        * Almost only black (narrow color histogram?)
    * Build haze data-base:
        * Data augmentation: crop + horizontal flip
        * Focus on I-Hazy first.
