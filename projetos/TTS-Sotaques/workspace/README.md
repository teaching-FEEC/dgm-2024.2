# TTS-Sotaques

This project uses resources of the [Coqui TTS project](https://github.com/coqui-ai/TTS)

Our workflow was carried on inside a Docker container created from a premade image provided [here](https://docs.coqui.ai/en/latest/docker_images.html)

To replicate our experiments or generate speech samples with one of the fine-tuned models, you can pull the premade image with:

```docker pull ghcr.io/coqui-ai/tts```

And then run the Coqui TTS container with:

```docker run --rm -it -p 5002:5002 -e NVIDIA_VISIBLE_DEVICES=<GPUs you'll use> --runtime=nvidia -v <path/to/this/repo/workspace>:/workspace --name <container_name> --entrypoint /bin/bash ghcr.io/coqui-ai/tts```

Additionally, you'll also need to download the dataset used and the model checkpoints.

The pretrained YourTTS model is available [here](https://github.com/coqui-ai/TTS/releases/download/v0.5.0_models/tts_models--multilingual--multi-dataset--your_tts.zip)