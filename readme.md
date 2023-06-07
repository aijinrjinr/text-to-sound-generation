# CS230 Final Project: Text-to-sound Generation Using Diffusion Mode
Our code is mainly developed based on **TANGO** (https://github.com/declare-lab/tango) and **Diffsound** (https://github.com/yangdongchao/Text-to-sound-Synthesis).

To train **TANGO** with FLAN-T5 on the AudioCaps dataset using

```bash
python train.py \
--text_encoder_name="google/flan-t5-large" \
--scheduler_name="stabilityai/stable-diffusion-2-1" \
--unet_model_config="configs/diffusion_model_config.json" \
--freeze_text_encoder --augment --snr_gamma 5 \
```

To train **TANGO** with CLIP on the AudioCaps dataset using

```bash
python train.py \
--text_encoder_name="stabilityai/stable-diffusion-2-1" \
--scheduler_name="stabilityai/stable-diffusion-2-1" \
--unet_model_config="configs/diffusion_model_config.json" \
--freeze_text_encoder --augment --snr_gamma 5 \
```

To run 
