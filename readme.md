# CS230 Final Project: Text-to-sound Generation Using Diffusion Model
Our code is mainly developed based on **TANGO** (https://github.com/declare-lab/tango) and **Diffsound** (https://github.com/yangdongchao/Text-to-sound-Synthesis).

To train **TANGO** with FLAN-T5 on the AudioCaps dataset, using

```bash
cd tango-master
python train.py \
--text_encoder_name="google/flan-t5-large" \
--scheduler_name="stabilityai/stable-diffusion-2-1" \
--unet_model_config="configs/diffusion_model_config.json" \
--freeze_text_encoder --augment --snr_gamma 5 \
```

To train **TANGO** with CLIP on the AudioCaps dataset, using

```bash
cd tango-master
python train.py \
--text_encoder_name="stabilityai/stable-diffusion-2-1" \
--scheduler_name="stabilityai/stable-diffusion-2-1" \
--unet_model_config="configs/diffusion_model_config.json" \
--freeze_text_encoder --augment --snr_gamma 5 \
```

To run inference on the AudioCaps dataset with **Diffsound**, using
```bash
cd Diffsound
python evaluation/generate_samples_batch.py
```
To run inference on the AudioCaps dataset with **TANGO**, using
```bash
python inference.py \
--original_args="saved/*/summary.jsonl" \
--model="saved/*/best/pytorch_model_2.bin" \
```
