import os
import copy
import json
import time
import torch
import argparse
import soundfile as sf
import wandb
from tqdm import tqdm
from diffusers import DDPMScheduler
from audioldm_eval import EvaluationHelper
from models import build_pretrained_models, AudioDiffusion
from transformers import AutoProcessor, ClapModel
import torchaudio
from tango import Tango
import pandas as pd



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
    parser.add_argument(
        "--original_args", type=str, default=None,
        help="Path for summary jsonl file saved during training."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path for saved model bin file."
    )
    parser.add_argument(
        "--test_file", type=str, default="data/test_audiocaps_subset.json",
        help="json file containing the test prompts for generation."
    )
    parser.add_argument(
        "--text_key", type=str, default="captions",
        help="Key containing the text in the json file."
    )
    parser.add_argument(
        "--test_references", type=str, default="data/audiocaps_test_references/subset",
        help="Folder containing the test reference wav files."
    )
    parser.add_argument(
        "--num_steps", type=int, default=200,
        help="How many denoising steps for generation.",
    )
    parser.add_argument(
        "--guidance", type=float, default=3,
        help="Guidance scale for classifier free guidance."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="How many samples per prompt.",
    )
    parser.add_argument(
        "--num_test_instances", type=int, default=-1,
        help="How many test instances to evaluate.",
    )
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    args.guidance = 3
    args.num_samples = 1
    args.num_steps = 200
    args.batch_size = 6
    args.cut = 0
    args.device = 'cuda:3'
    args.test_references = '/home/qywei/TTSS/audiocaps/test/'
    args.original_args = '/home/qywei/tango/saved/1685936659/summary.jsonl'
    args.model = r'/home/qywei/tango/saved/1685936659/best/pytorch_model_2.bin'
    # inference.py - - = "saved/1681728144/summary.jsonl" \
    #                                 - -model = "saved/1681728144/epoch_39/pytorch_model_2.bin" - -num_steps

    train_args = dotdict(json.loads(open(args.original_args).readlines()[0]))
    if "hf_model" not in train_args:
        train_args["hf_model"] = None
    
    # Load Models #
    if train_args.hf_model:
        tango = Tango(train_args.hf_model, "cpu")
        vae, stft, model = tango.vae.cuda(), tango.stft.cuda(), tango.model.cuda()
    else:
        name = "audioldm-s-full"
        vae, stft = build_pretrained_models(name)
        vae, stft = vae.cuda(args.device), stft.cuda(args.device)
        model = AudioDiffusion(
            train_args.text_encoder_name, train_args.scheduler_name, train_args.unet_model_name, train_args.unet_model_config, train_args.snr_gamma
        ).cuda(args.device)
        model.eval()
    # tango = Tango("declare-lab/tango-full-ft-audiocaps", device=args.device)
    # vae, stft = tango.vae, tango.stft
    # model = tango.model
    # model.eval()
    # Load Trained Weight #
    device = vae.device()
    model.load_state_dict(torch.load(args.model, args.device))

    scheduler = DDPMScheduler.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder="scheduler")
    evaluator = EvaluationHelper(16000, args.device)
    
    if args.num_samples > 1:
        clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
        clap.eval()
        clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    
    # wandb.init(project="Text to Audio Diffusion Evaluation")

    def audio_text_matching(waveforms, text, sample_freq=16000, max_len_in_seconds=10):
        new_freq = 48000
        resampled = []
        
        for wav in waveforms:
            x = torchaudio.functional.resample(torch.tensor(wav, dtype=torch.float).reshape(1, -1), orig_freq=sample_freq, new_freq=new_freq)[0].numpy()
            resampled.append(x[:new_freq*max_len_in_seconds])

        inputs = clap_processor(text=text, audios=resampled, return_tensors="pt", padding=True, sampling_rate=48000)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clap(**inputs)

        logits_per_audio = outputs.logits_per_audio
        ranks = torch.argsort(logits_per_audio.flatten(), descending=True).cpu().numpy()
        return ranks
    
    # Load Data #
    # if train_args.prefix:
    #     prefix = train_args.prefix
    # else:
    # prefix = ""

    # text_prompts = [json.loads(line)[args.text_key] for line in open(args.test_file).readlines()]
    # text_prompts = [prefix + inp for inp in text_prompts]
    text_prompts = list(pd.read_csv('/home/qywei/TTSS/audiocaps/test.csv').values[:, -1])
    text_id = list(pd.read_csv('/home/qywei/TTSS/audiocaps/test.csv').values[:, 0])
    T = len(text_prompts)
    cutT = T // 2
    if args.cut < 1:
        text_prompts = text_prompts[args.cut*cutT:args.cut*cutT + cutT]
        text_id = text_id[args.cut*cutT:args.cut*cutT + cutT]
    else:
        text_prompts = text_prompts[cutT*1:]
        text_id = text_id[cutT*1:]

    if args.num_test_instances != - 1:
        text_prompts = text_prompts[:args.num_test_instances]
    
    # Generate #
    num_steps, guidance, batch_size, num_samples = args.num_steps, args.guidance, args.batch_size, args.num_samples
    all_outputs = []
        
    for k in tqdm(range(0, len(text_prompts), batch_size)):
        text = text_prompts[k: k+batch_size]
        
        with torch.no_grad():
            latents = model.inference(text, scheduler, num_steps, guidance, num_samples, disable_progress=True)
            mel = vae.decode_first_stage(latents)
            wave = vae.decode_to_waveform(mel)
            all_outputs += [item for item in wave]
            
    # Save #
    exp_id = 'Audiocaps_clip_'#str(int(time.time()))
    if not os.path.exists("outputs/test/"):
        os.makedirs("outputs/test/", exist_ok=True)
    
    if num_samples == 1:
        # output_dir = "outputs/test/{}_{}_steps_{}_guidance_{}".format(exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance)
        output_dir = "outputs/test/{}_steps_{}_guidance_{}".format(exp_id, num_steps, guidance)

        os.makedirs(output_dir, exist_ok=True)
        for j, wav in enumerate(all_outputs):
            sf.write("{}/output_{}.wav".format(output_dir, text_id[j]), wav, samplerate=16000)

        # result = evaluator.main(output_dir, args.test_references)
        # result["Steps"] = num_steps
        # result["Guidance Scale"] = guidance
        # result["Test Instances"] = len(text_prompts)
        # wandb.log(result)
        #
        # result["scheduler_config"] = dict(scheduler.config)
        # result["args"] = dict(vars(args))
        # result["output_dir"] = output_dir
        #
        # with open("outputs/test/summary.jsonl", "a") as f:
        #     f.write(json.dumps(result) + "\n\n")
            
    else:
        for i in range(num_samples):
            output_dir = "outputs/test/{}_{}_steps_{}_guidance_{}/rank_{}".format(exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, i+1)
            os.makedirs(output_dir, exist_ok=True)

        groups = list(chunks(all_outputs, num_samples))
        for k in tqdm(range(len(groups))):
            wavs_for_text = groups[k]
            rank = audio_text_matching(wavs_for_text, text_prompts[k])
            ranked_wavs_for_text = [wavs_for_text[r] for r in rank]
            
            for i, wav in enumerate(ranked_wavs_for_text):
                output_dir = "outputs/test/{}_{}_steps_{}_guidance_{}/rank_{}".format(exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, i+1)
                sf.write("{}/output_{}.wav".format(output_dir, k), wav, samplerate=16000)
            
        # Compute results for each rank #
        for i in range(num_samples):
            output_dir = "outputs/{}_{}_steps_{}_guidance_{}/rank_{}".format(exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, i+1)
            result = evaluator.main(output_dir, args.test_references)
            result["Steps"] = num_steps
            result["Guidance Scale"] = guidance
            result["Instances"] = len(text_prompts)
            result["clap_rank"] = i+1
            
            wb_result = copy.deepcopy(result)
            wb_result = {"{}_rank{}".format(k, i+1): v for k, v in wb_result.items()}
            wandb.log(wb_result)
            
            result["scheduler_config"] = dict(scheduler.config)
            result["args"] = dict(vars(args))
            result["output_dir"] = output_dir

            with open("outputs/summary.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n\n")
        
if __name__ == "__main__":
    main()