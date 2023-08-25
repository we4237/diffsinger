import subprocess
import torch
# from inference.tts.fs import FastSpeechInfer
# from modules.tts.fs2_orig import FastSpeech2Orig
from inference.svs.base_svs_infer import BaseSVSInfer
from utils import load_ckpt
from utils.hparams import hparams
from usr.diff.shallow_diffusion_tts import GaussianDiffusion
from usr.diffsinger_task import DIFF_DECODERS
from modules.fastspeech.pe import PitchExtractor
import utils
import numpy as np
import parselmouth

class DiffSingerE2EInfer(BaseSVSInfer):
    def build_model(self):
        model = GaussianDiffusion(
            phone_encoder=self.ph_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model')

        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            self.pe = PitchExtractor().to(self.device)
            utils.load_ckpt(self.pe, hparams['pe_ckpt'], 'model', strict=True)
            self.pe.eval()
        return model

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_id = sample.get('spk_ids')
        with torch.no_grad():
            output = self.model(txt_tokens, spk_id=spk_id, ref_mels=None, infer=True,
                                pitch_midi=sample['pitch_midi'], midi_dur=sample['midi_dur'],
                                is_slur=sample['is_slur'])
            mel_out = output['mel_out']  # [B, T, 80]
            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                f0_pred = self.pe(mel_out)['f0_denorm_pred']  # pe predict from Pred mel
            else:
                f0_pred = output['f0_denorm']
            wav_out = self.run_vocoder(mel_out, f0=f0_pred)


        wav_out = wav_out.cpu().numpy()
        mel = mel_out.cpu()
        f0 = f0_pred.cpu()
        return wav_out[0],mel,f0

if __name__ == '__main__':

    transcription_path = 'data/raw/xiaobing/segments/transcriptions.txt'
    with open(transcription_path, 'r') as file:
        for line in file.readlines():
            input = line.split('|')
            name = input[0]
            text = input[1]
            ph_seq = input[2]
            note_seq = input[3]
            note_dur_seq = input[4]
            is_slur_seq = input[6]
            c = {
                'name':f'{name}',
                'text': f'{text}',
                'ph_seq': f'{ph_seq}',
                'note_seq': f'{note_seq}',
                'note_dur_seq': f'{note_dur_seq}',
                'is_slur_seq': f'{is_slur_seq}',
                'input_type': 'phoneme'
            } 
            DiffSingerE2EInfer.example_run(c)


            







# python inference/svs/ds_e2e.py --config usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml --exp_name 0228_opencpop_ds100_rel