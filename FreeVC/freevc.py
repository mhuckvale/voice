# FreeVC interface Class
#

import os
from types import SimpleNamespace
import torch
import librosa
from scipy.io.wavfile import write
import numpy as np

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder

import logging
logging.getLogger('numba').setLevel(logging.ERROR)
logging.getLogger('wavlm.WavLM').setLevel(logging.ERROR)
logging.getLogger('numba.core.byteflow').setLevel(logging.ERROR)

#
class FreeVC:
	def __init__(self):
		self.hps = None
		self.net_g = None
		self.cmodel = None
		self.smodel = None

	def load(self,args):
		self.outdir = args.outdir
		self.hps = utils.get_hparams_from_file(args.hpfile)
		print("Loading model...")
		self.net_g = SynthesizerTrn(self.hps.data.filter_length // 2 + 1,self.hps.train.segment_size // self.hps.data.hop_length,**self.hps.model).cuda()
		_ = self.net_g.eval()
		print("Loading checkpoint...")
		_ = utils.load_checkpoint(args.ptfile, self.net_g, None, True)
		print("Loading WavLM for content...")
		self.cmodel = utils.get_cmodel(0)
		print("Loading speaker encoder...")
		self.smodel = SpeakerEncoder(args.spfile)

	def embedding(self,tgtfile):
		with torch.no_grad():
			# target
			wav_tgt, _ = librosa.load(tgtfile, sr=self.hps.data.sampling_rate)
			wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
			g_tgt = self.smodel.embed_utterance(wav_tgt)
			return(g_tgt)

	def convert(self,srcfile,embedding,outfile):
		with torch.no_grad():
			# target
			#embedding=np.array(embedding,dtype=float)
			g_tgt = torch.from_numpy(embedding).unsqueeze(0).cuda()
			# source
			wav_src, _ = librosa.load(srcfile, sr=self.hps.data.sampling_rate)
			wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
			c = utils.get_content(self.cmodel, wav_src)
			audio = self.net_g.infer(c, g=g_tgt)
			audio = audio[0][0].data.cpu().float().numpy()
			write(os.path.join(self.outdir, outfile), self.hps.data.sampling_rate, audio)

	def convert_audio(self,srcfile,tgtfile,outfile):
		with torch.no_grad():
			# target
			wav_tgt, _ = librosa.load(tgtfile, sr=self.hps.data.sampling_rate)
			wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
			g_tgt = self.smodel.embed_utterance(wav_tgt)
			self.convert(srcfile,g_tgt,outfile)
