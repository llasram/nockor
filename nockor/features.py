import numpy
import numpy as np
import scipy.fftpack as fftpack

def f_mel(f):
    "Convert Hz to Mels."
    return 1127 * np.log(1 + f / 700)

def f_hz(f):
    "Convert Mels to Hz."
    return 700 * (np.exp(f / 1127) - 1)

def melfilts(M, rate, nfft):
    mels = np.linspace(0, f_mel(rate/2), M+2)
    f = np.floor((nfft + 1) * f_hz(mels) / rate)
    filts = np.zeros((M, int(nfft/2 + 1)))
    for j in range(M):
        m = j + 1
        for k in range(int(f[m - 1]), int(f[m]) + 1):
            filts[j, k] = (k - f[m - 1]) / (f[m] - f[m - 1])
        for k in range(int(f[m]), int(f[m + 1]) + 1):
            filts[j, k] = (f[m + 1] - k) / (f[m + 1] - f[m])
    return filts.T

class Extractor(object):
    def __init__(self, rate=16000, window=0.03, step=0.01, nfft=512, mel_m=26,
                 keep_m=13, delta_t=2, cmn_lambda=0.001, vad_energy=19):
        self.rate = rate
        self.window = int(rate * window)
        self.w = np.hanning(self.window)
        self.step = int(rate * step)
        self.nfft = nfft
        self.melfilts = melfilts(mel_m, rate, nfft)
        self.keep_m = keep_m
        self.delta_t = delta_t
        self.cmn_lambda = cmn_lambda
        self.vad_energy = vad_energy
        self.reset()

    def reset(self):
        self.signal = None
        self.mfeat = np.zeros(self.keep_m)
        self.feat = [None] * (2 * self.delta_t + 1)
        self.delta = [None] * (2 * self.delta_t + 1)

    def pump(self, samples):
        self._append(samples)
        delta_t = self.delta_t
        t = delta_t * 2 + 1
        delta_den = sum(abs(i - delta_t) for i in range(t))
        result = np.zeros((0, 2 * self.keep_m))
        nfft = self.nfft
        cmn_lambda = self.cmn_lambda
        while len(self.signal) >= self.window:
            signal = self.signal[:self.window]
            self.signal = self.signal[self.step:]
            signal = signal * self.w
            power = np.square(np.abs(np.fft.rfft(signal, nfft))) / nfft
            energy = np.log(1 + np.sum(power))
            if energy < self.vad_energy: continue
            feat = np.log(1 + np.dot(power, self.melfilts))
            feat = fftpack.dct(feat)[:self.keep_m]
            feat[0] = energy
            # delta features
            self.feat.append(feat)
            self.feat.pop(0)
            if self.feat[0] is None: continue
            delta = np.zeros(self.keep_m)
            for i in range(t):
                delta += np.sign(i - delta_t) * self.feat[i]
            delta = delta / delta_den
            # delta-delta features
            self.delta.append(delta)
            self.delta.pop(0)
            if self.delta[0] is None: continue
            ddelta = np.zeros(self.keep_m)
            for i in range(t):
                ddelta += np.sign(i - delta_t) * self.delta[i]
            ddelta = delta / delta_den
            # cepstral mean subtraction
            feat = self.feat[delta_t]
            self.mfeat = (1 - cmn_lambda) * self.mfeat + cmn_lambda * feat
            feat = feat - self.mfeat
            feat = np.concatenate((feat, delta))
            result = np.concatenate((result, np.reshape(feat, (1, len(feat)))))
        return result

    def _append(self, samples):
        if self.signal is None:
            self.signal = samples.astype('float')
        else:
            self.signal = np.concatenate((self.signal, samples))
