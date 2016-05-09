#! /usr/bin/env python

import sys
import os
import time
import pickle
import alsaaudio
import numpy as np
import configargparse as cap
import features

class Recognizer(object):
    def __init__(self, model_dir, rate=16000, alpha=0.01):
        self.extractor = features.Extractor(rate=rate)
        self.alpha = alpha
        self.prop = 0.0
        self.models = []
        self.mnames = []
        for entry in os.listdir(model_dir):
            if entry.startswith('.'): continue
            path = os.path.join(model_dir, entry)
            mname = os.path.splitext(entry)[0]
            with open(path, 'rb') as f:
                model = pickle.load(f)
            self.mnames.append(mname)
            self.models.append(model)
        self.logprob = np.zeros(len(self.mnames))

    def pump(self, data):
        alpha = self.alpha
        logprob = self.logprob
        prop = self.prop
        X = self.extractor.pump(data)
        if len(X) == 0:
            self.prop = (1- alpha) * prop
            self.logprob = (1 - alpha) * logprob
            return (None, 0.0, self.prop)
        logprobs = np.array([m.score_samples(X)[0] for m in self.models]).T
        for logprob1 in logprobs:
            prop = (1 - alpha) * prop + alpha
            logprob = (1 - alpha) * logprob + alpha * logprob1
        self.logprob = logprob
        self.prop = prop
        i = np.argmax(logprob)
        prob = np.exp(logprob)
        prob /= np.sum(prob)
        if prop < 0.3 or prob[i] < 0.95:
            return (None, 0.0, prop)
        return (self.mnames[i], prob[i], self.prop)

def main(argv=sys.argv):
    rate = 16000

    p = cap.ArgParser(default_config_files=['/etc/nockor.conf', '~/.nockor.conf'])
    p.add('--channels', default=1)
    p.add('--rate', default=rate)
    p.parse_known_args()
    print(p.format_values())

    ain = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL)
    ain.setchannels(1)
    ain.setrate(rate)
    ain.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    ain.setperiodsize(int(rate * 0.01))

    recog = Recognizer(argv[1], rate=rate)

    mname = None
    while True:
        _, data = ain.read()
        data = np.frombuffer(data, dtype='int16')
        mname, prob, prop = recog.pump(data)
        if mname is None: continue
        print((time.time(), mname, prob, prop))

    return 0

if __name__ == '__main__':
    sys.exit(main())
