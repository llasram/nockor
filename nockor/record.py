#! /usr/bin/env python

import sys
import os
import pickle
import alsaaudio
import numpy as np
import configargparse as cap
import features

class Recognizer(object):
    def __init__(self, model_dir, rate=16000, alpha=0.02):
        self.extractor = features.Extractor(rate=rate)
        self.alpha = alpha
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
        X = self.extractor.pump(data)
        if len(X) == 0: return None
        logprobs = np.array([m.score_samples(X)[0] for m in self.models]).T
        for logprob1 in logprobs:
            logprob = (1 - alpha) * logprob + alpha * logprob1
        self.logprob = logprob
        return self.mnames[np.argmax(logprob)]

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
    ain.setperiodsize(int(rate * 0.03))

    recog = Recognizer(argv[1], rate=rate)

    while True:
        _, data = ain.read()
        data = np.frombuffer(data, dtype='int16')
        mname = recog.pump(data)
        if mname is not None:
            print(mname)

    return 0

if __name__ == '__main__':
    sys.exit(main())
