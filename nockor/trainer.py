#! /usr/bin/env python

import sys
import pickle
import numpy as np
import scipy.io.wavfile as wavfile
from sklearn.mixture import GMM
import nockor.features as features

def main(argv=sys.argv):
    inpath, outpath = argv[1:]
    rate, data = wavfile.read(inpath)
    e = features.Extractor(rate=rate)
    X = e.pump(data)
    model = GMM(n_components=128)
    model.fit(X)
    with open(outpath, 'wb') as outf:
        pickle.dump(model, outf)
    return 0

if __name__ == '__main__':
    sys.exit(main())
