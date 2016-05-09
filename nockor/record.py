
import sys
import time
import getopt
import alsaaudio

import numpy as np

import configargparse as cap

p = cap.ArgParser(default_config_files=['/etc/nockor.conf', '~/.nockor.conf'])

p.add('--channels', default=1)
p.add('--rate', default=44100)
#p.add('--format', default=1)

p.parse_known_args()
print(p.format_values())

if __name__ == '__main__':

    card = 'default'

    opts, args = getopt.getopt(sys.argv[1:], 'c:')
#    for o, a in opts:
#        if o == '-c':
#            card = a


    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK, card)

    # Set attributes: Mono, 44100 Hz, 16 bit little endian samples
    inp.setchannels(1)
    inp.setrate(44100)
    inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)

    # The period size controls the internal number of frames per period.
    # The significance of this parameter is documented in the ALSA api.
    # For our purposes, it is suficcient to know that reads from the device
    # will return this many frames. Each frame being 2 bytes long.
    # This means that the reads below will return either 320 bytes of data
    # or 0 bytes of data. The latter is possible because we are in nonblocking
    # mode.
    inp.setperiodsize(160)
    
    loops = 1
    while loops > 0:
        loops -= 1
        # Read data from device
        l, data = inp.read()
      
        if l:
            buf = np.frombuffer(data, dtype='int16')
            print(buf)

