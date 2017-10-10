import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import PIL.Image as Image
import os

""" short time fourier transform of audio signal """

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    b = int(np.floor(frameSize / 2.0))
    #c= 512
    a = np.zeros(b)
    samples = np.append(a, sig)
    # cols for windowing
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """


def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins)  # ** factor

    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    f = lambda x: x * alpha if x <= f0 else (fmax - alpha * f0) / (fmax - f0) * (x - f0) + alpha * f0
    scale = map(f, scale)
    scale = np.fromiter(scale, dtype=np.float64)
    #scale = np.array(scale)
    scale *= (freqbins - 1) / max(scale)

    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = [0.0 for i in range(freqbins)]
    totw = [0.0 for i in range(freqbins)]
    for i in range(0, freqbins):
        if (i < 1 or i + 1 >= freqbins):
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            # scale[15] = 17.2
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))

            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down

            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up

    for i in range(len(freqs)):
        if (totw[i] > 1e-6):
            freqs[i] /= totw[i]

    return newspec, freqs


""" plot spectrogram"""


def plotstft(audiopath, binsize=2 ** 10, plotpath=None, colormap="gray", channel=0, name='tmp.png', alpha=1, offset=0):
    samplerate, samples = wav.read(audiopath)
    samples = samples[:, channel]
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=alpha)
    sshow = sshow[2:, :]
    v = np.abs(sshow)
    ims = 20. * np.log10( v / 10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    ims = np.transpose(ims)
    # ims = ims[0:256, offset:offset+768] # 0-11khz, ~9s interval
    ims = ims[0:256, :]  # 0-11khz, ~10s interval
    # print "ims.shape", ims.shape

    image = Image.fromarray(ims)
    image = image.convert('L')
    image.save(name)



from os import listdir
from os.path import isfile, join
#mypath = "/media/alissonsales/Files/base_dados/pt_05/temp/"
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#file = open('trainingData.csv', 'r')
#for iter, line in enumerate(
#        file.readlines()[1:]):  # first line of traininData.csv is header (only for trainingData.csv)
#    filepath = line.split(',')[0]
#    filename = filepath[:-4]


#for file in onlyfiles:
#wavfile = 'tmp2.wav'
#os.system('mpg123 -w ' + '/home/alissonsales/PycharmProjects/speech2text/'+ wavfile
#              + ' /home/alissonsales/PycharmProjects/speech2text/1_joao0_0.wav')# + filepath
"""
for augmentIdx in range(0, 20):
    alpha = np.random.uniform(0.9, 1.1)
    offset = np.random.randint(90)
    plotstft(wavfile, channel=0, name='/home/brainstorm/data/language/train/pngaugm/'
    +filename+'.'+str(augmentIdx)+'.png',
             alpha=alpha, offset=offset)
"""
    # we create only one spectrogram for each speach sample
    # we don't do vocal tract length perturbation (alpha=1.0)
    # also we don't crop 9s part from the speech
#    filename = file.split(".")[0]
    #wavfile = '00001_en.wav'
plotstft('/home/alissonsales/PycharmProjects/speech2text/joao.wav', channel=0,
         name='/home/alissonsales/PycharmProjects/speech2text/1_joao0_0.png', alpha=1.0)
os.remove('/media/alissonsales/Files/base_dados/pt_05/temp/'+wavfile)
    #print("processed: ", file)
