import wave

wf = wave.open("/home/alissonsales/Downloads/tmp.wav", 'rb')
print(wf.getframerate())