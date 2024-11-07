import soundfile
import speech_arena.streaming_helpers as sh
import numpy as np

stream, gem_model = sh.gemini_streaming("hai-gcp-accents-dialects")

t = soundfile.read("-1008642825401516622.wav")

for tok in stream((t[1], t[0])):
    print(tok)
