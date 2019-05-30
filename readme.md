#Audio Visualization

---

##Frequency and Amplitude using Pygame

The file `pygame_freq_amp/audio.py` script will use pygame to create audio visualizations based on frequency data and amplitude data.

More specifically, we take in an imput file, and convert it to wave format. We take the resulting amplitude data, read directly from the wave file, and run it through a fourier transform using scipy's fft function. We then make this set of frequency data points into a set of pygame rectangles, described in the code file itself.

The result is a somewhat decent visualization for the audio file. Note that it can be hard to notice and is easiest to see changes with songs with a larger range of frequency fluctuation, in terms of instruments used, tempo, and volume.

Note that a decent CPU is required in order for run smoothly.

---