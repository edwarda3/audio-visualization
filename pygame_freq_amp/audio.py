import pyaudio
import wave
import sys
from pydub import AudioSegment
from scipy.fftpack import fft
import numpy
import time
import pygame
from tqdm import trange

W = 1024
H = 512

class Audio:
    # This acts as a main function. We set up "global" variables (class instance) and start functions.
    # status call is a function callback that gets two parameters, the currenttime, and total time.
    # file is the name of the file in mp3 format.
    def __init__(self,file,statuscall):
        # Convert it to a wave file so we can manipulate it easily.
        self.tmpfile = self.convert(file)
        wf = wave.open(self.tmpfile,'rb')

        # Preprocess the FFT data at each chunk. This gives us slices of frequency data, which allows us to do the visualization.
        pa = pyaudio.PyAudio()
        self.duration = int(wf.getnframes() / wf.getframerate())
        self.time=0 #The time of the file.
        self.lastDisplayedTime=0 #This determines when to call the statuscall function, when this differs from the time.
        self.firstStamp=-1 #The first timestamp, which the file time is relative to.
        self.statusCallback = statuscall #Set the callback as a class variable
        self.currentindex = 0 #Keeps track of which chunk we are on for the fft data.

        self.preprocessFFT(wf)
        self.preprocessPoints(self.freq_data,self.amp_data)

        self.init_pygame()
        self.start_audio_stream(wf,pa)
        self.loop()   

    # Start up pygame. Boilerplate, mostly.
    def init_pygame(self):
        pygame.init()
        pygame.font.init()
        myfont = pygame.font.SysFont('Ariel', 20)
        self.screen = pygame.display.set_mode((W,H))

        self.screen.fill((220,220,220))

    # Start the audiostream. We use a callback style so that we dont block the main thread.
    def start_audio_stream(self,wavefile,pyaud):
        lastChunkSize = 0 # This will remain constant, we just need to update it once if we have a mismatch.
        def callback(in_data,frame_count,time_info,status):
            data = wavefile.readframes(frame_count) # Read the latest chunk of the audio we are playing.

            # Notify if there is a chunk size mismatch. We will update lastChunkSize so it only allows 1 print.
            if(len(data) != self.chunk_size and lastChunkSize != len(data)):
                print("Chunksize mismatch! Frequency data is probably offset.\nSource chunk: {}, Preprocessed Chunk size: {}".format(len(data),self.chunk_size),file=sys.stderr)
                lastChunkSize = len(data)

            self.time = int(time_info['current_time']) #Update current time based on wave file position.
            if(self.firstStamp == -1): #Get our baseline timestamp which makes the time relative to it.
                self.firstStamp = self.time

            # Update the current index of the fft data. Since the preprocessed data is not technically connected to this, we just update this and the main thread will go to this updated index.
            self.currentindex+=1 
            return (data,pyaudio.paContinue)

        stream = pyaud.open(format=pyaud.get_format_from_width(wavefile.getsampwidth()), channels=wavefile.getnchannels(), rate=wavefile.getframerate(), output=True, stream_callback=callback)
        stream.start_stream()

    # Main thread loop. This runs the pygame thread, and NOT the audio thread.
    def loop(self):
        while(True):
            # reprint the time when it changes
            if(self.time != self.lastDisplayedTime):
                realTime = self.time-self.firstStamp #Get the relative time
                self.lastDisplayedTime=self.time
                self.statusCallback(realTime,self.duration)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit() 
                    sys.exit()
            #Make the frequency data into a point list so it can be drawn
            if(-1 < self.currentindex < self.w_chunks_count):
                self.screen.fill((220,220,220))
                for rect in self.pointList[self.currentindex]:
                    pygame.draw.rect(self.screen,(100,100,100),rect)
            else:
                pygame.draw.lines(self.screen,(0,0,0),False,[(0,0),(W,0)])
            pygame.display.update()
    
    # Simply converts from mp3 to wav
    def convert(self,input):
        f = input+'.wav'
        sound = AudioSegment.from_mp3(input)
        sound.export(f,format='wav')
        return f

    # Converts a list of frequency data points into 2d points for pygame.
    # Change this function if a different visualization is desired.
    # Writes points into self.pointList as a 2-d array.
    def preprocessPoints(self,freqdata,ampdata):
        print('Preprocessing Points...',file=sys.stderr)
        # Given a number (original frequency value generated from FFT), map it to fit the screen.
        def bound_and_scale(fnums,anums):
                favg = sum([abs(n) for n in fnums])/len(fnums)
                aavg = sum([abs(n) for n in anums])/len(anums)
                f_approx_normalized = favg/70 #this can get way over 1, but most values should stay between 0-1
                scale_to_screen = H * (f_approx_normalized * aavg)
                bounded_amplitude = max(min(scale_to_screen,H),0)
                return int(bounded_amplitude)
        self.pointList = []

        # Because we can't fit all points in the chunk to the screeb (unless big minitor), we reduce the output of the array. This is relative to the width of the window.
        reduction_factor = (self.chunk_size//2) / W
        rect_width = 15
        width = int(reduction_factor * rect_width)
        # print("rf: {}, rw: {}, w: {}".format(reduction_factor,rect_width,width))

        for i in trange(freqdata.shape[0]):
            fdata = freqdata[i,:]
            adata = ampdata[i,:]

            # Every value is paired with an index (x-pos <-> freq Hz), and y-pos moved appropriately We average all values between idx and idx+1.
            # Explanation of the following list comprehension:
            #   objective:  Get a set of rectangles with each rectangle representing the frequency and amplitude at each frequency band
            #               note: Because our frequency band is longer than the screen size, and we don't want 1 pixel per freq, we reduce the
            #                     representation by taking an average.
            #               Our reduction_factor variable gives us how many elements in the data array we need to reduce into 1 value such that
            #                     1 value is proportional to 1 pixel.
            #               Our rect_width variable gives us how many pixels we want per single represented value. This is mostly personal preference.
            #               From these, we get the width variable, which is how many source elements will go into 1 represented element.
            #   What we do: We need to get the set of values that will reduce to one value, which will represent the height of our rectangle.
            #               Iterate through the data array for the number of total elements we will represent (len(fdata)//width)
            #               At every iteration, we create a 4-tuple corresponding to a pygame Rect: (left, top, width, height)
            #       left:   This will be the index * rect width, to get the x position for the screen.
            #       top:    The sub-arrays fdata[idx*width:(idx+1)*width] and adata[idx*width:(idx+1)*width] gives us width-number of elements 
            #               starting at idx. bound_and_scale() is a processing function that transforms this into a single bounded value for display. 
            #               We then subtract this from H to get the top positon, since the bars will go from bottom up.
            #       width:  rect_width
            #       height: The bounded, transformed value, which will make this rectangle end at H, the bottom of the screen.
            chunkpoints = [(idx*rect_width, H - bound_and_scale(fdata[idx*width:(idx+1)*width],adata[idx*width:(idx+1)*width]), rect_width-4, bound_and_scale(fdata[idx*width:(idx+1)*width],adata[idx*width:(idx+1)*width])) for idx in range(len(fdata)//width)]
            self.pointList.append(chunkpoints)


    # Preprocess the data and get a new set of arrays. Each row is a single chunk, and the data in each row is the frequency data for that chunk. Columns are representative of Hz.
    # Writes into the class instance variable self.freq_data, which is a 2-d numpy array.
    def preprocessFFT(self,wf):
        print('Preprocessing FFT...',file=sys.stderr)
        w_data_len = wf.getnframes()
        w_data = wf.readframes(w_data_len)
        wf.setpos(0) #Reset of reading position so it can be read when playing.
        self.chunk_size = 4096 #Assumption, because a good way to read the chunk size is not in the wave library (afaik).
        self.w_chunks_count = int(len(w_data)/self.chunk_size)
        # Preallocate memory to make this faster. We allocate #chunks rows, and half the chunksize for each chunk. This is because FFT data is symmetric, so we only need the first half
        self.freq_data = numpy.zeros((self.w_chunks_count,self.chunk_size//2)) 
        self.amp_data = numpy.zeros((self.w_chunks_count,self.chunk_size//2)) 
        #Go through each chunk and preform FFT.
        for i in trange(self.w_chunks_count):
            chunk_start_index = i     * self.chunk_size
            chunk_stop_index =  (i+1) * self.chunk_size
            # Since the amplitude data is from 0-255, we normalize it between 0-1
            chunk_data = [int(d)/255 for d in w_data[chunk_start_index : chunk_stop_index]]
            #fft_out = [f.real + f.imag for f in fft(chunk_data)]
            fft_out = [f.real for f in fft(chunk_data)]
            self.freq_data[i,:] = fft_out[:len(fft_out)//2]
            self.amp_data[i,:] = [chunk_data[r*2]+chunk_data[r*2+1] for r in range(len(chunk_data)//2)]

# Prints timing of the audio file. The times are in seconds
def printer(curtime,totaltime):
    curmin = curtime//60
    cursec = curtime%60
    totalmin = totaltime//60
    totalsec = totaltime%60
    current = "{}:{:02d}".format(curmin,cursec)
    total = "{}:{:02d}".format(totalmin,totalsec)
    if(curtime>0):
        print("{}/{}".format(current,total),end='\r',flush=True)

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('file',help="name of file use")
    args = parser.parse_args()

    try:
        a = Audio(args.file,printer)
    except KeyboardInterrupt:
        os.remove("{}.wav".format(args.file))
    os.remove("{}.wav".format(args.file))