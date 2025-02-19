from flask import Flask, render_template
import sounddevice as sd
from scipy.io.wavfile import write


app = Flask(__name__)

@app.route('/')
def FirstPage():
    return render_template('index.html')

@app.route('/process')
def SecondPage():
    # ai model here that records audio 
    #return page shows the results of the model and if you have alzheimers.
    fs = 44100
    seconds = 10

    recording = sd.rec(int(seconds*fs), samplerate=fs, channels=2)
    sd.wait()
    write('output.wav', fs, recording)
    return 'output.wav'

if __name__=="__main__":
    app.run(port=5002, debug=True)
