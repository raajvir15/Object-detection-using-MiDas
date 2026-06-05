import pyttsx3  #converts text to speech in offline method
import tempfile  #used to create tempoeary file that gets deleted automatically

def text_to_audio_file(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150) ## rate mtlb word per minute
    engine.setProperty('volume', 0.9)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        path = f.name
##save the audio to a .wav file
# runAndWait() stops the file till it is fully written

    engine.save_to_file(text, path)
    engine.runAndWait()
    return path