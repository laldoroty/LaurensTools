from playsound import playsound
import os

dirname = os.path.dirname(os.path.abspath(__file__))
cow=os.path.join(dirname,'audio/mixkit-cow-moo-in-the-barn-1751.wav')

def playcow():
    """
    LNA 20220217
    Plays the sound of a cow mooing. useful for alerting you that something is done running. 
    
    """
    playsound(cow)