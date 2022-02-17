
from playsound import playsound
import os

dirname = os.path.dirname(os.path.abspath(__file__))
cow=os.path.join(dirname,'audio/mixkit-cow-moo-in-the-barn-1751.wav')

def playcow():
    playsound(cow)