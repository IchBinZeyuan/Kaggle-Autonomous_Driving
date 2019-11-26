import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath( __file__))))
from src.Routine import Routine
from src.CenterNet import MyUNet

def main(settings):
    Main = Routine(settings, MyUNet)
    return Main.run()


if __name__ == '__main__':
    import Settings as settings
    main(settings)