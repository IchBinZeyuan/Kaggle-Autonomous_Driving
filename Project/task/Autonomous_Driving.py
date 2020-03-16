<<<<<<< HEAD
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath( __file__))))
from src.Routine import Routine
from src.CenterNet import MyUNet

=======
from src.Routine import Routine
from src.CenterNet import MyUNet


>>>>>>> b3b4e54ec26befba4553c393a79c43c299f88398
def main(settings):
    Main = Routine(settings, MyUNet)
    return Main.run()

<<<<<<< HEAD

=======
>>>>>>> b3b4e54ec26befba4553c393a79c43c299f88398
if __name__ == '__main__':
    import Settings as settings
    main(settings)