from Routine import Routine
from CenterNet import MyUNet


def main(settings):
    Main = Routine(settings, MyUNet)
    return Main.run()

if __name__ == '__main__':
    import Settings as settings
    main(settings)