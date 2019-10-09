""" Generate events using pythia8. """

import argparse

import pythia8

def parse_input():
    """ Parse input arguments. """
    parser = argparse.ArgumentParser(description='Generate nevents for W '
                                     'production for a given mass and width')
    parser.add_argument('--nevents', '-n', type=int, default=1000,
                        help='Number of events to generate')
    parser.add_argument('--mass', '-m', type=float, default=80.,
                        help='Mass of W-boson')
    parser.add_argument('--width', '-w', type=float, default=2.,
                        help='Width of W-boson')
    parser.add_argument('--filename', '-f', type=str, default='events.txt',
                        help='Name of output file')

    return parser.parse_args()

def main():
    """ Run pythia """
    args = parse_input()

    pythia = pythia8.Pythia()
    pythia.readString('Beams:eCM = 14000.')
    pythia.readString('WeakSingleBoson:ffbar2W = on')
    pythia.readString('24:m0 = {}'.format(args.mass))
    pythia.readString('24:mWidth = {}'.format(args.width))
    pythia.readString('24:onMode = off')
    pythia.readString('24:onIfMatch = 1 2')
    pythia.readString('24:onIfMatch = 1 4')
    pythia.readString('24:onIfMatch = 3 2')
    pythia.readString('24:onIfMatch = 3 4')
    pythia.readString('PartonLevel:FSR = off')
    pythia.readString('PartonLevel:ISR = off')
    pythia.readString('PartonLevel:MPI = off')
    pythia.readString('HadronLevel:all = off')
    pythia.readString('PartonLevel:Remnants = off')
    pythia.readString('Check:event = off')
    pythia.init()

    with open(args.filename, 'w') as outfile:
        for _ in range(args.nevents):
            if not pythia.next():
                continue

            for particle in pythia.event:
                if particle.isFinal():
                    outfile.write(
                        '[{: 8e}, {: 8e}, {: 8e}, {: 8e}] '.format(
                            particle.px(), particle.py(), particle.pz(), particle.e()))

            outfile.write('\n')

    pythia.stat()


if __name__ == '__main__':
    main()
