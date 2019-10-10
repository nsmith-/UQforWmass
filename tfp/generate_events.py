""" Generate events using pythia8. """

import argparse
import numpy as np

import pythia8

def parse_input():
    """ Parse input arguments. """
    parser = argparse.ArgumentParser(description='Generate nevents for W '
                                     'production for a given mass and width')
    parser.add_argument('--nevents', '-n', type=int, default=1000,
                        help='Number of events to generate')
    parser.add_argument('--mass_min', type=float, default=70.,
                        help='Minimum mass for W-boson')
    parser.add_argument('--mass_max', type=float, default=90.,
                        help='Maximum mass for W-boson')
    parser.add_argument('--mass_step', type=float, default=2.,
                        help='Mass step for W-boson')
    parser.add_argument('--width_min', type=float, default=0.8,
                        help='Minimum width for W-boson')
    parser.add_argument('--width_max', type=float, default=2.8,
                        help='Maximum width for W-boson')
    parser.add_argument('--width_step', type=float, default=0.2,
                        help='Width step for W-boson')

    return parser.parse_args()


def main():
    """ Loop over all combinations of mass and width. """
    args = parse_input()
    masses = np.linspace(args.mass_min, args.mass_max,
                         np.ceil((args.mass_max - args.mass_min) / args.mass_step)+1)
    widths = np.linspace(args.width_min, args.width_max,
                         np.ceil((args.width_max - args.width_min) / args.width_step)+1)

    for mass in masses:
        for width in widths:
            run(args.nevents, mass, width)


def run(nevents, mass, width):
    """ Run pythia """
    pythia = pythia8.Pythia()
    pythia.readString('Beams:eCM = 14000.')
    pythia.readString('WeakSingleBoson:ffbar2W = on')
    pythia.readString('24:m0 = {}'.format(mass))
    pythia.readString('24:mWidth = {}'.format(width))
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

    filename = 'events_{:.1f}_{:.1f}.txt'.format(mass, width)
    with open(filename, 'w') as outfile:
        for _ in range(nevents):
            if not pythia.next():
                continue

            for particle in pythia.event:
                if particle.isFinal():
                    outfile.write(
                        '[{: .8e}, {: .8e}, {: .8e}, {: .8e}] '.format(
                            particle.px(), particle.py(), particle.pz(), particle.e()))

            outfile.write('\n')

    pythia.stat()


if __name__ == '__main__':
    main()
