"""
This package contains tools for running a finite-difference
method simulation of the Schrödinger equation.

Written for Advanced Quantum Mechanics in Spring 2023.

------------------------------------------------------------

Written and tested with:
• python 3.9.13
• numpy 1.23.3
• pandas 1.4.4

------------------------------------------------------------
"""
__author__ = 'Will Ebmeyer'
__version__ = 'v0.0'

# Import modules
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from textwrap import dedent
from sys import stderr, stdout


# Main command-line program
if __name__ == "__main__":
    # Create root parser
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                            description=dedent(__doc__))
    subparsers = parser.add_subparsers(dest='subcommand', required=True, help='Subcommand to run')

    # Parse arguments
    args = parser.parse_args()
    stdout.write(str(args))
