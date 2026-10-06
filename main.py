"""Entry point for the high-frequency OCT application.

Launches the real-time user interface. Pass ``--sim`` to drive it with
synthetic interferograms instead of the spectrometer, which is useful for
developing or demonstrating the interface away from the setup.
"""

import argparse
import sys

from src.gui import run
from src.simulator import SimulatedWasatchAPI


def main() -> int:
    """Parse the command line and start the user interface.

    Returns:
        Qt application exit code.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim",
        action="store_true",
        help="use simulated interferograms instead of the spectrometer",
    )
    arguments = parser.parse_args()

    return run(SimulatedWasatchAPI() if arguments.sim else None)


if __name__ == "__main__":
    sys.exit(main())
