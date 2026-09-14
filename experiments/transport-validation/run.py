"""Run the canonical layered-absorption CUDA validation and record real results."""

from dpt.transport.experiments import main

if __name__ == "__main__":
    main("validation", __file__)
