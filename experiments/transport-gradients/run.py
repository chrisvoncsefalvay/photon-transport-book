"""Check complete expected-signal gradients using independent CUDA history batches."""

from dpt.transport.experiments import main

if __name__ == "__main__":
    main("gradients", __file__)
