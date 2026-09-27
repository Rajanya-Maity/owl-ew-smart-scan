"""Open-loop baseline scanner — Step 3.
No intelligence: sweeps bands 1,2,3,...,N,1,2,3,... in a fixed loop.
This is the permanent comparison benchmark.
"""
from env.spectrum_sim import SpectrumSimulator


def run_open_loop(sim: SpectrumSimulator):
    sim.reset()
    log = []
    band = 0
    done = False
    while not done:
        info, done = sim.step([band])
        log.append(info)
        band = (band + 1) % sim.n_bands
    return log
