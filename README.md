# resonant-chain-decommissioned
This script “shakes” adjacent planet pairs represented by AMUSE `Particles` set into mean-motion resonance, iterating outward to build a resonant chain. It reads an input system from HDF5, applies a migration-based resonance capture routine, and writes the updated system back to disk.

This repository is depricated.
The repository is continued publicly at: https://gitlab.strw.leidenuniv.nl/spz/Resonant_chain

# Resonant Chain Generator

This script “shakes” adjacent planet pairs represented by AMUSE `Particles` set into mean-motion resonance, iterating outward to build a resonant chain. It reads an input system from HDF5, applies a migration-based resonance capture routine, and writes the updated system back to disk.

Method designed and constructed by
-Shuo Huang
-Simon Portegies Zwart
-Dec 2024

## Overview
The resonant-chain construction proceeds as follows:
1. Identify the host star and orbiting planets.
2. Compute orbital elements for each planet.
3. Sort planets by semi-major axis.
4. Iteratively apply a resonance-capture routine to adjacent planet pairs:
    * (planet 0, planet 1) → (planet 1, planet 2) → …
5. Output the final resonant system in AMUSE HDF5 format.

## Installation / Setup

1. Ensure AMUSE is activated in your environment.
2. Prepare an input file containing your planetary system in HDF5 format.
3. Replace line 1 in ```globals.py``` with this file.
4. Run ```python resonance_chain_planetary_system.py```.

## File Structure
* ```generate_resonant_chain.py```: Main driver script that applies resonance capture to existing planetary systems.
* ```globals.py```: Global variables (input paths, flags, threshold for post-processing).
* ```launch_jobs.sh```:  Bash helper script for parallel execution over many systems.
* ```resonance_chain_planetary_system.py```: Script organising files and calling systems to be shaken into resonance.
* ```stitch_systems.py```: If numerous systems are processed, stitch together all final systems and re-embed them within the cluster environment they originated in. 
* ```test_resonance.py```: Test the resonance of the system.

## Usage
The code supports single-system and multi-system execution modes. This behavior is controlled with the flag ```single_system = True``` modifiable in ```globals.txt```.

### Single-system mode (default)
1. Loads a planetary system from input AMUSE HDF5 file (line 1 in ```globals.py```).
2. Identifies the host star as the most massive particle and planets as all other particles.
3. Computes and stores each planet’s semi-major axis (`sma`) from orbital elements.
4. Sorts bodies by `sma` and migrates planets into resonance through adjacent planet pairs starting with the inner-most pair:
   - (planet 0, planet 1), then (planet 1, planet 2), etc.
5. Writes the updated system to an output HDF5 file in the current working directory.

### Multi-system mode (in-code toggle)
1. Set `single_system = False`, the script will then loop over many systems, each identified by the atribute `syst_id`. 
2. Run ```python resonance_chain_planetary_system.py START END``` where ```START``` and ```END``` are integers whose range denotes the number of systems one core should process. 
    * Example: ```python resonance_chain_planetary_system.py 0 5``` will shake five independent systems. 
    * For automated parallel execution across many cores, run ```bash launch_jobs.sh```, with the list defined on line 3 setting the number of systems each core considers.
3. Individual resonant systems are written to: ```systems/resonant_system<syst_id>.hdf5```. These can be stitched back into cluster simulations via ```python stitch_systems.py```.