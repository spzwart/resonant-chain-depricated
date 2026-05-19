#!/usr/bin/env python


import glob
import numpy as np
import os
import sys

from amuse.ext.orbital_elements import orbital_elements
from amuse.io import write_set_to_file, read_set_from_file
from amuse.lab import Particles
from amuse.units import units, constants

from generate_resonant_chain import (
    bring_planet_pair_in_resonance, semi_to_orbital_period
)
from globals import _DEBUG_, particle_file, single_system


def resonant_chain_planetary_system(bodies, tau_a, t_evol, n_steps) -> Particles:
    """
    Iterate over adjacent planets and shake them into resonance.
    Args:
        bodies (Particles):  The particles representing the planetary system.
        tau_a (float):       Migration parameter in terms of the outer orbital period.
        t_evol (float):      Integration time in units of the outer orbital period.
        n_steps (int):       Number of steps for the integration.
    Returns:
        Particles: The updated particles after creating the resonant chain.
    """
    for pi in range(len(bodies)-2):
        bodies = resonant_pair_planetary_system(
            bodies=bodies,
            inner_planet_id=pi,
            tau_a=tau_a,
            t_evol=t_evol,
            n_steps=n_steps
            )
    return bodies

def resonant_pair_planetary_system(
    bodies, inner_planet_id=0, outer_planet_id=1,
    tau_a=-1e5, t_evol=100, n_steps=100
    ) -> Particles:
    """
    Create a resonant pair of planets in a planetary system.
    Assumes host to be most massive particle in the system, planets to be all other
    massive particles.
    Args:
        bodies (Particles):     The particles representing the planetary system.
        inner_planet_id (int):  Index of the inner planet.
        outer_planet_id (int):  Index of the outer planet.
        tau_a (float):          Migration parameter in terms of the outer orbital period.
        t_evol (float):         Integration time in units of the outer orbital period.
        n_steps (int):          Number of steps for the integration.
    """
    star = bodies[bodies.mass.argmax()]
    planets = bodies - star
    for pi in planets:
        kepler = orbital_elements(star + pi)
        pi.sma = kepler[2]

    planet_a = planets[inner_planet_id]
    planet_b = planets[outer_planet_id]
    Porb_a = semi_to_orbital_period(planet_a.sma, star.mass + planet_a.mass)[0]
    Porb_b = semi_to_orbital_period(planet_b.sma, star.mass + planet_b.mass)[0]
    if np.isnan(Porb_a.value_in(units.s)) or np.isnan(Porb_b.value_in(units.s)):
        print("Ill-defined orbital periods, returning original bodies")
        return bodies
    
    bring_planet_pair_in_resonance(
        planetary_system=bodies, 
        outer_planet=planet_b,
        tau_a=tau_a,
        t_evol=t_evol, 
        n_steps=n_steps
    )
    return bodies

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("--inner", dest="inner_planet_id",
                      type="int", default = 0,
                      help="inner planet id [%default]")
    result.add_option("--outer", dest="outer_planet_id",
                      type="int", default = 1,
                      help="outer planet id [%default]")
    result.add_option("-f", dest="infilename", 
                      default = "input_filename.amuse",
                      help="input infilename [%default]")
    result.add_option("-F", dest="outfilename", 
                      default = "output_filename.amuse",
                      help="output infilename [%default]")
    result.add_option("--n_steps", dest="n_steps", 
                      default = 100, type="int",
                      help="number of steps [%default]")
    result.add_option("--t_evol", dest="t_evol", 
                      default = 1000, type="float",
                      help="integration time in units of the outer orbital period [%default]")
    result.add_option("--tau", dest="tau_a", 
                      type="float", default = -1e5,
                      help="migration parameter (in terms of outer orbital period) [%default]")
    return result
    
if __name__ in ('__main__', '__plot__'):
    o, arguments  = new_option_parser().parse_args()
    bodies = read_set_from_file(particle_file)

    if single_system:
        if len(bodies)==2:
            print(f"System has 1 planet, no resonant chain to be created.")

        host = bodies[bodies.mass.argmax()]
        if not hasattr(bodies, "sma"):
            planets = bodies - host
            for p in planets:
                ke = orbital_elements(p+host, G=constants.G)
                p.sma = ke[2]

        bodies = bodies.sorted_by_attribute("sma")
        
        ### Remove asteroids from the system
        asteroids = bodies[bodies.mass == 0 | units.MSun].copy()
        bodies.remove_particles(asteroids)

        if _DEBUG_:
            print(f"Original attributes:")
            print(f"Masses: {bodies.mass}")
            p = bodies - host
            for pl in p:
                ke = orbital_elements(pl + host, G=constants.G)
                print(f"ecc={ke[3]}, ", end=" ") 
                print(f"sma={ke[2].in_(units.au)}", end=" ")
                print(f"inc={ke[5].in_(units.deg)}")

        bodies = resonant_chain_planetary_system(
            bodies=bodies, 
            tau_a=o.tau_a, 
            t_evol=o.t_evol, 
            n_steps=o.n_steps
            )
        
        if _DEBUG_:
            host = bodies[bodies.mass.argmax()]
            p = bodies - host
            print(f"After shaking:")
            print(f"Masses: {bodies.mass}")
            for pl in p:
                ke = orbital_elements(pl + host, G=constants.G)
                print(f"ecc={ke[3]}, ", end=" ") 
                print(f"sma={ke[2].in_(units.au)}", end=" ")
                print(f"inc={ke[5].in_(units.deg)}")
                
            if np.isnan(host.x.value_in(units.pc)):
                print("!!! Task unsuccessful: Resonance not found. !!!")
                exit(-1)

        ### Re-add asteroids to the system
        bodies.add_particles(asteroids)
        write_set_to_file(bodies, o.outfilename, "hdf5", append_to_file=False)

    else:
        os.makedirs("systems", exist_ok=True)
        os.makedirs("logfiles", exist_ok=True)
        
        try:
            bodies -= bodies[bodies.syst_id < 0]
        except Exception as e:
            raise AttributeError("No 'syst_id' attribute found in particle set. "
                                 "Make sure to include it in the input file.") from e

        syst_id_start, syst_id_end = int(sys.argv[1]), int(sys.argv[2])
        for syst_id in np.unique(bodies.syst_id)[syst_id_start:syst_id_end]:
            data = glob.glob("systems/*.hdf5")
            if f"systems/resonant_system{syst_id}.hdf5" in data:
                print(f"System {syst_id} already simulated")
                continue
            
            if syst_id > 0:
                system = bodies[bodies.syst_id == syst_id]
                
                ### Remove asteroids from the system
                asteroids = system[system.mass == 0 | units.MSun].copy()
                system.remove_particles(asteroids)

                print(f"Processing System #{syst_id}/{bodies.syst_id.max()}...")
                if len(system) > 2:
                    host = system[system.mass.argmax()]
                    original_pos = host.position
                    original_vel = host.velocity
                    copied_system = system.copy()

                    # Move system to origin
                    system.position -= original_pos
                    system.velocity -= original_vel
                    p = system - host
                    
                    print(f"    System masses: {system.mass.in_(units.MJupiter)}")
                    if _DEBUG_:
                        print(f"    Before:")
                        for pl in p:
                            ke = orbital_elements(pl + host, G=constants.G)
                            print(f"    ecc={ke[3]}, sma={ke[2].in_(units.au)}, inc={ke[5].in_(units.deg)}")

                    system = resonant_chain_planetary_system(
                        bodies=system, 
                        tau_a=o.tau_a, 
                        t_evol=o.t_evol, 
                        n_steps=o.n_steps
                        )
                    host = system[system.mass.argmax()]

                    if _DEBUG_:
                        print(f"   After:")
                        p = system - host
                        for pl in p:
                            ke = orbital_elements(pl + host, G=constants.G)
                            print(f"    ecc={ke[3]}, sma={ke[2].in_(units.au)}, inc={ke[5].in_(units.deg)}")

                    # Move back to cluster reference frame
                    system.position -= host.position
                    system.velocity -= host.velocity
                    system.position += original_pos
                    system.velocity += original_vel
                    
                    drij = (original_pos - host.position).lengths()
                    dvij = (original_vel - host.velocity).lengths()
                    assert np.all(drij < 1e-5 | units.pc), "Position restoration failed!"
                    assert np.all(dvij < 1e-5 | units.kms), "Velocity restoration failed!"

                ### Re-add asteroids to the system
                system.add_particles(asteroids)
                write_set_to_file(system, f"systems/resonant_system{syst_id}.hdf5")