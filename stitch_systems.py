import glob
import numpy as np

from amuse.lab import constants, Particles, units, read_set_from_file, write_set_to_file
from amuse.ext.orbital_elements import new_binary_from_orbital_elements, orbital_elements
from globals import _DEBUG_, particle_file, shift_tolerance

def new_rotation_matrix_from_euler_angles(phi, theta, chi):
    """
    Rotation matrix for planetary system orientation
    Args:
        phi (float):   Rotation angle
        theta (float): Rotation angle
        chi (float):   Rotation angle
    Returns:
        matrix (array): Rotation matrix
    """
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    cost = np.cos(theta)
    sint = np.sin(theta)
    cosc = np.cos(chi)
    sinc = np.sin(chi)
    return np.array(
        [[cost*cosc, -cosp*sinc+sinp*sint*cosc, sinp*sinc+cosp*sint*cosc], 
         [cost*sinc, cosp*cosc+sinp*sint*sinc, -sinp*cosc+cosp*sint*sinc],
         [-sint,  sinp*cost,  cosp*cost]]
        )


def rotate(position, velocity, phi, theta, psi):
    """
    Rotate planetary system
    Args:
        position (array): Position vector
        velocity (array): Velocity vector
        phi (float):      Rotation angle
        theta (float):    Rotation angle
        psi (float):      Rotation angle
    Returns:
        matrix (array): Rotated position and velocity vector
    """
    Runit = position.unit
    Vunit = velocity.unit
    matrix = new_rotation_matrix_from_euler_angles(phi, theta, psi)
    return (np.dot(matrix, position.value_in(Runit)) | Runit,
            np.dot(matrix, velocity.value_in(Vunit)) | Vunit)


original = read_set_from_file(particle_file)

bodies = Particles()
bodies.add_particles(original[original.syst_id < 0])  # Add all isolated/rogue bodies since they were not shaken

## Re-assemble all shaken systems into one Particles set alongside the isolated bodies
shaken = glob.glob(f"systems/*")
for i, p in enumerate(shaken):
    print(f"Processing {i+1}/{len(shaken)}: {p}")
    modified_system = read_set_from_file(p)
    bodies.add_particles(modified_system)

    system_id = modified_system.syst_id[0]
    original_system = original[original.syst_id == system_id]
    dN = len(modified_system) - len(original_system)
    if len(modified_system) == len(original_system):
        dr = (modified_system.position - original_system.position).lengths()
        to_remove = modified_system[dr > (shift_tolerance | units.AU)]
        if to_remove:
            print(f"    Curious: removing {len(to_remove)} particles due to large shift")
            bodies.remove_particles(to_remove)
            modified_system.remove_particles(to_remove)
    else:
        print(f"    During shaking, system size changed by {dN} particles due to merger; checking for large shifts...")
        syst_com = original_system.center_of_mass()
        dr = (modified_system.position - syst_com).lengths()
        to_remove = modified_system[dr > (2. * shift_tolerance | units.AU)]
        if to_remove:
            print(f"    Curious: removing {len(to_remove)} particles due to large shift")
            bodies.remove_particles(to_remove)
            modified_system.remove_particles(to_remove)


sma = [ ]
ecc = [ ]

### Reorient planetary systems randomly
to_remove = Particles()
for id in np.unique(bodies.syst_id):
    if id < 0:
        continue

    system = bodies[bodies.syst_id == id]
    host = system[system.mass.argmax()]
    planets = system - host
    for p in planets:
        ke = orbital_elements(p+host, G=constants.G)
        if ke[2] < 0 | units.au or ke[3] > 1:
            print(f"    Removing particle with sma={ke[2]}, ecc={ke[3]}")
            to_remove.add_particle(p)
        p.sma = ke[2]
        p.ecc = ke[3]
        p.ta  = ke[4]
        p.inc = ke[5]
        p.loa = ke[6]
        p.aop = ke[7]

    ### New orientations
    theta0 = np.radians((np.random.normal(-90.0, 90.0, 1)[0]))
    theta_inclination = np.radians(np.random.normal(0, 1.0, (1+len(planets))))
    theta_inclination[0] = 0
    theta = theta0 + theta_inclination
    psi = np.radians(np.random.uniform(0.0, 180.0, 1))[0]
    phi = np.radians(np.random.uniform(0.0, 90.0, 1)[0])
    for i, p in enumerate(planets):
        binary_set = new_binary_from_orbital_elements(
            mass1=p.mass, 
            mass2=host.mass,
            semimajor_axis=p.sma,
            eccentricity=p.ecc,
            inclination=p.inc,
            longitude_of_the_ascending_node=p.loa,
            true_anomaly=p.ta,
            argument_of_periapsis=p.aop,
            G=constants.G
            )
                
        binary_set.position -= binary_set[1].position
        binary_set.velocity -= binary_set[1].velocity
        p.position = binary_set[0].position
        p.velocity = binary_set[0].velocity
        
        pos = p.position
        vel = p.velocity
        
        pos, vel = rotate(pos, vel, 0, 0, psi)
        pos, vel = rotate(pos, vel, 0, p.inc.value_in(units.rad), 0)
        pos, vel = rotate(pos, vel, phi, 0, 0)
        p.position = pos
        p.velocity = vel
        
        p.position += host.position
        p.velocity += host.velocity
        ke = orbital_elements(p+host, G=constants.G)
        try:
            assert (ke[2] - p.sma)/p.sma < 1e-8
            assert abs(ke[3] - p.ecc) < 0.01
        except AssertionError:
            print("Warning: Orbital elements do not match after rotation")
            print(f"Relative error in sma: {(ke[2] - p.sma)/p.sma}")
            print(f"Error in ecc: {abs(ke[3] - p.ecc)}")

print(f"Removing {len(to_remove)} planets")
bodies.remove_particles(to_remove)

if _DEBUG_:
    import matplotlib.pyplot as plt
    plt.scatter(bodies.x.value_in(units.AU), bodies.y.value_in(units.AU))
    plt.scatter(original.x.value_in(units.AU), original.y.value_in(units.AU), alpha=0.3)
    plt.show()

write_set_to_file(bodies, f"modified_{particle_file}", "hdf5", overwrite_file=True)