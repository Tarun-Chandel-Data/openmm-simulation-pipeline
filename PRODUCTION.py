import os
from sys import stdout, exit, stderr
import math, fnmatch
import openmm as mm
from openmm import *
from openmm.app import *
from openmm.unit import *

# --- Path Configuration ---
#define path here by using pwd,/home/name/dir/dir/...
workDir = '__PROJECT_DIR__'
Equilibrated_PDB = 'prot_lig_equil.pdb'
State_file = 'prot_lig_equil.rst'
Ligand_Force_field = "GAFF2"
Jobname = '500ns'

top = os.path.join(workDir, "complex.prmtop")
crd = os.path.join(workDir, "complex.inpcrd")
pdb = os.path.join(workDir, "complex.pdb")

# --- Replica Configuration ---
# Change REPLICA_ID to 1, 2, or 3 for each replicate run.
# Each value produces a unique random velocity seed so trajectories diverge.
REPLICA_ID = 1                          # <-- SET THIS: 1, 2, or 3
VELOCITY_SEEDS = {1: 42, 2: 137, 3: 251}
velocity_seed = VELOCITY_SEEDS[REPLICA_ID]

# --- Performance Parameters ---
Stride_Time = "500"           # ns per stride
Number_of_strides = "1"       # number of strides
Integration_timestep = "4"    # fs (HMR enabled)
Temperature = 310             # K
Pressure = 1                  # bar
Write_the_trajectory = "100"  # ps
Write_the_log = "10"          # ps

# --- MD Parameter Calculation ---
jobname = os.path.join(workDir, f"{Jobname}_rep{REPLICA_ID}")
coordinatefile = crd
pdbfile = os.path.join(workDir, Equilibrated_PDB)
topologyfile = top
equil_rst_file = os.path.join(workDir, State_file)

stride_time_ps = float(Stride_Time) * 1000
stride_time    = float(stride_time_ps) * picosecond
nstride        = int(Number_of_strides)
dt             = int(Integration_timestep) * femtosecond
temperature    = float(Temperature) * kelvin
savcrd_freq    = int(Write_the_trajectory) * picosecond
print_freq     = int(Write_the_log) * picosecond
pressure       = float(Pressure) * bar

nsteps  = int(stride_time.value_in_unit(picosecond) / dt.value_in_unit(picosecond))
nprint  = int(print_freq.value_in_unit(picosecond)  / dt.value_in_unit(picosecond))
nsavcrd = int(savcrd_freq.value_in_unit(picosecond) / dt.value_in_unit(picosecond))

print("\n> Simulation details:\n")
print(f"\tReplica ID             = {REPLICA_ID}")
print(f"\tVelocity seed          = {velocity_seed}")
print(f"\tIntegration timestep   = {dt}")
print(f"\tHMR Setting            = 3.0 amu")
print(f"\tTotal steps per stride = {nsteps}")
print(f"\tTrajectory saved every = {nsavcrd} steps ({Write_the_trajectory} ps)")
print(f"\tLog written every      = {nprint} steps ({Write_the_log} ps)")

# --- System Setup ---
print("\n> Setting the system:\n")
prmtop = AmberPrmtopFile(topologyfile)
inpcrd = AmberInpcrdFile(coordinatefile)

print("\t- Creating system with HMR (3.0 amu) and HBonds constraints...")
nonbondedMethod      = PME
nonbondedCutoff      = 1.0 * nanometers
ewaldErrorTolerance  = 0.0005
constraints          = HBonds
rigidWater           = True
constraintTolerance  = 0.000001
friction             = 1.0

system = prmtop.createSystem(
    nonbondedMethod=nonbondedMethod,
    nonbondedCutoff=nonbondedCutoff,
    constraints=constraints,
    rigidWater=rigidWater,
    ewaldErrorTolerance=ewaldErrorTolerance,
    hydrogenMass=3.0 * amu          # HMR: enables 4 fs timestep
)

print("\t- Setting barostat (NPT)...")
system.addForce(MonteCarloBarostat(pressure, temperature))

print("\t- Setting LangevinMiddleIntegrator...")
integrator = LangevinMiddleIntegrator(temperature, friction, dt)
integrator.setConstraintTolerance(constraintTolerance)

# --- Platform: Explicit CUDA mixed precision ---
try:
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {"Precision": "mixed"}
    print("\t- Using CUDA platform (mixed precision)")
except Exception:
    print("\t- CUDA not found, falling back to CPU Reference platform.")
    platform = mm.Platform.getPlatformByName("Reference")
    properties = {}

simulation = Simulation(prmtop.topology, system, integrator, platform, properties)
simulation.context.setPositions(inpcrd.positions)
if inpcrd.boxVectors is not None:
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

# --- Production Loop ---
for n in range(1, nstride + 1):
    print(f"\n\n>>> Simulating Stride #{n} (Replica {REPLICA_ID}) <<<")
    dcd_file     = f"{jobname}_{n}.dcd"
    log_file     = f"{jobname}_{n}.log"
    rst_file     = f"{jobname}_{n}.rst"
    prv_rst_file = f"{jobname}_{n-1}.rst"
    pdb_file     = f"{jobname}_{n}.pdb"

    if os.path.exists(rst_file):
        print(f"> Stride #{n} already finished. Skipping...")
        continue

    if n == 1:
        print(f"> Loading state from equilibration: {equil_rst_file}")
        with open(equil_rst_file, 'r') as f:
            simulation.context.setState(XmlSerializer.deserialize(f.read()))
            simulation.currentStep = 0
            simulation.context.setTime(0)

        # Randomize velocities using replica-specific seed for trajectory divergence
        print(f"> Randomizing velocities with seed {velocity_seed} (Replica {REPLICA_ID})...")
        simulation.context.setVelocitiesToTemperature(temperature, velocity_seed)

    else:
        print(f"> Loading state from previous stride: {prv_rst_file}")
        with open(prv_rst_file, 'r') as f:
            simulation.context.setState(XmlSerializer.deserialize(f.read()))

    # --- Reporters ---
    simulation.reporters.append(DCDReporter(dcd_file, nsavcrd))

    simulation.reporters.append(StateDataReporter(
        stdout, nprint,
        step=True, time=True, speed=True, progress=True,
        remainingTime=True, totalSteps=(nsteps * nstride),
        separator='\t\t'
    ))

    simulation.reporters.append(StateDataReporter(
        log_file, nprint,
        step=True, time=True,
        kineticEnergy=True, potentialEnergy=True, totalEnergy=True,
        temperature=True, density=True, volume=True,   # density added for publication
        speed=True, separator='\t'
    ))

    print(f"\n> Simulating {nsteps} steps ({Stride_Time} ns)...")
    simulation.step(nsteps)
    simulation.reporters.clear()

    # --- Save State ---
    print(f"> Saving state: {rst_file}")
    state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        enforcePeriodicBox=True        # ensures molecules wrapped inside box
    )
    with open(rst_file, 'w') as f:
        f.write(XmlSerializer.serialize(state))

    # --- Save PDB ---
    print(f"> Saving PDB: {pdb_file}")
    positions = simulation.context.getState(
        getPositions=True,
        enforcePeriodicBox=True        # consistent with RST
    ).getPositions()
    with open(pdb_file, 'w') as f:
        PDBFile.writeFile(simulation.topology, positions, f)

print(f"\n> Production Finished! (Replica {REPLICA_ID})\n")
