import os
from sys import stdout
import openmm as mm
from openmm import *
from openmm.app import *
from openmm.unit import *

# --- Path Configuration ---
workDir = '__PROJECT_DIR__'
top_path = os.path.join(workDir, "SYS_gaff2.prmtop")
crd_path = os.path.join(workDir, "SYS_gaff2.crd")
equil_rst = os.path.join(workDir, "prot_lig_equil.rst")
equil_pdb = os.path.join(workDir, "prot_lig_equil.pdb")
equil_dcd = os.path.join(workDir, "prot_lig_equil.dcd")
equil_log = os.path.join(workDir, "prot_lig_equil.log")

# --- Parameters ---
Integration_timestep = 2   # fs
Temperature = 310          # K
Pressure = 1               # bar
NVT_Time_ns = 1            # 1000 ps
NPT_Time_ns = 1            # 1000 ps
Log_interval_ps = 10       # every 10 ps

dt = Integration_timestep * femtosecond
temperature = Temperature * kelvin
pressure = Pressure * bar

nvt_steps = int((NVT_Time_ns * nanosecond) / dt)   
npt_steps = int((NPT_Time_ns * nanosecond) / dt)   
log_freq  = int((Log_interval_ps * picosecond) / dt)  

print(f"\n> NVT steps : {nvt_steps} ({NVT_Time_ns} ns)")
print(f"> NPT steps : {npt_steps} ({NPT_Time_ns} ns)")
print(f"> Log every : {log_freq} steps ({Log_interval_ps} ps)")

# --- System Creation ---
print(f"\n> Loading system from {top_path}...")
prmtop = AmberPrmtopFile(top_path)
inpcrd = AmberInpcrdFile(crd_path)

system = prmtop.createSystem(
    nonbondedMethod=PME,
    nonbondedCutoff=1.0 * nanometers,
    constraints=HBonds,
    rigidWater=True,
    ewaldErrorTolerance=0.0005,
    hydrogenMass=1.0 * amu
)

# --- ADVANCED FIX: Add Barostat Early but Keep Disabled ---
# Setting frequency to 0 keeps it completely inactive during NVT without requiring reinitialization later
barostat = MonteCarloBarostat(pressure, temperature, 0)
system.addForce(barostat)

# --- ADVANCED FIX: Setup Harmonic Restraints Force ---
# Targets protein heavy atoms and ligand heavy atoms (excludes water/ions)
restraint_force = mm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
restraint_force.addGlobalParameter("k", 1000.0 * kilojoules_per_mole / nanometer**2) # default placeholder
restraint_force.addPerParticleParameter("x0")
restraint_force.addPerParticleParameter("y0")
restraint_force.addPerParticleParameter("z0")

for atom in prmtop.topology.atoms():
    if atom.residue.name not in ['HOH', 'WAT', 'H2O', 'NA', 'CL']:
        if atom.element and atom.element.symbol == 'H':
            continue # Skip hydrogens
        restraint_force.addParticle(atom.index, inpcrd.positions[atom.index])
system.addForce(restraint_force)

integrator = LangevinMiddleIntegrator(temperature, 1.0, dt)
integrator.setConstraintTolerance(0.00001)

# Platform selection (CUDA)
try:
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {"Precision": "mixed"}
    print("> Using CUDA platform (mixed precision)")
except:
    print("> CUDA not found, falling back to CPU.")
    platform = mm.Platform.getPlatformByName("Reference")
    properties = {}

simulation = Simulation(prmtop.topology, system, integrator, platform, properties)
simulation.context.setPositions(inpcrd.positions)
if inpcrd.boxVectors is not None:
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

# --- Stage 1: Staged Energy Minimization ---
print("\n> Stage 1: Heavy-Restrained Energy minimization (waters relaxing around solute)...")
# Set stiff restraints to lock solute down while waters pack in
simulation.context.setParameter("k", 10000.0 * kilojoules_per_mole / nanometer**2)
simulation.minimizeEnergy(maxIterations=10000, tolerance=100)

print("> Relaxing restraints for final minimization pass...")
# Drop restraints to let sidechains settle gently
simulation.context.setParameter("k", 1000.0 * kilojoules_per_mole / nanometer**2)
simulation.minimizeEnergy(maxIterations=40000, tolerance=10)

# --- Stage 2: NVT Warmup with Position Restraints ---
print(f"\n> Stage 2: NVT warmup under restraints ({NVT_Time_ns} ns = {nvt_steps} steps)...")
simulation.context.setVelocitiesToTemperature(temperature)

simulation.reporters.append(StateDataReporter(
    equil_log, log_freq,
    step=True, time=True, temperature=True,
    kineticEnergy=True, potentialEnergy=True, totalEnergy=True,
    speed=True, separator='\t', append=False
))
simulation.reporters.append(StateDataReporter(
    stdout, log_freq * 5,
    step=True, time=True, temperature=True, speed=True,
    progress=True, totalSteps=nvt_steps, separator='\t'
))

simulation.step(nvt_steps)
simulation.reporters.clear()
print("  NVT done.")

# --- Stage 3: NPT Equilibration (Release Restraints, Activate Barostat) ---
print(f"\n> Stage 3: NPT equilibration ({NPT_Time_ns} ns = {npt_steps} steps)...")
# 1. Turn off harmonic restraints completely
simulation.context.setParameter("k", 0.0 * kilojoules_per_mole / nanometer**2)
# 2. Activate the barostat by setting its frequency to every 25 steps
barostat.setFrequency(25)

simulation.reporters.append(DCDReporter(equil_dcd, log_freq))
simulation.reporters.append(StateDataReporter(
    equil_log, log_freq,
    step=True, time=True, temperature=True, density=True,
    kineticEnergy=True, potentialEnergy=True, totalEnergy=True,
    volume=True, speed=True, separator='\t', append=True
))
simulation.reporters.append(StateDataReporter(
    stdout, log_freq * 5,
    step=True, time=True, temperature=True, density=True,
    speed=True, progress=True, totalSteps=npt_steps, separator='\t'
))

simulation.step(npt_steps)
simulation.reporters.clear()
print("  NPT done.")

# --- Final Save ---
print(f"\n> Saving final equilibration files...")
state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)

with open(equil_rst, 'w') as f:
    f.write(XmlSerializer.serialize(state))

with open(equil_pdb, 'w') as f:
    PDBFile.writeFile(simulation.topology, state.getPositions(), f)

print(f"\nDone! Saved Production-Ready files:")
print(f"  RST : {equil_rst}")
print(f"  PDB : {equil_pdb}")
print(f"  DCD : {equil_dcd}")
print(f"  LOG : {equil_log}\n")
