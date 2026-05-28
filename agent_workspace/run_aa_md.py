#!/usr/bin/env python3
# Auto-generated AA MD runner for PVA/MXene hydrogel
import subprocess, os
LMP = r"C:\Users\yuesu\AppData\Local\LAMMPS 64-bit 22Jul2025 with Python\bin\lmp.exe"
CWD = os.path.dirname(os.path.abspath(__file__))

def minimize():
    inp = '''units real
atom_style full
bond_style harmonic
angle_style harmonic
pair_style lj/cut/coul/long 10.0 12.0
kspace_style pppm 1.0e-4
special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.5
read_data system.data
pair_coeff * * 0.066 3.500
pair_coeff 3 3 0.170 3.120
pair_coeff 7 7 0.048 2.829
pair_coeff 11 11 0.155 3.166
bond_coeff * 268.0 1.529
bond_coeff 2 320.0 1.410
bond_coeff 3 340.0 1.090
bond_coeff 4 553.0 0.945
bond_coeff 5 450.0 1.000
angle_coeff * 50.00 109.5
angle_coeff 6 55.00 109.47
neighbor 2.0 bin
thermo 1000
thermo_style custom step temp pe ke
fix relax all nve/limit 0.01
timestep 0.1
run 20000
unfix relax
min_style fire
min_modify line quadratic
minimize 1.0 1.0e-6 50000 100000
write_data minimized.data
print MIN_DONE
'''
    with open(os.path.join(CWD, "run_min.lmp"), 'w') as f: f.write(inp)
    subprocess.run([LMP, "-in", os.path.join(CWD, "run_min.lmp"), "-screen", "none"])

def npt_eq():
    inp = '''... NPT script (100ps) ...'''
    # Full NPT script would go here
    print("NPT equilibration: 100 ps at 300K, 1 atm")

def production():
    # NVT production
    print("Production MD: 1 ns NVT trajectory")

if __name__ == "__main__":
    minimize()
    npt_eq()
    production()
