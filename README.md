# Preconditioned Davidson Algorithm for TDDFT excitation energy calculations.
This is the demonstrative toy-code to show a faster Davidson algorithm with sTDA preconditioner. It contains all the codes and data points mentioned in our paper https://doi.org/10.1063/5.0071013

## This repo is nologer in usage, pleas refer to https://github.com/John-zzh/Davidson


## Installation
Required Python package: `PySCF`, see https://pyscf.org/install.html for detailed installation guidance.

Required python libraries: `Numpy`, `opt_einsum`, `scipy`

Required parallel environment: `openmpi/2.0.1`

Aside from above requirements, no extra installation is required. Because this is just a toy python3 code to show the TDDFT speedup, not a serious quantum chemistry software. Software developers from `PySCF`, `Turbomole`, `ORCA`, `Gaussian`, `Q-chem`, etc., are encouraged to incorporate this semiempirical-preconditioned Davidson method to their softwares accordingly :-)

## Usage
`python <path_to_your_dir>/improved-Davidson-Algorithm/source/Davidson.py -x molecule.xyz -b def2-tzvp -m RKS -f FUNCTIONAL -n NSTATE -df True -TDA true -it 1e-3 -pt 1e-2 -o 0 1 -M PySCF_MEM -v 3 -chk True -TV 40 -max 35 > molecule.out`

### keywords
- **-x** atom coordinates file, in `.xyz` format    
- **-b** basis set, such as `def2-SVP` or `def2-TZVP`
- **-m** RKS method, and only UKS was implemented :-)
- **-f** functional, `lc-b3lyp`, `wb97`,
    `wb97x`,
    `wb97x-d3`,
    `cam-b3lyp`,`b3lyp`, `tpssh`, `m05-2x`, `pbe0`, `m06`, `m06-2x`
- **-n** the number of excited state to compute
- **-df** rue or false, turn on the density fitting
- **-TDA** true or false, use TDA instead of full TDDFT
- **-it** the convergence tolerance for sTDA initial guess
- **-pt** the convergence tolerance for sTDA preconditioning
- **-o** options for initial guess and preconditioner, 0-3

  0: sTDA initial guess add sTDA preconditioner

  1: diagonal initial guess add diagonal preconditioner

  2: diagonal initial guess add sTDA preconditioner

  3: sTDA initial guess add diagonal preconditioner

- **chk** true or false, start the excited state calculation form a pre-calculated ground state PySCF SCF check point file `*.chk`.
- **-TV** the threshold to truncate the virtual orbitals in sTDA two electron integrals, in purpose of lowering the extra preconditioning cost.
- **-max** the maximum number iterations of Davidson algorithm
- **-M** the total memory allocated for the calculation
- **-v** verbose level, same as PySCF ones. 3 is simplest and 5 is for debug.







## Citations
Zhou, Z., & Parker, S. M. (2021). Accelerating molecular property calculations with semiempirical preconditioning. The Journal of Chemical Physics, 155(20), 204111.
