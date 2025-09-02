<div align="left">
  <img src="https://github.com/masaya0222/PyGraSO/blob/main/logo.png" height="160px"/>
</div>

PyGraSO
==================================

## Features

## Installation
- Prerequisites
  - Python version 3.11 or higher
  - GNU Make 3.82+ (tested on 3.82)
  - CMake 3.22+ (tested on 3.22.1)
  - Gaussian 16 (g16) installed and available on `PATH` (i.e., `g16` is executable)
  - `rwfdump` available on `PATH` (i.e., `rwfdump` is executable)

```bash
git clone --recurse-submodules git@github.com:masaya0222/PyGraSO.git
cd PyGraSO

echo "export PYTHONPATH=$(pwd):\$PYTHONPATH" >> ~/.bashrc
echo "export PATH=$(pwd)/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc

pip install -r requirements.txt

bash build_clisp.sh
bash build_libcint.sh
```

## Tests & Examples
Please run the tests on a machine with at least 32 CPU cores.
```bash
pytest
cd examples/from_log_2tThy
bash run_from_log.sh
cd ../from_dump_2tThy
bash run_from_dump.sh
```

## Usage
### Notes (read before running)
- Prepare an **XYZ file** that contains atomic symbols and **Cartesian coordinates** (e.g., `molecule.xyz`).
- In each **Gaussian input (.gjf)**, add:
  - The RWF directive:  
    `%rwf="NameOfFile".rwf`  (e.g., `molecule_s1_freq.rwf`)
  - Route options (example):  
    `freq 6D 10F IOP(10/33=2,10/95=9)`
- **Log size caution:** `IOP(10/33=2)` can make `.log` files very large. PyGraSO supports two workflows:
  1) keep `.log` files and parse them, or
  2) avoid large logs by dumping the required derivative info to files and then run PyGraSO.

> Minimal `.gjf` route example
> ```
> %chk=molecule_s1_freq.chk
> %rwf=molecule_s1_freq.rwf
> #p <functional/basis> td freq 6D 10F IOP(10/33=2,10/95=9)
> ```

---

### 1) Keep log files (example: T1–S1)
```bash
g16 molecule_s1_freq.gjf
g16 molecule_t1_freq.gjf
pyGraso -c config_T1_S1.yaml
```
`config_T1_S1.yaml` example:
```
triplet: "T1"
singlet: "S1"
dump: False
xyz_file: "molecule.xyz"
zeff_type: "orca"

triplet_log_file: "molecule_t1_freq.log"
triplet_rwf_file: "molecule_t1_freq.rwf"
singlet_log_file: "molecule_s1_freq.log"
singlet_rwf_file: "molecule_s1_freq.rwf"
```


### 2) Avoid large logs (dump derivative info instead)
Use `tg16` to run Gaussian and dump the required coefficient-derivative information without keeping bulky logs, then run PyGraSO:
```
tg16 molecule_s1_freq.gjf
tg16 molecule_t1_freq.gjf
pyGraso -c config_T1_S1.yaml
```
`config_T1_S1.yaml` example:
```
triplet: "T1"
singlet: "S1"
dump: True
xyz_file: "molecule.xyz"
zeff_type: "orca"

triplet_json_file: "molecule_t1_freq_log.json"
triplet_npz_file: "molecule_t1_freq_mat.npz"
singlet_json_file: "molecule_s1_freq_log.json"
singlet_npz_file: "molecule_s1_freq_mat.npz"
```
