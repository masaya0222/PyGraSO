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
git clone git@github.com:masaya0222/PyGraSO.git
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
