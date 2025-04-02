# PySOC_deriv

## Features

## Installation
```bash
git clone git@github.com:masaya0222/pysoc_deriv.git
cd pysoc_deriv

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
