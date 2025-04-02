import os
import tempfile
import shutil
import subprocess
import numpy as np
import time
import argparse

from .parser import gaussian_perser, decode_gaussian_parser


def extract_info(mol_name, log_file_path, rwf_file_path, deriv=True):
    parser = gaussian_perser(log_file_path, rwf_file_path)
    parser.read_basis()

    dump_json_file = f"{mol_name}_log.json"

    parser.dump_json(dump_json_file)
    ao_ovlp = parser.get_ao_ovlp()
    ao_ovlp_deriv = np.empty(0)
    if deriv:
        ao_ovlp_deriv = parser.get_ao_ovlp_deriv()
    mo_coeff = parser.get_mo_coeff()
    mo_coeff_deriv = np.empty(0)
    if deriv:
        mo_coeff_deriv = parser.get_mo_coeff_deriv()
    x_coeff, y_coeff, x_coeff_deriv, y_coeff_deriv = (
        np.empty(0),
        np.empty(0),
        np.empty(0),
        np.empty(0),
    )
    x_coeff = parser.get_xy_coeff()
    if parser.nxy == 2:
        x_coeff, y_coeff = x_coeff

    if deriv:
        x_coeff_deriv = parser.get_xy_coeff_deriv()
        if parser.nxy == 2:
            x_coeff_deriv, y_coeff_deriv = x_coeff_deriv
    dump_npz_file = f"{mol_name}_mat.npz"

    np.savez(
        dump_npz_file,
        ao_ovlp=ao_ovlp,
        ao_ovlp_deriv=ao_ovlp_deriv,
        mo_coeff=mo_coeff,
        mo_coeff_deriv=mo_coeff_deriv,
        x_coeff=x_coeff,
        y_coeff=y_coeff,
        x_coeff_deriv=x_coeff_deriv,
        y_coeff_deriv=y_coeff_deriv,
    )


def check_file_exist(file_path):
    if os.path.isfile(file_path):
        return
    else:
        raise FileNotFoundError(f"Can't find {file_path}")


def tg16(inp_file, work_dir=None, deriv=True):
    if work_dir is None:
        work_dir = os.environ.get("GAUSS_SCRDIR")
    check_file_exist(inp_file)
    with tempfile.TemporaryDirectory(dir=work_dir, prefix="tg16_") as temp_dir:
        cp_inp_file = os.path.join(temp_dir, inp_file)
        shutil.copy(inp_file, cp_inp_file)

        st = time.time()
        try:
            subprocess.run(["g16", cp_inp_file], cwd=temp_dir, check=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error in Gaussian execution: {e}")
        et = time.time()
        print(f"calculating g16 time: {et - st}")
        mol_name = os.path.splitext(inp_file)[0]
        log_file_path = os.path.join(temp_dir, f"{mol_name}.log")
        check_file_exist(log_file_path)
        rwf_file_path = os.path.join(temp_dir, f"{mol_name}.rwf")
        check_file_exist(rwf_file_path)
        st = time.time()
        extract_info(mol_name, log_file_path, rwf_file_path, deriv)
        et = time.time()
        print(f"extract info time: {et - st}")
