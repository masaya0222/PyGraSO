import os
import numpy as np

from pygraso.parser import gaussian_perser
from pygraso.num_deriv import numerical_deriv


def test_h2_mo_coeff():
    mol_name = "h2_tda"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)

    log_file_name = os.path.join(current_dir, f"data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/{mol_name}.rwf")

    g_parser = gaussian_perser(log_file_name, rwf_file_name)
    mo_coeff = g_parser.get_mo_coeff()
    answer = np.load(current_dir + "/data/h2_mo_coeff.npy")
    assert np.allclose(mo_coeff, answer)


def test_h2_mo_coeff_deriv_anal():
    mol_name = "h2_tda"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)

    log_file_name = os.path.join(current_dir, f"data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/{mol_name}.rwf")

    g_parser = gaussian_perser(log_file_name, rwf_file_name, method="2")
    mo_coeff_deriv = g_parser.get_mo_coeff_deriv()
    # np.save(current_dir+'/data/h2_mo_coeff_deriv.npy', mo_coeff_deriv)
    answer = np.load(current_dir + "/data/h2_mo_coeff_deriv.npy")
    assert np.allclose(mo_coeff_deriv, answer)


def test_h2_mo_coeff_deriv_num():
    mol_name = "h2_tda"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)
    calc_dir = os.path.join(current_dir, f"data/num/")

    num_deriv = numerical_deriv(
        mol_name, atoms, coordinates, calc_dir=calc_dir, calc_type="tda"
    )
    mo_coeff_deriv_num = num_deriv.execute_num_deriv("mo_coeff")
    answer = np.load(current_dir + "/data/h2_mo_coeff_deriv.npy")
    assert np.allclose(mo_coeff_deriv_num, answer, atol=1e-4)


def test_h2o_mo_coeff():
    mol_name = "h2o_tda"
    atoms = ["O", "H", "H"]
    coordinates = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.957200],
        [0.000000, 0.757160, -0.478600],
    ]
    current_dir = os.path.dirname(__file__)

    log_file_name = os.path.join(current_dir, f"data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/{mol_name}.rwf")

    g_parser = gaussian_perser(log_file_name, rwf_file_name)
    mo_coeff = g_parser.get_mo_coeff()
    answer = np.load(current_dir + "/data/h2o_mo_coeff.npy")
    assert np.allclose(mo_coeff, answer)


def test_h2o_mo_coeff_deriv_anal():
    mol_name = "h2o_tda"
    atoms = ["O", "H", "H"]
    coordinates = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.957200],
        [0.000000, 0.757160, -0.478600],
    ]

    current_dir = os.path.dirname(__file__)

    log_file_name = os.path.join(current_dir, f"data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/{mol_name}.rwf")

    g_parser = gaussian_perser(log_file_name, rwf_file_name, method="2")
    mo_coeff_deriv = g_parser.get_mo_coeff_deriv()
    # np.save(current_dir+'/data/h2o_mo_coeff_deriv.npy', mo_coeff_deriv)
    answer = np.load(current_dir + "/data/h2o_mo_coeff_deriv.npy")
    assert np.allclose(mo_coeff_deriv, answer)


def test_h2o_mo_coeff_deriv_num():
    mol_name = "h2o_tda"
    atoms = ["O", "H", "H"]
    coordinates = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.957200],
        [0.000000, 0.757160, -0.478600],
    ]

    current_dir = os.path.dirname(__file__)
    calc_dir = os.path.join(current_dir, f"data/num/")

    num_deriv = numerical_deriv(
        mol_name, atoms, coordinates, calc_dir=calc_dir, calc_type="tda"
    )
    mo_coeff_deriv_num = num_deriv.execute_num_deriv("mo_coeff")
    answer = np.load(current_dir + "/data/h2o_mo_coeff_deriv.npy")
    assert np.allclose(mo_coeff_deriv_num, answer, atol=1e-4)
