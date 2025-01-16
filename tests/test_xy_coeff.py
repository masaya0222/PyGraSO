import os
import sys
import numpy as np

from parser import gaussian_perser
from num_deriv import numerical_deriv


def test_h2_tda_x_coeff():
    mol_name = "h2_tda"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)

    log_file_name = os.path.join(current_dir, f"data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/{mol_name}.rwf")

    g_parser = gaussian_perser(log_file_name, rwf_file_name)
    x_coeff = g_parser.get_xy_coeff()
    assert isinstance(x_coeff, np.ndarray)

    answer = np.load(current_dir + "/data/h2_tda_x_coeff.npy")
    assert np.allclose(x_coeff, answer)


def test_h2_tda_x_coeff_deriv_anal():
    mol_name = "h2_tda"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)

    log_file_name = os.path.join(current_dir, f"data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/{mol_name}.rwf")

    g_parser = gaussian_perser(log_file_name, rwf_file_name)
    x_coeff_deriv = g_parser.get_xy_coeff_deriv()
    assert isinstance(x_coeff_deriv, np.ndarray)

    answer = np.load(current_dir + "/data/h2_tda_x_coeff_deriv.npy")
    assert np.allclose(x_coeff_deriv, answer)


def test_h2_tda_x_coeff_deriv_num():
    mol_name = "h2_tda"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)
    calc_dir = os.path.join(current_dir, f"data/num/")

    num_deriv = numerical_deriv(
        mol_name, atoms, coordinates, calc_dir=calc_dir, calc_type="tda"
    )
    x_coeff_deriv_num = num_deriv.execute_num_deriv("xy_coeff")
    assert isinstance(x_coeff_deriv_num, np.ndarray)
    answer = np.load(current_dir + "/data/h2_tda_x_coeff_deriv.npy")
    assert np.allclose(x_coeff_deriv_num, answer, atol=1e-5)


def test_h2_td_xy_coeff():
    mol_name = "h2_td"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)

    log_file_name = os.path.join(current_dir, f"data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/{mol_name}.rwf")

    g_parser = gaussian_perser(log_file_name, rwf_file_name)
    xy_coeff = g_parser.get_xy_coeff()
    assert isinstance(xy_coeff, tuple)

    # np.save(current_dir+'/data/h2_td_xy_coeff.npy', xy_coeff)
    answer = np.load(current_dir + "/data/h2_td_xy_coeff.npy")
    assert np.allclose(xy_coeff[0], answer[0])
    assert np.allclose(xy_coeff[1], answer[1])


def test_h2_td_xy_coeff_deriv_anal():
    mol_name = "h2_td"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)

    log_file_name = os.path.join(current_dir, f"data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/{mol_name}.rwf")

    g_parser = gaussian_perser(log_file_name, rwf_file_name)
    xy_coeff_deriv = g_parser.get_xy_coeff_deriv()
    assert isinstance(xy_coeff_deriv, tuple)

    answer = np.load(current_dir + "/data/h2_td_xy_coeff_deriv.npy")
    assert np.allclose(xy_coeff_deriv[0], answer[0])
    assert np.allclose(xy_coeff_deriv[1], answer[1])


def test_h2_td_xy_coeff_deriv_num():
    mol_name = "h2_td"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)
    calc_dir = os.path.join(current_dir, f"data/num/")

    num_deriv = numerical_deriv(
        mol_name, atoms, coordinates, calc_dir=calc_dir, calc_type="td"
    )
    xy_coeff_deriv_num = num_deriv.execute_num_deriv("xy_coeff")
    assert isinstance(xy_coeff_deriv_num, tuple)

    answer = np.load(current_dir + "/data/h2_td_xy_coeff_deriv.npy")
    assert np.allclose(xy_coeff_deriv_num[0], answer[0], atol=1e-5)
    assert np.allclose(xy_coeff_deriv_num[1], answer[1], atol=1e-5)


def test_h2o_tda_x_coeff():
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
    x_coeff = g_parser.get_xy_coeff()
    assert isinstance(x_coeff, np.ndarray)

    # np.save(current_dir+'/data/h2o_tda_x_coeff.npy', x_coeff)
    answer = np.load(current_dir + "/data/h2o_tda_x_coeff.npy")
    assert np.allclose(x_coeff, answer)


def test_h2o_tda_x_coeff_deriv_anal():
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
    x_coeff_deriv = g_parser.get_xy_coeff_deriv()
    assert isinstance(x_coeff_deriv, np.ndarray)

    # np.save(current_dir+'/data/h2o_tda_x_coeff_deriv.npy', x_coeff_deriv)
    answer = np.load(current_dir + "/data/h2o_tda_x_coeff_deriv.npy")
    assert np.allclose(x_coeff_deriv, answer)


def test_h2_tda_x_coeff_deriv_num():
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
    x_coeff_deriv_num = num_deriv.execute_num_deriv("xy_coeff")
    assert isinstance(x_coeff_deriv_num, np.ndarray)
    answer = np.load(current_dir + "/data/h2o_tda_x_coeff_deriv.npy")
    assert np.allclose(x_coeff_deriv_num, answer, atol=1e-5)


def test_h2o_td_xy_coeff():
    mol_name = "h2o_td"
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
    xy_coeff = g_parser.get_xy_coeff()
    assert isinstance(xy_coeff, tuple)

    # np.save(current_dir+'/data/h2o_td_xy_coeff.npy', xy_coeff)
    answer = np.load(current_dir + "/data/h2o_td_xy_coeff.npy")
    assert np.allclose(xy_coeff[0], answer[0])
    assert np.allclose(xy_coeff[1], answer[1])


def test_h2o_td_xy_coeff_deriv_anal():
    mol_name = "h2o_td"
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
    xy_coeff_deriv = g_parser.get_xy_coeff_deriv()
    assert isinstance(xy_coeff_deriv, tuple)

    # np.save(current_dir+'/data/h2o_td_xy_coeff_deriv.npy', xy_coeff_deriv)
    answer = np.load(current_dir + "/data/h2o_td_xy_coeff_deriv.npy")
    assert np.allclose(xy_coeff_deriv[0], answer[0])
    assert np.allclose(xy_coeff_deriv[1], answer[1])


def test_h2_td_xy_coeff_deriv_num():
    mol_name = "h2o_td"
    atoms = ["O", "H", "H"]
    coordinates = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.957200],
        [0.000000, 0.757160, -0.478600],
    ]

    current_dir = os.path.dirname(__file__)
    calc_dir = os.path.join(current_dir, f"data/num/")

    num_deriv = numerical_deriv(
        mol_name, atoms, coordinates, calc_dir=calc_dir, calc_type="td"
    )
    xy_coeff_deriv_num = num_deriv.execute_num_deriv("xy_coeff")
    assert isinstance(xy_coeff_deriv_num, tuple)

    answer = np.load(current_dir + "/data/h2o_td_xy_coeff_deriv.npy")
    assert np.allclose(xy_coeff_deriv_num[0], answer[0], atol=1e-4)
    assert np.allclose(xy_coeff_deriv_num[1], answer[1], atol=1e-4)
