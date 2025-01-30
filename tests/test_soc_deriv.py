import os
import numpy as np

from pysoc_deriv.parser import gaussian_perser
from pysoc_deriv.num_deriv import numerical_deriv
from pysoc_deriv.calc_ao_element import calc_ao_element
from pysoc_deriv.calc_soc import calc_soc_s0t1_deriv, calc_soc_s1t1_deriv


def test_h2o_ao_soc_deriv():
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
    ao_soc_deriv_num = num_deriv.execute_num_deriv("ao_soc")

    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.rwf")
    g_parser = gaussian_perser(log_file_name, rwf_file_name)

    ao_calculator = calc_ao_element(atoms, coordinates, basis=g_parser.read_basis())
    ao_soc_deriv_anal = ao_calculator.get_ao_soc_deriv()

    assert np.allclose(ao_soc_deriv_anal, ao_soc_deriv_num)


def test_h2o_d_ao_soc_deriv():
    mol_name = "h2o_d_td"
    atoms = ["O", "H", "H"]
    coordinates = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.957200],
        [0.000000, 0.757160, -0.478600],
    ]

    current_dir = os.path.dirname(__file__)

    calc_dir = os.path.join(current_dir, f"data/num/")

    num_deriv = numerical_deriv(
        mol_name,
        atoms,
        coordinates,
        calc_dir=calc_dir,
        calc_type="td",
        basis="6-31G(d)",
    )
    ao_soc_deriv_num = num_deriv.execute_num_deriv("ao_soc")

    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.rwf")
    g_parser = gaussian_perser(log_file_name, rwf_file_name)
    ao_calculator = calc_ao_element(atoms, coordinates, basis=g_parser.read_basis())
    ao_soc_deriv_anal = ao_calculator.get_ao_soc_deriv()
    assert np.allclose(ao_soc_deriv_anal, ao_soc_deriv_num)


def test_ch2o_ao_soc_deriv():
    mol_name = "ch2o_td"
    atoms = ["C", "O", "H", "H"]
    coordinates = [
        [-0.131829, -0.000001, -0.000286],
        [1.065288, 0.000001, 0.000090],
        [-0.718439, 0.939705, 0.000097],
        [-0.718441, -0.939705, 0.000136],
    ]

    current_dir = os.path.dirname(__file__)

    calc_dir = os.path.join(current_dir, f"data/num/")

    num_deriv = numerical_deriv(
        mol_name, atoms, coordinates, calc_dir=calc_dir, calc_type="td"
    )
    ao_soc_deriv_num = num_deriv.execute_num_deriv("ao_soc")

    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.rwf")
    g_parser = gaussian_perser(log_file_name, rwf_file_name)

    ao_calculator = calc_ao_element(atoms, coordinates, basis=g_parser.read_basis())
    ao_soc_deriv_anal = ao_calculator.get_ao_soc_deriv()

    assert np.allclose(ao_soc_deriv_anal, ao_soc_deriv_num)


def test_h2o_td_soc_s0t1_deriv():
    mol_name = "h2o_td"
    atoms = ["O", "H", "H"]
    coordinates = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.957200],
        [0.000000, 0.757160, -0.478600],
    ]
    current_dir = os.path.dirname(__file__)

    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.rwf")

    calc_dir = os.path.join(current_dir, f"data/num/")
    soc_s0t1_deriv_anal = calc_soc_s0t1_deriv(
        atoms, coordinates, log_file_name, rwf_file_name
    )

    num_deriv = numerical_deriv(mol_name, atoms, coordinates, calc_dir=calc_dir)
    soc_s0t1_deriv_num = num_deriv.execute_num_deriv("soc_s0t1")

    assert np.allclose(soc_s0t1_deriv_anal, soc_s0t1_deriv_num, atol=5e-4)


def test_h2o_td_soc_s1t1_deriv():
    mol_name = "h2o_td"
    atoms = ["O", "H", "H"]
    coordinates = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.957200],
        [0.000000, 0.757160, -0.478600],
    ]
    current_dir = os.path.dirname(__file__)

    s1_log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_s1.log")
    s1_rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_s1.rwf")
    t1_log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.log")
    t1_rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.rwf")

    calc_dir = os.path.join(current_dir, f"data/num/")
    soc_s1t1_deriv_anal = calc_soc_s1t1_deriv(
        atoms,
        coordinates,
        s1_log_file_name,
        s1_rwf_file_name,
        t1_log_file_name,
        t1_rwf_file_name,
    )

    num_deriv = numerical_deriv(mol_name, atoms, coordinates, calc_dir=calc_dir)
    soc_s1t1_deriv_num = num_deriv.execute_num_deriv("soc_s1t1")

    assert np.allclose(soc_s1t1_deriv_anal, soc_s1t1_deriv_num, atol=5e-4)


def test_h2o_d_td_soc_s0t1_deriv():
    mol_name = "h2o_d_td"
    atoms = ["O", "H", "H"]
    coordinates = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.957200],
        [0.000000, 0.757160, -0.478600],
    ]
    current_dir = os.path.dirname(__file__)

    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.rwf")

    calc_dir = os.path.join(current_dir, f"data/num/")
    soc_s0t1_deriv_anal = calc_soc_s0t1_deriv(
        atoms, coordinates, log_file_name, rwf_file_name, basis="6-31G(d)"
    )

    num_deriv = numerical_deriv(
        mol_name, atoms, coordinates, calc_dir=calc_dir, basis="6-31G(d)"
    )
    soc_s0t1_deriv_num = num_deriv.execute_num_deriv("soc_s0t1")

    assert np.allclose(soc_s0t1_deriv_anal, soc_s0t1_deriv_num, atol=5e-4)


def test_h2o_d_td_soc_s1t1_deriv():
    mol_name = "h2o_d_td"
    atoms = ["O", "H", "H"]
    coordinates = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.957200],
        [0.000000, 0.757160, -0.478600],
    ]
    current_dir = os.path.dirname(__file__)

    s1_log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_s1.log")
    s1_rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_s1.rwf")
    t1_log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.log")
    t1_rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.rwf")

    calc_dir = os.path.join(current_dir, f"data/num/")
    soc_s1t1_deriv_anal = calc_soc_s1t1_deriv(
        atoms,
        coordinates,
        s1_log_file_name,
        s1_rwf_file_name,
        t1_log_file_name,
        t1_rwf_file_name,
        basis="6-31G(d)",
    )

    num_deriv = numerical_deriv(
        mol_name, atoms, coordinates, calc_dir=calc_dir, basis="6-31G(d)"
    )
    soc_s1t1_deriv_num = num_deriv.execute_num_deriv("soc_s1t1")

    assert np.allclose(soc_s1t1_deriv_anal, soc_s1t1_deriv_num, atol=5e-4)


def test_ch2o_td_soc_s0t1_deriv():
    atoms = ["C", "O", "H", "H"]
    coordinates = [
        [-0.131829, -0.000001, -0.000286],
        [1.065288, 0.000001, 0.000090],
        [-0.718439, 0.939705, 0.000097],
        [-0.718441, -0.939705, 0.000136],
    ]
    current_dir = os.path.dirname(__file__)

    mol_name = "ch2o_td"
    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.rwf")

    calc_dir = os.path.join(current_dir, f"data/num/")
    soc_s0t1_deriv_anal = calc_soc_s0t1_deriv(
        atoms, coordinates, log_file_name, rwf_file_name
    )

    num_deriv = numerical_deriv(mol_name, atoms, coordinates, calc_dir=calc_dir)
    soc_s0t1_deriv_num = num_deriv.execute_num_deriv("soc_s0t1")

    assert np.allclose(soc_s0t1_deriv_anal, soc_s0t1_deriv_num, atol=5e-4)


def test_ch2o_td_soc_s1t1_deriv():
    atoms = ["C", "O", "H", "H"]
    coordinates = [
        [-0.131829, -0.000001, -0.000286],
        [1.065288, 0.000001, 0.000090],
        [-0.718439, 0.939705, 0.000097],
        [-0.718441, -0.939705, 0.000136],
    ]
    current_dir = os.path.dirname(__file__)

    mol_name = "ch2o_td"

    s1_log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_s1.log")
    s1_rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_s1.rwf")
    t1_log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.log")
    t1_rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}_t1.rwf")

    calc_dir = os.path.join(current_dir, f"data/num/")
    soc_s1t1_deriv_anal = calc_soc_s1t1_deriv(
        atoms,
        coordinates,
        s1_log_file_name,
        s1_rwf_file_name,
        t1_log_file_name,
        t1_rwf_file_name,
    )

    num_deriv = numerical_deriv(mol_name, atoms, coordinates, calc_dir=calc_dir)
    soc_s1t1_deriv_num = num_deriv.execute_num_deriv("soc_s1t1")

    assert np.allclose(soc_s1t1_deriv_anal, soc_s1t1_deriv_num, atol=5e-4)
