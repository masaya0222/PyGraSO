import os
import numpy as np

from parser import gaussian_perser
from num_deriv import numerical_deriv
from calc_ao_element import calc_ao_element


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

    ao_calculator = calc_ao_element(atoms, coordinates)
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

    ao_calculator = calc_ao_element(atoms, coordinates)
    ao_soc_deriv_anal = ao_calculator.get_ao_soc_deriv()

    assert np.allclose(ao_soc_deriv_anal, ao_soc_deriv_num)
