import sys
import numpy as np
import periodictable

from parser import gaussian_perser
from num_deriv import numerical_deriv
from calc_ao_element import calc_ao_element


def main():
    mol_name = "h2"
    # mol_name = "h2_rpa"
    mol_name = "h2o"
    # mol_name = "ch2o"
    if mol_name == "h2" or mol_name == "h2_rpa":
        atoms = ["H", "H"]
        coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]
    elif mol_name == "h2o":
        atoms = ["O", "H", "H"]
        coordinates = [
            [0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 0.957200],
            [0.000000, 0.757160, -0.478600],
        ]
    elif mol_name == "ch2o":
        atoms = ["C", "O", "H", "H"]
        coordinates = [
            [-0.131829, -0.000001, -0.000286],
            [1.065288, 0.000001, 0.000090],
            [-0.718439, 0.939705, 0.000097],
            [-0.718441, -0.939705, 0.000136],
        ]

    log_file_name = f"{mol_name}.log"
    rwf_file_name = f"{mol_name}.rwf"

    g_parser = gaussian_perser(log_file_name, rwf_file_name)
    mo_coeff = g_parser.get_mo_coeff()
    ao_ovlp, ao_ovlp_deriv = g_parser.get_ao_ovlp_and_deriv()
    mo_coeff_deriv = g_parser.get_mo_coeff_deriv()

    ao_calculator = calc_ao_element(atoms, coordinates)
    ao_dip = ao_calculator.get_ao_dip()

    x_coeff, y_coeff = g_parser.get_xy_coeff()
    x_coeff_deriv, y_coeff_deriv = g_parser.get_xy_coeff_deriv()

    td = g_parser.get_tdip()

    mo_coeff_i = mo_coeff[: g_parser.nfc + g_parser.noa, :]
    mo_coeff_a = mo_coeff[g_parser.nfc + g_parser.noa :, :]
    mo_coeff_deriv_i = mo_coeff_deriv[:, :, : g_parser.nfc + g_parser.noa, :]
    mo_coeff_deriv_a = mo_coeff_deriv[:, :, g_parser.nfc + g_parser.noa :, :]
    tdip = -2.0 * np.einsum(
        "kpq,ia,ip,aq->k", ao_dip, xpy_coeff, mo_coeff_i, mo_coeff_a
    )
    print(tdip)
    print(td)


if __name__ == "__main__":
    main()
