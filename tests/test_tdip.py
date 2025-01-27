import os
import numpy as np

from pysoc_deriv.parser import gaussian_perser
from pysoc_deriv.num_deriv import numerical_deriv
from pysoc_deriv.calc_ao_element import calc_ao_element


def test_h2_tda_tdip():
    mol_name = "h2_tda"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)

    log_file_name = os.path.join(current_dir, f"data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/{mol_name}.rwf")

    g_parser = gaussian_perser(log_file_name, rwf_file_name)
    tdip = g_parser.get_tdip()
    answer = np.array([0.0, 0.0, 1.41789])

    assert np.allclose(tdip, answer)

    mo_coeff = g_parser.get_mo_coeff()
    mo_coeff_i = mo_coeff[: g_parser.nfc + g_parser.noa, :]
    mo_coeff_a = mo_coeff[g_parser.nfc + g_parser.noa :, :]

    x_coeff = g_parser.get_xy_coeff()
    ao_calculator = calc_ao_element(atoms, coordinates)
    ao_dip = ao_calculator.get_ao_dip()

    calc_tdip = -2.0 * np.einsum(
        "kpq,ia,ip,aq->k", ao_dip, x_coeff, mo_coeff_i, mo_coeff_a
    )
    assert np.allclose(calc_tdip, answer)


def test_h2_tda_tdip_deriv_anal():
    mol_name = "h2_tda"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)

    log_file_name = os.path.join(current_dir, f"data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/{mol_name}.rwf")

    g_parser = gaussian_perser(log_file_name, rwf_file_name)

    answer = np.array(
        [
            [
                -0.101394e01,
                0.000000e00,
                0.000000e00,
                0.101394e01,
                0.000000e00,
                0.000000e00,
            ],
            [
                0.000000e00,
                -0.101394e01,
                0.000000e00,
                0.000000e00,
                0.101394e01,
                0.000000e00,
            ],
            [
                0.000000e00,
                0.000000e00,
                -0.605018e00,
                0.000000e00,
                0.000000e00,
                0.605018e00,
            ],
        ]
    ).T
    answer = answer.reshape((len(atoms), 3, 3))

    mo_coeff = g_parser.get_mo_coeff()
    mo_coeff_i = mo_coeff[: g_parser.nfc + g_parser.noa, :]
    mo_coeff_a = mo_coeff[g_parser.nfc + g_parser.noa :, :]

    mo_coeff_deriv = g_parser.get_mo_coeff_deriv()
    mo_coeff_deriv_i = mo_coeff_deriv[:, :, : g_parser.nfc + g_parser.noa, :]
    mo_coeff_deriv_a = mo_coeff_deriv[:, :, g_parser.nfc + g_parser.noa :, :]

    x_coeff = g_parser.get_xy_coeff()
    x_coeff_deriv = g_parser.get_xy_coeff_deriv()

    ao_calculator = calc_ao_element(atoms, coordinates)
    ao_dip = ao_calculator.get_ao_dip()
    ao_dip_deriv = ao_calculator.get_ao_dip_deriv()

    tdip_deriv1 = -2.0 * np.einsum(
        "rdkpq,ia,ip,aq->rdk", ao_dip_deriv, x_coeff, mo_coeff_i, mo_coeff_a
    )
    tdip_deriv2 = -2.0 * (
        np.einsum("kpq,rdia,ip,aq->rdk", ao_dip, x_coeff_deriv, mo_coeff_i, mo_coeff_a)
        + np.einsum(
            "kpq,ia,rdip,aq->rdk", ao_dip, x_coeff, mo_coeff_deriv_i, mo_coeff_a
        )
        + np.einsum(
            "kpq,ia,ip,rdaq->rdk", ao_dip, x_coeff, mo_coeff_i, mo_coeff_deriv_a
        )
    )
    tdip_deriv = tdip_deriv1 + tdip_deriv2
    assert np.allclose(tdip_deriv, answer)


def test_h2_tda_tdip_deriv_num():
    mol_name = "h2_tda"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)
    calc_dir = os.path.join(current_dir, f"data/num/")

    answer = np.array(
        [
            [
                -0.101394e01,
                0.000000e00,
                0.000000e00,
                0.101394e01,
                0.000000e00,
                0.000000e00,
            ],
            [
                0.000000e00,
                -0.101394e01,
                0.000000e00,
                0.000000e00,
                0.101394e01,
                0.000000e00,
            ],
            [
                0.000000e00,
                0.000000e00,
                -0.605018e00,
                0.000000e00,
                0.000000e00,
                0.605018e00,
            ],
        ]
    ).T
    answer = answer.reshape((len(atoms), 3, 3))

    num_deriv = numerical_deriv(
        mol_name, atoms, coordinates, calc_dir=calc_dir, calc_type="tda", step_size=1e-2
    )
    tdip_deriv_num = num_deriv.execute_num_deriv("tdip")
    assert np.allclose(tdip_deriv_num, answer, atol=2e-4)


def test_h2_td_tdip():
    mol_name = "h2_td"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)

    log_file_name = os.path.join(current_dir, f"data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/{mol_name}.rwf")

    g_parser = gaussian_perser(log_file_name, rwf_file_name)
    tdip = g_parser.get_tdip()
    answer = np.array([0.0, 0.0, 1.31032])

    assert np.allclose(tdip, answer)

    mo_coeff = g_parser.get_mo_coeff()
    mo_coeff_i = mo_coeff[: g_parser.nfc + g_parser.noa, :]
    mo_coeff_a = mo_coeff[g_parser.nfc + g_parser.noa :, :]

    x_coeff, y_coeff = g_parser.get_xy_coeff()
    xpy_coeff = x_coeff + y_coeff
    ao_calculator = calc_ao_element(atoms, coordinates)
    ao_dip = ao_calculator.get_ao_dip()

    calc_tdip = -2.0 * np.einsum(
        "kpq,ia,ip,aq->k", ao_dip, xpy_coeff, mo_coeff_i, mo_coeff_a
    )
    assert np.allclose(calc_tdip, answer)


def test_h2_td_tdip_deriv_anal():
    mol_name = "h2_td"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)

    log_file_name = os.path.join(current_dir, f"data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/{mol_name}.rwf")

    g_parser = gaussian_perser(log_file_name, rwf_file_name)

    answer = np.array(
        [
            [
                -0.937014e00,
                0.000000e00,
                0.000000e00,
                0.937014e00,
                0.000000e00,
                0.000000e00,
            ],
            [
                0.000000e00,
                -0.937014e00,
                0.000000e00,
                0.000000e00,
                0.937014e00,
                0.000000e00,
            ],
            [
                0.000000e00,
                0.000000e00,
                -0.451372e00,
                0.000000e00,
                0.000000e00,
                0.451372e00,
            ],
        ]
    ).T
    answer = answer.reshape((len(atoms), 3, 3))

    mo_coeff = g_parser.get_mo_coeff()
    mo_coeff_i = mo_coeff[: g_parser.nfc + g_parser.noa, :]
    mo_coeff_a = mo_coeff[g_parser.nfc + g_parser.noa :, :]

    mo_coeff_deriv = g_parser.get_mo_coeff_deriv()
    mo_coeff_deriv_i = mo_coeff_deriv[:, :, : g_parser.nfc + g_parser.noa, :]
    mo_coeff_deriv_a = mo_coeff_deriv[:, :, g_parser.nfc + g_parser.noa :, :]

    x_coeff, y_coeff = g_parser.get_xy_coeff()
    xpy_coeff = x_coeff + y_coeff

    x_coeff_deriv, y_coeff_deriv = g_parser.get_xy_coeff_deriv()
    xpy_coeff_deriv = x_coeff_deriv + y_coeff_deriv

    ao_calculator = calc_ao_element(atoms, coordinates)
    ao_dip = ao_calculator.get_ao_dip()
    ao_dip_deriv = ao_calculator.get_ao_dip_deriv()

    tdip_deriv1 = -2.0 * np.einsum(
        "rdkpq,ia,ip,aq->rdk", ao_dip_deriv, xpy_coeff, mo_coeff_i, mo_coeff_a
    )
    tdip_deriv2 = -2.0 * (
        np.einsum(
            "kpq,rdia,ip,aq->rdk", ao_dip, xpy_coeff_deriv, mo_coeff_i, mo_coeff_a
        )
        + np.einsum(
            "kpq,ia,rdip,aq->rdk", ao_dip, xpy_coeff, mo_coeff_deriv_i, mo_coeff_a
        )
        + np.einsum(
            "kpq,ia,ip,rdaq->rdk", ao_dip, xpy_coeff, mo_coeff_i, mo_coeff_deriv_a
        )
    )
    tdip_deriv = tdip_deriv1 + tdip_deriv2
    assert np.allclose(tdip_deriv, answer)


def test_h2_td_tdip_deriv_num():
    mol_name = "h2_td"
    atoms = ["H", "H"]
    coordinates = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.740000]]

    current_dir = os.path.dirname(__file__)
    calc_dir = os.path.join(current_dir, f"data/num/")

    answer = np.array(
        [
            [
                -0.937014e00,
                0.000000e00,
                0.000000e00,
                0.937014e00,
                0.000000e00,
                0.000000e00,
            ],
            [
                0.000000e00,
                -0.937014e00,
                0.000000e00,
                0.000000e00,
                0.937014e00,
                0.000000e00,
            ],
            [
                0.000000e00,
                0.000000e00,
                -0.451372e00,
                0.000000e00,
                0.000000e00,
                0.451372e00,
            ],
        ]
    ).T
    answer = answer.reshape((len(atoms), 3, 3))

    num_deriv = numerical_deriv(
        mol_name, atoms, coordinates, calc_dir=calc_dir, calc_type="td", step_size=1e-2
    )
    tdip_deriv_num = num_deriv.execute_num_deriv("tdip")
    assert np.allclose(tdip_deriv_num, answer, atol=2e-4)


def test_h2o_tda_tdip():
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
    tdip = g_parser.get_tdip()
    answer = np.array(
        [
            0.243568,
            0.0,
            0.0,
        ]
    )

    assert np.allclose(tdip, answer)

    mo_coeff = g_parser.get_mo_coeff()
    mo_coeff_i = mo_coeff[: g_parser.nfc + g_parser.noa, :]
    mo_coeff_a = mo_coeff[g_parser.nfc + g_parser.noa :, :]

    x_coeff = g_parser.get_xy_coeff()
    ao_calculator = calc_ao_element(atoms, coordinates)
    ao_dip = ao_calculator.get_ao_dip()

    calc_tdip = -2.0 * np.einsum(
        "kpq,ia,ip,aq->k", ao_dip, x_coeff, mo_coeff_i, mo_coeff_a
    )
    assert np.allclose(calc_tdip, answer)


def test_h2o_tda_tdip_deriv_anal():
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

    answer = np.array(
        [
            [
                0.000000e00,
                -0.174157e-01,
                0.182933e00,
                0.000000e00,
                0.630176e-01,
                -0.291425e00,
                0.000000e00,
                -0.456019e-01,
                0.108492e00,
            ],
            [
                0.255356e00,
                0.000000e00,
                0.000000e00,
                -0.851181e-01,
                0.000000e00,
                0.000000e00,
                -0.170238e00,
                0.000000e00,
                0.000000e00,
            ],
            [
                0.134656e00,
                0.000000e00,
                0.000000e00,
                -0.134660e00,
                0.000000e00,
                0.000000e00,
                0.351948e-05,
                0.000000e00,
                0.000000e00,
            ],
        ]
    ).T
    answer = answer.reshape((len(atoms), 3, 3))

    mo_coeff = g_parser.get_mo_coeff()
    mo_coeff_i = mo_coeff[: g_parser.nfc + g_parser.noa, :]
    mo_coeff_a = mo_coeff[g_parser.nfc + g_parser.noa :, :]

    mo_coeff_deriv = g_parser.get_mo_coeff_deriv()
    mo_coeff_deriv_i = mo_coeff_deriv[:, :, : g_parser.nfc + g_parser.noa, :]
    mo_coeff_deriv_a = mo_coeff_deriv[:, :, g_parser.nfc + g_parser.noa :, :]

    x_coeff = g_parser.get_xy_coeff()
    x_coeff_deriv = g_parser.get_xy_coeff_deriv()

    ao_calculator = calc_ao_element(atoms, coordinates)
    ao_dip = ao_calculator.get_ao_dip()
    ao_dip_deriv = ao_calculator.get_ao_dip_deriv()

    tdip_deriv1 = -2.0 * np.einsum(
        "rdkpq,ia,ip,aq->rdk", ao_dip_deriv, x_coeff, mo_coeff_i, mo_coeff_a
    )
    tdip_deriv2 = -2.0 * (
        np.einsum("kpq,rdia,ip,aq->rdk", ao_dip, x_coeff_deriv, mo_coeff_i, mo_coeff_a)
        + np.einsum(
            "kpq,ia,rdip,aq->rdk", ao_dip, x_coeff, mo_coeff_deriv_i, mo_coeff_a
        )
        + np.einsum(
            "kpq,ia,ip,rdaq->rdk", ao_dip, x_coeff, mo_coeff_i, mo_coeff_deriv_a
        )
    )
    tdip_deriv = tdip_deriv1 + tdip_deriv2

    assert np.allclose(tdip_deriv, answer, atol=1e-6)


def test_h2o_tda_tdip_deriv_num():
    mol_name = "h2o_tda"
    atoms = ["O", "H", "H"]
    coordinates = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.957200],
        [0.000000, 0.757160, -0.478600],
    ]

    current_dir = os.path.dirname(__file__)
    calc_dir = os.path.join(current_dir, f"data/num/")

    answer = np.array(
        [
            [
                0.000000e00,
                -0.174157e-01,
                0.182933e00,
                0.000000e00,
                0.630176e-01,
                -0.291425e00,
                0.000000e00,
                -0.456019e-01,
                0.108492e00,
            ],
            [
                0.255356e00,
                0.000000e00,
                0.000000e00,
                -0.851181e-01,
                0.000000e00,
                0.000000e00,
                -0.170238e00,
                0.000000e00,
                0.000000e00,
            ],
            [
                0.134656e00,
                0.000000e00,
                0.000000e00,
                -0.134660e00,
                0.000000e00,
                0.000000e00,
                0.351948e-05,
                0.000000e00,
                0.000000e00,
            ],
        ]
    ).T
    answer = answer.reshape((len(atoms), 3, 3))

    num_deriv = numerical_deriv(
        mol_name, atoms, coordinates, calc_dir=calc_dir, calc_type="tda", step_size=1e-2
    )
    tdip_deriv_num = num_deriv.execute_num_deriv("tdip")
    assert np.allclose(tdip_deriv_num, answer, atol=2e-4)


def test_h2o_td_tdip():
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
    tdip = g_parser.get_tdip()
    answer = np.array([0.23864, 0.0, 0.0])

    assert np.allclose(tdip, answer)

    mo_coeff = g_parser.get_mo_coeff()
    mo_coeff_i = mo_coeff[: g_parser.nfc + g_parser.noa, :]
    mo_coeff_a = mo_coeff[g_parser.nfc + g_parser.noa :, :]

    x_coeff, y_coeff = g_parser.get_xy_coeff()
    xpy_coeff = x_coeff + y_coeff
    ao_calculator = calc_ao_element(atoms, coordinates)
    ao_dip = ao_calculator.get_ao_dip()

    calc_tdip = -2.0 * np.einsum(
        "kpq,ia,ip,aq->k", ao_dip, xpy_coeff, mo_coeff_i, mo_coeff_a
    )
    assert np.allclose(calc_tdip, answer)


def test_h2o_td_tdip_deriv_anal():
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
    answer = np.array(
        [
            [
                0.000000e00,
                -0.136298e-01,
                0.180833e00,
                0.000000e00,
                0.618784e-01,
                -0.289558e00,
                0.000000e00,
                -0.482486e-01,
                0.108725e00,
            ],
            [
                0.250191e00,
                0.000000e00,
                0.000000e00,
                -0.833964e-01,
                0.000000e00,
                0.000000e00,
                -0.166795e00,
                0.000000e00,
                0.000000e00,
            ],
            [
                0.131931e00,
                0.000000e00,
                0.000000e00,
                -0.131935e00,
                0.000000e00,
                0.000000e00,
                0.379981e-05,
                0.000000e00,
                0.000000e00,
            ],
        ]
    ).T
    answer = answer.reshape((len(atoms), 3, 3))

    mo_coeff = g_parser.get_mo_coeff()
    mo_coeff_i = mo_coeff[: g_parser.nfc + g_parser.noa, :]
    mo_coeff_a = mo_coeff[g_parser.nfc + g_parser.noa :, :]

    mo_coeff_deriv = g_parser.get_mo_coeff_deriv()
    mo_coeff_deriv_i = mo_coeff_deriv[:, :, : g_parser.nfc + g_parser.noa, :]
    mo_coeff_deriv_a = mo_coeff_deriv[:, :, g_parser.nfc + g_parser.noa :, :]

    x_coeff, y_coeff = g_parser.get_xy_coeff()
    xpy_coeff = x_coeff + y_coeff

    x_coeff_deriv, y_coeff_deriv = g_parser.get_xy_coeff_deriv()
    xpy_coeff_deriv = x_coeff_deriv + y_coeff_deriv

    ao_calculator = calc_ao_element(atoms, coordinates)
    ao_dip = ao_calculator.get_ao_dip()
    ao_dip_deriv = ao_calculator.get_ao_dip_deriv()

    tdip_deriv1 = -2.0 * np.einsum(
        "rdkpq,ia,ip,aq->rdk", ao_dip_deriv, xpy_coeff, mo_coeff_i, mo_coeff_a
    )
    tdip_deriv2 = -2.0 * (
        np.einsum(
            "kpq,rdia,ip,aq->rdk", ao_dip, xpy_coeff_deriv, mo_coeff_i, mo_coeff_a
        )
        + np.einsum(
            "kpq,ia,rdip,aq->rdk", ao_dip, xpy_coeff, mo_coeff_deriv_i, mo_coeff_a
        )
        + np.einsum(
            "kpq,ia,ip,rdaq->rdk", ao_dip, xpy_coeff, mo_coeff_i, mo_coeff_deriv_a
        )
    )
    tdip_deriv = tdip_deriv1 + tdip_deriv2
    assert np.allclose(tdip_deriv, answer, atol=1e-6)


def test_h2_td_tdip_deriv_num():
    mol_name = "h2o_td"
    atoms = ["O", "H", "H"]
    coordinates = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.957200],
        [0.000000, 0.757160, -0.478600],
    ]

    current_dir = os.path.dirname(__file__)
    calc_dir = os.path.join(current_dir, f"data/num/")

    answer = np.array(
        [
            [
                0.000000e00,
                -0.136298e-01,
                0.180833e00,
                0.000000e00,
                0.618784e-01,
                -0.289558e00,
                0.000000e00,
                -0.482486e-01,
                0.108725e00,
            ],
            [
                0.250191e00,
                0.000000e00,
                0.000000e00,
                -0.833964e-01,
                0.000000e00,
                0.000000e00,
                -0.166795e00,
                0.000000e00,
                0.000000e00,
            ],
            [
                0.131931e00,
                0.000000e00,
                0.000000e00,
                -0.131935e00,
                0.000000e00,
                0.000000e00,
                0.379981e-05,
                0.000000e00,
                0.000000e00,
            ],
        ]
    ).T
    answer = answer.reshape((len(atoms), 3, 3))

    num_deriv = numerical_deriv(
        mol_name, atoms, coordinates, calc_dir=calc_dir, calc_type="td", step_size=1e-2
    )
    tdip_deriv_num = num_deriv.execute_num_deriv("tdip")
    assert np.allclose(tdip_deriv_num, answer, atol=2e-4)
