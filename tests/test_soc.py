import os
import numpy as np

from pysoc_deriv.parser import gaussian_perser
from pysoc_deriv.num_deriv import numerical_deriv
from pysoc_deriv.calc_ao_element import calc_ao_element


def test_h2o_td_soc():
    atoms = ["O", "H", "H"]
    coordinates = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.957200],
        [0.000000, 0.757160, -0.478600],
    ]

    current_dir = os.path.dirname(__file__)
    mol_name = "h2o_td_s1"
    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.rwf")

    g_parser_s1 = gaussian_perser(log_file_name, rwf_file_name)

    mol_name = "h2o_td_t1"
    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.rwf")

    g_parser_t1 = gaussian_perser(log_file_name, rwf_file_name)

    mo_coeff = g_parser_s1.get_mo_coeff()

    ao_calculator = calc_ao_element(atoms, coordinates, basis=g_parser_t1.read_basis())
    ao_soc = ao_calculator.get_ao_soc()
    mo_soc = np.einsum("kpq,ip,jq->kij", ao_soc, mo_coeff, mo_coeff)

    x_coeff_s1, y_coeff_s1 = g_parser_s1.get_xy_coeff()
    xpy_coeff_s1 = x_coeff_s1 + y_coeff_s1
    norm_s1 = np.sqrt(np.trace(xpy_coeff_s1 @ xpy_coeff_s1.T) * 2.0)
    xpy_coeff_s1 = xpy_coeff_s1 / norm_s1

    x_coeff_t1, y_coeff_t1 = g_parser_t1.get_xy_coeff()
    xpy_coeff_t1 = x_coeff_t1 + y_coeff_t1
    norm_t1 = np.sqrt(np.trace(xpy_coeff_t1 @ xpy_coeff_t1.T) * 2.0)
    xpy_coeff_t1 = xpy_coeff_t1 / norm_t1

    fine_stru = 7.297352568e-3
    mo_soc *= 0.5 * fine_stru**2
    mo_soc_ij = mo_soc[
        :, : g_parser_t1.nfc + g_parser_t1.noa, : g_parser_t1.nfc + g_parser_t1.noa
    ]
    mo_soc_ia = mo_soc[
        :, : g_parser_t1.nfc + g_parser_t1.noa, g_parser_t1.nfc + g_parser_t1.noa :
    ]
    mo_soc_ab = mo_soc[
        :, g_parser_t1.nfc + g_parser_t1.noa :, g_parser_t1.nfc + g_parser_t1.noa :
    ]

    au2wavnum = 219474.6
    soc_s0t1_1 = (1.0 / np.sqrt(2)) * complex(
        -np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[0, :, :]),
        -np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[1, :, :]),
    )
    soc_s0t1_2 = complex(np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[2, :, :]), 0.0)
    soc_s0t1_3 = -1.0 * soc_s0t1_1
    soc_s0t1 = np.array([soc_s0t1_1, soc_s0t1_2, soc_s0t1_3])
    soc_s0t1 *= au2wavnum
    answer = np.array(
        [-0.00000 + 25.48577j, 37.02488 + 0.00000j, 0.00000 - 25.48577j]
    )  # from pysoc

    assert np.allclose(soc_s0t1, answer, atol=1e-6, rtol=5e-3)

    soc_s1t1_1 = complex(
        1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[0, :, :])
        - 1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[0, :, :]),
        1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[1, :, :])
        - 1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[1, :, :]),
    )

    soc_s1t1_2 = complex(
        -1.0 * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[2, :, :])
        + 1.0 * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[2, :, :]),
        0.0,
    )
    soc_s1t1_3 = -1.0 * soc_s1t1_1
    soc_s1t1 = np.array([soc_s1t1_1, soc_s1t1_2, soc_s1t1_3])
    soc_s1t1 *= au2wavnum
    answer = np.array(
        [0.01834 + 0.00000j, 0.00000 + 0.00000j, -0.01834 - 0.00000j]
    )  # from pysoc

    assert np.allclose(soc_s1t1, answer, atol=1e-6, rtol=5e-3)


def test_h2o_td_soc_molsoc():
    atoms = ["O", "H", "H"]
    coordinates = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.957200],
        [0.000000, 0.757160, -0.478600],
    ]

    current_dir = os.path.dirname(__file__)
    mol_name = "h2o_td_s1"
    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.rwf")

    g_parser_s1 = gaussian_perser(log_file_name, rwf_file_name)

    mol_name = "h2o_td_t1"
    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.rwf")

    g_parser_t1 = gaussian_perser(log_file_name, rwf_file_name)

    mo_coeff = g_parser_s1.get_mo_coeff()

    ao_soc0_tmp = """
    0.00000   0.00000   0.00000   0.18040  -0.44666   0.00000   0.00000   0.02007  -0.04969  -0.01980  -0.01128   0.02121   0.00990
    0.00000   0.00000   0.00000   0.08223  -0.20049   0.00000   0.00000   0.07316  -0.16938  -0.04665  -0.04079   0.05000   0.03727
    0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    -0.18040  -0.08223   0.00000   0.00000  65.81387  -0.03098   0.00000  -0.07259   7.59285   3.61633   1.84331  -2.29645  -1.00467
    0.44666   0.20049   0.00000 -65.81387   0.00000   0.07279   0.00000  -7.52243   0.07259  -0.01699   0.02810  -3.66122  -1.53961
    0.00000   0.00000   0.00000   0.03098  -0.07279   0.00000   0.00000   0.04823  -0.09553  -0.02344  -0.02694   0.02437   0.02619
    0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    -0.02007  -0.07316   0.00000   0.07259   7.52243  -0.04823   0.00000   0.00000   2.43796   1.35614   0.76508  -0.81907  -0.43023
    0.04969   0.16938   0.00000  -7.59285  -0.07259   0.09553   0.00000  -2.43796   0.00000   0.00467   0.03011  -1.28034  -0.62049
    0.01980   0.04665   0.00000  -3.61633   0.01699   0.02344   0.00000  -1.35614  -0.00467   0.00000   0.00000  -0.29950  -0.26229
    0.01128   0.04079   0.00000  -1.84331  -0.02810   0.02694   0.00000  -0.76508  -0.03011   0.00000   0.00000  -0.29288  -0.18804
    -0.02121  -0.05000   0.00000   2.29645   3.66122  -0.02437   0.00000   0.81907   1.28034   0.29950   0.29288   0.00000   0.00000
    -0.00990  -0.03727   0.00000   1.00467   1.53961  -0.02619   0.00000   0.43023   0.62049   0.26229   0.18804   0.00000   0.00000
    """
    ao_soc0 = np.array(
        [float(ao_soc0_t) for ao_soc0_t in ao_soc0_tmp.strip().split()]
    ).reshape(g_parser_s1.nbasis, g_parser_s1.nbasis)

    ao_soc1_tmp = """
    0.00000   0.00000  -0.18040   0.00000   0.00000   0.00000  -0.02007   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000  -0.08223   0.00000   0.00000   0.00000  -0.07316   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.18040   0.08223   0.00000  -0.13670 -66.03013   0.03098   0.00000  -0.02186  -7.74227  -3.64228  -1.87736   2.34596   1.01344
    0.00000   0.00000   0.13670   0.00000   0.00000   0.00000   0.09445   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000  66.03013   0.00000   0.00000   0.00000   7.55702   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000  -0.03098   0.00000   0.00000   0.00000  -0.04823   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.02007   0.07316   0.00000  -0.09445  -7.55702   0.04823   0.00000  -0.04397  -2.50752  -1.36376  -0.78353   0.83991   0.43824
    0.00000   0.00000   0.02186   0.00000   0.00000   0.00000   0.04397   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000   7.74227   0.00000   0.00000   0.00000   2.50752   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000   3.64228   0.00000   0.00000   0.00000   1.36376   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000   1.87736   0.00000   0.00000   0.00000   0.78353   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000  -2.34596   0.00000   0.00000   0.00000  -0.83991   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000  -1.01344   0.00000   0.00000   0.00000  -0.43824   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    """
    ao_soc1 = np.array(
        [float(ao_soc1_t) for ao_soc1_t in ao_soc1_tmp.strip().split()]
    ).reshape(g_parser_s1.nbasis, g_parser_s1.nbasis)

    ao_soc2_tmp = """
    0.00000   0.00000   0.44666   0.00000   0.00000   0.00000   0.04969   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000   0.20049   0.00000   0.00000   0.00000   0.16938   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    -0.44666  -0.20049   0.00000  66.15097   0.13670  -0.07279   0.00000   7.76395   0.02186  -0.02300  -0.01845   3.66205   1.56986
    0.00000   0.00000 -66.15097   0.00000   0.00000   0.00000  -7.64911   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000  -0.13670   0.00000   0.00000   0.00000  -0.09445   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000   0.07279   0.00000   0.00000   0.00000   0.09553   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    -0.04969  -0.16938   0.00000   7.64911   0.09445  -0.09553   0.00000   2.56254   0.04397  -0.02494  -0.02866   1.27689   0.63757
    0.00000   0.00000  -7.76395   0.00000   0.00000   0.00000  -2.56254   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000  -0.02186   0.00000   0.00000   0.00000  -0.04397   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000   0.02300   0.00000   0.00000   0.00000   0.02494   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000   0.01845   0.00000   0.00000   0.00000   0.02866   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000  -3.66205   0.00000   0.00000   0.00000  -1.27689   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    0.00000   0.00000  -1.56986   0.00000   0.00000   0.00000  -0.63757   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
    """
    ao_soc2 = np.array(
        [float(ao_soc2_t) for ao_soc2_t in ao_soc2_tmp.strip().split()]
    ).reshape(g_parser_s1.nbasis, g_parser_s1.nbasis)
    ao_soc = np.array([ao_soc0, ao_soc1, ao_soc2])

    mo_soc = np.einsum("kpq,ip,jq->kij", ao_soc, mo_coeff, mo_coeff)

    coeff_thresh = 1e-5

    x_coeff_s1, y_coeff_s1 = g_parser_s1.get_xy_coeff()
    xpy_coeff_s1 = x_coeff_s1 + y_coeff_s1
    norm_s1 = np.sqrt(np.trace(xpy_coeff_s1 @ xpy_coeff_s1.T) * 2.0)
    xpy_coeff_s1 = xpy_coeff_s1 / norm_s1
    xpy_coeff_s1[np.abs(xpy_coeff_s1) < coeff_thresh] = 0.0  # pysoc treatment

    x_coeff_t1, y_coeff_t1 = g_parser_t1.get_xy_coeff()
    xpy_coeff_t1 = x_coeff_t1 + y_coeff_t1
    norm_t1 = np.sqrt(np.trace(xpy_coeff_t1 @ xpy_coeff_t1.T) * 2.0)
    xpy_coeff_t1 = xpy_coeff_t1 / norm_t1
    xpy_coeff_t1[np.abs(xpy_coeff_t1) < coeff_thresh] = 0.0  # pysoc treatment

    fine_stru = 7.297352568e-3
    mo_soc *= 0.5 * fine_stru**2
    mo_soc_ij = mo_soc[
        :, : g_parser_t1.nfc + g_parser_t1.noa, : g_parser_t1.nfc + g_parser_t1.noa
    ]
    mo_soc_ia = mo_soc[
        :, : g_parser_t1.nfc + g_parser_t1.noa, g_parser_t1.nfc + g_parser_t1.noa :
    ]
    mo_soc_ab = mo_soc[
        :, g_parser_t1.nfc + g_parser_t1.noa :, g_parser_t1.nfc + g_parser_t1.noa :
    ]

    au2wavnum = 219474.6
    soc_s0t1_1 = (1.0 / np.sqrt(2)) * complex(
        -np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[0, :, :]),
        -np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[1, :, :]),
    )
    soc_s0t1_2 = complex(np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[2, :, :]), 0.0)
    soc_s0t1_3 = -1.0 * soc_s0t1_1
    soc_s0t1 = np.array([soc_s0t1_1, soc_s0t1_2, soc_s0t1_3])
    soc_s0t1 *= au2wavnum
    answer = np.array(
        [-0.00000 + 25.48577j, 37.02488 + 0.00000j, 0.00000 - 25.48577j]
    )  # from pysoc

    assert np.allclose(soc_s0t1, answer, atol=1e-6, rtol=1e-4)

    soc_s1t1_1 = complex(
        1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[0, :, :])
        - 1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[0, :, :]),
        1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[1, :, :])
        - 1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[1, :, :]),
    )

    soc_s1t1_2 = complex(
        -1.0 * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[2, :, :])
        + 1.0 * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[2, :, :]),
        0.0,
    )
    soc_s1t1_3 = -1.0 * soc_s1t1_1
    soc_s1t1 = np.array([soc_s1t1_1, soc_s1t1_2, soc_s1t1_3])
    soc_s1t1 *= au2wavnum
    answer = np.array(
        [0.01834 + 0.00000j, 0.00000 + 0.00000j, -0.01834 - 0.00000j]
    )  # from pysoc

    assert np.allclose(soc_s1t1, answer, atol=1e-6, rtol=1e-3)


def test_h2o_d_td_soc():
    atoms = ["O", "H", "H"]
    coordinates = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.957200],
        [0.000000, 0.757160, -0.478600],
    ]

    current_dir = os.path.dirname(__file__)
    mol_name = "h2o_d_td_s1"
    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.rwf")

    g_parser_s1 = gaussian_perser(log_file_name, rwf_file_name)

    mol_name = "h2o_d_td_t1"
    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.rwf")

    g_parser_t1 = gaussian_perser(log_file_name, rwf_file_name)

    mo_coeff = g_parser_s1.get_mo_coeff()

    ao_calculator = calc_ao_element(atoms, coordinates, basis=g_parser_t1.read_basis())
    ao_soc = ao_calculator.get_ao_soc()
    mo_soc = np.einsum("kpq,ip,jq->kij", ao_soc, mo_coeff, mo_coeff)

    x_coeff_s1, y_coeff_s1 = g_parser_s1.get_xy_coeff()
    xpy_coeff_s1 = x_coeff_s1 + y_coeff_s1
    norm_s1 = np.sqrt(np.trace(xpy_coeff_s1 @ xpy_coeff_s1.T) * 2.0)
    xpy_coeff_s1 = xpy_coeff_s1 / norm_s1

    x_coeff_t1, y_coeff_t1 = g_parser_t1.get_xy_coeff()
    xpy_coeff_t1 = x_coeff_t1 + y_coeff_t1
    norm_t1 = np.sqrt(np.trace(xpy_coeff_t1 @ xpy_coeff_t1.T) * 2.0)
    xpy_coeff_t1 = xpy_coeff_t1 / norm_t1

    fine_stru = 7.297352568e-3
    mo_soc *= 0.5 * fine_stru**2
    mo_soc_ij = mo_soc[
        :, : g_parser_t1.nfc + g_parser_t1.noa, : g_parser_t1.nfc + g_parser_t1.noa
    ]
    mo_soc_ia = mo_soc[
        :, : g_parser_t1.nfc + g_parser_t1.noa, g_parser_t1.nfc + g_parser_t1.noa :
    ]
    mo_soc_ab = mo_soc[
        :, g_parser_t1.nfc + g_parser_t1.noa :, g_parser_t1.nfc + g_parser_t1.noa :
    ]

    au2wavnum = 219474.6
    soc_s0t1_1 = (1.0 / np.sqrt(2)) * complex(
        -np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[0, :, :]),
        -np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[1, :, :]),
    )
    soc_s0t1_2 = complex(np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[2, :, :]), 0.0)
    soc_s0t1_3 = -1.0 * soc_s0t1_1
    soc_s0t1 = np.array([soc_s0t1_1, soc_s0t1_2, soc_s0t1_3])
    soc_s0t1 *= au2wavnum
    answer = np.array(
        [0.00000 + 25.15476j, 35.63319 + 0.00000j, 0.00000 - 25.15476j]
    )  # from pysoc

    assert np.allclose(soc_s0t1, answer, atol=1e-6, rtol=5e-3)

    soc_s1t1_1 = complex(
        1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[0, :, :])
        - 1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[0, :, :]),
        1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[1, :, :])
        - 1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[1, :, :]),
    )

    soc_s1t1_2 = complex(
        -1.0 * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[2, :, :])
        + 1.0 * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[2, :, :]),
        0.0,
    )
    soc_s1t1_3 = -1.0 * soc_s1t1_1
    soc_s1t1 = np.array([soc_s1t1_1, soc_s1t1_2, soc_s1t1_3])
    soc_s1t1 *= au2wavnum
    answer = np.array(
        [0.03615 + 0.00000j, 0.00000 + 0.00000j, -0.03615 - 0.00000j]
    )  # from pysoc

    assert np.allclose(soc_s1t1, answer, atol=1e-6, rtol=5e-3)


def test_ch2o_td_soc():
    atoms = ["C", "O", "H", "H"]
    coordinates = [
        [-0.131829, -0.000001, -0.000286],
        [1.065288, 0.000001, 0.000090],
        [-0.718439, 0.939705, 0.000097],
        [-0.718441, -0.939705, 0.000136],
    ]

    current_dir = os.path.dirname(__file__)
    mol_name = "ch2o_td_s1"
    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.rwf")

    g_parser_s1 = gaussian_perser(log_file_name, rwf_file_name)

    mol_name = "ch2o_td_t1"
    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.rwf")

    g_parser_t1 = gaussian_perser(log_file_name, rwf_file_name)

    mo_coeff = g_parser_s1.get_mo_coeff()

    ao_calculator = calc_ao_element(atoms, coordinates, basis=g_parser_t1.read_basis())
    ao_soc = ao_calculator.get_ao_soc()
    mo_soc = np.einsum("kpq,ip,jq->kij", ao_soc, mo_coeff, mo_coeff)

    x_coeff_s1, y_coeff_s1 = g_parser_s1.get_xy_coeff()
    xpy_coeff_s1 = x_coeff_s1 + y_coeff_s1
    norm_s1 = np.sqrt(np.trace(xpy_coeff_s1 @ xpy_coeff_s1.T) * 2.0)
    xpy_coeff_s1 = xpy_coeff_s1 / norm_s1

    x_coeff_t1, y_coeff_t1 = g_parser_t1.get_xy_coeff()
    xpy_coeff_t1 = x_coeff_t1 + y_coeff_t1
    norm_t1 = np.sqrt(np.trace(xpy_coeff_t1 @ xpy_coeff_t1.T) * 2.0)
    xpy_coeff_t1 = xpy_coeff_t1 / norm_t1

    fine_stru = 7.297352568e-3
    mo_soc *= 0.5 * fine_stru**2
    mo_soc_ij = mo_soc[
        :, : g_parser_t1.nfc + g_parser_t1.noa, : g_parser_t1.nfc + g_parser_t1.noa
    ]
    mo_soc_ia = mo_soc[
        :, : g_parser_t1.nfc + g_parser_t1.noa, g_parser_t1.nfc + g_parser_t1.noa :
    ]
    mo_soc_ab = mo_soc[
        :, g_parser_t1.nfc + g_parser_t1.noa :, g_parser_t1.nfc + g_parser_t1.noa :
    ]

    au2wavnum = 219474.6
    soc_s0t1_1 = (1.0 / np.sqrt(2)) * complex(
        -np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[0, :, :]),
        -np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[1, :, :]),
    )
    soc_s0t1_2 = complex(np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[2, :, :]), 0.0)
    soc_s0t1_3 = -1.0 * soc_s0t1_1
    soc_s0t1 = np.array([soc_s0t1_1, soc_s0t1_2, soc_s0t1_3])
    soc_s0t1 *= au2wavnum
    answer = np.array(
        [68.05414 + 0.00024j, -0.01202 + 0.00000, -68.05414 - 0.00024j]
    )  # from pysoc

    assert np.allclose(soc_s0t1, answer, atol=1e-3)

    soc_s1t1_1 = complex(
        1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[0, :, :])
        - 1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[0, :, :]),
        1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[1, :, :])
        - 1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[1, :, :]),
    )

    soc_s1t1_2 = complex(
        -1.0 * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[2, :, :])
        + 1.0 * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[2, :, :]),
        0.0,
    )
    soc_s1t1_3 = -1.0 * soc_s1t1_1
    soc_s1t1 = np.array([soc_s1t1_1, soc_s1t1_2, soc_s1t1_3])
    soc_s1t1 *= au2wavnum
    answer = np.array(
        [0.00000 - 0.00161j, 0.00000 + 0.00000j, -0.00000 + 0.00161j]
    )  # from pysoc

    assert np.allclose(soc_s1t1, answer, atol=3e-4)
    # assert np.allclose(soc_s1t1, answer, atol=1e-6, rtol=1e-3)


def test_ch2o_td_soc_molsoc():
    atoms = ["C", "O", "H", "H"]
    coordinates = [
        [-0.131829, -0.000001, -0.000286],
        [1.065288, 0.000001, 0.000090],
        [-0.718439, 0.939705, 0.000097],
        [-0.718441, -0.939705, 0.000136],
    ]

    current_dir = os.path.dirname(__file__)
    mol_name = "ch2o_td_s1"
    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.rwf")

    g_parser_s1 = gaussian_perser(log_file_name, rwf_file_name)

    mol_name = "ch2o_td_t1"
    log_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.log")
    rwf_file_name = os.path.join(current_dir, f"data/soc_data/{mol_name}.rwf")

    g_parser_t1 = gaussian_perser(log_file_name, rwf_file_name)

    mo_coeff = g_parser_s1.get_mo_coeff()

    ao_soc0_tmp = """
    0.00000   0.00000   0.00000   0.00071  -0.00000   0.00000   0.00000   0.00010  -0.00000   0.00000   0.00000   0.00000   0.00001   0.00000   0.00000   0.00000   0.00005   0.00000   0.00003   0.00003  -0.00003  -0.00003
    0.00000   0.00000   0.00000   0.00033  -0.00000   0.00000   0.00000   0.00030  -0.00000   0.00000   0.00000   0.00000   0.00079   0.00000   0.00000   0.00000   0.00033   0.00000   0.00007   0.00009  -0.00007  -0.00009
    0.00000   0.00000   0.00000   0.00026   0.00000   0.00000   0.00000   0.00020   0.00000   0.00000   0.00000   0.00000   0.00161   0.00000   0.00000   0.00000   0.00054   0.00000  -0.00004   0.00003   0.00004  -0.00003
    -0.00071  -0.00033  -0.00026   0.00000  18.50527  -0.00015  -0.00005  -0.00000   2.92462   0.00001   0.00023  -0.00048   0.00000   1.21860   0.00044  -0.00093  -0.00000   1.67181   0.00056   0.00039   0.00061   0.00044
    0.00000   0.00000  -0.00000 -18.50527   0.00000   0.00000  -0.00000  -3.04196   0.00000   0.00000   0.00000   0.00000  -1.21727  -0.00000   0.00000  -0.00000  -1.69416   0.00000  -1.26057  -1.05190   1.26057   1.05190
    0.00000   0.00000   0.00000   0.00015  -0.00000   0.00000   0.00000   0.00018  -0.00000   0.00000   0.00000   0.00000   0.00057   0.00000   0.00000   0.00000   0.00024   0.00000   0.00003   0.00005  -0.00003  -0.00005
    0.00000   0.00000   0.00000   0.00005   0.00000   0.00000   0.00000   0.00010   0.00000   0.00000   0.00000   0.00000   0.00103   0.00000   0.00000   0.00000   0.00037   0.00000  -0.00002   0.00001   0.00002  -0.00001
    -0.00010  -0.00030  -0.00020   0.00000   3.04196  -0.00018  -0.00010   0.00000   1.41206   0.00002   0.00012  -0.00017   0.00000   2.07194   0.00016  -0.00027   0.00000   1.25675   0.00021   0.00016   0.00024   0.00018
    0.00000   0.00000  -0.00000  -2.92462  -0.00000   0.00000  -0.00000  -1.41206   0.00000   0.00000   0.00000   0.00000  -2.06802  -0.00000   0.00000   0.00000  -1.24917  -0.00000  -0.54591  -0.52487   0.54591   0.52487
    0.00000   0.00000   0.00000  -0.00001   0.00000   0.00000   0.00000  -0.00002   0.00000   0.00000   0.00000   0.00000  -0.00056   0.00000   0.00000   0.00000  -0.00006   0.00000  -0.00000  -0.00000   0.00000   0.00000
    0.00000   0.00000   0.00000  -0.00023   0.00000   0.00000   0.00000  -0.00012   0.00000   0.00000   0.00000   0.00000  -0.00025   0.00000   0.00000   0.00000  -0.00023   0.00000  -0.00001  -0.00003   0.00001   0.00003
    0.00000   0.00000   0.00000   0.00048   0.00000   0.00000   0.00000   0.00017   0.00000   0.00000   0.00000   0.00000   0.00025   0.00000   0.00000   0.00000   0.00020   0.00000   0.00002   0.00004  -0.00002  -0.00004
    -0.00001  -0.00079  -0.00161  -0.00000   1.21727  -0.00057  -0.00103  -0.00000   2.06802   0.00056   0.00025  -0.00025   0.00000  66.53600   0.00010  -0.00005  -0.00000   7.88632  -0.00000   0.00001  -0.00000   0.00002
    0.00000   0.00000   0.00000  -1.21860   0.00000   0.00000   0.00000  -2.07194   0.00000   0.00000   0.00000   0.00000 -66.53600   0.00000   0.00000   0.00000  -7.89361   0.00000  -0.01602  -0.34496   0.01602   0.34496
    0.00000   0.00000   0.00000  -0.00044  -0.00000   0.00000   0.00000  -0.00016   0.00000   0.00000   0.00000   0.00000  -0.00010   0.00000   0.00000   0.00000  -0.00017   0.00000  -0.00004  -0.00005   0.00004   0.00005
    0.00000   0.00000   0.00000   0.00093   0.00000   0.00000   0.00000   0.00027   0.00000   0.00000   0.00000   0.00000   0.00005   0.00000   0.00000   0.00000   0.00016   0.00000   0.00010   0.00009  -0.00010  -0.00009
    -0.00005  -0.00033  -0.00054   0.00000   1.69416  -0.00024  -0.00037  -0.00000   1.24917   0.00006   0.00023  -0.00020   0.00000   7.89361   0.00017  -0.00016   0.00000   2.86004   0.00002   0.00004   0.00002   0.00005
    0.00000   0.00000   0.00000  -1.67181  -0.00000   0.00000   0.00000  -1.25675   0.00000   0.00000   0.00000   0.00000  -7.88632  -0.00000   0.00000   0.00000  -2.86004   0.00000  -0.13848  -0.30118   0.13848   0.30118
    -0.00003  -0.00007   0.00004  -0.00056   1.26057  -0.00003   0.00002  -0.00021   0.54591   0.00000   0.00001  -0.00002   0.00000   0.01602   0.00004  -0.00010  -0.00002   0.13848   0.00000   0.00000   0.00006   0.00011
    -0.00003  -0.00009  -0.00003  -0.00039   1.05190  -0.00005  -0.00001  -0.00016   0.52487   0.00000   0.00003  -0.00004  -0.00001   0.34496   0.00005  -0.00009  -0.00004   0.30118   0.00000   0.00000   0.00011   0.00011
    0.00003   0.00007  -0.00004  -0.00061  -1.26057   0.00003  -0.00002  -0.00024  -0.54591  -0.00000  -0.00001   0.00002   0.00000  -0.01602  -0.00004   0.00010  -0.00002  -0.13848  -0.00006  -0.00011   0.00000   0.00000
    0.00003   0.00009   0.00003  -0.00044  -1.05190   0.00005   0.00001  -0.00018  -0.52487  -0.00000  -0.00003   0.00004  -0.00002  -0.34496  -0.00005   0.00009  -0.00005  -0.30118  -0.00011  -0.00011   0.00000   0.00000
    """
    ao_soc0 = np.array(
        [float(ao_soc0_t) for ao_soc0_t in ao_soc0_tmp.strip().split()]
    ).reshape(g_parser_s1.nbasis, g_parser_s1.nbasis)

    ao_soc1_tmp = """
    0.00000   0.00000  -0.00071   0.00000   1.40471   0.00000  -0.00010   0.00000   0.19294  -0.00000  -0.00001   0.00000   0.00000   0.02809  -0.00002  -0.00001   0.00000   0.11104   0.00005   0.00005   0.00005   0.00005
    0.00000   0.00000  -0.00033   0.00000   0.65477   0.00000  -0.00030   0.00000   0.61685  -0.00000  -0.00002  -0.00076   0.00000   2.46819  -0.00004  -0.00026   0.00000   0.96955   0.00010   0.00013   0.00010   0.00014
    0.00071   0.00033   0.00000  -0.00000 -17.63304   0.00015  -0.00015  -0.00000  -2.25238  -0.00001  -0.00026  -0.00109   0.00000   3.83720  -0.00047   0.00042  -0.00000  -0.03288  -0.00051  -0.00029  -0.00056  -0.00032
    0.00000   0.00000   0.00000   0.00000  -0.00000   0.00000   0.00000   0.00000  -0.00000   0.00000   0.00000  -0.00000  -0.00001   0.00000   0.00000  -0.00000  -0.00005   0.00000   0.00014   0.00005  -0.00014  -0.00005
    -1.40471  -0.65477  17.63304   0.00000   0.00000  -0.29440   2.86067   0.00000   0.00015   0.03218   0.75375  -0.35115   0.00000   0.00110   1.49536  -1.47544   0.00000  -0.00050  -0.88425  -0.77485  -0.88425  -0.77485
    0.00000   0.00000  -0.00015   0.00000   0.29440   0.00000  -0.00018   0.00000   0.37261  -0.00000  -0.00001  -0.00056   0.00000   1.79138  -0.00002  -0.00021   0.00000   0.71707   0.00005   0.00008   0.00005   0.00008
    0.00010   0.00030   0.00015  -0.00000  -2.86067   0.00018   0.00000  -0.00000  -1.07103  -0.00002  -0.00013  -0.00085   0.00000   1.15020  -0.00017  -0.00009  -0.00000  -0.13570  -0.00020  -0.00011  -0.00022  -0.00013
    0.00000   0.00000   0.00000   0.00000  -0.00000   0.00000   0.00000   0.00000  -0.00000   0.00000   0.00000  -0.00000  -0.00001   0.00000   0.00000  -0.00000  -0.00002   0.00000   0.00005   0.00003  -0.00005  -0.00004
    -0.19294  -0.61685   2.25238   0.00000  -0.00015  -0.37261   1.07103   0.00000   0.00000   0.05669   0.41046   1.51625   0.00000   0.00085   0.58533   0.27906   0.00000   0.00005  -0.43086  -0.47673  -0.43086  -0.47673
    0.00000   0.00000   0.00001  -0.00000  -0.03218   0.00000   0.00002  -0.00000  -0.05669   0.00000   0.00000   0.00056   0.00000  -1.95961   0.00000   0.00006   0.00000  -0.21801  -0.00000  -0.00001  -0.00000  -0.00001
    0.00001   0.00002   0.00026  -0.00000  -0.75375   0.00001   0.00013  -0.00000  -0.41046   0.00000   0.00000   0.00025   0.00000  -0.88898   0.00000   0.00023   0.00000  -0.79388  -0.00002  -0.00005  -0.00002  -0.00006
    -0.00000   0.00076   0.00109   0.00000   0.35115   0.00056   0.00085   0.00000  -1.51625  -0.00056  -0.00025   0.00000  -0.00000 -65.72826  -0.00010  -0.00015  -0.00000  -7.24285   0.00003   0.00007   0.00004   0.00006
    -0.00000  -0.00000  -0.00000   0.00001   0.00000  -0.00000  -0.00000   0.00001   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000  -0.00001  -0.00001   0.00001   0.00001
    -0.02809  -2.46819  -3.83720   0.00000  -0.00110  -1.79138  -1.15020   0.00000  -0.00085   1.95961   0.88898  65.72826   0.00000   0.00000   0.33621   7.74130   0.00000   0.00015  -0.02015  -0.61731  -0.02015  -0.61731
    0.00002   0.00004   0.00047  -0.00000  -1.49536   0.00002   0.00017  -0.00000  -0.58533   0.00000   0.00000   0.00010   0.00000  -0.33621   0.00000   0.00017   0.00000  -0.58933  -0.00008  -0.00010  -0.00008  -0.00010
    0.00001   0.00026  -0.00042   0.00000   1.47544   0.00021   0.00009   0.00000  -0.27906  -0.00006  -0.00023   0.00015  -0.00000  -7.74130  -0.00017   0.00000  -0.00000  -2.32781   0.00015   0.00013   0.00015   0.00013
    -0.00000  -0.00000   0.00000   0.00005  -0.00000  -0.00000   0.00000   0.00002  -0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000  -0.00005  -0.00003   0.00006   0.00003
    -0.11104  -0.96955   0.03288   0.00000   0.00050  -0.71707   0.13570   0.00000  -0.00005   0.21801   0.79388   7.24285   0.00000  -0.00015   0.58933   2.32781   0.00000   0.00000  -0.11609  -0.38270  -0.11609  -0.38270
    -0.00005  -0.00010   0.00051  -0.00014   0.88425  -0.00005   0.00020  -0.00005   0.43086   0.00000   0.00002  -0.00003   0.00001   0.02015   0.00008  -0.00015   0.00005   0.11609   0.00000   0.00000   0.00000   0.00000
    -0.00005  -0.00013   0.00029  -0.00005   0.77485  -0.00008   0.00011  -0.00003   0.47673   0.00001   0.00005  -0.00007   0.00001   0.61731   0.00010  -0.00013   0.00003   0.38270   0.00000   0.00000   0.00000   0.00001
    -0.00005  -0.00010   0.00056   0.00014   0.88425  -0.00005   0.00022   0.00005   0.43086   0.00000   0.00002  -0.00004  -0.00001   0.02015   0.00008  -0.00015  -0.00006   0.11609  -0.00000  -0.00000   0.00000   0.00000
    -0.00005  -0.00014   0.00032   0.00005   0.77485  -0.00008   0.00013   0.00004   0.47673   0.00001   0.00006  -0.00006  -0.00001   0.61731   0.00010  -0.00013  -0.00003   0.38270  -0.00000  -0.00001   0.00000   0.00000
    """
    ao_soc1 = np.array(
        [float(ao_soc1_t) for ao_soc1_t in ao_soc1_tmp.strip().split()]
    ).reshape(g_parser_s1.nbasis, g_parser_s1.nbasis)

    ao_soc2_tmp = """
    0.00000   0.00000   0.00000  -1.40471   0.00000   0.00000   0.00000  -0.19294   0.00000   0.00000   0.00000   0.00000  -0.02709  -0.00000   0.00000  -0.00000  -0.10779  -0.00000  -0.06627  -0.06267   0.06627   0.06267
    0.00000   0.00000   0.00000  -0.65477   0.00000   0.00000   0.00000  -0.61685   0.00000   0.00000   0.00000   0.00000  -2.46003  -0.00000   0.00000  -0.00000  -0.93754  -0.00000  -0.13438  -0.17613   0.13438   0.17613
    -0.00000  -0.00000   0.00000  17.40935   0.00000  -0.00000  -0.00000   2.20999   0.00000   0.00000   0.00000   0.00000  -3.82970  -0.00000   0.00000   0.00000   0.01675   0.00000   1.28776   0.87988  -1.28776  -0.87988
    1.40471   0.65477 -17.40935   0.00000   0.00026   0.29440  -2.70095   0.00000   0.00005  -0.03209  -0.74500   0.33764   0.00000   0.00049  -1.45169   1.40858   0.00000   0.00098   0.68608   0.67760   0.68608   0.67760
    0.00000   0.00000  -0.00000  -0.00026   0.00000   0.00000  -0.00000  -0.00020   0.00000  -0.00000  -0.00000   0.00000  -0.00160   0.00000  -0.00000   0.00000  -0.00050   0.00000  -0.00010  -0.00008   0.00011   0.00008
    0.00000   0.00000   0.00000  -0.29440   0.00000   0.00000   0.00000  -0.37261   0.00000   0.00000   0.00000   0.00000  -1.78706  -0.00000   0.00000   0.00000  -0.69831  -0.00000  -0.06570  -0.10872   0.06570   0.10872
    -0.00000  -0.00000   0.00000   2.70095   0.00000  -0.00000   0.00000   1.00196   0.00000   0.00000   0.00000   0.00000  -1.14961  -0.00000   0.00000   0.00000   0.11820   0.00000   0.56480   0.45199  -0.56480  -0.45199
    0.19294   0.61685  -2.20999  -0.00000   0.00020   0.37261  -1.00196   0.00000   0.00010  -0.05664  -0.40802  -1.51831   0.00000   0.00017  -0.57050  -0.29863   0.00000   0.00029   0.36135   0.41821   0.36135   0.41821
    0.00000   0.00000  -0.00000  -0.00005   0.00000   0.00000  -0.00000  -0.00010   0.00000  -0.00000  -0.00000   0.00000  -0.00102   0.00000  -0.00000   0.00000  -0.00035   0.00000  -0.00003  -0.00004   0.00004   0.00005
    0.00000   0.00000   0.00000   0.03209   0.00000   0.00000   0.00000   0.05664   0.00000   0.00000   0.00000   0.00000   1.95961   0.00000   0.00000   0.00000   0.21801   0.00000   0.00017   0.00920  -0.00017  -0.00920
    0.00000   0.00000   0.00000   0.74500   0.00000   0.00000   0.00000   0.40802   0.00000   0.00000   0.00000   0.00000   0.88898   0.00000   0.00000   0.00000   0.79388   0.00000   0.02811   0.09323  -0.02811  -0.09323
    0.00000   0.00000   0.00000  -0.33764  -0.00000   0.00000   0.00000   1.51831  -0.00000   0.00000   0.00000   0.00000  65.71648   0.00000   0.00000   0.00000   7.24059   0.00000  -0.04767   0.20220   0.04767  -0.20220
    0.02709   2.46003   3.82970   0.00000   0.00160   1.78706   1.14961   0.00000   0.00102  -1.95961  -0.88898 -65.71648   0.00000   0.00025  -0.33621  -7.73174   0.00000   0.00005   0.03474   0.63867   0.03474   0.63867
    0.00000   0.00000   0.00000  -0.00049   0.00000   0.00000   0.00000  -0.00017   0.00000   0.00000   0.00000  -0.00000  -0.00025   0.00000   0.00000  -0.00000  -0.00020   0.00000  -0.00001  -0.00003   0.00001   0.00003
    -0.00000   0.00000   0.00000   1.45169   0.00000   0.00000   0.00000   0.57050   0.00000   0.00000   0.00000   0.00000   0.33621   0.00000   0.00000   0.00000   0.58933   0.00000   0.13544   0.17443  -0.13544  -0.17443
    0.00000   0.00000  -0.00000  -1.40858  -0.00000   0.00000   0.00000   0.29863  -0.00000   0.00000   0.00000   0.00000   7.73174   0.00000   0.00000   0.00000   2.31612   0.00000  -0.19314  -0.02973   0.19314   0.02973
    0.10779   0.93754  -0.01675  -0.00000   0.00050   0.69831  -0.11820   0.00000   0.00035  -0.21801  -0.79388  -7.24059   0.00000   0.00020  -0.58933  -2.31612   0.00000   0.00016   0.21001   0.43004   0.21001   0.43004
    0.00000   0.00000  -0.00000  -0.00098   0.00000   0.00000  -0.00000  -0.00029   0.00000   0.00000   0.00000  -0.00000  -0.00005   0.00000   0.00000  -0.00000  -0.00016   0.00000  -0.00005  -0.00006   0.00004   0.00006
    0.06627   0.13438  -1.28776  -0.68608   0.00010   0.06570  -0.56480  -0.36135   0.00003  -0.00017  -0.02811   0.04767  -0.03474   0.00001  -0.13544   0.19314  -0.21001   0.00005   0.00000   0.00000   0.10362   0.21932
    0.06267   0.17613  -0.87988  -0.67760   0.00008   0.10872  -0.45199  -0.41821   0.00004  -0.00920  -0.09323  -0.20220  -0.63867   0.00003  -0.17443   0.02973  -0.43004   0.00006   0.00000   0.00000   0.21932   0.28732
    -0.06627  -0.13438   1.28776  -0.68608  -0.00011  -0.06570   0.56480  -0.36135  -0.00004   0.00017   0.02811  -0.04767  -0.03474  -0.00001   0.13544  -0.19314  -0.21001  -0.00004  -0.10362  -0.21932   0.00000   0.00000
    -0.06267  -0.17613   0.87988  -0.67760  -0.00008  -0.10872   0.45199  -0.41821  -0.00005   0.00920   0.09323   0.20220  -0.63867  -0.00003   0.17443  -0.02973  -0.43004  -0.00006  -0.21932  -0.28732   0.00000   0.00000
    """
    ao_soc2 = np.array(
        [float(ao_soc2_t) for ao_soc2_t in ao_soc2_tmp.strip().split()]
    ).reshape(g_parser_s1.nbasis, g_parser_s1.nbasis)
    ao_soc = np.array([ao_soc0, ao_soc1, ao_soc2])

    mo_soc = np.einsum("kpq,ip,jq->kij", ao_soc, mo_coeff, mo_coeff)

    coeff_thresh = 1e-5

    x_coeff_s1, y_coeff_s1 = g_parser_s1.get_xy_coeff()
    xpy_coeff_s1 = x_coeff_s1 + y_coeff_s1
    norm_s1 = np.sqrt(np.trace(xpy_coeff_s1 @ xpy_coeff_s1.T) * 2.0)
    xpy_coeff_s1 = xpy_coeff_s1 / norm_s1
    xpy_coeff_s1[np.abs(xpy_coeff_s1) < coeff_thresh] = 0.0  # pysoc treatment

    x_coeff_t1, y_coeff_t1 = g_parser_t1.get_xy_coeff()
    xpy_coeff_t1 = x_coeff_t1 + y_coeff_t1
    norm_t1 = np.sqrt(np.trace(xpy_coeff_t1 @ xpy_coeff_t1.T) * 2.0)
    xpy_coeff_t1 = xpy_coeff_t1 / norm_t1
    xpy_coeff_t1[np.abs(xpy_coeff_t1) < coeff_thresh] = 0.0  # pysoc treatment

    fine_stru = 7.297352568e-3
    mo_soc *= 0.5 * fine_stru**2
    mo_soc_ij = mo_soc[
        :, : g_parser_t1.nfc + g_parser_t1.noa, : g_parser_t1.nfc + g_parser_t1.noa
    ]
    mo_soc_ia = mo_soc[
        :, : g_parser_t1.nfc + g_parser_t1.noa, g_parser_t1.nfc + g_parser_t1.noa :
    ]
    mo_soc_ab = mo_soc[
        :, g_parser_t1.nfc + g_parser_t1.noa :, g_parser_t1.nfc + g_parser_t1.noa :
    ]

    au2wavnum = 219474.6
    soc_s0t1_1 = (1.0 / np.sqrt(2)) * complex(
        -np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[0, :, :]),
        -np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[1, :, :]),
    )
    soc_s0t1_2 = complex(np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[2, :, :]), 0.0)
    soc_s0t1_3 = -1.0 * soc_s0t1_1
    soc_s0t1 = np.array([soc_s0t1_1, soc_s0t1_2, soc_s0t1_3])
    soc_s0t1 *= au2wavnum
    answer = np.array(
        [68.05414 + 0.00024j, -0.01202 + 0.00000, -68.05414 - 0.00024j]
    )  # from pysoc

    assert np.allclose(soc_s0t1, answer, atol=1e-3)

    soc_s1t1_1 = complex(
        1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[0, :, :])
        - 1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[0, :, :]),
        1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[1, :, :])
        - 1.0
        / np.sqrt(2.0)
        * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[1, :, :]),
    )

    soc_s1t1_2 = complex(
        -1.0 * np.einsum("ia,ja,ji->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij[2, :, :])
        + 1.0 * np.einsum("ia,ib,ab->", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab[2, :, :]),
        0.0,
    )
    soc_s1t1_3 = -1.0 * soc_s1t1_1
    soc_s1t1 = np.array([soc_s1t1_1, soc_s1t1_2, soc_s1t1_3])
    soc_s1t1 *= au2wavnum
    answer = np.array(
        [0.00000 - 0.00161j, 0.00000 + 0.00000j, -0.00000 + 0.00161j]
    )  # from pysoc

    assert np.allclose(soc_s1t1, answer, atol=1e-3)
