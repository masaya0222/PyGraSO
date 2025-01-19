import numpy as np

from parser import gaussian_perser
from calc_ao_element import calc_ao_element

fine_stru = 7.297352568e-3
au2wavnum = 219474.6
coeff_thresh = 1e-5


def calc_soc_s0t1(atoms, coordinates, t1_log_file_name, t1_rwf_file_name):
    g_parser_t1 = gaussian_perser(t1_log_file_name, t1_rwf_file_name)
    mo_coeff = g_parser_t1.get_mo_coeff()
    x_coeff_t1, y_coeff_t1 = g_parser_t1.get_xy_coeff()
    xpy_coeff_t1 = x_coeff_t1 + y_coeff_t1
    norm_t1 = np.sqrt(np.trace(xpy_coeff_t1 @ xpy_coeff_t1.T) * 2.0)
    xpy_coeff_t1 = xpy_coeff_t1 / norm_t1
    xpy_coeff_t1[np.abs(xpy_coeff_t1) < coeff_thresh] = 0.0

    ao_calculator = calc_ao_element(atoms, coordinates)
    ao_soc = ao_calculator.get_ao_soc()
    mo_soc = np.einsum("kpq,ip,jq->kij", ao_soc, mo_coeff, mo_coeff)

    mo_soc *= 0.5 * fine_stru**2
    mo_soc_ij = mo_soc[:, : g_parser_t1.nfc + g_parser_t1.noa, :][
        :, :, : g_parser_t1.nfc + g_parser_t1.noa
    ]
    mo_soc_ia = mo_soc[:, : g_parser_t1.nfc + g_parser_t1.noa, :][
        :, :, g_parser_t1.nfc + g_parser_t1.noa :
    ]
    mo_soc_ab = mo_soc[:, g_parser_t1.nfc + g_parser_t1.noa :, :][
        :, :, g_parser_t1.nfc + g_parser_t1.noa :
    ]

    soc_s0t1_1 = (1.0 / np.sqrt(2)) * complex(
        -np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[0, :, :]),
        -np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[1, :, :]),
    )
    soc_s0t1_2 = complex(np.einsum("ia,ia->", xpy_coeff_t1, mo_soc_ia[2, :, :]), 0.0)
    soc_s0t1_3 = -1.0 * soc_s0t1_1
    soc_s0t1 = np.array([soc_s0t1_1, soc_s0t1_2, soc_s0t1_3])
    soc_s0t1 *= au2wavnum

    return soc_s0t1


def calc_soc_s1t1(
    atoms,
    coordinates,
    s1_log_file_name,
    s1_rwf_file_name,
    t1_log_file_name,
    t1_rwf_file_name,
):
    g_parser_s1 = gaussian_perser(s1_log_file_name, s1_rwf_file_name)
    mo_coeff = g_parser_s1.get_mo_coeff()
    x_coeff_s1, y_coeff_s1 = g_parser_s1.get_xy_coeff()
    xpy_coeff_s1 = x_coeff_s1 + y_coeff_s1
    norm_s1 = np.sqrt(np.trace(xpy_coeff_s1 @ xpy_coeff_s1.T) * 2.0)
    xpy_coeff_s1 = xpy_coeff_s1 / norm_s1
    xpy_coeff_s1[np.abs(xpy_coeff_s1) < coeff_thresh] = 0.0

    g_parser_t1 = gaussian_perser(t1_log_file_name, t1_rwf_file_name)
    x_coeff_t1, y_coeff_t1 = g_parser_t1.get_xy_coeff()
    xpy_coeff_t1 = x_coeff_t1 + y_coeff_t1
    norm_t1 = np.sqrt(np.trace(xpy_coeff_t1 @ xpy_coeff_t1.T) * 2.0)
    xpy_coeff_t1 = xpy_coeff_t1 / norm_t1
    xpy_coeff_t1[np.abs(xpy_coeff_t1) < coeff_thresh] = 0.0

    ao_calculator = calc_ao_element(atoms, coordinates)
    ao_soc = ao_calculator.get_ao_soc()
    mo_soc = np.einsum("kpq,ip,jq->kij", ao_soc, mo_coeff, mo_coeff)

    mo_soc *= 0.5 * fine_stru**2
    mo_soc_ij = mo_soc[:, : g_parser_t1.nfc + g_parser_t1.noa, :][
        :, :, : g_parser_t1.nfc + g_parser_t1.noa
    ]
    mo_soc_ia = mo_soc[:, : g_parser_t1.nfc + g_parser_t1.noa, :][
        :, :, g_parser_t1.nfc + g_parser_t1.noa :
    ]
    mo_soc_ab = mo_soc[:, g_parser_t1.nfc + g_parser_t1.noa :, :][
        :, :, g_parser_t1.nfc + g_parser_t1.noa :
    ]

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

    return soc_s1t1
