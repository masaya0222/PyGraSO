import numpy as np

from .parser import gaussian_perser
from .calc_ao_element import calc_ao_element

fine_stru = 7.297352568e-3
au2wavnum = 219474.6
coeff_thresh = 1e-5


def calc_soc_s0t1(
    atoms,
    coordinates,
    t1_log_file_name,
    t1_rwf_file_name,
    normalize=True,
    basis=None,
    Z="one",
):
    g_parser_t1 = gaussian_perser(t1_log_file_name, t1_rwf_file_name)
    mo_coeff = g_parser_t1.get_mo_coeff()
    x_coeff_t1, y_coeff_t1 = g_parser_t1.get_xy_coeff()
    xpy_coeff_t1 = x_coeff_t1 + y_coeff_t1
    if normalize:
        norm_t1 = np.sqrt(np.trace(xpy_coeff_t1 @ xpy_coeff_t1.T) * 2.0)
        xpy_coeff_t1 = xpy_coeff_t1 / norm_t1
    # xpy_coeff_t1[np.abs(xpy_coeff_t1) < coeff_thresh] = 0.0

    ao_calculator = calc_ao_element(atoms, coordinates, basis=g_parser_t1.read_basis())
    ao_soc = ao_calculator.get_ao_soc(Z=Z)

    mo_soc = mo_coeff @ ao_soc @ mo_coeff.T

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

    soc_s0t1_1 = (1.0 / np.sqrt(2)) * (
        -np.trace(xpy_coeff_t1.T @ mo_soc_ia[0, :, :])
        - np.trace(xpy_coeff_t1.T @ mo_soc_ia[1, :, :]) * 1.0j
    )
    soc_s0t1_2 = np.trace(xpy_coeff_t1.T @ mo_soc_ia[2, :, :])

    soc_s0t1_3 = (1.0 / np.sqrt(2)) * (
        np.trace(xpy_coeff_t1.T @ mo_soc_ia[0, :, :])
        - np.trace(xpy_coeff_t1.T @ mo_soc_ia[1, :, :]) * 1.0j
    )
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
    normalize=True,
    basis=None,
    Z="one",
):
    g_parser_s1 = gaussian_perser(s1_log_file_name, s1_rwf_file_name)
    mo_coeff = g_parser_s1.get_mo_coeff()
    x_coeff_s1, y_coeff_s1 = g_parser_s1.get_xy_coeff()
    xpy_coeff_s1 = x_coeff_s1 + y_coeff_s1
    if normalize:
        norm_s1 = np.sqrt(np.trace(xpy_coeff_s1 @ xpy_coeff_s1.T) * 2.0)
        xpy_coeff_s1 = xpy_coeff_s1 / norm_s1
    # xpy_coeff_s1[np.abs(xpy_coeff_s1) < coeff_thresh] = 0.0

    g_parser_t1 = gaussian_perser(t1_log_file_name, t1_rwf_file_name)
    x_coeff_t1, y_coeff_t1 = g_parser_t1.get_xy_coeff()
    xpy_coeff_t1 = x_coeff_t1 + y_coeff_t1
    if normalize:
        norm_t1 = np.sqrt(np.trace(xpy_coeff_t1 @ xpy_coeff_t1.T) * 2.0)
        xpy_coeff_t1 = xpy_coeff_t1 / norm_t1
    # xpy_coeff_t1[np.abs(xpy_coeff_t1) < coeff_thresh] = 0.0

    ao_calculator = calc_ao_element(atoms, coordinates, basis=g_parser_t1.read_basis())
    ao_soc = ao_calculator.get_ao_soc(Z=Z)

    mo_soc = mo_coeff @ ao_soc @ mo_coeff.T

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

    soc_s1t1_1 = (
        1.0
        / np.sqrt(2.0)
        * np.trace(xpy_coeff_s1.T @ mo_soc_ij[0, :, :].T @ xpy_coeff_t1)
        - 1.0
        / np.sqrt(2.0)
        * np.trace(xpy_coeff_s1 @ mo_soc_ab[0, :, :] @ xpy_coeff_t1.T)
        + 1.0
        / np.sqrt(2.0)
        * np.trace(xpy_coeff_s1.T @ mo_soc_ij[1, :, :].T @ xpy_coeff_t1)
        * 1.0j
        - 1.0
        / np.sqrt(2.0)
        * np.trace(xpy_coeff_s1 @ mo_soc_ab[1, :, :] @ xpy_coeff_t1.T)
        * 1.0j
    )

    soc_s1t1_2 = -1.0 * np.trace(
        xpy_coeff_s1.T @ mo_soc_ij[2, :, :].T @ xpy_coeff_t1
    ) + 1.0 * np.trace(xpy_coeff_s1 @ mo_soc_ab[2, :, :] @ xpy_coeff_t1.T)

    soc_s1t1_3 = (
        -1.0
        / np.sqrt(2.0)
        * np.trace(xpy_coeff_s1.T @ mo_soc_ij[0, :, :].T @ xpy_coeff_t1)
        + 1.0
        / np.sqrt(2.0)
        * np.trace(xpy_coeff_s1 @ mo_soc_ab[0, :, :] @ xpy_coeff_t1.T)
        + 1.0
        / np.sqrt(2.0)
        * np.trace(xpy_coeff_s1.T @ mo_soc_ij[1, :, :].T @ xpy_coeff_t1)
        * 1.0j
        - 1.0
        / np.sqrt(2.0)
        * np.trace(xpy_coeff_s1 @ mo_soc_ab[1, :, :] @ xpy_coeff_t1.T)
        * 1.0j
    )
    soc_s1t1 = np.array([soc_s1t1_1, soc_s1t1_2, soc_s1t1_3])
    soc_s1t1 *= au2wavnum

    return soc_s1t1


def calc_soc_s0t1_deriv(
    atoms,
    coordinates,
    t1_log_file_name,
    t1_rwf_file_name,
    normalize=True,
    basis=None,
):
    g_parser_t1 = gaussian_perser(t1_log_file_name, t1_rwf_file_name)
    mo_coeff = g_parser_t1.get_mo_coeff()
    mo_coeff_deriv = g_parser_t1.get_mo_coeff_deriv()

    mo_coeff_i = mo_coeff[: g_parser_t1.nfc + g_parser_t1.noa, :]
    mo_coeff_a = mo_coeff[g_parser_t1.nfc + g_parser_t1.noa :, :]
    mo_coeff_deriv_i = mo_coeff_deriv[:, :, : g_parser_t1.nfc + g_parser_t1.noa, :]
    mo_coeff_deriv_a = mo_coeff_deriv[:, :, g_parser_t1.nfc + g_parser_t1.noa :, :]

    x_coeff_t1, y_coeff_t1 = g_parser_t1.get_xy_coeff()
    xpy_coeff_t1 = x_coeff_t1 + y_coeff_t1
    if normalize:
        norm_t1 = np.sqrt(np.trace(xpy_coeff_t1 @ xpy_coeff_t1.T) * 2.0)
        xpy_coeff_t1 = xpy_coeff_t1 / norm_t1
    # xpy_coeff_t1[np.abs(xpy_coeff_t1) < coeff_thresh] = 0.0

    x_coeff_deriv_t1, y_coeff_deriv_t1 = g_parser_t1.get_xy_coeff_deriv()
    xpy_coeff_deriv_t1 = x_coeff_deriv_t1 + y_coeff_deriv_t1
    if normalize:
        xpy_coeff_deriv_t1 = (
            xpy_coeff_deriv_t1
            - 2.0
            * np.einsum(
                "jb,ia,rdia->rdjb", xpy_coeff_t1, xpy_coeff_t1, xpy_coeff_deriv_t1
            )
        ) / norm_t1

    ao_calculator = calc_ao_element(atoms, coordinates, basis=g_parser_t1.read_basis())
    ao_soc = ao_calculator.get_ao_soc()
    ao_soc_deriv = ao_calculator.get_ao_soc_deriv()

    mo_soc = np.einsum("kpq,ip,jq->kij", ao_soc, mo_coeff, mo_coeff)

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

    mo_soc_deriv = (
        np.einsum("rdkpq,ip,jq->rdkij", ao_soc_deriv, mo_coeff, mo_coeff)
        + np.einsum("kpq,rdip,jq->rdkij", ao_soc, mo_coeff_deriv, mo_coeff)
        + np.einsum("kpq,ip,rdjq->rdkij", ao_soc, mo_coeff, mo_coeff_deriv)
    )

    mo_soc_deriv *= 0.5 * fine_stru**2
    mo_soc_ij_deriv = mo_soc_deriv[
        :,
        :,
        :,
        : g_parser_t1.nfc + g_parser_t1.noa,
        : g_parser_t1.nfc + g_parser_t1.noa,
    ]
    mo_soc_ia_deriv = mo_soc_deriv[
        :,
        :,
        :,
        : g_parser_t1.nfc + g_parser_t1.noa,
        g_parser_t1.nfc + g_parser_t1.noa :,
    ]
    mo_soc_ab_deriv = mo_soc_deriv[
        :,
        :,
        :,
        g_parser_t1.nfc + g_parser_t1.noa :,
        g_parser_t1.nfc + g_parser_t1.noa :,
    ]

    soc_s0t1_1_deriv_re = (1.0 / np.sqrt(2)) * (
        -np.einsum("rdia,ia->rd", xpy_coeff_deriv_t1, mo_soc_ia[0, :, :])
        - np.einsum("ia,rdia->rd", xpy_coeff_t1, mo_soc_ia_deriv[:, :, 0, :, :])
    )
    soc_s0t1_1_deriv_im = (1.0 / np.sqrt(2)) * (
        -np.einsum("rdia,ia->rd", xpy_coeff_deriv_t1, mo_soc_ia[1, :, :])
        - np.einsum("ia,rdia->rd", xpy_coeff_t1, mo_soc_ia_deriv[:, :, 1, :, :])
    )
    soc_s0t1_1_deriv = soc_s0t1_1_deriv_re + soc_s0t1_1_deriv_im * 1.0j

    soc_s0t1_2_deriv_re = np.einsum(
        "rdia,ia->rd", xpy_coeff_deriv_t1, mo_soc_ia[2, :, :]
    ) + np.einsum("ia,rdia->rd", xpy_coeff_t1, mo_soc_ia_deriv[:, :, 2, :, :])
    soc_s0t1_2_deriv = soc_s0t1_2_deriv_re
    soc_s0t1_3_deriv_re = -soc_s0t1_1_deriv_re
    soc_s0t1_3_deriv_im = soc_s0t1_1_deriv_im
    soc_s0t1_3_deriv = soc_s0t1_3_deriv_re + soc_s0t1_3_deriv_im * 1.0j

    soc_s0t1_deriv = np.array(
        [soc_s0t1_1_deriv, soc_s0t1_2_deriv, soc_s0t1_3_deriv]
    ).transpose(1, 2, 0)
    soc_s0t1_deriv *= au2wavnum
    return soc_s0t1_deriv


def calc_soc_s1t1_deriv(
    atoms,
    coordinates,
    s1_log_file_name,
    s1_rwf_file_name,
    t1_log_file_name,
    t1_rwf_file_name,
    normalize=True,
    basis=None,
):
    g_parser_s1 = gaussian_perser(s1_log_file_name, s1_rwf_file_name)
    mo_coeff = g_parser_s1.get_mo_coeff()
    mo_coeff_deriv = g_parser_s1.get_mo_coeff_deriv()
    mo_coeff_i = mo_coeff[: g_parser_s1.nfc + g_parser_s1.noa, :]
    mo_coeff_a = mo_coeff[g_parser_s1.nfc + g_parser_s1.noa :, :]
    mo_coeff_deriv_i = mo_coeff_deriv[:, :, : g_parser_s1.nfc + g_parser_s1.noa, :]
    mo_coeff_deriv_a = mo_coeff_deriv[:, :, g_parser_s1.nfc + g_parser_s1.noa :, :]

    x_coeff_s1, y_coeff_s1 = g_parser_s1.get_xy_coeff()
    xpy_coeff_s1 = x_coeff_s1 + y_coeff_s1
    if normalize:
        norm_s1 = np.sqrt(np.trace(xpy_coeff_s1 @ xpy_coeff_s1.T) * 2.0)
        xpy_coeff_s1 = xpy_coeff_s1 / norm_s1
    # xpy_coeff_s1[np.abs(xpy_coeff_s1) < coeff_thresh] = 0.0

    x_coeff_deriv_s1, y_coeff_deriv_s1 = g_parser_s1.get_xy_coeff_deriv()
    xpy_coeff_deriv_s1 = x_coeff_deriv_s1 + y_coeff_deriv_s1
    if normalize:
        xpy_coeff_deriv_s1 = (
            xpy_coeff_deriv_s1
            - 2.0
            * np.einsum(
                "jb,ia,rdia->rdjb", xpy_coeff_s1, xpy_coeff_s1, xpy_coeff_deriv_s1
            )
        ) / norm_s1

    g_parser_t1 = gaussian_perser(t1_log_file_name, t1_rwf_file_name)
    x_coeff_t1, y_coeff_t1 = g_parser_t1.get_xy_coeff()
    xpy_coeff_t1 = x_coeff_t1 + y_coeff_t1
    if normalize:
        norm_t1 = np.sqrt(np.trace(xpy_coeff_t1 @ xpy_coeff_t1.T) * 2.0)
        xpy_coeff_t1 = xpy_coeff_t1 / norm_t1
    # xpy_coeff_t1[np.abs(xpy_coeff_t1) < coeff_thresh] = 0.0

    x_coeff_deriv_t1, y_coeff_deriv_t1 = g_parser_t1.get_xy_coeff_deriv()
    xpy_coeff_deriv_t1 = x_coeff_deriv_t1 + y_coeff_deriv_t1
    if normalize:
        xpy_coeff_deriv_t1 = (
            xpy_coeff_deriv_t1
            - 2.0
            * np.einsum(
                "jb,ia,rdia->rdjb", xpy_coeff_t1, xpy_coeff_t1, xpy_coeff_deriv_t1
            )
        ) / norm_t1

    ao_calculator = calc_ao_element(atoms, coordinates, basis=g_parser_t1.read_basis())
    ao_soc = ao_calculator.get_ao_soc()
    ao_soc_deriv = ao_calculator.get_ao_soc_deriv()

    mo_soc = np.einsum("kpq,ip,jq->kij", ao_soc, mo_coeff, mo_coeff)

    mo_soc *= 0.5 * fine_stru**2
    mo_soc_ij = mo_soc[
        :, : g_parser_s1.nfc + g_parser_s1.noa, : g_parser_s1.nfc + g_parser_s1.noa
    ]
    mo_soc_ia = mo_soc[
        :, : g_parser_s1.nfc + g_parser_s1.noa, g_parser_s1.nfc + g_parser_s1.noa :
    ]
    mo_soc_ab = mo_soc[
        :, g_parser_s1.nfc + g_parser_s1.noa :, g_parser_s1.nfc + g_parser_s1.noa :
    ]

    mo_soc_deriv = (
        np.einsum("rdkpq,ip,jq->rdkij", ao_soc_deriv, mo_coeff, mo_coeff)
        + np.einsum("kpq,rdip,jq->rdkij", ao_soc, mo_coeff_deriv, mo_coeff)
        + np.einsum("kpq,ip,rdjq->rdkij", ao_soc, mo_coeff, mo_coeff_deriv)
    )

    mo_soc_deriv *= 0.5 * fine_stru**2
    mo_soc_ij_deriv = mo_soc_deriv[
        :,
        :,
        :,
        : g_parser_s1.nfc + g_parser_s1.noa,
        : g_parser_s1.nfc + g_parser_s1.noa,
    ]
    mo_soc_ia_deriv = mo_soc_deriv[
        :,
        :,
        :,
        : g_parser_s1.nfc + g_parser_s1.noa,
        g_parser_s1.nfc + g_parser_s1.noa :,
    ]
    mo_soc_ab_deriv = mo_soc_deriv[
        :,
        :,
        :,
        g_parser_s1.nfc + g_parser_s1.noa :,
        g_parser_s1.nfc + g_parser_s1.noa :,
    ]

    soc_s1t1_1_deriv_re = (
        1.0
        / np.sqrt(2.0)
        * (
            np.einsum(
                "rdia,ja,ji->rd", xpy_coeff_deriv_s1, xpy_coeff_t1, mo_soc_ij[0, :, :]
            )
            + np.einsum(
                "ia,rdja,ji->rd", xpy_coeff_s1, xpy_coeff_deriv_t1, mo_soc_ij[0, :, :]
            )
            + np.einsum(
                "ia,ja,rdji->rd",
                xpy_coeff_s1,
                xpy_coeff_t1,
                mo_soc_ij_deriv[:, :, 0, :, :],
            )
            - np.einsum(
                "rdia,ib,ab->rd", xpy_coeff_deriv_s1, xpy_coeff_t1, mo_soc_ab[0, :, :]
            )
            - np.einsum(
                "ia,rdib,ab->rd", xpy_coeff_s1, xpy_coeff_deriv_t1, mo_soc_ab[0, :, :]
            )
            - np.einsum(
                "ia,ib,rdab->rd",
                xpy_coeff_s1,
                xpy_coeff_t1,
                mo_soc_ab_deriv[:, :, 0, :, :],
            )
        )
    )
    soc_s1t1_1_deriv_im = (
        1.0
        / np.sqrt(2.0)
        * (
            np.einsum(
                "rdia,ja,ji->rd", xpy_coeff_deriv_s1, xpy_coeff_t1, mo_soc_ij[1, :, :]
            )
            + np.einsum(
                "ia,rdja,ji->rd", xpy_coeff_s1, xpy_coeff_deriv_t1, mo_soc_ij[1, :, :]
            )
            + np.einsum(
                "ia,ja,rdji->rd",
                xpy_coeff_s1,
                xpy_coeff_t1,
                mo_soc_ij_deriv[:, :, 1, :, :],
            )
            - np.einsum(
                "rdia,ib,ab->rd", xpy_coeff_deriv_s1, xpy_coeff_t1, mo_soc_ab[1, :, :]
            )
            - np.einsum(
                "ia,rdib,ab->rd", xpy_coeff_s1, xpy_coeff_deriv_t1, mo_soc_ab[1, :, :]
            )
            - np.einsum(
                "ia,ib,rdab->rd",
                xpy_coeff_s1,
                xpy_coeff_t1,
                mo_soc_ab_deriv[:, :, 1, :, :],
            )
        )
    )
    soc_s1t1_1_deriv = soc_s1t1_1_deriv_re + soc_s1t1_1_deriv_im * 1.0j

    soc_s1t1_2_deriv_re = (
        -1.0
        * np.einsum(
            "rdia,ja,ji->rd", xpy_coeff_deriv_s1, xpy_coeff_t1, mo_soc_ij[2, :, :]
        )
        - 1.0
        * np.einsum(
            "ia,rdja,ji->rd", xpy_coeff_s1, xpy_coeff_deriv_t1, mo_soc_ij[2, :, :]
        )
        - 1.0
        * np.einsum(
            "ia,ja,rdji->rd", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ij_deriv[:, :, 2, :, :]
        )
        + 1.0
        * np.einsum(
            "rdia,ib,ab->rd", xpy_coeff_deriv_s1, xpy_coeff_t1, mo_soc_ab[2, :, :]
        )
        + 1.0
        * np.einsum(
            "ia,rdib,ab->rd", xpy_coeff_s1, xpy_coeff_deriv_t1, mo_soc_ab[2, :, :]
        )
        + 1.0
        * np.einsum(
            "ia,ib,rdab->rd", xpy_coeff_s1, xpy_coeff_t1, mo_soc_ab_deriv[:, :, 2, :, :]
        )
    )
    soc_s1t1_2_deriv = soc_s1t1_2_deriv_re + 0.0j

    soc_s1t1_3_deriv_re = -soc_s1t1_1_deriv_re
    soc_s1t1_3_deriv_im = soc_s1t1_1_deriv_im
    soc_s1t1_3_deriv = soc_s1t1_3_deriv_re + soc_s1t1_3_deriv_im * 1.0j

    soc_s1t1_deriv = np.array(
        [soc_s1t1_1_deriv, soc_s1t1_2_deriv, soc_s1t1_3_deriv]
    ).transpose(1, 2, 0)

    soc_s1t1_deriv *= au2wavnum

    return soc_s1t1_deriv
