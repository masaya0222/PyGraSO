import numpy as np

from .parser import gaussian_perser
from .calc_ao_element import calc_ao_element

fine_stru = 7.297352568e-3
au2wavnum = 219474.6
coeff_thresh = 1e-5


def calc_nac_tntm(
    atoms, coordinates, charge, g_parser_tn, g_parser_tm, normalize=False, basis=None
):
    mo_coeff = g_parser_tn.get_mo_coeff()
    mo_coeff_deriv = g_parser_tn.get_mo_coeff_deriv()
    mo_coeff_i = mo_coeff[: g_parser_tn.nfc + g_parser_tn.noa, :]
    mo_coeff_a = mo_coeff[g_parser_tn.nfc + g_parser_tn.noa :, :]
    mo_coeff_deriv_i = mo_coeff_deriv[:, :, : g_parser_tn.nfc + g_parser_tn.noa, :]
    mo_coeff_deriv_a = mo_coeff_deriv[:, :, g_parser_tn.nfc + g_parser_tn.noa :, :]

    if g_parser_tn.nxy == 1:
        x_coeff_tn, _ = g_parser_tn.get_xy_coeff()
    elif g_parser_tn.nxy == 2:
        x_coeff_tn, y_coeff_tn = g_parser_tn.get_xy_coeff()
    else:
        raise ValueError(f"g_parser_tn.nxy must be 1 or 2 but {g_parser_tn.nxy}")

    if normalize:
        raise ValueError("Normalize can't be used for NAC calculation")

    if g_parser_tn.nxy == 1:
        x_coeff_deriv_tn, _ = g_parser_tn.get_xy_coeff_deriv()
    elif g_parser_tn.nxy == 2:
        x_coeff_deriv_tn, y_coeff_deriv_tn = g_parser_tn.get_xy_coeff_deriv()
    else:
        raise ValueError(f"g_parser_tn.nxy must be 1 or 2 but {g_parser_tn.nxy}")

    if g_parser_tm.nxy == 1:
        x_coeff_tm, _ = g_parser_tm.get_xy_coeff()
    elif g_parser_tm.nxy == 2:
        x_coeff_tm, y_coeff_tm = g_parser_tm.get_xy_coeff()
    else:
        raise ValueError(f"g_parser_tm.nxy must be 1 or 2 but {g_parser_tm.nxy}")

    if g_parser_tm.nxy == 1:
        x_coeff_deriv_tm, _ = g_parser_tm.get_xy_coeff_deriv()
    elif g_parser_tm.nxy == 2:
        x_coeff_deriv_tm, y_coeff_deriv_tm = g_parser_tm.get_xy_coeff_deriv()
    else:
        raise ValueError(f"g_parser_tm.nxy must be 1 or 2 but {g_parser_tm.nxy}")

    ao_calculator = calc_ao_element(
        atoms, coordinates, charge, basis=g_parser_tn.read_basis()
    )
    ao_ovlp = ao_calculator.get_ao_ovlp()
    ao_ovlp_deriv = ao_calculator.get_ao_ovlp_deriv()
    # mo_nac_ab = np.einsum("ap,rdbq,pq->rdab",mo_coeff_a,mo_coeff_deriv_a,ao_ovlp) \
    #     + np.einsum("ap,bq,rdpq->rdab",mo_coeff_a,mo_coeff_a, ao_ovlp_deriv)
    mo_nac_ab = (mo_coeff_deriv_a @ (ao_ovlp.T) @ (mo_coeff_a.T)).transpose(
        0, 1, 3, 2
    ) + (
        (ao_ovlp_deriv @ (mo_coeff_a.T)).transpose(0, 1, 3, 2) @ (mo_coeff_a.T)
    ).transpose(0, 1, 3, 2)
    # mo_nac_ij = np.einsum("ip,rdjq,pq->rdij",mo_coeff_i,mo_coeff_deriv_i,ao_ovlp) \
    #     + np.einsum("ip,jq,rdpq",mo_coeff_i,mo_coeff_i, ao_ovlp_deriv)
    mo_nac_ij = (mo_coeff_deriv_i @ (ao_ovlp.T) @ (mo_coeff_i.T)).transpose(
        0, 1, 3, 2
    ) + (
        (ao_ovlp_deriv @ (mo_coeff_i.T)).transpose(0, 1, 3, 2) @ (mo_coeff_i.T)
    ).transpose(0, 1, 3, 2)

    # nac_tntm_1 = np.einsum("ia, rdia->rd", x_coeff_tn, x_coeff_deriv_tm)
    nac_tntm_1 = np.trace(x_coeff_deriv_tm @ (x_coeff_tn.T), axis1=-2, axis2=-1)
    if g_parser_tm.nxy == 2:
        # nac_tntm_1 -= np.einsum("ia, rdia->rd", y_coeff_tn, y_coeff_deriv_tm)
        nac_tntm_1 -= np.trace(y_coeff_deriv_tm @ (y_coeff_tn).T, axis1=-2, axis2=-1)
    # rab = np.einsum("ia,ib->ab", x_coeff_tn,x_coeff_tm)
    rab = (x_coeff_tn.T) @ x_coeff_tm
    if g_parser_tm.nxy == 2:
        # rab += np.einsum("ia,ib->ab", y_coeff_tn, y_coeff_tm)
        rab += ((y_coeff_tn.T) @ y_coeff_tm).T
    # nac_tntm_2 = np.einsum("rdab,ab->rd", mo_nac_ab, rab)
    nac_tntm_2 = np.trace(mo_nac_ab @ (rab.T), axis1=-2, axis2=-1)

    # rij = -np.einsum("ia,ja->ij", x_coeff_tn,x_coeff_tm)
    rij = -(x_coeff_tn @ (x_coeff_tm.T)).T
    if g_parser_tm.nxy == 2:
        # rij += np.einsum("ia,ja->ij", y_coeff_tn, y_coeff_tm)
        rij += -y_coeff_tn @ (y_coeff_tm.T)
    # nac_tntm_3 = np.einsum("rdij,ij->rd",mo_nac_ij, rij)
    nac_tntm_3 = np.trace(mo_nac_ij @ (rij.T), axis1=-2, axis2=-1)

    nac_tntm = nac_tntm_1 + nac_tntm_2 + nac_tntm_3

    return nac_tntm * 2


def calc_nac_t0tm(atoms, coordinates, charge, g_parser_tm, normalize=False, basis=None):
    mo_coeff = g_parser_tm.get_mo_coeff()
    mo_coeff_deriv = g_parser_tm.get_mo_coeff_deriv()
    mo_coeff_i = mo_coeff[: g_parser_tm.nfc + g_parser_tm.noa, :]
    mo_coeff_a = mo_coeff[g_parser_tm.nfc + g_parser_tm.noa :, :]
    mo_coeff_deriv_i = mo_coeff_deriv[:, :, : g_parser_tm.nfc + g_parser_tm.noa, :]
    mo_coeff_deriv_a = mo_coeff_deriv[:, :, g_parser_tm.nfc + g_parser_tm.noa :, :]

    if g_parser_tm.nxy == 1:
        xmy_coeff_tm, _ = g_parser_tm.get_xy_coeff()
    elif g_parser_tm.nxy == 2:
        x_coeff_tm, y_coeff_tm = g_parser_tm.get_xy_coeff()
        xmy_coeff_tm = x_coeff_tm - y_coeff_tm
    else:
        raise ValueError(f"g_parser_tm.nxy must be 1 or 2 but {g_parser_tm.nxy}")

    if normalize:
        raise ValueError("Normalize can't be used for NAC calculation")

    ao_calculator = calc_ao_element(
        atoms, coordinates, charge, basis=g_parser_tm.read_basis()
    )
    ao_ovlp = ao_calculator.get_ao_ovlp()
    ao_ovlp_deriv = ao_calculator.get_ao_ovlp_deriv()
    # mo_nac_ia = np.einsum("ip,rdaq,pq->rdia",mo_coeff_i,mo_coeff_deriv_a,ao_ovlp) \
    #     + np.einsum("ip,aq,rdpq->rdia",mo_coeff_i,mo_coeff_a, ao_ovlp_deriv)
    mo_nac_ia = (mo_coeff_deriv_a @ (ao_ovlp.T) @ (mo_coeff_i.T)).transpose(
        0, 1, 3, 2
    ) + (
        (ao_ovlp_deriv @ (mo_coeff_a.T)).transpose(0, 1, 3, 2) @ (mo_coeff_i.T)
    ).transpose(0, 1, 3, 2)

    nac_t0tm = np.trace(mo_nac_ia @ (xmy_coeff_tm.T), axis1=-2, axis2=-1)
    return nac_t0tm * 2


def calc_nac_tnt0(atoms, coordinates, charge, g_parser_tn, normalize=False, basis=None):
    return -calc_nac_t0tm(
        atoms, coordinates, charge, g_parser_tn, normalize=normalize, basis=basis
    )
