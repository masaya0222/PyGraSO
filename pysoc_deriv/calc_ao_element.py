import os
import sys
import ctypes
import numpy as np
from pathlib import Path

_cint = np.ctypeslib.load_library(
    "libcint",
    os.path.abspath(
        os.path.join(Path(__file__).resolve().parent, "../thridparty/libcint/build")
    ),
)

from pyscf import gto, lib


def sozeff(atom, zeff_type="one"):
    """
    与えられた原子番号に対して、有効核電荷 (Z_eff) を計算する。
    """
    assert zeff_type in ["one", "orca", "pyscf"], f"{zeff_type=} is not valid"
    neval = {
        1: 1,
        2: 2,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 5,
        8: 6,
        9: 7,
        10: 8,
        11: 1,
        12: 2,
        13: 3,
        14: 4,
        15: 5,
        16: 6,
        17: 7,
        18: 8,
        19: 1,
        20: 2,
        21: 3,
        22: 4,
        23: 5,
        24: 6,
        25: 7,
        26: 8,
        27: 9,
        28: 10,
        29: 11,
        30: 12,
        31: 3,
        32: 4,
        33: 5,
        34: 6,
        35: 7,
        36: 8,
        37: 1,
        38: 2,
        39: 3,
        40: 4,
        41: 5,
        42: 6,
        43: 7,
        44: 8,
        45: 9,
        46: 10,
        47: 11,
        48: 12,
        49: 3,
        50: 4,
        51: 5,
        52: 6,
        53: 7,
        54: 8,
    }
    if zeff_type == "one":
        return atom

    if zeff_type == "pyscf":
        if atom == 1:
            return 1.0
        elif atom == 2:
            return 2.0
        elif 3 <= atom <= 10:
            return (0.2517 + 0.0626 * neval[atom]) * atom
        elif 11 <= atom <= 18:
            return (0.7213 + 0.0144 * neval[atom]) * atom
        elif (19 <= atom <= 20) or (31 <= atom <= 36):
            return (0.8791 + 0.0039 * neval[atom]) * atom
        elif (37 <= atom <= 38) or (49 <= atom <= 54):
            return (0.9228 + 0.0017 * neval[atom]) * atom
        elif atom == 26:
            return 0.583289 * atom
        elif atom == 30:
            return 330.0
        elif 21 <= atom <= 30:
            return atom * (0.385 + 0.025 * (neval[atom] - 2))
        elif 39 <= atom <= 48:
            return atom * (4.680 + 0.060 * (neval[atom] - 2))
        elif atom == 72:
            return 1025.28
        elif atom == 73:
            return 1049.74
        elif atom == 74:
            return 1074.48
        elif atom == 75:
            return 1099.5
        elif atom == 76:
            return 1124.8
        elif atom == 77:
            return 1150.38
        elif atom == 78:
            return 1176.24
        elif atom == 79:
            return 1202.38
        elif atom == 80:
            return 1228.8
        else:
            raise ValueError(f"SOZEFF is not available for atomic number {atom}")
    if zeff_type == "orca":
        if atom == 1:
            return 1.0
        elif atom == 2:
            return 2.0
        elif 3 <= atom < 10:
            return (0.4 + 0.05 * neval[atom]) * atom
        elif 11 <= atom <= 18:
            return (0.925 - 0.0125 * neval[atom]) * atom
        else:
            raise ValueError(f"SOZEFF is not available for atomic number {atom}")


class calc_ao_element:
    def __init__(self, atoms, coordinates, basis=None):
        self.atoms = atoms
        self.coordinates = coordinates
        self.basis = basis
        self.mol = self.setup_mol()
        self.permu_basis, self.basis_idx = self.setup_permu()
        self.nbasis = len(self.permu_basis)
        self.natoms = len(atoms)

        self._ao_ovlp = None
        self._ao_norm = None
        self.get_ao_ovlp()

        self._ao_dip = None
        self._ao_dip_deriv = None

        self._ao_soc = None
        self._ao_soc_deriv = None

    def setup_mol(self):
        mol = gto.Mole()
        mol.atom = [
            (symbol, coord) for symbol, coord in zip(self.atoms, self.coordinates)
        ]
        mol.unit = "angstrom"
        mol.basis = self.basis
        mol.symmetry = False
        mol.cart = True
        mol.build()
        return mol

    def setup_permu(self):
        # In pyscf, the order of p orbital is [px,py,pz], the order of d orbital is [dxx,dxy,dxz,dyy,dyz,dzz]
        # In Gaussian16, the order of p orbital is [px,py,pz], the order of d orbital is [dxx,dyy,dzz,dxy,dxz,dyz]
        permu_tmp = []
        basis_idx = []
        angular2order = [[0], [0, 1, 2], [0, 3, 5, 1, 2, 4]]
        angular2length = [1, 3, 6]
        permu = []
        base_idx = 0
        for i in range(self.mol.natm):
            basis = self.mol.basis[self.mol.atom_symbol(i)]
            angular_counter = [0, 0, 0]

            pyscf2gau = []
            for j, bas in enumerate(basis):
                l = bas[0]
                assert l < 3, "This code is not implemented for f function"
                pyscf2gau.append([l * 10 + angular_counter[l], j])
                angular_counter[l] += 1
                basis_idx.extend([i] * angular2length[l])

            pyscf2gau = np.array([j for (i, j) in sorted(pyscf2gau)])
            permu_tmp = []
            for idx in pyscf2gau:
                bas = basis[idx]
                l = bas[0]
                permu_tmp.append([idx, [m + base_idx for m in angular2order[l]]])
                base_idx += angular2length[l]
            permu_tmp = [j for (i, j) in sorted(permu_tmp)]
            for pt in permu_tmp:
                permu += pt
        permu = np.array(permu)
        return permu, basis_idx

    def get_ao_ovlp(self):
        if self._ao_ovlp is not None:
            return self._ao_ovlp
        try:
            intor_name = "int1e_ovlp_cart"
            ao_ovlp_tmp = gto.getints(
                intor_name, self.mol._atm, self.mol._bas, env=self.mol._env
            )
            ao_ovlp = ao_ovlp_tmp[np.ix_(self.permu_basis, self.permu_basis)]
            self._ao_norm = np.sqrt(np.diag(ao_ovlp))
            ao_ovlp = ao_ovlp / self._ao_norm[None, :] / self._ao_norm[:, None]
            self._ao_ovlp = ao_ovlp
        except Exception as e:
            raise ValueError(f"Error occur while reading and ao_ovlp:{e}")
        return self._ao_ovlp

    def get_ao_dip(self):
        if self._ao_dip is not None:
            return self._ao_dip
        try:
            intor_name = "int1e_r_cart"
            ao_dip_tmp = gto.getints(
                intor_name, self.mol._atm, self.mol._bas, env=self.mol._env
            )
            ao_dip = np.real(
                ao_dip_tmp[np.ix_(np.arange(3), self.permu_basis, self.permu_basis)]
            )
            ao_dip = (
                ao_dip / self._ao_norm[None, None, :] / self._ao_norm[None, :, None]
            )
            self._ao_dip = ao_dip
        except Exception as e:
            raise ValueError(f"Error occur while reading and ao_dip:{e}")
        return self._ao_dip

    def get_ao_dip_deriv(self):
        if self._ao_dip_deriv is not None:
            return self._ao_dip_deriv
        try:
            intor_name = "int1e_irp_cart"
            ao_dip_deriv_tmp = gto.getints(
                intor_name, self.mol._atm, self.mol._bas, env=self.mol._env
            )
            ao_dip_deriv_tmp = np.real(
                ao_dip_deriv_tmp[
                    np.ix_(np.arange(9), self.permu_basis, self.permu_basis)
                ]
            ).reshape(3, 3, self.nbasis, self.nbasis)
            ao_dip_deriv = np.zeros((self.natoms, 3, 3, self.nbasis, self.nbasis))
            for i in range(3):  # x,y,z
                for j in range(3):  # dx,dy,dz
                    for k in range(self.nbasis):
                        for l in range(self.nbasis):
                            v = ao_dip_deriv_tmp[i, j, k, l]
                            ao_dip_deriv[self.basis_idx[l], j, i, k, l] += -v
                            ao_dip_deriv[self.basis_idx[l], j, i, l, k] += -v
            ao_dip_deriv = (
                ao_dip_deriv
                / self._ao_norm[None, None, None, None, :]
                / self._ao_norm[None, None, None, :, None]
            )
            self._ao_dip_deriv = ao_dip_deriv
        except Exception as e:
            raise ValueError(f"Error occur while reading ao_dip_deriv:{e}")
        return self._ao_dip_deriv

    def get_ao_soc_old(self):
        if self._ao_soc is not None:
            return self._ao_soc
        try:
            intor_name = "int1e_pnucxp_cart"

            ao_soc_tmp = gto.getints(
                intor_name, self.mol._atm, self.mol._bas, env=self.mol._env
            )
            ao_soc = -1.0 * np.real(
                ao_soc_tmp[np.ix_(np.arange(3), self.permu_basis, self.permu_basis)]
            )
            ao_soc = (
                ao_soc / self._ao_norm[None, None, :] / self._ao_norm[None, :, None]
            )
            self._ao_soc = ao_soc

        except Exception as e:
            raise ValueError(f"Error occur while reading ao_soc:{e}")
        return self._ao_soc

    def get_ao_soc(self, Z="one"):
        if self._ao_soc is not None:
            return self._ao_soc
        try:
            zeff_list = [
                sozeff(self.mol.atom_charge(i), zeff_type=Z) for i in range(self.natoms)
            ]

            ao_soc = []

            intor_name = "int1e_prinvxp_cart"
            for k in range(self.mol.natm):
                self.mol.set_rinv_origin(self.mol.atom_coord(k))
                ao_soc_tmp = gto.getints(
                    intor_name, self.mol._atm, self.mol._bas, env=self.mol._env
                )
                ao_soc_tmp = 1.0 * np.real(
                    ao_soc_tmp[np.ix_(np.arange(3), self.permu_basis, self.permu_basis)]
                )
                ao_soc_tmp = (
                    ao_soc_tmp / self._ao_norm[None, None, :] / self._ao_norm[None, :, None]
                )
                ao_soc.append(ao_soc_tmp * zeff_list[k])
            ao_soc = sum(ao_soc)
            self._ao_soc = ao_soc
        except Exception as e:
            raise ValueError(f"Error occur while reading ao_soc:{e}")
        return self._ao_soc

    def get_ao_soc_deriv(self):
        if self._ao_soc_deriv is not None:
            return self._ao_soc_deriv
        try:
            intor_name = "cint1e_prinvxpp_cart"
            fn = getattr(_cint, intor_name)

            mat = np.zeros(
                (self.natoms, 3, 3, self.nbasis, self.nbasis)
            )  # atom, xyz, q, i, j
            am2nb = [1, 3, 6]
            for k in range(self.mol.natm):
                self.mol.set_rinv_origin(self.mol.atom_coord(k))
                Z_k = self.mol._atm[k, 0]
                idx_i = 0
                for i in range(self.mol.nbas):
                    am_i = self.mol._bas[i, 1]
                    nb_i = am2nb[am_i]
                    idx_j = 0
                    for j in range(self.mol.nbas):
                        am_j = self.mol._bas[j, 1]
                        nb_j = am2nb[am_j]
                        buf = np.zeros((3, 3, nb_j, nb_i))
                        fn(
                            buf.ctypes.data_as(ctypes.c_void_p),
                            (ctypes.c_int * 2)(i, j),
                            self.mol._atm.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(self.mol.natm),
                            self.mol._bas.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(self.mol.nbas),
                            self.mol._env.ctypes.data_as(ctypes.c_void_p),
                        )
                        buf = buf.transpose(1, 0, 3, 2)
                        mat[
                            k,
                            :,
                            :,
                            idx_i : idx_i + nb_i,
                            idx_j : idx_j + nb_j,
                        ] = buf * Z_k
                        idx_j += nb_j
                    idx_i += nb_i

            mat = 1.0 * np.real(
                mat[
                    np.ix_(
                        np.arange(self.natoms),
                        np.arange(3),
                        np.arange(3),
                        self.permu_basis,
                        self.permu_basis,
                    )
                ]
            )  # atom, q, xyz, i, j

            mat_t = -mat.transpose(0, 1, 2, 4, 3)
            res_l = np.sum(mat_t, axis=0)
            res_r = np.sum(mat, axis=0)
            res = mat + mat_t  # atom, q, xyz, i, j

            bas2nuc = []
            for i in range(self.mol.nbas):
                bas2nuc.extend([self.mol._bas[i, 0]] * (am2nb[self.mol._bas[i, 1]]))
            bas2nuc = np.array(bas2nuc)[self.permu_basis]

            for i, t_i in enumerate(bas2nuc):
                res[t_i, :, :, i, :] -= res_l[:, :, i, :]
                res[t_i, :, :, :, i] -= res_r[:, :, :, i]
            ao_soc_deriv = (
                res
                / self._ao_norm[None, None, None, None, :]
                / self._ao_norm[None, None, None, :, None]
            )
            self._ao_soc_deriv = ao_soc_deriv
        except Exception as e:
            raise ValueError(f"Error occur while reading ao_soc_deriv:{e}")
        return self._ao_soc_deriv
