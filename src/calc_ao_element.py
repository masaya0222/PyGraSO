import os
import sys
import ctypes
import numpy as np
import numpy

_cint = np.ctypeslib.load_library(
    "libcint", os.path.abspath(os.path.join(__file__, "/home/hagai/libcint/build"))
)
from pyscf import gto, lib


def make_cintopt(atm, bas, env, intor):
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order="C")
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order="C")
    c_env = numpy.asarray(env, dtype=numpy.double, order="C")
    natm = c_atm.shape[0]
    nbas = c_bas.shape[0]
    cintopt = lib.c_null_ptr()
    foptinit = getattr(_cint, intor + "_optimizer")
    foptinit(
        ctypes.byref(cintopt),
        c_atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(natm),
        c_bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbas),
        c_env.ctypes.data_as(ctypes.c_void_p),
    )
    return cintopt


class calc_ao_element:
    def __init__(self, atoms, coordinates):
        self.atoms = atoms
        self.coordinates = coordinates
        self.mol = self.setup_mol()
        self.permu_basis, self.basis_idx = self.setup_permu()
        self.nbasis = len(self.permu_basis)
        self.natoms = len(atoms)

        self._ao_dip = None
        self._ao_dip_deriv = None

    def setup_mol(self):
        mol = gto.Mole()
        mol.atom = [
            (symbol, coord) for symbol, coord in zip(self.atoms, self.coordinates)
        ]
        mol.unit = "angstrom"
        mol.basis = "6-31G"
        mol.symmetry = False
        mol.cart = True
        mol.build()
        return mol

    def setup_permu(self):
        permu_tmp = []
        basis_idx = []
        angular2num = {"s": "0", "px": "1", "py": "2", "pz": "3"}
        for i, ao_label in enumerate(self.mol.ao_labels()):
            atom_idx, atom_symbol, orb_type = ao_label.split()
            for angular_symbol in angular2num.keys():
                if angular_symbol in orb_type:
                    orb_type = orb_type.replace(
                        angular_symbol, angular2num[angular_symbol]
                    )
            permu_tmp.append([atom_idx + orb_type, i])
            basis_idx.append(int(atom_idx))
        permu = np.array([j for (i, j) in sorted(permu_tmp)])
        return permu, basis_idx

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
            self._ao_dip = ao_dip
        except Exception as e:
            raise ValueError(f"Error occur while reading ao_ovlp and ao_dip:{e}")
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
            self._ao_dip_deriv = ao_dip_deriv
        except Exception as e:
            raise ValueError(f"Error occur while reading ao_ovlp and ao_dip:{e}")
        return self._ao_dip_deriv

    def test(self):
        # intor_name = "cint1e_nuc_cart"
        # intor_name = "cint1e_so_cart"
        intor_name = "cint1e_pnucxp_cart"
        # intor_name = "cint1e_ia01p_cart"
        # intor_name = "cint1e_r_cart"
        intor_name_tmp = "cint1e_nuc_cart"
        intor_name_tmp = intor_name
        print(self.mol._bas)

        fn1 = getattr(_cint, intor_name)
        for i in range(self.mol.nbas):
            am_i = self.mol._bas[i, 1]
            for j in range(self.mol.nbas):
                ref = self.mol.intor_by_shell(intor_name_tmp, [i, j], comp=3)
                # buf = np.empty_like(ref)
                am_j = self.mol._bas[j, 1]
                buf = np.zeros((3, am_i * 2 + 1, am_j * 2 + 1))
                # buf = np.empty((3,am_i*2+1,am_j*2+1))
                fn1(
                    buf.ctypes.data_as(ctypes.c_void_p),
                    (ctypes.c_int * 2)(i, j),
                    self.mol._atm.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(self.mol.natm),
                    self.mol._bas.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(self.mol.nbas),
                    self.mol._env.ctypes.data_as(ctypes.c_void_p),
                )
                print(i, j, buf.shape)
                print(ref)
                print(buf)
        exit()
        print(f"{self.mol._atm=}")
        print(f"{self.mol._bas=}")
        print(f"{self.mol._env=}")
        self.mol._atm[0, 0] = 1.0
        self.mol._atm[1, 0] = 1.0
        tmp = gto.getints(intor_name, self.mol._atm, self.mol._bas, env=self.mol._env)
        tmp = np.real(tmp)
        print(tmp)
