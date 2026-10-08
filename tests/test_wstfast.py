"""Fast checks of the measurement, the theory building blocks and the desilike graph."""

import numpy as np
import pytest
from scipy.special import gammaln

import wstfast  # noqa: F401
from wstfast import WSTConfig, select_coefficients
from wstfast.measure import Lattice, PowerMultipoles, measure_wst
from wstfast.theory import Assembly, all_coefficients
from wstfast.theory.perturbation import OneLoopMatter, log_interpolator


def gaussian_field(nmesh=64, seed=1):
    lattice = Lattice(nmesh)
    k = lattice.kmag
    pk = np.where(k > 0, (k + 1e-2) ** -1.5 * np.exp(-((k / 1.5) ** 2)), 0.0)
    noise = np.fft.rfftn(np.random.default_rng(seed).standard_normal((nmesh,) * 3))
    return np.fft.irfftn(noise * np.sqrt(pk), s=(nmesh,) * 3).astype(np.float32), lattice


def test_s1_of_gaussian_field_is_chi_moment():
    delta, lattice = gaussian_field()
    config = WSTConfig(J=2, L=2, sigma0=0.8, cellsize=1.0)
    result = {name: value[0] for name, value in measure_wst(delta, config, lattice=lattice).items()}
    fk = np.fft.rfftn(delta)
    mult = np.full(fk.shape, 2.0)
    mult[..., 0] = 1.0
    mult[..., -1] = 1.0
    for j in range(config.J + 1):
        for ell in range(config.L + 1):
            n = 2 * ell + 1
            weight = lattice.radial(config.sigma0 * 2**j, ell) ** 2
            s2 = np.sum(mult * np.abs(fk) ** 2 * weight) / delta.size**2 / n  # Parseval
            expected = (2 * s2) ** (config.q / 2) * np.exp(gammaln((n + config.q) / 2) - gammaln(n / 2))
            assert result["S1"][j, ell] == pytest.approx(expected, rel=2e-2)
    assert np.isnan(result["S2"][1, 0, 0]) and np.isfinite(result["S2"][0, 1, 1])


def test_one_loop_signs():
    k = np.geomspace(1e-4, 10, 512)
    pk = log_interpolator(k, 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.5))
    loop = OneLoopMatter(np.array([0.05, 0.3]), nq=128, nq13=512)
    assert np.all(np.asarray(loop.p22(pk)) > 0)
    assert np.all(np.asarray(loop.p13(pk)) < 0)


def test_assembly_keeps_selection_order():
    config = WSTConfig()
    selected = select_coefficients(config)[::-1]
    assembly = Assembly(config, selected)
    basis = all_coefficients(config)
    n1 = sum(c.kind == "S1" for c in basis)
    s1_terms = np.tile(np.arange(1.0, n1 + 1.0)[:, None], (1, 6)) * np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s21_terms = np.ones((len(basis) - n1, 3))
    out = np.asarray(assembly(s1_terms, s21_terms, 0.0, np.zeros(len(assembly.noise_keys))))
    s1_rows = {c: i for i, c in enumerate(c for c in basis if c.kind == "S1")}
    for value, c in zip(out, selected):
        if c.kind == "S1":
            n = 2 * c.ell + 1
            variance = (s1_rows[c] + 1.0) / n
            gamma = np.exp(gammaln((n + config.q) / 2) - gammaln(n / 2))
            assert value == pytest.approx(gamma * (2 * variance) ** (config.q / 2))


@pytest.mark.slow
def test_likelihood_graph_evaluates():
    from desilike import build

    from wstfast.calculators import WSTLikelihood, WSTTheory, build_cosmology

    config = WSTConfig()
    coefficients = select_coefficients(config)
    theory = WSTTheory(coefficients, config=config, cosmo=build_cosmology(), shotnoise=7.45)
    prediction = np.asarray(build(theory)({}))
    assert prediction.shape == (len(coefficients),) and np.all(np.isfinite(prediction))
    likelihood = WSTLikelihood(theory, prediction, np.diag((0.01 * prediction) ** 2))
    assert float(build(likelihood)({})) == pytest.approx(0.0, abs=1e-8)


def test_half_octave_superset_contains_dyadic_configuration():
    delta, lattice = gaussian_field(nmesh=48)
    dyadic = WSTConfig(J=2, L=2, sigma0=0.8, cellsize=1.0)
    superset = WSTConfig(J=4, L=2, sigma0=0.8, step=2**0.5, min_dj=2, cellsize=1.0)
    qs = [0.5, 0.8]
    a = measure_wst(delta, dyadic, qs=qs, lattice=lattice)
    b = measure_wst(delta, superset, qs=qs, lattice=lattice, spectra=PowerMultipoles(lattice, 48.0, kmax=1.0))
    np.testing.assert_allclose(b["S1"][:, ::2], a["S1"], rtol=1e-5)
    for j1, j2 in dyadic.second_layer_pairs():
        np.testing.assert_allclose(b["S2"][:, 2 * j1, 2 * j2], a["S2"][:, j1, j2], rtol=1e-5)
    # First-layer auto-spectra are stored and positive.
    assert np.all(b["PUU"][:, :, 0] > 0)


def test_torch_backend_matches_numpy():
    torch = pytest.importorskip("torch")
    from wstfast.measure_torch import TorchLattice, TorchPowerMultipoles, measure_wst_torch, paint_cic_torch
    from wstfast.quijote import paint_cic

    delta, lattice = gaussian_field(nmesh=32)
    config = WSTConfig(J=3, L=3, L2=2, min_dj=2, sigma0=0.8, step=2**0.5, cellsize=1.0)
    qs = [0.5, 0.8, 2.0]
    reference = measure_wst(delta, config, qs=qs, lattice=lattice, spectra=PowerMultipoles(lattice, 32.0, kmax=1.5))
    tlattice = TorchLattice(32, device=torch.device("cpu"))
    result = measure_wst_torch(delta, config, qs=qs, lattice=tlattice,
                               spectra=TorchPowerMultipoles(tlattice, 32.0, kmax=1.5))
    assert set(result) == set(reference)
    for name in ("S0", "S1", "S2", "Umean", "Pdd", "PUU"):
        np.testing.assert_allclose(result[name], reference[name], rtol=2e-4, atol=1e-9)
    np.testing.assert_allclose(result["PUd"], reference["PUd"], rtol=2e-3, atol=1e-6 * np.abs(reference["PUU"]).max())

    positions = [np.random.default_rng(2).uniform(0, 100.0, (5000, 3))]
    np.testing.assert_allclose(paint_cic_torch(positions, 16, 100.0, device="cpu"), paint_cic(positions, 16, 100.0),
                               atol=1e-5)


def test_tree_trispectrum_is_symmetric_and_f3_conserves_momentum():
    import jax.numpy as jnp

    from wstfast.theory.cumulants import _f3, tree_trispectrum

    rng = np.random.default_rng(3)
    k = list(rng.normal(size=(3, 5, 3)) * 0.1)
    k.append(-sum(k))
    pk = log_interpolator(np.geomspace(1e-4, 10, 512), 2e4 / (1 + (np.geomspace(1e-4, 10, 512) / 0.02) ** 2))
    reference = np.asarray(tree_trispectrum([jnp.asarray(x) for x in k], pk))
    for order in ((1, 0, 2, 3), (2, 3, 0, 1), (3, 1, 2, 0)):
        np.testing.assert_allclose(np.asarray(tree_trispectrum([jnp.asarray(k[i]) for i in order], pk)), reference,
                                   rtol=1e-10)
    # F3(q1, q2, q3) ~ |q1 + q2 + q3|^2 as the total momentum vanishes.
    q1, q2 = jnp.asarray([[0.1, 0.02, 0.0]]), jnp.asarray([[-0.03, 0.08, 0.05]])
    small = [float(_f3(q1, q2, -q1 - q2 + jnp.asarray([[eps, 0.0, 0.0]]))[0]) for eps in (1e-3, 2e-3)]
    assert small[1] / small[0] == pytest.approx(4.0, rel=0.05)


def test_leg_sampler_integrates_gaussian():
    import jax.numpy as jnp

    from wstfast.theory.modulus_noise import LegSampler

    # Legs u1, u2, u1 + u2 - U, u1, u2: the integrand exp(-sum |leg|^2 / 2) = exp(-(v.M v - 2 b.v + U^2) / 2)
    # per Cartesian component, with M = [[3, 1], [1, 3]] and b = (U, U) for the z component only.
    sampler = LegSampler([[1, 0], [0, 1], [1, 1], [1, 0], [0, 1]], [0, 0, -1, 0, 0], 1.0, 2**14, 0)
    unode = 0.7
    value = sampler.integrate(lambda legs: jnp.exp(-0.5 * sum(jnp.sum(x**2, axis=-1) for x in legs)),
                              np.array([unode]), 1.0)[0]
    M, b = np.array([[3.0, 1.0], [1.0, 3.0]]), np.full(2, unode)
    exact = np.exp(0.5 * (b @ np.linalg.solve(M, b) - unode**2)) / (np.linalg.det(M) ** 1.5 * (2 * np.pi) ** 3)
    assert float(value) == pytest.approx(exact, rel=1e-3)


def test_line_of_sight_resolved_moduli():
    delta, lattice = gaussian_field(nmesh=48, seed=4)
    config = WSTConfig(J=2, L=2, sigma0=0.8, min_dj=1, cellsize=1.0)
    qs = [0.8, 2.0]
    spectra = PowerMultipoles(lattice, 48.0, kmax=1.0)
    plain = measure_wst(delta, config, qs=qs, lattice=lattice, spectra=spectra)
    out = measure_wst(delta, config, qs=qs, lattice=lattice, spectra=spectra, los=True)
    # The m-summed outputs do not change.
    for name in ("S1", "S2", "PUU", "PUd", "Umean"):
        np.testing.assert_allclose(out[name], plain[name], rtol=1e-6)
    # sum_|m| U_|m|^2 = U^2, so at q = 2 the |m|-resolved S1 add up to S1 (and likewise in the second layer).
    np.testing.assert_allclose(np.nansum(out["S1m"][1], axis=-1), out["S1"][1], rtol=1e-5)
    for j1, j2 in config.second_layer_pairs():
        for ell in range(config.lmax2 + 1):
            total = sum(np.nansum(out["S2m"][1, j1, j2, ell, m1]) for m1 in range(ell + 1))
            assert np.isfinite(total)
    # Isotropic Gaussian field: each |m| block is a chi variable with 1 (m = 0) or 2 components.
    fk = np.fft.rfftn(delta)
    mult = np.full(fk.shape, 2.0)
    mult[..., 0] = mult[..., -1] = 1.0
    for j in range(config.J + 1):
        for ell in range(config.L + 1):
            n = 2 * ell + 1
            s2 = np.sum(mult * np.abs(fk) ** 2 * lattice.radial(config.sigma0 * 2**j, ell) ** 2) / delta.size**2 / n
            for m in range(ell + 1):
                nm = 1 if m == 0 else 2
                expected = (2 * s2) ** 0.4 * np.exp(gammaln((nm + 0.8) / 2) - gammaln(nm / 2))
                assert out["S1m"][0, j, ell, m] == pytest.approx(expected, rel=3e-2)
            assert np.all(np.isnan(out["S1m"][:, j, ell, ell + 1:]))


def test_torch_line_of_sight_matches_numpy():
    torch = pytest.importorskip("torch")
    from wstfast.measure_torch import TorchLattice, TorchPowerMultipoles, measure_wst_torch

    delta, lattice = gaussian_field(nmesh=32, seed=5)
    config = WSTConfig(J=3, L=2, L2=2, min_dj=2, sigma0=0.8, step=2**0.5, cellsize=1.0)
    qs = [0.8, 2.0]
    reference = measure_wst(delta, config, qs=qs, lattice=lattice, spectra=PowerMultipoles(lattice, 32.0, kmax=1.5),
                            los=True)
    tlattice = TorchLattice(32, device=torch.device("cpu"))
    result = measure_wst_torch(delta, config, qs=qs, lattice=tlattice,
                               spectra=TorchPowerMultipoles(tlattice, 32.0, kmax=1.5), los=True)
    assert set(result) == set(reference)
    for name in ("S1m", "S2m", "Umean_m", "PUU_m"):
        np.testing.assert_allclose(result[name], reference[name], rtol=2e-4, atol=1e-9)
