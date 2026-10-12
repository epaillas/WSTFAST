"""Survey-geometry WST estimator: box limit, local line of sight, painting."""

import numpy as np
import pytest

from wstfast import WSTConfig
from wstfast.measure import Lattice, measure_wst
from wstfast.survey import (LOSRotation, SurveyMesh, _harmonic_matrix, axis_rotation, erode, harmonic_rotation,
                            m_powers, measure_wst_survey, paint_cic_weighted)

CONFIG = WSTConfig(J=2, L=3, L2=2, sigma0=1.5, min_dj=1, cellsize=1.0)  # away from the Nyquist plane
QS = [0.8, 1.0]


def anisotropic_field(nmesh=32, axis=2, seed=3):
    """Gaussian field with a Kaiser-like anisotropy along ``axis``."""
    k = 2 * np.pi * np.fft.fftfreq(nmesh)
    kz = 2 * np.pi * np.fft.rfftfreq(nmesh)
    kvec = np.meshgrid(k, k, kz, indexing="ij")
    kmag = np.sqrt(sum(c**2 for c in kvec))
    mu2 = np.where(kmag > 0, kvec[axis] ** 2 / np.where(kmag > 0, kmag, 1) ** 2, 0)
    pk = np.where(kmag > 0, (kmag + 0.05) ** -1.5 * np.exp(-(kmag / 1.5) ** 2) * (1 + 1.5 * mu2) ** 2, 0)
    noise = np.fft.rfftn(np.random.default_rng(seed).standard_normal((nmesh,) * 3))
    return np.fft.irfftn(noise * np.sqrt(pk), s=(nmesh,) * 3, axes=(0, 1, 2)).astype(np.float32)


def assert_same(box, survey, keys=("S0", "S1", "S2", "S1m", "S2m")):
    for key in keys:
        np.testing.assert_allclose(survey[key], box[key], rtol=2e-5, equal_nan=True, err_msg=key)


def test_full_box_reproduces_measure_wst():
    delta = anisotropic_field()
    lattice = Lattice(delta.shape[0])
    box = measure_wst(delta, CONFIG, qs=QS, lattice=lattice, los=True)
    assert_same(box, measure_wst_survey(delta, CONFIG, qs=QS, lattice=lattice, los="z"))
    assert_same(box, measure_wst_survey(delta, CONFIG, qs=QS, lattice=lattice), keys=("S0", "S1", "S2"))


@pytest.mark.parametrize("axis, vector", [(0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0])])
def test_constant_line_of_sight_matches_rotated_box(axis, vector):
    """|m| about x (or y) of a field anisotropic along x equals |m| about z of the field with x and z swapped."""
    delta = anisotropic_field(axis=axis)
    lattice = Lattice(delta.shape[0])
    order = {0: (2, 1, 0), 1: (0, 2, 1)}[axis]
    box = measure_wst(np.ascontiguousarray(np.transpose(delta, order)), CONFIG, qs=QS, lattice=lattice, los=True)
    assert_same(box, measure_wst_survey(delta, CONFIG, qs=QS, lattice=lattice, los=np.array(vector)))
    per_cell = np.repeat(np.array(vector)[:, None], delta.size, axis=1)
    assert_same(box, measure_wst_survey(delta, CONFIG, qs=QS, lattice=lattice, los=per_cell), keys=("S1m",))


@pytest.mark.parametrize("ell", [1, 2, 3, 4, 6])
def test_harmonic_rotation_is_orthogonal(ell):
    rotation = axis_rotation(0, 0.3) @ axis_rotation(2, 1.1) @ axis_rotation(1, -0.7)
    d = harmonic_rotation(ell, rotation)
    np.testing.assert_allclose(d @ d.T, np.eye(2 * ell + 1), atol=1e-10)
    np.testing.assert_allclose(harmonic_rotation(ell, np.eye(3)), np.eye(2 * ell + 1), atol=1e-10)


@pytest.mark.parametrize("ell", [1, 2, 3, 4, 6])
def test_mode_along_line_of_sight_is_m0(ell):
    """The harmonic vector Y(u) of a plane wave along u, seen about the line of sight n = u, is pure m = 0; seen
    about any n, its |m| powers are those of Y_l|m| at the angle between u and n."""
    rng = np.random.default_rng(ell)
    n = rng.standard_normal((500, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    rotation = LOSRotation(ell)
    theta = np.arccos(n[:, 2])
    phi = np.arctan2(n[:, 1], n[:, 0])
    powers = m_powers(rotation(_harmonic_matrix(ell, n), theta, phi))
    np.testing.assert_allclose(powers[0], 1.0, atol=1e-10)
    u = rng.standard_normal((500, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    powers = m_powers(rotation(_harmonic_matrix(ell, u), theta, phi))
    cos = np.sum(u * n, axis=1)
    expected = m_powers(_harmonic_matrix(ell, np.stack([np.sqrt(1 - cos**2), 0 * cos, cos], axis=1)))
    for p, e in zip(powers, expected):
        np.testing.assert_allclose(p, e, atol=1e-10)


def test_paint_conserves_weight_and_rejects_outside():
    rng = np.random.default_rng(0)
    pos = rng.uniform(10, 90, (1000, 3))
    w = rng.uniform(0.5, 2, 1000)
    grid = paint_cic_weighted(pos, w, 32, 100.0)
    assert grid.sum() == pytest.approx(w.sum())
    with pytest.raises(ValueError):
        paint_cic_weighted(pos + 50, w, 32, 100.0)


def test_survey_mesh_of_uniform_sphere():
    rng = np.random.default_rng(1)

    def ball(n):
        x = rng.uniform(-200, 200, (3 * n, 3))
        return x[np.linalg.norm(x, axis=1) < 200][:n]

    mesh = SurveyMesh.from_catalogues(ball(20000), ball(400000), nmesh=48, padding=100.0, observer=(0, 0, -1000))
    assert mesh.alpha == pytest.approx(0.05)
    inner = erode(mesh.support, 3)
    assert abs(mesh.delta[inner].mean()) < 0.05
    nbar = 20000 / (4 / 3 * np.pi * 200**3)
    assert np.median(mesh.nbar[inner]) == pytest.approx(nbar, rel=0.1)
    los = mesh.line_of_sight()
    assert los.shape == (3, mesh.support.sum()) and np.all(los[2] > 0.9)
    weights = mesh.fkp_weights()
    assert np.all(weights[~mesh.support] == 0) and weights[inner].min() > 0
