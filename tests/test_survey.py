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


def test_wide_angle_mocks_with_distant_observer_are_plane_parallel():
    """With the observer far along -z, the wide-angle Kaiser field tends to the plane-parallel one."""
    from wstfast.survey import GaussianMocks

    n, cell = 32, 10.0
    support = np.ones((n,) * 3, dtype=bool)
    nbar = np.full((n,) * 3, 1e-3)

    def pk(k):
        return 1e4 * k / (1 + (k / 0.02) ** 2)

    plane = GaussianMocks(n, cell, pk, 2.0, 0.8, nbar, support)(5)
    far = GaussianMocks(n, cell, pk, 2.0, 0.8, nbar, support, observer=(160.0, 160.0, -1e7))(5)
    np.testing.assert_allclose(far, plane, atol=1e-4 * np.abs(plane).max())


def test_window_ratio_of_full_support_is_one():
    from wstfast.survey import GaussianMocks, window_ratio

    n = 32
    support = np.ones((n,) * 3, dtype=bool)
    mocks = GaussianMocks(n, 10.0, lambda k: 1e4 * k / (1 + (k / 0.02) ** 2), 2.0, 0.8, np.full((n,) * 3, 1e-3), support)
    out = window_ratio(mocks, CONFIG, support, qs=[0.8], los="z", realizations=2)
    ratio = out["S1m"]["ratio"]
    np.testing.assert_allclose(ratio[np.isfinite(ratio)], 1.0, rtol=1e-6)


def test_box_to_survey_reproduces_periodic_painting():
    """A full-mesh geometry with uniform density, a box equal to the mesh and no displacement: the periodic CIC delta
    (away from the last plane, which the non-periodic painting does not wrap)."""
    from wstfast.quijote import paint_cic
    from wstfast.survey import SurveyGeometry, box_to_survey

    rng = np.random.default_rng(2)
    n, box = 32, 320.0
    pos = rng.uniform(0, box - 2 * box / n, (20000, 3))
    support = np.ones((n,) * 3, dtype=bool)
    nbar = len(pos) / box**3
    geometry = SurveyGeometry(support, np.full((n,) * 3, nbar), box, np.zeros(3), np.array([0.0, 0.0, -1e4]))
    delta = box_to_survey(pos, box, geometry)
    np.testing.assert_allclose(delta[:-1, :-1, :-1], paint_cic([pos], n, box)[:-1, :-1, :-1], atol=1e-4)


def test_cone_geometry_sampling():
    from wstfast.survey import SurveyGeometry

    geometry = SurveyGeometry.cone(48, 960.0, (480.0, 480.0, -500.0), 20.0, 600.0, 1300.0, lambda r: 1e-4 + 0 * r)
    assert geometry.support.any() and not geometry.support.all()
    prob = geometry.sampling(2e-4)
    np.testing.assert_allclose(prob, 0.5)
    with pytest.raises(ValueError):
        geometry.sampling(5e-5)


@pytest.mark.parametrize("ell", [1, 3, 4])
def test_tabulated_rotation_matches_direct_rotation(ell):
    rng = np.random.default_rng(ell)
    n = rng.standard_normal((200, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    theta, phi = np.arccos(n[:, 2]), np.arctan2(n[:, 1], n[:, 0])
    x = rng.standard_normal((2 * ell + 1, 200)).astype(np.float32)
    rotation = LOSRotation(ell)
    np.testing.assert_allclose(rotation.rotate(x, rotation.tables(theta, phi)), rotation(x, theta, phi), atol=1e-5)
    np.testing.assert_allclose(rotation.rotate(x, rotation.tables(0.4, -1.2)), rotation(x, 0.4, -1.2), atol=1e-5)


def _two_region_cone(nmesh=48):
    from wstfast.survey import SurveyGeometry

    geometry = SurveyGeometry.cone(nmesh, 960.0, (480.0, 480.0, -500.0), 25.0, 600.0, 1300.0,
                                   lambda r: 1e-4 * (1 + r / 1000.0))
    x = np.arange(nmesh)[:, None, None] * geometry.cellsize + 0 * geometry.support
    geometry.region = np.where(geometry.support, (x > 480.0).astype(int), -1)
    return geometry


def test_radial_integral_constraint_nulls_every_shell_jointly():
    from wstfast.survey import RadialIntegralConstraint

    a, b = _two_region_cone(), _two_region_cone()
    b.region = np.where(b.support, 1, -1)  # region 1 continues on a second mesh, as S across the two caps
    ric = RadialIntegralConstraint(a, b, width=40.0)
    rng = np.random.default_rng(0)
    da, db = (0.3 * rng.standard_normal(a.support.shape) for _ in range(2))
    ca, cb = ric(da, db)
    np.testing.assert_array_equal(ca[~a.support], da[~a.support])
    for region in (0, 1):
        for shell in range(int(1300 / 40) + 1):
            sums = []
            for g, c in ((a, ca), (b, cb)):
                cells = g.support & (g.region == region) & (np.floor(g.distance() / 40.0) == shell)
                sums.append((g.nbar[cells] * c[cells]).sum())
            assert abs(sum(sums)) < 1e-9 * max(1.0, a.nbar[a.support].sum())


def test_radial_integral_constraint_removes_radial_modes_only():
    from wstfast.survey import RadialIntegralConstraint

    g = _two_region_cone()
    ric = RadialIntegralConstraint(g, width=40.0)
    radial = 0.2 * np.sin(np.floor(g.distance() / 40.0))  # constant on every shell
    np.testing.assert_allclose(ric(radial)[g.support], 0.0, atol=1e-12)
    angular = 0.05 * np.cos(np.arange(48))[:, None, None] + 0 * radial  # transverse to the line of sight
    assert np.abs(ric(angular)[g.support]).max() > 0.01


def test_calibrate_window_accepts_survey_reference_pairs():
    from wstfast.survey import calibrate_window

    n = 24
    support = np.ones((n,) * 3, dtype=bool)
    fields = [anisotropic_field(n, seed=s) for s in range(2)]
    config = WSTConfig(J=1, L=2, sigma0=1.5, min_dj=10, cellsize=1.0)
    out = calibrate_window(((f, f) for f in fields), config, support, qs=[0.8], los="z")["S1m"]
    np.testing.assert_allclose(out["ratio"][np.isfinite(out["ratio"])], 1.0, rtol=1e-6)
    out = calibrate_window(((2 * f, f) for f in fields), config, support, qs=[1.0], los="z")["S1m"]
    np.testing.assert_allclose(out["ratio"][np.isfinite(out["ratio"])], 2.0, rtol=1e-5)


def test_lambda_window_of_full_support_is_one_and_gaussian_s1m_matches():
    from wstfast.survey import LambdaWindow, noise_variance_maps, s1m_gaussian, signal_variances

    n = 24
    config = WSTConfig(J=1, L=2, sigma0=1.5, min_dj=10, cellsize=1.0)
    lattice = Lattice(n)
    support = np.ones((n,) * 3, dtype=bool)
    los = np.zeros((3, n, n, n), dtype=np.float32)
    los[2] = 1.0
    window = LambdaWindow(config, support, los, lattice)
    fields = [anisotropic_field(n, seed=s) for s in range(4)]
    for f in fields:
        window.add(f, f)
    for t in window.transfer().values():
        np.testing.assert_allclose(t, 1.0, rtol=1e-6)
    # the summed squares of the components are the lattice variance on average over cells
    pcell = np.abs(np.fft.rfftn(fields[0])) ** 2 / n**3
    lam = signal_variances(pcell, config, lattice)
    for (j, ell), x2 in LambdaWindow(config, support, los, lattice).squares(fields[0]):
        np.testing.assert_allclose(x2.mean(axis=1), lam[j, ell], rtol=1e-4)
    noise = noise_variance_maps(np.full(support.shape, 0.5), config, support, lattice)
    pred = s1m_gaussian(lam, config, 0.8, np.ones(n**3), lam_noise=noise)
    assert np.isfinite(pred[:, :, 0]).all()
    white = signal_variances(np.full(lattice.kmag.shape, 0.5), config, lattice)
    for (j, ell), v in noise.items():
        np.testing.assert_allclose(v, white[j, ell].mean(), rtol=1e-4)


def test_torch_survey_backend_matches_numpy():
    torch = pytest.importorskip("torch")
    from wstfast.measure_torch import TorchLattice, measure_wst_survey_torch
    from wstfast.survey import LambdaWindow, noise_variance_maps, signal_variances

    geometry = _two_region_cone(32)
    config = WSTConfig(J=2, L=3, L2=2, sigma0=1.2, min_dj=1, cellsize=1.0)
    delta = anisotropic_field(32, seed=7)
    los = geometry.line_of_sight(full=True)
    weights = geometry.weights
    lattice, tlattice = Lattice(32), TorchLattice(32, device=torch.device("cpu"))
    kwargs = dict(support=geometry.support, weights=weights, qs=[0.8, 2.0], los=los)
    reference = measure_wst_survey(delta, config, lattice=lattice, **kwargs)
    result = measure_wst_survey_torch(delta, config, lattice=tlattice, **kwargs)
    for key in ("S0", "S1", "S2", "S1m", "S2m"):
        np.testing.assert_allclose(result[key], reference[key], rtol=2e-4, atol=1e-9, equal_nan=True, err_msg=key)

    windows = [LambdaWindow(config, geometry.support, los, lat, nbins=8) for lat in (lattice, tlattice)]
    for w in windows:
        w.add(np.where(geometry.support, delta, 0.0), delta)
    for b in windows[0].blocks:
        np.testing.assert_allclose(windows[1].coverage[b], windows[0].coverage[b], atol=1e-5)
        np.testing.assert_allclose(windows[1].correction()[b], windows[0].correction()[b], rtol=1e-3)
    sigma2 = np.full(geometry.support.shape, 0.3)
    a = noise_variance_maps(sigma2, config, geometry.support, lattice)
    b = noise_variance_maps(sigma2, config, geometry.support, tlattice)
    pcell = np.abs(np.fft.rfftn(delta)) ** 2 / 32**3
    c, d = signal_variances(pcell, config, lattice), signal_variances(pcell, config, tlattice)
    for key in a:
        np.testing.assert_allclose(b[key], a[key], rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(d[key], c[key], rtol=1e-4)


def test_lambda_window_second_layer_matches_the_estimator():
    from wstfast.survey import LambdaWindow

    geometry = _two_region_cone(32)
    config = WSTConfig(J=2, L=2, L2=2, sigma0=1.2, min_dj=2, cellsize=1.0)
    delta = anisotropic_field(32, seed=11)
    los = geometry.line_of_sight(full=True)
    lattice = Lattice(32)
    window = LambdaWindow(config, geometry.support, los, lattice, nbins=8, second_layer=True)
    masked = measure_wst_survey(delta, config, support=geometry.support, qs=[2.0], los=los, lattice=lattice)["S2m"][0]
    box = measure_wst_survey(delta, config, moment_support=geometry.support, qs=[2.0], los=los,
                             lattice=lattice)["S2m"][0]
    for survey, reference in ((True, masked), (False, box)):
        for (j1, j2, ell, m1), x2 in window.second_layer_squares(delta, survey):
            powers = [x2[ell]] + [x2[ell - m] + x2[ell + m] for m in range(1, ell + 1)]
            np.testing.assert_allclose([p.mean() for p in powers], reference[j1, j2, ell, m1, :ell + 1], rtol=2e-4)

    full = np.ones((32,) * 3, dtype=bool)
    window = LambdaWindow(config, full, los, lattice, nbins=8, second_layer=True)
    window.add(delta, delta)
    for ratio in window.s2m_ratio(np.ones(32**3), 0.8).values():
        np.testing.assert_allclose(ratio, 1.0, rtol=1e-5)


def test_binned_second_layer_mean():
    torch = pytest.importorskip("torch")
    from wstfast.measure_torch import TorchLattice, measure_wst_survey_torch
    from wstfast.survey import LambdaWindow, second_layer_bins

    config = WSTConfig(J=2, L=2, L2=2, sigma0=1.2, min_dj=2, cellsize=1.0)
    delta = anisotropic_field(32, seed=13)
    lattice, tlattice = Lattice(32), TorchLattice(32, device=torch.device("cpu"))
    # box limit: one bin per (j1, l), the same as the global mean
    full = np.ones((32,) * 3, dtype=bool)
    bins = second_layer_bins(full, np.ones(full.shape), config, lattice)
    assert all(np.unique(b).size == 1 for b in bins.values())
    a = measure_wst_survey(delta, config, qs=[0.8], los="z", lattice=lattice)["S2m"]
    b = measure_wst_survey(delta, config, qs=[0.8], los="z", lattice=lattice, second_layer_bins=bins)["S2m"]
    np.testing.assert_allclose(b, a, rtol=1e-6, equal_nan=True)

    geometry = _two_region_cone(32)
    los = geometry.line_of_sight(full=True)
    bins = second_layer_bins(geometry.support, geometry.nbar, config, lattice, ncoverage=4, nnbar=2)
    assert max(np.unique(b).size for b in bins.values()) > 2
    kwargs = dict(support=geometry.support, qs=[2.0], los=los, second_layer_bins=bins)
    masked = measure_wst_survey(delta, config, lattice=lattice, **kwargs)["S2m"][0]
    np.testing.assert_allclose(measure_wst_survey_torch(delta, config, lattice=tlattice, **kwargs)["S2m"][0], masked,
                               rtol=2e-4, equal_nan=True)
    for lat in (lattice, tlattice):
        window = LambdaWindow(config, geometry.support, los, lat, nbins=8, second_layer=True, second_layer_bins=bins)
        for (j1, j2, ell, m1), x2 in window.second_layer_squares(delta, True):
            powers = [x2[ell]] + [x2[ell - m] + x2[ell + m] for m in range(1, ell + 1)]
            np.testing.assert_allclose([p.mean() for p in powers], masked[j1, j2, ell, m1, :ell + 1], rtol=5e-4)
    box = measure_wst_survey(delta, config, moment_support=geometry.support, qs=[2.0], los=los, lattice=lattice)["S2m"][0]
    window = LambdaWindow(config, geometry.support, los, tlattice, nbins=8, second_layer=True, second_layer_bins=bins)
    for (j1, j2, ell, m1), x2 in window.second_layer_squares(delta, False):
        powers = [x2[ell]] + [x2[ell - m] + x2[ell + m] for m in range(1, ell + 1)]
        np.testing.assert_allclose([p.mean() for p in powers], box[j1, j2, ell, m1, :ell + 1], rtol=5e-4)
