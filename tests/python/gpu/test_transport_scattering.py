"""Direct scientific checks of the canonical free-electron conditional sampler.

Independent numerical integration of the differential law determines expected
bin masses; no simulator-generated histogram is treated as the reference.
"""

# Warp's decorators/annotations are runtime DSL, while host tests remain typed.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false, reportUnknownParameterType=false
import importlib
import math
from typing import Any

import pytest

from dpt.validation.transport import compton_energy, klein_nishina_total_ratio

wp: Any = importlib.import_module("warp")
kernels: Any = importlib.import_module("dpt.transport.kernels")
pytestmark = pytest.mark.gpu


@wp.kernel
def _sample(
    energy: wp.float64,
    out_values: wp.array(dtype=wp.vec4d),
):
    history = wp.tid()
    out_values[history] = kernels.compton_scatter(
        energy, wp.uint64(95371), wp.uint64(history), wp.uint32(0), 4096
    )


def _integral(energy: float, lower: float, upper: float, cells: int = 4096) -> float:
    # Composite Simpson quadrature of the differential cross section; independent
    # expression, using the physical outgoing-energy oracle in ordinary binary64.
    step = (upper - lower) / cells

    def density(cosine: float) -> float:
        ratio = float(compton_energy(energy, cosine)) / energy
        return ratio**2 * (ratio + 1 / ratio - (1 - cosine**2))

    return (
        step
        / 3
        * math.fsum(
            (1 if index in (0, cells) else (4 if index % 2 else 2)) * density(lower + index * step)
            for index in range(cells + 1)
        )
    )


@pytest.mark.parametrize("energy", [1.0, 80.0, 1000.0])
def test_klein_nishina_angular_distribution_and_energy_conservation(energy: float) -> None:
    histories = 2**17
    values = wp.empty(histories, dtype=wp.vec4d, device="cuda:0")
    wp.launch(_sample, dim=histories, inputs=[energy, values], device="cuda:0")
    observed = values.numpy()
    assert (observed[:, 3] == 0).all()
    integral = _integral(energy, -1.0, 1.0)
    assert integral * 3 / 8 == pytest.approx(float(klein_nishina_total_ratio(energy)), rel=1e-9)
    for lower, upper in ((-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0)):
        expected = _integral(energy, lower, upper) / integral
        fraction = float(((observed[:, 0] >= lower) & (observed[:, 0] < upper)).mean())
        assert abs(fraction - expected) < 7 * math.sqrt(expected * (1 - expected) / histories)
    for cosine, azimuth, outgoing, _ in observed[::1024]:
        assert 0 <= float(azimuth) < 2 * math.pi
        assert float(outgoing) == pytest.approx(
            float(compton_energy(energy, float(cosine))), rel=1e-14
        )
