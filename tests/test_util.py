from xarray_multiscale.util import broadcast_to_rank


def test_broadcast_to_rank():
    assert broadcast_to_rank(2, 1) == (2,)
    assert broadcast_to_rank(2, 2) == (2, 2)
    assert broadcast_to_rank((2, 3), 2) == (2, 3)
    assert broadcast_to_rank({0: 2}, 3) == (2, 1, 1)