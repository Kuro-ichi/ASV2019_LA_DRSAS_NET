def test_imports():
    import drsas_net
    from drsas_net.models.model import DRSASNet, build_model
    m = build_model()
    assert m is not None
