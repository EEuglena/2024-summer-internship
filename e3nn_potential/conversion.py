def kcal2meV(e: float) -> float:
    """Convert kcal/mol to meV/molecule"""
    return e * 43.36


def meV2kcal(e: float) -> float:
    """Convert meV/molecule to kcal/mol"""
    return e * 2.306e-2


def Ha2meV(e: float) -> float:
    return e * 2.721e4


def Ha2MeV(e: float) -> float:
    return e * 2.721e-2


def MeV2kcal(e: float) -> float:
    return e * 2.306e7


def Ha2kcal(e: float) -> float:
    return e * 6.275e2
