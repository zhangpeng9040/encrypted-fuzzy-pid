"""Use the recorded fuzzy mapping without fitting or generating any keys."""
import ast
from pathlib import Path
import numpy as np
from fuzzy_controller import FuzzyLongitudinalController


def fuzzy_engine():
    source = Path(__file__).with_name("fuzzy_controller.py")
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "FuzzyLongitudinalController")
    init = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "__init__")
    instance = object.__new__(FuzzyLongitudinalController)
    for node in ast.walk(init):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                try:
                    setattr(instance, target.attr, ast.literal_eval(node.value))
                except (ValueError, TypeError):
                    pass
    return instance


def features(z):
    z = np.asarray(z, dtype=float)
    e, d, integral = z.T
    return np.column_stack([np.ones(len(z)), e, d, integral, e*e, e*d,
                            e*integral, d*d, d*integral, integral*integral])


def target(z, engine):
    z = np.clip(np.asarray(z, dtype=float), [-15, -3, -20], [15, 3, 20])
    e, d, integral = z.T
    s = np.array([engine._plain_throttle(float(a), float(b)) for a, b in zip(e, d)])
    kp = np.clip(1.0 * (1 + .5*s), .5, 1.5)
    ki = np.clip(.3 * (1 - .3*s), .15, .45)
    kd = np.clip(.001 * (1 + .2*abs(d)), .0005, .0015)
    return np.clip(kp*e + ki*integral + kd*d, -1, 1)


def region_indices(z, edges):
    z = np.asarray(z, dtype=float)
    shape = tuple(len(axis)-1 for axis in edges)
    indices = tuple(np.searchsorted(axis[1:-1], z[:, n], side="right") for n, axis in enumerate(edges))
    return np.ravel_multi_index(indices, shape)
