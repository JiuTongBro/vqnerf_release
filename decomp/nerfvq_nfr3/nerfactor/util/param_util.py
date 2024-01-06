from enum import Enum


class ParameterNames(Enum):
    # flash for 1
    DIFFUSE = "diffuse.png"
    SPECULAR = "specular.png"
    ROUGHNESS = "roughness.png"
    NORMAL = "normal.exr"
    DEPTH = "depth.exr"
    MASK = "mask.png"
    # INPUT_ENV = "cam2_env.exr"
    INPUT = "cam2.exr"
    # INPUT_LDR = "cam2.png"
    SGS = "sgs.npy"
    # RERENDER = "rerender%d.exr"


class Stages(Enum):
    INITIAL = 0
    REFINEMENT = 1