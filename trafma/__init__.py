"""TRAF-MA: SUMO simulation layer (Phase 1 + SUMO-side data generation of Phase 2).

Hierarchy:  vehicle agents --V2I--> 4 RSUs (one per road) --I2C--> edge coordinator
"""
__version__ = "0.1.0"
MSG_SCHEMA = "trafma.msg"
MSG_VERSION = "1.0"
LABEL_SCHEMA = "trafma.labels/1"
APPROACHES = ("N", "E", "S", "W")  # clockwise; index arithmetic is used for turn geometry
