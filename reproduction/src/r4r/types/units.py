"""Registered physical units only; identifiers and hashes have no unit."""
from enum import StrEnum


class Unit(StrEnum):
    MVA = "MVA"
    MVAr = "MVAr"
    MW = "MW"
    MWh = "MWh"
    HOUR = "h"
    KV = "kV"
    PU = "pu"
