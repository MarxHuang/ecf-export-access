"""Completed-record interface for the manuscript's conservative delivery test.

No latent availability argument is accepted. Network feasibility is an explicit
external input and is not certified by a componentwise delivery lower bound.
"""
from dataclasses import dataclass
import math
from typing import Sequence

@dataclass(frozen=True)
class Qualification:
    record_valid: bool
    qualifies: bool
    reason: str
    idle_indices: tuple[int, ...]
    lower_availability: tuple[float, ...]
    lower_replay_delivery: tuple[float, ...]
    outside_access_gain: float
    outside_delivery_gain: float
    total_delivery_gain: float

def assess_bounded_record(*, allocation: Sequence[float], delivery: Sequence[float],
                          replay_allocation: Sequence[float],
                          record: Sequence[float] | None,
                          error_radius: Sequence[float] | None,
                          record_valid: bool, replay_physical_valid: bool,
                          tolerance: float) -> Qualification:
    """All powers/radii/tolerance must share one unit (MW in the experiment).

The caller must validate the radius independently. ``record_valid`` is an
explicit declaration, not a claim that these arrays prove the error bound.
Invalid data fail closed; the calling ceiling update must use capacity recovery.
"""
    def failed(reason, idle=()):
        return Qualification(False,False,reason,idle,(),(),0.,0.,0.)
    if not isinstance(record_valid,bool) or not isinstance(replay_physical_valid,bool):
        return failed('INVALID_VALIDITY_FLAG')
    if not record_valid:return failed('RECORD_NOT_VALIDATED')
    if record is None or error_radius is None:return failed('MISSING_RECORD_OR_RADIUS')
    try:
        x,y,xr,h,d=[tuple(float(v) for v in a) for a in
                     (allocation,delivery,replay_allocation,record,error_radius)]
        tol=float(tolerance)
    except (TypeError,ValueError,OverflowError):return failed('INVALID_NUMERIC_INPUT')
    if not x or len({len(a) for a in (x,y,xr,h,d)})!=1:return failed('INVALID_VECTOR_LENGTH')
    if not math.isfinite(tol) or tol<0:return failed('INVALID_TOLERANCE')
    if any(not math.isfinite(v) or v<0 for a in (x,y,xr,h,d) for v in a):
        return failed('INVALID_POWER_OR_RADIUS')
    if any(yi>xi for xi,yi in zip(x,y)):return failed('DELIVERY_EXCEEDS_AWARD')
    idle=tuple(i for i,(xi,yi) in enumerate(zip(x,y)) if xi-yi>tol)
    # An upper error bound below metered delivery contradicts y <= a.
    if any(hi+di<yi for hi,di,yi in zip(h,d,y)):
        return failed('RECORD_BOUND_CONTRADICTS_DELIVERY',idle)
    lower=tuple(max(yi,hi-di) for yi,hi,di in zip(y,h,d))
    yr=tuple(min(xi,ai) for xi,ai in zip(xr,lower))
    outside=set(range(len(x)))-set(idle)
    try:
        access=math.fsum(max(xr[i]-x[i],0.) for i in outside)
        out=math.fsum(max(yr[i]-y[i],0.) for i in outside)
        total=math.fsum(yr)-math.fsum(y)
    except OverflowError:return failed('NONFINITE_AGGREGATE',idle)
    if not all(math.isfinite(v) for v in (access,out,total)):
        return failed('NONFINITE_AGGREGATE',idle)
    if not idle:reason='NO_UNUSED_AWARD'
    elif not replay_physical_valid:reason='PHYSICAL_CHECK_NOT_PASSED'
    elif access<=tol:reason='NO_OUTSIDE_ACCESS_RELEASE'
    elif out<=tol:reason='NO_LOWER_BOUND_OUTSIDE_RECOVERY'
    elif total<=tol:reason='NO_LOWER_BOUND_TOTAL_RECOVERY'
    else:reason='LOWER_BOUND_RECOVERY'
    return Qualification(True,reason=='LOWER_BOUND_RECOVERY',reason,idle,lower,yr,access,out,total)
