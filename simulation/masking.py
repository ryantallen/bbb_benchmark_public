# masking.py
import re
from copy import deepcopy
from typing import List, Dict, Literal, Callable, Union

MaskMode = Literal["unmasked", "masked"]

# --- Aliases you can adjust as needed ---
SIM_NAME = "EnergyCo"

AGM_ALIAS_LONG  = "AeroBond Matrix"      # replaces "Absorbed Glass Mat"
AGM_ALIAS_SHORT = "ABM"                  # replaces "AGM"

SC_ALIAS_LONG   = "Quantum Storage Cell" # replaces "Supercapacitor"
SC_ALIAS_SHORT  = "QSC"                  # replaces "SC"

WAREHOUSE_EQP_LONG = "Industrial Handling Systems"  # replaces "Warehouse equipment"
IHS_ACRO           = "IHS"

UPS_LONG = "Continuity Power Modules"    # replaces "Uninterruptible power systems/supplies"
UPS_ACRO = "CPM"                         # replaces "UPS"

AUTO_SING = "Light Transport Vehicle"    # replaces "automobile"
AUTO_PLUR = "Light Transport Vehicles"   # replaces "automobiles"
AUTO_ACRO = "LTV"

# --- Performance dimensions (aliases) ---
ED_LONG  = "Specific Energy Index (SEI)"      # replaces "Energy Density"
ED_SHORT = "SEI"

RC_LONG  = "Cycle Endurance Rating (CER)"     # replaces "Recharge Cycles"
RC_SHORT = "CER"

SD_LONG  = "Standby Loss Interval (SLI)"      # replaces "Self Discharge"
SD_SHORT = "SLI"

RT_LONG  = "Recovery Time Index (RTI)"        # replaces "Recharge Time" / "Recharge Duration"
RT_SHORT = "RTI"

_Repl = Union[str, Callable[[re.Match], str]]

# Custom boundaries that treat underscore as a separator too.
_BPRE  = r"(?<![A-Za-z0-9])"
_BPOST = r"(?![A-Za-z0-9])"

def _ci(p: str) -> re.Pattern:
    return re.compile(p, re.IGNORECASE)

# -------------------------
# Identifier-first rules
# -------------------------
_RULES_IDENT: list[tuple[re.Pattern, _Repl]] = [
    # --- Performance dimensions in identifiers: use acronyms ---
    # Energy Density
    (_ci(r"(?<=[_-])energy(?:[_\s-]*density)(?=$|[_-])"), ED_SHORT),
    (_ci(r"^energy(?:[_\s-]*density)(?=$|[_-])"), ED_SHORT),

    # Recharge Cycles
    (_ci(r"(?<=[_-])recharge(?:[_\s-]*cycles?)(?=$|[_-])"), RC_SHORT),
    (_ci(r"^recharge(?:[_\s-]*cycles?)(?=$|[_-])"), RC_SHORT),

    # Self Discharge (self-discharge / selfdischarge)
    (_ci(r"(?<=[_-])self(?:[_\s-]*discharge)(?=$|[_-])"), SD_SHORT),
    (_ci(r"^self(?:[_\s-]*discharge)(?=$|[_-])"), SD_SHORT),

    # Recharge Time
    (_ci(r"(?<=[_-])recharge(?:[_\s-]*time)(?=$|[_-])"), RT_SHORT),
    (_ci(r"^recharge(?:[_\s-]*time)(?=$|[_-])"), RT_SHORT),

    # Recharge Duration
    (_ci(r"(?<=[_-])recharge(?:[_\s-]*duration)(?=$|[_-])"), RT_SHORT),
    (_ci(r"^recharge(?:[_\s-]*duration)(?=$|[_-])"), RT_SHORT),

    # --- Technologies / segments in identifiers ---
    # Supercapacitor -> QSC
    (_ci(_BPRE + r"supercapacitor(s)?(?=[_-])"),
        lambda m: SC_ALIAS_SHORT + ("s" if m.group(1) else "")),
    (_ci(_BPRE + r"sc(?=[_-])"), SC_ALIAS_SHORT),

    # Absorbed Glass Mat -> ABM
    (_ci(_BPRE + r"absorbed\s+glass\s+mat(?=[_-])"), AGM_ALIAS_SHORT),
    (_ci(_BPRE + r"agm(?=[_-])"), AGM_ALIAS_SHORT),

    # Industry segments -> acronyms
    (_ci(_BPRE + r"uninterruptible\s+power\s+(?:systems?|suppl(?:y|ies))(?=[_-])"), UPS_ACRO),
    (_ci(_BPRE + r"ups(?=[_-])"), UPS_ACRO),

    (_ci(_BPRE + r"automobiles?(?=[_-])"), AUTO_ACRO),
    (_ci(_BPRE + r"automotive(?=[_-])"),  AUTO_ACRO),

    (_ci(_BPRE + r"warehouse[-\s]*equipment(?=[_-])"), IHS_ACRO),
    (_ci(_BPRE + r"ihs(?=[_-])"), IHS_ACRO),
]

# -------------------------
# General prose / tables
# -------------------------
_RULES: list[tuple[re.Pattern, _Repl]] = [
    # --- Simulation / company name ---
    (_ci(_BPRE + r"back\s*bay\s*battery" + _BPOST), SIM_NAME),
    (_ci(_BPRE + r"back\s*bay" + _BPOST),           SIM_NAME),

    # --- Technologies ---
    (_ci(_BPRE + r"absorbed\s+glass\s+mat" + _BPOST), AGM_ALIAS_LONG),
    (_ci(_BPRE + r"agm" + _BPOST),                    AGM_ALIAS_SHORT),

    # Supercapacitor (plural + acronym)
    (_ci(_BPRE + r"supercapacitor(s)?" + _BPOST),
        lambda m: SC_ALIAS_LONG + ("s" if m.group(1) else "")),
    (_ci(_BPRE + r"sc" + _BPOST),                      SC_ALIAS_SHORT),

    # --- Industry segments ---
    (_ci(_BPRE + r"warehouse[-\s]*equipment" + _BPOST), WAREHOUSE_EQP_LONG),
    (_ci(_BPRE + r"ihs" + _BPOST),                       IHS_ACRO),

    (_ci(_BPRE + r"uninterruptible\s+power\s+(systems?|suppl(?:y|ies))" + r"(?:\s*\(ups\))?" + _BPOST), UPS_LONG),
    (_ci(_BPRE + r"ups(s)?" + _BPOST),
        lambda m: UPS_ACRO + ("s" if m.group(1) else "")),

    (_ci(_BPRE + r"automobiles" + _BPOST), AUTO_PLUR),
    (_ci(_BPRE + r"automobile"  + _BPOST), AUTO_SING),
    (_ci(_BPRE + r"automotive"  + _BPOST), f"{AUTO_ACRO} sector"),

    # --- Performance dimensions in prose/tables (long forms) ---
    # Energy Density
    (_ci(_BPRE + r"energy[\s-]*density" + _BPOST), ED_LONG),

    # Recharge Cycles (prefer label-like contexts)
    (_ci(_BPRE + r"Recharge[\s-]*Cycles?" + _BPOST), RC_LONG),            # title/label case
    (_ci(_BPRE + r"recharge[\s-]*cycles?(?=\s*\()" + _BPOST), RC_LONG),   # followed by units, e.g., "(Cycles)"
    # If you want to force masking everywhere, uncomment:
    # (_ci(_BPRE + r"recharge[\s-]*cycles?" + _BPOST), RC_LONG),

    # Self Discharge (optionally 'to 50%')
    (_ci(_BPRE + r"self[\s-]*discharge(?P<to50>\s*to\s*50\%)?" + _BPOST),
        lambda m: SD_LONG + (m.group('to50') or "")),

    # Recharge Time / Duration → RTI
    (_ci(_BPRE + r"recharge[\s-]*time" + _BPOST), RT_LONG),
    (_ci(_BPRE + r"recharge[\s-]*duration" + _BPOST), RT_LONG),

    # Readability polish: replace awkward CER phrasing in narrative
    (_ci(r"\brequire\s+(?:regular|frequent)\s+Cycle\s+Endurance\s+Rating\s*\(CER\)"),
        "require frequent cycling (high Cycle Endurance Rating (CER))"),

    # --- Optional normalization / de-dup (place after dimension replacements) ---
    # Always include acronyms if the long form appears without them
    (_ci(r"\bSpecific Energy Index\b(?!\s*\()"), ED_LONG),
    (_ci(r"\bCycle Endurance Rating\b(?!\s*\()"), RC_LONG),
    (_ci(r"\bStandby Loss Interval\b(?!\s*\()"), SD_LONG),
    (_ci(r"\bRecovery Time Index\b(?!\s*\()"), RT_LONG),

    # Collapse duplicate RTI when both "recharge time" and "recharge duration" were present
    (_ci(r"(Recovery\s*Time\s*Index\s*\(RTI\))\s*,\s*and\s*Recovery\s*Time\s*Index\b"), r"\1"),
    (_ci(r"(Recovery\s*Time\s*Index\s*\(RTI\))\s*,\s*Recovery\s*Time\s*Index\b"),       r"\1"),
]

def mask_messages(
    messages: List[Dict[str, str]],
    mode: MaskMode = "unmasked",
) -> List[Dict[str, str]]:
    """
    If mode == 'masked', returns a deep-copied list with masked content.
    If mode == 'unmasked', returns a deep copy unchanged.
    messages: list like [{"role": "user"|"system"|"assistant", "content": "..."}]
    """
    out = deepcopy(messages)
    if mode != "masked":
        return out

    for msg in out:
        content = msg.get("content")
        if isinstance(content, str):
            # identifier-first pass
            for pattern, repl in _RULES_IDENT:
                content = pattern.sub(repl, content)
            # general prose/tables pass
            for pattern, repl in _RULES:
                content = pattern.sub(repl, content)
            msg["content"] = content
    return out
