# clinical_rules.py - Unified Clinical Safety & Burn Severity Rule Engine
# Single Source of Truth for BurnAid AI (Web Client, API Server, Streamlit, Database)
# Strict Pure English Clinical Protocol

from typing import Dict, Any, List, Optional, Tuple

# High-risk anatomical areas (GS. Le The Trung 22-region standard + international clinical burn criteria)
# Critical priority functional zones: Face, Neck, Hands, Feet, Perineum, Major Thigh regions
CRITICAL_BODY_REGIONS = {
    "FACE", "HEAD_FACE", "HEAD_NECK", "NECK", "PERINEUM",
    "HAND_R", "HAND_L", "FOOT_R", "FOOT_L", "THIGH_R", "THIGH_L"
}

VALID_BURN_DEGREES = {
    "degree_1", "degree_2", "degree_3",
    "Degree I", "Degree II", "Degree III",
    "Độ 1", "Độ 2", "Độ 3",
    "1", "2", "3",
    "no_burn", "No Burn", "Không bỏng", "none", "Healthy Skin",
    "unassessed", "Unassessed", "Chưa thể đánh giá", "Unknown", "UNASSESSED", None, ""
}

def evaluate_safety_and_severity(
    tbsa: Optional[float],
    age: int,
    burn_degree: Optional[str],
    has_inhalation: bool,
    has_high_voltage: bool = False,
    has_circumferential: bool = False,
    body_regions: Optional[List[str]] = None,
    burn_cause: str = "thermal"
) -> Dict[str, Any]:
    """
    Unified Clinical Rule Engine:
    - Safety Rules ALWAYS execute first, independent of AI classification confidence.
    - Strict Pure English clinical reasons and recommendations.
    - Unified taxonomic mapping to Database Schema v9.2 check constraints.
    """
    reasons: List[str] = []
    actions: List[str] = []
    
    # 0. Validate and normalize input degree
    raw_degree_str = str(burn_degree).strip() if burn_degree is not None else ""
    is_invalid_degree = (burn_degree is not None and raw_degree_str != "" and raw_degree_str not in VALID_BURN_DEGREES)
    
    # Check anatomical critical areas
    regions_set = set(body_regions or [])
    has_critical_area = bool(regions_set.intersection(CRITICAL_BODY_REGIONS))
    
    # -------------------------------------------------------------
    # 1. IMMEDIATE LIFE-THREATENING EMERGENCY RULES (OVERRIDE ALL)
    # -------------------------------------------------------------
    is_emergency_override = False
    
    if has_inhalation:
        is_emergency_override = True
        reasons.append("Suspected acute toxic smoke inhalation / upper airway compromise")
        actions.append("Immediately activate Emergency EMS 115 / 911. Maintain open airway with 100% humidified oxygen if available.")
        
    if has_high_voltage or burn_cause == "electrical":
        is_emergency_override = True
        reasons.append("High-voltage electrical burn detected (severe risk of cardiac arrhythmia and hidden deep tissue necrosis)")
        actions.append("Immediately dial Emergency 115 / 911. Out of thermal scope: Requires urgent burn ICU inpatient monitoring.")
        
    if has_circumferential:
        is_emergency_override = True
        reasons.append("Circumferential burn of limb or thorax (imminent compartment syndrome / respiratory restriction risk)")
        actions.append("Do NOT apply tight circumferential wraps. Transfer immediately for specialist escharotomy evaluation.")

    # -------------------------------------------------------------
    # 2. DEGREE CLASSIFICATION NORMALIZATION
    # -------------------------------------------------------------
    deg_lower = raw_degree_str.lower()
    
    is_deg3 = (deg_lower in ["degree iii", "độ 3", "degree_3", "3"] or "iii" in deg_lower or "degree 3" in deg_lower)
    is_deg2 = (deg_lower in ["degree ii", "độ 2", "degree_2", "2"] or "ii" in deg_lower or "degree 2" in deg_lower)
    is_deg1 = (deg_lower in ["degree i", "độ 1", "degree_1", "1"] or ("i" in deg_lower and not is_deg2 and not is_deg3))
    is_no_burn = (deg_lower in ["no burn", "không bỏng", "none", "healthy skin", "no_burn"])
    is_unassessed = (is_invalid_degree or deg_lower in ["unassessed", "chưa thể đánh giá", "unknown", "pending", "none_selected", ""] or burn_degree is None)
    
    # -------------------------------------------------------------
    # 3. BURN SEVERITY STRATIFICATION (Adults >= 18)
    # Standard: Ministry of Health Decision 635/QD-BYT & ABA Criteria
    # -------------------------------------------------------------
    safe_tbsa = float(tbsa) if (tbsa is not None and tbsa >= 0) else 0.0
    
    severity_level = "NORMAL"
    action_level = "HOME_CARE"
    
    if is_emergency_override or is_deg3 or safe_tbsa >= 20.0:
        severity_level = "EMERGENCY"
        action_level = "EMERGENCY_115"
        if is_deg3:
            reasons.append("Full-thickness burn (Degree III) - Full dermis destruction requiring urgent specialist surgical transfer")
            actions.append("Dial Emergency 115 / 911 immediately. Drape wound loosely with clean dry sterile sheet; keep patient warm.")
        if safe_tbsa >= 20.0:
            reasons.append(f"Extensive thermal burn surface area (TBSA >= 20%: {safe_tbsa:.1f}%) - Severe burn shock risk")
            actions.append("Immediate 115 transfer for aggressive fluid resuscitation and burn center ICU admission.")
            
    elif (safe_tbsa >= 10.0 and age > 50) or (has_critical_area and is_deg2):
        severity_level = "SEVERE"
        action_level = "HOSPITAL_TRANSFER"
        if safe_tbsa >= 10.0 and age > 50:
            reasons.append(f"Elderly patient (>50 years) with extensive burn (TBSA >= 10%: {safe_tbsa:.1f}%)")
        if has_critical_area and is_deg2:
            reasons.append("Partial-thickness burn involving critical functional or aesthetic zones (Face, Neck, Hands, Feet, or Perineum)")
        actions.append("Specialist outpatient burn clinic or burn center referral required within 24 hours.")
        
    elif is_deg2 or (10.0 <= safe_tbsa < 20.0):
        severity_level = "MODERATE"
        action_level = "CLINIC_VISIT"
        if is_deg2:
            reasons.append("Partial-thickness dermal burn with blistering / weeping base (Degree II)")
        if 10.0 <= safe_tbsa < 20.0:
            reasons.append(f"Moderate thermal burn extent ({safe_tbsa:.1f}% TBSA)")
        actions.append("Cool with running tap water for 20 minutes. Keep blister roofs intact. Consult local medical clinic.")
        
    elif is_unassessed:
        severity_level = "UNASSESSED"
        # If unassessed in critical anatomical area (Face, Neck, Hand), escalate vigilance to CLINIC_VISIT
        if has_critical_area:
            action_level = "CLINIC_VISIT"
            reasons.append("AI could not reliably assess burn depth in a high-priority anatomical zone (Face/Neck/Hands)")
            actions.append("Due to sensitive anatomical location, clinical examination by a physician is strongly advised.")
        else:
            action_level = "CLINIC_VISIT"
            if is_invalid_degree:
                reasons.append("Invalid or unrecognized burn degree label; safety protocol enforces professional clinical triage")
            else:
                reasons.append("AI system unable to reliably grade burn depth from current image")
            actions.append("Administer standard 20-minute clean cool water irrigation. Seek medical assessment if pain or blisters develop.")
            
    elif is_deg1:
        severity_level = "MILD"
        action_level = "HOME_CARE"
        reasons.append("Superficial epidermal burn (Degree I) - Erythema without blistering")
        actions.append("Stream cool running tap water for 15-20 minutes. Apply pure aloe vera or petroleum jelly. No ice or butter.")
        
    else:  # is_no_burn
        severity_level = "NO_BURN"
        if not is_emergency_override:
            action_level = "NORMAL_SKIN"
            reasons.append("Visual analysis indicates healthy, unburned skin")
            actions.append("Continue normal skin monitoring. Re-evaluate if redness, burning sensation, or pain develops.")

    return {
        "severity_level": severity_level,       # NO_BURN, MILD, MODERATE, SEVERE, EMERGENCY, UNASSESSED
        "action_level": action_level,           # NORMAL_SKIN, HOME_CARE, CLINIC_VISIT, HOSPITAL_TRANSFER, EMERGENCY_115
        "is_urgent_emergency": (action_level in ["EMERGENCY_115", "HOSPITAL_TRANSFER"] or is_emergency_override),
        "reasons": reasons,
        "recommended_actions": actions,
        "has_critical_area": has_critical_area,
        "is_emergency_override": is_emergency_override
    }

def map_to_db_schema(eval_result: Dict[str, Any]) -> Tuple[str, str, bool, List[str]]:
    """
    Translates evaluate_safety_and_severity() output into exact Database Schema v9.2 constraints:
    - overall_risk_level: ('emergency', 'high', 'moderate', 'mild', 'none', 'undetermined')
    - medical_escalation_level: ('emergency', 'urgent', 'routine', 'none')
    - show_emergency_call_prompt: bool
    - emergency_reasons: List[str]
    """
    sev = eval_result.get("severity_level", "NORMAL")
    act = eval_result.get("action_level", "HOME_CARE")
    reasons = eval_result.get("reasons", [])
    
    if sev == "EMERGENCY" or act == "EMERGENCY_115":
        return "emergency", "emergency", True, reasons
    elif sev == "SEVERE" or act == "HOSPITAL_TRANSFER":
        return "high", "urgent", False, reasons
    elif sev == "MODERATE" or act == "CLINIC_VISIT":
        return "moderate", "urgent" if eval_result.get("has_critical_area") else "routine", False, reasons
    elif sev == "UNASSESSED":
        return "undetermined", "urgent" if eval_result.get("has_critical_area") else "routine", False, reasons
    elif sev == "MILD" or act == "HOME_CARE":
        return "mild", "none", False, reasons
    elif sev in ["NO_BURN", "NORMAL"]:
        return "none", "none", False, reasons
    else:
        return "mild", "none", False, reasons
