# -*- coding: utf-8 -*-
"""
BurnAid AI Diagnostic System - Database Layer
Schema Version: v9.2 Multi-Area Case Architecture & Production Clinical Data
Reference: BurnAid_Database_v9.1_Full_Production.sql (Schema v9.1 Full Production)

Medical Formulations:
- Standard %TBSA Reference: Prof. Le The Trung (1999), 22 Anatomical Adult Regions (1-3-6-9-18).
- TBSA Formula: tbsa_percent = reference_tbsa * affected_proportion / 100.
- Thermal Burns Exclusivity: Strictly thermal etiology (burn_cause = 'thermal').
- Emergency Escalation: Degree 3 burns unconditionally trigger Emergency 115.
- Abbreviated Burn Severity Index (ABSI): Validated clinical severity calculation.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import clinical_rules

_DIR = os.path.dirname(os.path.abspath(__file__))
_PDIR = os.path.dirname(_DIR)
def _find_db():
    p1 = os.path.join(_DIR, "burnaid.db")
    if os.path.exists(p1): return p1
    p2 = os.path.join(_PDIR, "burnaid.db")
    if os.path.exists(p2): return p2
    return p1
DB_PATH = _find_db()

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Creates a connection to the SQLite database with Foreign Keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

# ============================================================================
# DDL SCHEMA DEFINITIONS (v9.2 Full Production - Strictly Thermal Standard)
# ============================================================================

SCHEMA_SQL = """
-- 1. ACCOUNT (Table 1)
CREATE TABLE IF NOT EXISTS account (
    account_id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    phone_number TEXT,
    date_of_birth TEXT, -- YYYY-MM-DD
    sex TEXT CHECK (sex IN ('male', 'female', 'other', 'unspecified')),
    role TEXT NOT NULL CHECK (role IN ('patient', 'medical_advisor', 'it_staff')),
    account_status TEXT NOT NULL DEFAULT 'active' CHECK (account_status IN ('active', 'locked', 'deactivated')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT
);

-- 2. AI MODEL REGISTRY (Table 2)
CREATE TABLE IF NOT EXISTS ai_model (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type TEXT NOT NULL CHECK (task_type IN ('burn_classification', 'burn_segmentation', 'image_quality_check')),
    architecture TEXT NOT NULL,
    training_strategy TEXT,
    dataset_version TEXT NOT NULL,
    version TEXT NOT NULL,
    decision_threshold REAL CHECK (decision_threshold IS NULL OR (decision_threshold >= 0 AND decision_threshold <= 1)),
    evaluation_metrics TEXT, -- JSON string
    model_size TEXT,
    inference_time_ms INTEGER,
    file_path TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    uploaded_by INTEGER REFERENCES account(account_id),
    uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    activated_at TEXT,
    UNIQUE(task_type, architecture, version)
);

-- 3. BODY REGION REFERENCE (Table 3, GS. Le The Trung 1999 - 22 Regions)
CREATE TABLE IF NOT EXISTS body_region_reference (
    region_code TEXT PRIMARY KEY,
    region_name TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('left', 'right', 'none')),
    reference_tbsa_percent REAL NOT NULL CHECK (reference_tbsa_percent >= 0 AND reference_tbsa_percent <= 100),
    is_critical_priority INTEGER NOT NULL DEFAULT 0 CHECK (is_critical_priority IN (0, 1))
);

-- 4. FIRST AID GUIDELINE (Table 4)
CREATE TABLE IF NOT EXISTS first_aid_guideline (
    guideline_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guideline_code TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    body_region_code TEXT REFERENCES body_region_reference(region_code),
    burn_degree TEXT CHECK (burn_degree IS NULL OR burn_degree IN ('degree_1', 'degree_2', 'degree_3')),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    warnings TEXT,
    source_reference TEXT,
    applicability_rules TEXT,
    status TEXT DEFAULT 'approved' CHECK (status IS NULL OR status IN ('draft', 'approved', 'archived')),
    created_by INTEGER REFERENCES account(account_id),
    approved_by INTEGER REFERENCES account(account_id),
    approved_at TEXT,
    updated_at TEXT,
    UNIQUE(guideline_code, version)
);

-- 5. BURN CASE (Parent Encounter, Table 5 - Strictly Thermal)
CREATE TABLE IF NOT EXISTS burn_case (
    case_id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER REFERENCES account(account_id),
    age_at_assessment INTEGER NOT NULL CHECK (age_at_assessment >= 18),
    sex_at_assessment TEXT NOT NULL CHECK (sex_at_assessment IN ('male', 'female', 'other', 'unspecified')),
    burn_cause TEXT NOT NULL DEFAULT 'thermal' CHECK (burn_cause = 'thermal'),
    injury_time TEXT,
    smoke_exposure_reported INTEGER NOT NULL CHECK (smoke_exposure_reported IN (0, 1)),
    respiratory_symptoms INTEGER NOT NULL CHECK (respiratory_symptoms IN (0, 1)),
    suspected_inhalation_injury INTEGER NOT NULL CHECK (suspected_inhalation_injury IN (0, 1)),
    total_tbsa_percent REAL CHECK (total_tbsa_percent IS NULL OR (total_tbsa_percent >= 0 AND total_tbsa_percent <= 100)),
    absi_score INTEGER CHECK (absi_score IS NULL OR (absi_score >= 2 AND absi_score <= 18)),
    absi_risk_category TEXT,
    absi_not_applicable_reason TEXT CHECK (absi_not_applicable_reason IS NULL OR absi_not_applicable_reason IN ('case_incomplete', 'tbsa_zero_superficial_only', 'sex_not_binary')),
    overall_risk_level TEXT NOT NULL DEFAULT 'undetermined' CHECK (overall_risk_level IN ('undetermined', 'none', 'mild', 'moderate', 'high', 'emergency')),
    medical_escalation_level TEXT NOT NULL DEFAULT 'undetermined' CHECK (medical_escalation_level IN ('undetermined', 'none', 'routine', 'urgent', 'emergency')),
    show_emergency_call_prompt INTEGER NOT NULL DEFAULT 0 CHECK (show_emergency_call_prompt IN (0, 1)),
    emergency_reasons TEXT, -- JSON string
    case_status TEXT NOT NULL DEFAULT 'draft' CHECK (case_status IN ('draft', 'in_progress', 'completed', 'inconclusive')),
    finalized_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT CK_case_inhalation CHECK (
        (suspected_inhalation_injury = 1 AND (smoke_exposure_reported = 1 OR respiratory_symptoms = 1))
        OR (suspected_inhalation_injury = 0 AND smoke_exposure_reported = 0 AND respiratory_symptoms = 0)
    ),
    CONSTRAINT CK_case_absi_consistency CHECK (
        (absi_score IS NOT NULL AND absi_risk_category IS NOT NULL AND absi_not_applicable_reason IS NULL)
        OR (absi_score IS NULL AND absi_risk_category IS NULL)
    ),
    CONSTRAINT CK_case_total_tbsa_status CHECK (case_status = 'completed' OR total_tbsa_percent IS NULL)
);

-- 6. CASE REFERRAL (Clinician Referral, Table 6)
CREATE TABLE IF NOT EXISTS case_referral (
    referral_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL UNIQUE REFERENCES burn_case(case_id) ON DELETE CASCADE,
    assigned_to INTEGER REFERENCES account(account_id),
    sent_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    referral_status TEXT NOT NULL DEFAULT 'sent' CHECK (referral_status IN ('sent', 'viewed', 'reviewed')),
    viewed_at TEXT,
    doctor_notes TEXT,
    reviewed_at TEXT
);

-- 7. BURN AREA ASSESSMENT (Child Photographic Assessment, Table 7)
CREATE TABLE IF NOT EXISTS burn_area_assessment (
    area_assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES burn_case(case_id) ON DELETE CASCADE,
    region_code TEXT NOT NULL REFERENCES body_region_reference(region_code),
    affected_proportion_percent REAL NOT NULL CHECK (affected_proportion_percent >= 1 AND affected_proportion_percent <= 100),
    burn_image_path TEXT,
    burn_mask_path TEXT,
    image_deleted_at TEXT,
    image_retention_status TEXT NOT NULL DEFAULT 'pending' CHECK (image_retention_status IN ('pending', 'active', 'deleted', 'archived')),
    quality_status TEXT NOT NULL DEFAULT 'pending' CHECK (quality_status IN ('pending', 'acceptable', 'rejected')),
    quality_issues TEXT, -- JSON string
    quality_check_model_id INTEGER REFERENCES ai_model(model_id),
    area_status TEXT NOT NULL DEFAULT 'draft' CHECK (area_status IN ('draft', 'processing', 'completed', 'inconclusive')),
    model_predicted_degree TEXT CHECK (model_predicted_degree IS NULL OR model_predicted_degree IN ('degree_1', 'degree_2', 'degree_3')),
    model_confidence_score REAL CHECK (model_confidence_score IS NULL OR (model_confidence_score >= 0 AND model_confidence_score <= 1)),
    classification_model_id INTEGER REFERENCES ai_model(model_id),
    segmentation_model_id INTEGER REFERENCES ai_model(model_id),
    threshold_used REAL,
    prediction_decision TEXT CHECK (prediction_decision IS NULL OR prediction_decision IN ('accepted', 'rejected')),
    rejection_reason TEXT,
    predicted_degree TEXT CHECK (predicted_degree IS NULL OR predicted_degree IN ('degree_1', 'degree_2', 'degree_3')),
    confidence_score REAL CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)),
    is_low_confidence INTEGER NOT NULL DEFAULT 0 CHECK (is_low_confidence IN (0, 1)),
    tbsa_percent REAL CHECK (tbsa_percent IS NULL OR (tbsa_percent >= 0 AND tbsa_percent <= 100)),
    clinical_review_status TEXT NOT NULL DEFAULT 'pending' CHECK (clinical_review_status IN ('pending', 'approved', 'flagged', 'rejected')),
    assessed_at TEXT
);

-- 8. ASSESSMENT GUIDELINE JUNCTION (Table 8)
CREATE TABLE IF NOT EXISTS assessment_guideline (
    assessment_guideline_id INTEGER PRIMARY KEY AUTOINCREMENT,
    area_assessment_id INTEGER NOT NULL REFERENCES burn_area_assessment(area_assessment_id) ON DELETE CASCADE,
    guideline_id INTEGER NOT NULL REFERENCES first_aid_guideline(guideline_id),
    display_order INTEGER NOT NULL,
    assigned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(area_assessment_id, guideline_id)
);

-- 9. FEEDBACK & CLINICIAN REVIEW LOG (Table 9)
CREATE TABLE IF NOT EXISTS feedback_review_log (
    review_entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER REFERENCES account(account_id),
    area_assessment_id INTEGER REFERENCES burn_area_assessment(area_assessment_id) ON DELETE SET NULL,
    guideline_id INTEGER REFERENCES first_aid_guideline(guideline_id),
    entry_type TEXT NOT NULL CHECK (entry_type IN ('user_feedback', 'low_confidence_case')),
    rating INTEGER CHECK (rating IS NULL OR (rating >= 1 AND rating <= 5)),
    comment TEXT,
    reviewed_by INTEGER REFERENCES account(account_id),
    review_decision TEXT,
    review_notes TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'reviewed')),
    submitted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TEXT
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS IX_case_account ON burn_case(account_id);
CREATE INDEX IF NOT EXISTS IX_case_status ON burn_case(case_status);
CREATE INDEX IF NOT EXISTS IX_referral_doctor ON case_referral(assigned_to, referral_status);
CREATE INDEX IF NOT EXISTS IX_area_case ON burn_area_assessment(case_id);
CREATE INDEX IF NOT EXISTS IX_area_region ON burn_area_assessment(region_code);
CREATE INDEX IF NOT EXISTS IX_area_status ON burn_area_assessment(area_status);
CREATE INDEX IF NOT EXISTS IX_ag_area ON assessment_guideline(area_assessment_id);
CREATE INDEX IF NOT EXISTS IX_feedback_area ON feedback_review_log(area_assessment_id);
CREATE INDEX IF NOT EXISTS IX_feedback_guideline ON feedback_review_log(guideline_id);
CREATE INDEX IF NOT EXISTS IX_guideline_region ON first_aid_guideline(body_region_code);

-- Standard Schema Views (Matching Full Production Schema v9.1)
CREATE VIEW IF NOT EXISTS v_case_user_result AS
SELECT c.case_id, c.account_id, c.case_status, c.total_tbsa_percent,
       c.overall_risk_level, c.medical_escalation_level, c.show_emergency_call_prompt, c.finalized_at
FROM burn_case c;

CREATE VIEW IF NOT EXISTS v_area_user_result AS
SELECT a.area_assessment_id, a.case_id, r.region_name, r.side,
       a.area_status, a.predicted_degree, a.tbsa_percent
FROM burn_area_assessment a
JOIN body_region_reference r ON r.region_code = a.region_code;

CREATE VIEW IF NOT EXISTS v_case_clinician_summary AS
SELECT c.*, f.referral_id, f.assigned_to, f.sent_at, f.referral_status, f.viewed_at, f.doctor_notes, f.reviewed_at
FROM burn_case c
JOIN case_referral f ON f.case_id = c.case_id;

-- Backward Compatibility View for Single Area Assessment
CREATE VIEW IF NOT EXISTS burn_assessment AS
SELECT 
    a.area_assessment_id AS assessment_id,
    c.account_id,
    c.age_at_assessment,
    c.sex_at_assessment,
    c.burn_cause,
    c.injury_time,
    c.smoke_exposure_reported,
    c.respiratory_symptoms,
    c.suspected_inhalation_injury,
    a.region_code,
    a.affected_proportion_percent,
    a.burn_image_path,
    a.burn_mask_path,
    a.area_status AS assessment_status,
    a.predicted_degree,
    a.confidence_score,
    a.classification_model_id,
    a.segmentation_model_id,
    a.is_low_confidence,
    a.tbsa_percent,
    CASE WHEN a.predicted_degree = 'degree_3' THEN 1 ELSE 0 END AS full_thickness_burn_present,
    1 AS is_only_burn_area,
    c.overall_risk_level,
    c.medical_escalation_level,
    c.show_emergency_call_prompt,
    c.emergency_reasons,
    a.clinical_review_status AS review_status,
    a.assessed_at
FROM burn_area_assessment a
JOIN burn_case c ON a.case_id = c.case_id;

CREATE VIEW IF NOT EXISTS v_single_area_assessment AS
SELECT * FROM burn_assessment;
"""

# ============================================================================
# MEDICAL CALCULATION FUNCTIONS
# ============================================================================

def calculate_tbsa(reference_tbsa: float, affected_proportion_percent: float, degree: Optional[str] = None) -> float:
    """Calculates TBSA based on GS. Le The Trung standard (1-3-6-9-18)."""
    if degree == 'degree_1' or not affected_proportion_percent or affected_proportion_percent <= 0:
        return 0.0
    return round((reference_tbsa * affected_proportion_percent) / 100.0, 2)

def calculate_absi(age: int, sex: str, tbsa_pct: Optional[float], has_inhalation: bool, has_deg3: bool) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Calculates Abbreviated Burn Severity Index (ABSI) according to clinical standards."""
    if tbsa_pct is None or tbsa_pct <= 0:
        return None, None, 'tbsa_zero_superficial_only'
    if sex not in ('male', 'female'):
        return None, None, 'sex_not_binary'

    score = 0
    if sex == 'female':
        score += 1

    if age <= 20:
        score += 1
    elif age <= 40:
        score += 2
    elif age <= 60:
        score += 3
    elif age <= 80:
        score += 4
    else:
        score += 5

    if has_inhalation:
        score += 1
    if has_deg3:
        score += 1

    if tbsa_pct <= 10.0:
        score += 1
    elif tbsa_pct <= 20.0:
        score += 2
    elif tbsa_pct <= 30.0:
        score += 3
    elif tbsa_pct <= 40.0:
        score += 4
    elif tbsa_pct <= 50.0:
        score += 5
    elif tbsa_pct <= 60.0:
        score += 6
    elif tbsa_pct <= 70.0:
        score += 7
    elif tbsa_pct <= 80.0:
        score += 8
    elif tbsa_pct <= 90.0:
        score += 9
    else:
        score += 10

    if score <= 3:
        cat = 'Low Risk (Score 2-3, >99% survival)'
    elif score <= 5:
        cat = 'Moderate Risk (Score 4-5, ~98% survival)'
    elif score <= 7:
        cat = 'Serious Risk (Score 6-7, ~80-90% survival)'
    elif score <= 9:
        cat = 'Severe Risk (Score 8-9, ~50-70% survival)'
    elif score <= 11:
        cat = 'Maximum Risk (Score 10-11, ~20-40% survival)'
    else:
        cat = 'Extremely Severe Risk (Score >=12, <10% survival)'
    return score, cat, None

def determine_escalation(
    degree: Optional[str],
    tbsa_percent: Optional[float],
    burn_cause: str,
    suspected_inhalation: bool,
    region_code: str,
    age: int = 28
) -> Tuple[str, str, bool, List[str]]:
    """Determines triage escalation level and emergency prompt strictly unified via clinical_rules.py."""
    eval_res = clinical_rules.evaluate_safety_and_severity(
        tbsa=tbsa_percent,
        age=age,
        burn_degree=degree,
        has_inhalation=suspected_inhalation,
        body_regions=[region_code] if region_code else [],
        burn_cause=burn_cause
    )
    return clinical_rules.map_to_db_schema(eval_res)

# ============================================================================
# INITIALIZATION & SEEDING
# ============================================================================

def init_db(db_path: str = DB_PATH) -> None:
    """Initializes tables, views and indexes in SQLite."""
    conn = get_connection(db_path)
    with conn:
        conn.executescript(SCHEMA_SQL)
    conn.close()

def seed_reference_data(db_path: str = DB_PATH) -> None:
    """Seeds rich, medically validated production data matching Schema v9.2."""
    conn = get_connection(db_path)
    with conn:
        # 1. Accounts
        accounts = [
            (1, 'admin@burnaid.vn', '$2a$12$e8Yh7YgHqB8JmG1kLk1lYe2o8A5hT3wF7p6R1gV4z9b7Y1u9m6qWy', 'Nguyễn Hoàng Long', '0912345678', '1988-04-12', 'male', 'it_staff', 'active', '2026-01-01 08:00:00'),
            (2, 'mlops.lead@burnaid.vn', '$2a$12$K8d2M3n4P5q6R7s8T9u0v1w2x3y4z5A6b7C8d9E0F1G2H3I4J5K6L', 'Trần Minh Tuấn', '0988776655', '1992-09-25', 'male', 'it_staff', 'active', '2026-01-05 09:30:00'),
            (3, 'dr.thang.tran@burnaid.vn', '$2a$12$a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8s9T0u1V2w3X4y5Z6a', 'TS.BS. Trần Đình Thắng', '0903214567', '1975-02-18', 'male', 'medical_advisor', 'active', '2026-01-10 10:00:00'),
            (4, 'dr.ha.nguyen@burnaid.vn', '$2a$12$z9Y8x7W6v5U4t3S2r1Q0p9O8n7M6l5K4j3I2h1G0f9E8d7C6b5A4z', 'ThS.BS. Nguyễn Thu Hà', '0913987654', '1982-11-05', 'female', 'medical_advisor', 'active', '2026-01-10 10:15:00'),
            (5, 'dr.bach.le@burnaid.vn', '$2a$12$b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0U1v2W3x4Y5z6A7b', 'BSCKII. Lê Văn Bách', '0934567890', '1978-07-22', 'male', 'medical_advisor', 'active', '2026-01-12 14:00:00'),
            (6, 'patient.nam@gmail.com', '$2a$12$p1A2t3I4e5N6t7H8a9S0h1V2a3L4i5D6k7E8y9F0u1L2l3S4e5E6d', 'Phạm Hoài Nam', '0971239845', '1998-03-14', 'male', 'patient', 'active', '2026-02-01 11:20:00'),
            (7, 'lethimai.bn@yahoo.com', '$2a$12$m1A2i3L4e5T6h7I8m9A0i1H2a3S4h5V6a7L8i9D0k1E2y3F4u5L6l', 'Lê Thị Mai', '0965432189', '1981-08-30', 'female', 'patient', 'active', '2026-02-05 15:40:00'),
            (8, 'vu.quoc.anh@outlook.com', '$2a$12$q1U2o3C4a5N6h7V8u9Q0u1O2c3A4n5H6v7A8l9I0d1K2e3Y4f5U6l', 'Vũ Quốc Anh', '0943216789', '1992-12-04', 'male', 'patient', 'active', '2026-02-14 09:10:00'),
            (9, 'hoangthilan.nd@gmail.com', '$2a$12$l1A2n3H4o5A6n7G8t9H0i1L2a3N4h5A6s7H8v9A0l1I2d3K4e5Y6f', 'Hoàng Thị Lan', '0922334455', '1974-05-19', 'female', 'patient', 'active', '2026-03-01 16:05:00'),
            (10, 'dang.van.tai@fpt.edu.vn', '$2a$12$t1A2i3D4a5N6g7V8a9N0t1A2i3H4a5S6h7V8a9L0i1D2k3E4y5F6u', 'Đặng Văn Tài', '0981122334', '2002-10-10', 'male', 'patient', 'active', '2026-03-15 08:45:00'),
            (11, 'do.thanh.huong@hanoimail.vn', '$2a$12$h1U2o3N4g5D6o7T8h9A0n1H2h3U4o5N6g7H8a9S0h1V2a3L4i5D6k', 'Đỗ Thanh Hương', '0933445566', '1965-06-15', 'female', 'patient', 'active', '2026-03-20 13:15:00'),
            (12, 'bui.xuan.kien@gmail.com', '$2a$12$k1I2e3N4b5U6i7X8u9A0n1K2i3E4n5H6a7S8h9V0a1L2i3D4k5E6y', 'Bùi Xuân Kiên', '0919887766', '1986-01-28', 'male', 'patient', 'active', '2026-04-02 18:22:00')
        ]
        for acc in accounts:
            conn.execute("""
                INSERT OR IGNORE INTO account (account_id, email, password_hash, full_name, phone_number, date_of_birth, sex, role, account_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, acc)

        # 2. AI Model Registry
        models = [
            (1, 'burn_classification', 'ConvNeXtV2-Nano-FCMAE', 'Transfer Learning on 4-Class Thermal Burn Dataset (Degree 1, 2, 3, No Burn)', 'v2.4-augmented', '2.4.0', 0.7000, '{"val_accuracy": 0.9480, "macro_f1": 0.9412, "auc_roc": 0.9820}', '60MB', 42, 'models/convnextv2_burn_4class_best.pth', 1, 2, '2026-01-15 10:00:00', '2026-01-15 14:00:00'),
            (2, 'burn_segmentation', 'U-NetPlusPlus-ResNet34', 'Random curriculum learning with Dice-BCE loss and scSE attention heads', 'v2.4-augmented', '2.1.0', 0.5000, '{"dice_score": 0.8870, "iou_score": 0.7980, "boundary_f1": 0.8620}', '104MB', 85, 'models/best_random_curriculum_seed2026.pt', 1, 2, '2026-01-16 11:30:00', '2026-01-16 15:00:00'),
            (3, 'image_quality_check', 'Laplacian-Illumination-Filter', 'Pre-inference QA filter: blur variance, underexposure, glare detection', 'v1.2-quality', '1.2.0', 0.6500, '{"blur_detection_accuracy": 0.9650, "exposure_auc": 0.9710}', '9MB', 12, 'models/quality_check_filter.pt', 1, 2, '2026-01-18 09:00:00', '2026-01-18 10:00:00')
        ]
        for m in models:
            conn.execute("""
                INSERT OR IGNORE INTO ai_model (model_id, task_type, architecture, training_strategy, dataset_version, version, decision_threshold, evaluation_metrics, model_size, inference_time_ms, file_path, is_active, uploaded_by, uploaded_at, activated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, m)

        # 3. Body Region Reference (22 Regions matching Schema v9.2)
        regions = [
            ('FACE', 'Face (scalp excluded)', 'none', 6.00, 1),
            ('NECK', 'Neck / Nape', 'none', 3.00, 0),
            ('ARM_TOTAL_L', 'Left upper limb (whole)', 'left', 9.00, 0),
            ('ARM_UPPER_L', 'Left upper arm', 'left', 3.00, 0),
            ('FOREARM_L', 'Left forearm', 'left', 3.00, 0),
            ('HAND_L', 'Left hand', 'left', 3.00, 1),
            ('ARM_TOTAL_R', 'Right upper limb (whole)', 'right', 9.00, 0),
            ('ARM_UPPER_R', 'Right upper arm', 'right', 3.00, 0),
            ('FOREARM_R', 'Right forearm', 'right', 3.00, 0),
            ('HAND_R', 'Right hand', 'right', 3.00, 1),
            ('CHEST', 'Chest', 'none', 9.00, 0),
            ('ABDOMEN', 'Abdomen', 'none', 9.00, 0),
            ('UPPER_BACK', 'Upper back', 'none', 9.00, 0),
            ('LOWER_BACK', 'Lower back', 'none', 9.00, 0),
            ('LEG_TOTAL_L', 'Left lower limb (whole)', 'left', 18.00, 0),
            ('THIGH_L', 'Left thigh', 'left', 9.00, 1),
            ('LEG_L', 'Left lower leg', 'left', 6.00, 0),
            ('FOOT_L', 'Left foot', 'left', 3.00, 1),
            ('LEG_TOTAL_R', 'Right lower limb (whole)', 'right', 18.00, 0),
            ('THIGH_R', 'Right thigh', 'right', 9.00, 1),
            ('LEG_R', 'Right lower leg', 'right', 6.00, 0),
            ('FOOT_R', 'Right foot', 'right', 3.00, 1)
        ]
        for r in regions:
            conn.execute("""
                INSERT OR IGNORE INTO body_region_reference (region_code, region_name, side, reference_tbsa_percent, is_critical_priority)
                VALUES (?, ?, ?, ?, ?)
            """, r)

        # 4. First Aid Guidelines
        guidelines = [
            (1, 'GUIDE-DEG1-GENERAL', 1, None, 'degree_1', 'First-Degree Thermal Burn Standard Care Protocol',
             'Superficial thermal injury affecting epidermis only (erythema, pain, dry surface, no blisters). Cool under running tap water (15-20°C) for 15-20 minutes. Keep clean and hydrated with pure petroleum jelly. Provide over-the-counter paracetamol for analgesia.',
             'Strictly forbidden to apply ice, iced water, butter, fish sauce, egg whites, or toothpaste.',
             'WHO Burn First Aid 2023, Ministry of Health Decision 635/QĐ-BYT', 'Applicable for superficial thermal burns with intact epidermis.', 'approved', 2, 3, '2026-01-20 10:00:00'),

            (2, 'GUIDE-DEG2-GENERAL', 1, None, 'degree_2', 'Second-Degree Partial-Thickness Thermal Burn Care Protocol',
             'Partial-thickness dermal burn with active blistering, severe pain, and exudative base. Cool immediately under running clean tap water (15-20°C) for 15-20 minutes. Preserve blister roofs intact as natural sterile barrier. Cover loosely with sterile paraffin gauze or clean non-adherent dressing. Elevate limb.',
             'DO NOT pop or burst blisters. DO NOT apply tight restrictive tourniquets or folk remedies.',
             'American Burn Association (ABA) Practice Guidelines 2023, Vietnam National Burn Hospital', 'Applicable for superficial and deep partial-thickness thermal burns.', 'approved', 2, 3, '2026-01-20 10:30:00'),

            (3, 'GUIDE-DEG3-EMERGENCY', 1, None, 'degree_3', 'Third-Degree Full-Thickness Thermal Burn Emergency Protocol',
             'Full-thickness dermal destruction (leathery white, charred, or translucent parchment appearance, reduced sensation). MEDICAL EMERGENCY: Call Emergency Services (115) immediately. Maintain open airway, monitor breathing. Drape clean dry sheet loosely over wound. Keep patient warm with blanket to prevent hypothermia during transport.',
             'DO NOT immerse large burns in cold water (severe hypothermia risk). DO NOT apply ointments, topical creams, or adhesive bandages.',
             'Advanced Burn Life Support (ABLS), Ministry of Health Decision 635/QĐ-BYT', 'Mandatory emergency escalation for any full-thickness third-degree thermal burn.', 'approved', 2, 3, '2026-01-20 11:00:00'),

            (4, 'GUIDE-HAND-SPECIALIZED', 1, 'HAND_L', 'degree_2', 'Specialized Second-Degree Thermal Burn Protocol for Hand & Fingers',
             'Hands represent critical functional anatomical zones. Active cooling for 20 minutes with gentle water. Remove rings, bracelets, and watches immediately before edema develops. Keep fingers separated with sterile paraffin gauze to prevent web-space contractures. Elevate hand on pillows above heart level.',
             'Do not allow burned fingers to adhere together. Prompt physician evaluation required.',
             'ABA Specialized Functional Anatomical Guidelines', 'Applicable for thermal burns involving hands or digits.', 'approved', 2, 4, '2026-01-22 14:00:00'),

            (5, 'GUIDE-FACE-SPECIALIZED', 1, 'FACE', 'degree_2', 'Specialized Thermal Burn Care for Facial Region',
             'Facial burns carry heightened risks of airway compromise and corneal injury. Rinse gently with clean cool water. Keep head elevated 30-45 degrees (Semi-Fowler position) to minimize facial edema. Assess breathing and singed nasal hairs. Seek immediate medical evaluation.',
             'Never apply pressure dressings or occlusive bandages over face or neck.',
             'Vietnam National Burn Hospital - Facial Burn Protocol', 'Applicable for facial and anterior neck thermal burns.', 'approved', 2, 3, '2026-01-22 15:00:00')
        ]
        for g in guidelines:
            conn.execute("""
                INSERT OR IGNORE INTO first_aid_guideline (guideline_id, guideline_code, version, body_region_code, burn_degree, title, content, warnings, source_reference, applicability_rules, status, created_by, approved_by, approved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, g)

        # 5. Burn Cases (10 Clinical Cases: Strictly Thermal, Valid ABSI)
        cases = [
            (1, 6, 28, 'male', 'thermal', '2026-02-10 14:15:00', 0, 0, 0, 4.50, 3, 'Low Risk (Score 2-3, >99% survival)', None, 'mild', 'routine', 0, None, 'completed', '2026-02-10 14:35:00', '2026-02-10 14:20:00'),
            (2, 7, 45, 'female', 'thermal', '2026-02-15 18:30:00', 0, 0, 0, 12.00, 6, 'Serious Risk (Score 6-7, ~80-90% survival)', None, 'moderate', 'urgent', 0, None, 'completed', '2026-02-15 19:10:00', '2026-02-15 18:50:00'),
            (3, 8, 34, 'male', 'thermal', '2026-02-20 08:20:00', 0, 0, 0, 6.00, 4, 'Moderate Risk (Score 4-5, ~98% survival)', None, 'emergency', 'emergency', 1, '["full_thickness_degree_3_thermal_burn", "critical_functional_hand_injury"]', 'completed', '2026-02-20 08:45:00', '2026-02-20 08:30:00'),
            (4, 9, 52, 'female', 'thermal', '2026-03-05 21:00:00', 1, 1, 1, 27.00, 9, 'Severe Risk (Score 8-9, ~50-70% survival)', None, 'emergency', 'emergency', 1, '["suspected_acute_inhalation_injury", "smoke_exposure_with_dyspnea", "extensive_tbsa_exceeding_20_percent"]', 'completed', '2026-03-05 21:40:00', '2026-03-05 21:15:00'),
            (5, 11, 61, 'female', 'thermal', '2026-03-22 10:10:00', 0, 0, 0, 4.50, 6, 'Serious Risk (Score 6-7, ~80-90% survival)', None, 'moderate', 'urgent', 0, None, 'completed', '2026-03-22 10:45:00', '2026-03-22 10:20:00'),
            (6, 10, 29, 'male', 'thermal', '2026-03-28 15:40:00', 0, 0, 0, 4.50, 3, 'Low Risk (Score 2-3, >99% survival)', None, 'mild', 'routine', 0, None, 'completed', '2026-03-28 16:15:00', '2026-03-28 15:55:00'),
            (7, 7, 24, 'female', 'thermal', '2026-04-01 11:30:00', 0, 0, 0, 0.60, None, None, 'tbsa_zero_superficial_only', 'mild', 'routine', 0, None, 'completed', '2026-04-01 11:55:00', '2026-04-01 11:40:00'),
            (8, 12, 40, 'male', 'thermal', '2026-04-03 20:10:00', 0, 0, 0, None, None, None, 'case_incomplete', 'undetermined', 'routine', 0, None, 'inconclusive', None, '2026-04-03 20:15:00'),
            (9, 6, 31, 'male', 'thermal', '2026-04-05 13:00:00', 0, 0, 0, None, None, None, 'case_incomplete', 'undetermined', 'undetermined', 0, None, 'in_progress', None, '2026-04-05 13:10:00'),
            (10, 8, 68, 'male', 'thermal', '2026-04-08 09:20:00', 0, 0, 0, 18.00, 6, 'Serious Risk (Score 6-7, ~80-90% survival)', None, 'moderate', 'urgent', 0, None, 'completed', '2026-04-08 10:05:00', '2026-04-08 09:35:00')
        ]
        for c in cases:
            conn.execute("""
                INSERT OR IGNORE INTO burn_case (
                    case_id, account_id, age_at_assessment, sex_at_assessment, burn_cause, injury_time,
                    smoke_exposure_reported, respiratory_symptoms, suspected_inhalation_injury,
                    total_tbsa_percent, absi_score, absi_risk_category, absi_not_applicable_reason,
                    overall_risk_level, medical_escalation_level, show_emergency_call_prompt, emergency_reasons,
                    case_status, finalized_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, c)

        # 6. Burn Area Assessments
        areas = [
            (1, 1, 'FOREARM_L', 75.00, 'uploads/cases/case_001/area_01_forearm_l.jpg', 'uploads/masks/case_001/mask_01_forearm_l.png', None, 'active', 'acceptable', '{"quality_passed": true}', 3, 'completed', 'degree_2', 0.9140, 1, 2, 0.7000, 'accepted', None, 'degree_2', 0.9140, 0, 2.25, 'approved', '2026-02-10 14:30:00'),
            (2, 1, 'HAND_L', 75.00, 'uploads/cases/case_001/area_02_hand_l.jpg', 'uploads/masks/case_001/mask_02_hand_l.png', None, 'active', 'acceptable', '{"quality_passed": true}', 3, 'completed', 'degree_2', 0.8850, 1, 2, 0.7000, 'accepted', None, 'degree_2', 0.8850, 0, 2.25, 'approved', '2026-02-10 14:32:00'),
            (3, 2, 'ARM_UPPER_R', 100.00, 'uploads/cases/case_002/area_01_arm_upper_r.jpg', 'uploads/masks/case_002/mask_01_arm_upper_r.png', None, 'active', 'acceptable', '{"quality_passed": true}', 3, 'completed', 'degree_2', 0.9420, 1, 2, 0.7000, 'accepted', None, 'degree_2', 0.9420, 0, 3.00, 'approved', '2026-02-15 19:02:00'),
            (4, 2, 'CHEST', 100.00, 'uploads/cases/case_002/area_02_chest.jpg', 'uploads/masks/case_002/mask_02_chest.png', None, 'active', 'acceptable', '{"quality_passed": true}', 3, 'completed', 'degree_2', 0.9150, 1, 2, 0.7000, 'accepted', None, 'degree_2', 0.9150, 0, 9.00, 'approved', '2026-02-15 19:06:00'),
            (5, 3, 'HAND_R', 100.00, 'uploads/cases/case_003/area_01_hand_r.jpg', 'uploads/masks/case_003/mask_01_hand_r.png', None, 'active', 'acceptable', '{"quality_passed": true}', 3, 'completed', 'degree_3', 0.9680, 1, 2, 0.7000, 'accepted', None, 'degree_3', 0.9680, 0, 3.00, 'approved', '2026-02-20 08:38:00'),
            (6, 3, 'FOREARM_R', 100.00, 'uploads/cases/case_003/area_02_forearm_r.jpg', 'uploads/masks/case_003/mask_02_forearm_r.png', None, 'active', 'acceptable', '{"quality_passed": true}', 3, 'completed', 'degree_3', 0.9350, 1, 2, 0.7000, 'accepted', None, 'degree_3', 0.9350, 0, 3.00, 'approved', '2026-02-20 08:42:00'),
            (7, 4, 'CHEST', 100.00, 'uploads/cases/case_004/area_01_chest.jpg', 'uploads/masks/case_004/mask_01_chest.png', None, 'active', 'acceptable', '{"quality_passed": true}', 3, 'completed', 'degree_2', 0.8840, 1, 2, 0.7000, 'accepted', None, 'degree_2', 0.8840, 0, 9.00, 'approved', '2026-03-05 21:28:00'),
            (8, 4, 'ABDOMEN', 100.00, 'uploads/cases/case_004/area_02_abdomen.jpg', 'uploads/masks/case_004/mask_02_abdomen.png', None, 'active', 'acceptable', '{"quality_passed": true}', 3, 'completed', 'degree_2', 0.8920, 1, 2, 0.7000, 'accepted', None, 'degree_2', 0.8920, 0, 9.00, 'approved', '2026-03-05 21:30:00'),
            (9, 4, 'THIGH_L', 100.00, 'uploads/cases/case_004/area_03_thigh_l.jpg', 'uploads/masks/case_004/mask_03_thigh_l.png', None, 'active', 'acceptable', '{"quality_passed": true}', 3, 'completed', 'degree_3', 0.9520, 1, 2, 0.7000, 'accepted', None, 'degree_3', 0.9520, 0, 9.00, 'approved', '2026-03-05 21:35:00'),
            (10, 5, 'FOOT_R', 50.00, 'uploads/cases/case_005/area_01_foot_r.jpg', 'uploads/masks/case_005/mask_01_foot_r.png', None, 'active', 'acceptable', '{"quality_passed": true}', 3, 'completed', 'degree_2', 0.8790, 1, 2, 0.7000, 'accepted', None, 'degree_2', 0.8790, 0, 1.50, 'approved', '2026-03-22 10:35:00'),
            (11, 5, 'HAND_R', 100.00, 'uploads/cases/case_005/area_02_hand_r.jpg', 'uploads/masks/case_005/mask_02_hand_r.png', None, 'active', 'acceptable', '{"quality_passed": true}', 3, 'completed', 'degree_2', 0.9020, 1, 2, 0.7000, 'accepted', None, 'degree_2', 0.9020, 0, 3.00, 'approved', '2026-03-22 10:40:00'),
            (12, 6, 'THIGH_L', 50.00, 'uploads/cases/case_006/area_01_thigh_l.jpg', 'uploads/masks/case_006/mask_01_thigh_l.png', None, 'active', 'acceptable', '{"quality_passed": true}', 3, 'completed', 'degree_2', 0.8650, 1, 2, 0.7000, 'accepted', None, 'degree_2', 0.8650, 0, 4.50, 'approved', '2026-03-28 16:05:00'),
            (13, 7, 'HAND_L', 20.00, 'uploads/cases/case_007/area_01_hand_l.jpg', 'uploads/masks/case_007/mask_01_hand_l.png', None, 'active', 'acceptable', '{"quality_passed": true}', 3, 'completed', 'degree_1', 0.9410, 1, 2, 0.7000, 'accepted', None, 'degree_1', 0.9410, 0, 0.60, 'approved', '2026-04-01 11:48:00'),
            (14, 8, 'LEG_TOTAL_L', 10.00, 'uploads/cases/case_008/area_01_leg_blurry.jpg', None, None, 'active', 'rejected', '{"blur_detected": true, "variance_score": 42.5}', 3, 'inconclusive', 'degree_2', 0.5420, 1, 2, 0.7000, 'rejected', 'image_blur_exceeds_threshold', None, None, 1, None, 'flagged', None)
        ]
        for a in areas:
            conn.execute("""
                INSERT OR IGNORE INTO burn_area_assessment (
                    area_assessment_id, case_id, region_code, affected_proportion_percent,
                    burn_image_path, burn_mask_path, image_deleted_at, image_retention_status,
                    quality_status, quality_issues, quality_check_model_id,
                    area_status, model_predicted_degree, model_confidence_score,
                    classification_model_id, segmentation_model_id, threshold_used,
                    prediction_decision, rejection_reason, predicted_degree, confidence_score,
                    is_low_confidence, tbsa_percent, clinical_review_status, assessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, a)

        # 7. Case Referrals
        referrals = [
            (1, 2, 4, '2026-02-15 19:15:00', 'reviewed', '2026-02-15 19:30:00', 'Confirmed 2nd degree thermal burn on right arm and chest. Prescribe antimicrobial dressings and schedule wound check in 48 hours.', '2026-02-15 20:00:00'),
            (2, 3, 3, '2026-02-20 08:48:00', 'reviewed', '2026-02-20 08:52:00', 'Emergency 115 transport confirmed. Deep 3rd degree burn on hand requires surgical debridement and compartment syndrome assessment at tertiary burn center.', '2026-02-20 09:10:00'),
            (3, 4, 3, '2026-03-05 21:42:00', 'reviewed', '2026-03-05 21:45:00', 'Severe burn with inhalation injury. Emergency intubation protocol initiated during ICU transfer.', '2026-03-05 22:00:00'),
            (4, 10, 5, '2026-04-08 10:10:00', 'viewed', '2026-04-08 10:25:00', 'Elderly patient with extensive partial-thickness burn. Recommend hospitalization for fluid management and infection prevention.', None)
        ]
        for ref in referrals:
            conn.execute("""
                INSERT OR IGNORE INTO case_referral (referral_id, case_id, assigned_to, sent_at, referral_status, viewed_at, doctor_notes, reviewed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, ref)

        # 8. Assessment Guidelines
        ag_links = [
            (1, 1, 2, 1, '2026-02-10 14:30:00'),
            (2, 2, 4, 1, '2026-02-10 14:32:00'),
            (3, 3, 2, 1, '2026-02-15 19:02:00'),
            (4, 4, 2, 1, '2026-02-15 19:06:00'),
            (5, 5, 3, 1, '2026-02-20 08:38:00'),
            (6, 6, 3, 1, '2026-02-20 08:42:00'),
            (7, 7, 2, 1, '2026-03-05 21:28:00'),
            (8, 8, 2, 1, '2026-03-05 21:30:00'),
            (9, 9, 3, 1, '2026-03-05 21:35:00'),
            (10, 10, 2, 1, '2026-03-22 10:35:00'),
            (11, 11, 4, 1, '2026-03-22 10:40:00'),
            (12, 12, 2, 1, '2026-03-28 16:05:00'),
            (13, 13, 1, 1, '2026-04-01 11:48:00')
        ]
        for ag in ag_links:
            conn.execute("""
                INSERT OR IGNORE INTO assessment_guideline (assessment_guideline_id, area_assessment_id, guideline_id, display_order, assigned_at)
                VALUES (?, ?, ?, ?, ?)
            """, ag)

        # 9. Feedback & Review Logs
        feedbacks = [
            (1, 6, 1, 2, 'user_feedback', 5, 'AI identified the burn wound very rapidly, 20-minute cool water instruction was clear and timely.', 3, 'approved', 'Accurate triage and compliant first-aid steps delivered.', 'reviewed', '2026-02-10 15:00:00', '2026-02-10 16:30:00'),
            (2, 7, 4, 2, 'user_feedback', 4, 'Very helpful guidance on keeping blisters intact and cooling the chest burn properly.', 4, 'approved', 'Compliant pre-hospital advice.', 'reviewed', '2026-02-15 20:30:00', '2026-02-16 08:15:00'),
            (3, 12, 14, None, 'low_confidence_case', None, 'Blurry image on lower leg, confidence score 54.2% below safety threshold (70%). System safely rejected and requested retake.', 5, 'reviewed', 'Safety rejection confirmed: blurry image handled appropriately with retake request.', 'reviewed', '2026-04-03 20:26:00', '2026-04-04 09:00:00')
        ]
        for fb in feedbacks:
            conn.execute("""
                INSERT OR IGNORE INTO feedback_review_log (review_entry_id, account_id, area_assessment_id, guideline_id, entry_type, rating, comment, reviewed_by, review_decision, review_notes, status, submitted_at, reviewed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, fb)

    conn.close()

# ============================================================================
# HIGH-LEVEL DATA ACCESS OBJECT (DAO) FUNCTIONS
# ============================================================================

def create_single_area_assessment(
    account_id: Optional[int],
    age: int,
    sex: str,
    burn_cause: str,
    region_code: str,
    affected_proportion_percent: float,
    smoke_exposure: bool,
    respiratory_symptoms: bool,
    is_only_burn_area: bool,
    predicted_degree: Optional[str],
    confidence_score: Optional[float],
    burn_image_path: Optional[str] = None,
    burn_mask_path: Optional[str] = None,
    classification_model_id: int = 1,
    segmentation_model_id: Optional[int] = 2,
    db_path: str = DB_PATH
) -> Dict[str, Any]:
    """
    Creates a completed or inconclusive single-area assessment according to production clinical rules.
    Thermal burns only, valid ABSI score, Degree 3 unconditionally triggers Emergency 115.
    """
    conn = get_connection(db_path)
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Enforce strict thermal etiology
    clean_cause = str(burn_cause).strip().lower()
    if clean_cause != 'thermal':
        conn.close()
        raise ValueError(f"Out-of-scope burn etiology: '{clean_cause}'. Database schema strictly enforces 'thermal' burns only.")

    # Canonical region code alias normalization
    REGION_ALIASES = {
        "LOWER_LEG_L": "LEG_L",
        "LOWER_LEG_R": "LEG_R",
        "HEAD_NECK": "FACE",
        "HEAD_FACE": "FACE",
        "BACK_UPPER": "UPPER_BACK",
        "BACK_LOWER": "LOWER_BACK",
        "ARM_LOWER_L": "FOREARM_L",
        "ARM_LOWER_R": "FOREARM_R",
        "ARM_UPPER_L": "ARM_UPPER_L",
        "ARM_UPPER_R": "ARM_UPPER_R",
        "PERINEUM": "ABDOMEN",
        "GENITAL": "ABDOMEN",
        "GROIN": "ABDOMEN",
        "BUTTOCK": "LOWER_BACK"
    }
    canonical_region = REGION_ALIASES.get(str(region_code).strip(), str(region_code).strip())

    cursor = conn.cursor()
    cursor.execute("SELECT reference_tbsa_percent, is_critical_priority FROM body_region_reference WHERE region_code = ?", (canonical_region,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Unknown region_code: {canonical_region}. Must be in body_region_reference.")
    reference_tbsa = row["reference_tbsa_percent"]
    is_critical_loc = bool(row["is_critical_priority"])
    suspected_inhalation = bool(smoke_exposure or respiratory_symptoms)
    clean_prop = max(1.0, min(100.0, float(affected_proportion_percent))) if (affected_proportion_percent and affected_proportion_percent > 0) else 1.0

    # Defensive validation: Valid degree REQUIRES non-null confidence score >= 0.70
    has_sufficient_conf = bool(confidence_score is not None and float(confidence_score) >= 0.70)
    is_valid_degree = bool(predicted_degree in ['degree_1', 'degree_2', 'degree_3'] and has_sufficient_conf)

    if is_valid_degree:
        clean_degree = predicted_degree
        clean_conf = float(confidence_score)
        clean_tbsa = calculate_tbsa(reference_tbsa, clean_prop, clean_degree)
        case_st = 'completed'
        area_st = 'completed'
        decision = 'accepted'
        rej_reason = None
        area_assessed_at = now_iso
        absi_sc, absi_cat, absi_not_app = calculate_absi(age, sex, clean_tbsa, suspected_inhalation, clean_degree == 'degree_3')
    else:
        # Defensive validation: When rejected or low confidence, total_tbsa_percent MUST BE NULL to satisfy CK_case_total_tbsa_status
        clean_degree = None
        clean_conf = None
        clean_tbsa = None
        case_st = 'inconclusive'
        area_st = 'inconclusive'
        decision = 'rejected'
        rej_reason = 'unassessed_or_low_quality' if predicted_degree is None else 'low_confidence_requires_medical_review'
        area_assessed_at = None
        absi_sc, absi_cat, absi_not_app = None, None, 'case_incomplete'

    # Triage and emergency calculation (passes real patient age and canonical region)
    risk_level, escalation, show_115, reasons = determine_escalation(
        clean_degree if is_valid_degree else 'unassessed',
        clean_tbsa if clean_tbsa is not None else 0.0,
        clean_cause,
        suspected_inhalation,
        canonical_region,
        age=age
    )

    with conn:
        # 1. Create parent burn_case
        cursor.execute("""
            INSERT INTO burn_case (
                account_id, age_at_assessment, sex_at_assessment, burn_cause,
                smoke_exposure_reported, respiratory_symptoms, suspected_inhalation_injury,
                total_tbsa_percent, absi_score, absi_risk_category, absi_not_applicable_reason,
                overall_risk_level, medical_escalation_level,
                show_emergency_call_prompt, emergency_reasons, case_status, finalized_at, created_at
            ) VALUES (?, ?, ?, 'thermal', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            account_id, age, sex,
            1 if smoke_exposure else 0,
            1 if respiratory_symptoms else 0,
            1 if suspected_inhalation else 0,
            clean_tbsa,
            absi_sc, absi_cat, absi_not_app,
            risk_level, escalation,
            1 if show_115 else 0,
            json.dumps({"reasons": reasons}) if reasons else None,
            case_st,
            now_iso if is_valid_degree else None,
            now_iso
        ))
        case_id = cursor.lastrowid

        # 2. Create child burn_area_assessment
        cursor.execute("""
            INSERT INTO burn_area_assessment (
                case_id, region_code, affected_proportion_percent,
                burn_image_path, burn_mask_path, image_retention_status,
                quality_status, quality_issues, quality_check_model_id,
                area_status, model_predicted_degree, model_confidence_score,
                classification_model_id, segmentation_model_id, threshold_used,
                prediction_decision, rejection_reason, predicted_degree, confidence_score,
                is_low_confidence, tbsa_percent, clinical_review_status, assessed_at
            ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, 3, ?, ?, ?, ?, ?, 0.70, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (
            case_id, canonical_region, clean_prop,
            burn_image_path, burn_mask_path,
            'acceptable' if is_valid_degree else 'rejected',
            '{"quality_passed": true}' if is_valid_degree else '{"quality_passed": false, "reason": "unassessed"}',
            area_st,
            clean_degree, clean_conf,
            classification_model_id, segmentation_model_id,
            decision, rej_reason,
            clean_degree, clean_conf,
            0 if is_valid_degree else 1,
            clean_tbsa,
            area_assessed_at
        ))
        area_id = cursor.lastrowid

        # 3. Create case referral for doctor review
        assigned_doctor = 4 if risk_level in ['low', 'moderate'] else 3
        cursor.execute("""
            INSERT INTO case_referral (case_id, assigned_to, sent_at, referral_status)
            VALUES (?, ?, ?, 'sent')
        """, (case_id, assigned_doctor, now_iso))

        # 4. Map relevant first aid guideline if degree is valid
        if is_valid_degree:
            cursor.execute("SELECT guideline_id FROM first_aid_guideline WHERE burn_degree = ? LIMIT 1", (clean_degree,))
            g_row = cursor.fetchone()
            if g_row:
                cursor.execute("""
                    INSERT OR IGNORE INTO assessment_guideline (area_assessment_id, guideline_id, display_order, assigned_at)
                    VALUES (?, ?, 1, ?)
                """, (area_id, g_row["guideline_id"], now_iso))

    conn.close()
    return {
        "assessment_id": area_id,
        "case_id": case_id,
        "region_code": canonical_region,
        "reference_tbsa": reference_tbsa,
        "predicted_degree": clean_degree,
        "tbsa_percent": clean_tbsa,
        "absi_score": absi_sc,
        "absi_risk_category": absi_cat,
        "overall_risk_level": risk_level,
        "medical_escalation_level": escalation,
        "show_emergency_call_prompt": show_115,
        "emergency_reasons": reasons,
        "case_status": case_st,
        "assessed_at": now_iso
    }

def get_assessment_history(account_id: Optional[int] = None, limit: int = 50, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves assessment history joined with body region reference strictly for thermal burns."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    query = """
        SELECT 
            b.assessment_id, b.account_id, b.age_at_assessment, b.sex_at_assessment,
            b.burn_cause, b.region_code, r.region_name, r.side, r.reference_tbsa_percent,
            b.affected_proportion_percent, b.tbsa_percent, b.predicted_degree,
            b.confidence_score, b.is_only_burn_area,
            b.overall_risk_level, b.medical_escalation_level, b.show_emergency_call_prompt,
            b.burn_image_path, b.burn_mask_path, b.assessed_at
        FROM burn_assessment b
        JOIN body_region_reference r ON b.region_code = r.region_code
        WHERE (? IS NULL OR b.account_id = ?)
        ORDER BY b.assessment_id DESC
        LIMIT ?
    """
    cursor.execute(query, (account_id, account_id, limit))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
