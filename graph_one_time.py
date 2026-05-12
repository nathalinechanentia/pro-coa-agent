"""
Neo4j AuraDB connection and query functions for PRO COA AI Agent
University of Cambridge × Evinova (AstraZeneca)
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv


class Neo4jConnection:
    """Manages connection to Neo4j AuraDB and provides query functions."""

    def __init__(self, uri, username, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
        except Exception as e:
            raise RuntimeError(f"Neo4j connection failed: {e}")

    def close(self):
        if self.driver:
            self.driver.close()
            print("✓ Neo4j connection closed")

    def run_query(self, query: str, params: dict = None) -> list:
        """Execute a Cypher query and return results as a list of dicts."""
        params = params or {}
        try:
            with self.driver.session() as session:
                result = session.run(query, **params)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Query failed: {e}")
            return []

    def test_connection(self) -> list:
        return self.run_query("MATCH (n) RETURN labels(n) as label, count(n) as count")

    # -------------------------------------------------------------------------
    # HELPER: match a scalar or array Neo4j field against a search term
    # This resolves the bug where primary_disease_area is stored as "MM" (scalar)
    # but some fields are stored as ["MM", "MDS"] (array).
    # -------------------------------------------------------------------------
    @staticmethod
    def _field_contains_cypher(field_name: str, param_name: str) -> str:
        """
        Returns Cypher expression that safely matches a scalar or array field
        against a parameter, bidirectionally (field contains param OR param contains field).
        """
        return f"""(
            toLower(toString({field_name})) CONTAINS toLower(${param_name})
            OR toLower(${param_name}) CONTAINS toLower(toString({field_name}))
            OR (
                {field_name} IS NOT NULL AND
                size([x IN (
                    CASE WHEN {field_name} STARTS WITH '['
                         THEN {field_name}
                         ELSE [{field_name}]
                    END
                ) WHERE
                    toLower(toString(x)) CONTAINS toLower(${param_name})
                    OR toLower(${param_name}) CONTAINS toLower(toString(x))
                ]) > 0
            )
        )"""

    # -------------------------------------------------------------------------
    # QUERY 1: Get trial instruments by indication
    # -------------------------------------------------------------------------
    # def get_instruments_by_indication(self, indications=None, phase="", endpoint=""):
    #     """
    #     Returns TrialInstrument records joined to Trial, Drug, and Instrument nodes.
    #     Searches across trial_patient_population (array), primary_disease_area (scalar or array),
    #     and disease_classification (scalar or array).
    #     Supports multiple synonyms via indications list.
    #     """
    #     if not indications:
    #         indications = [""]

    #     query = """
    #     UNWIND $indications AS ind
    #     MATCH (ti:TrialInstrument)--(t:Trial)--(d:Drug)
    #     WHERE (
    #         ind = ""
    #         OR ANY(pop IN CASE
    #                 WHEN t.trial_patient_population IS NULL THEN []
    #                 WHEN NOT t.trial_patient_population STARTS WITH '[' THEN [t.trial_patient_population]
    #                 ELSE t.trial_patient_population
    #               END
    #            WHERE toLower(toString(pop)) CONTAINS toLower(ind)
    #               OR toLower(ind) CONTAINS toLower(toString(pop)))
    #         OR toLower(toString(d.primary_disease_area)) CONTAINS toLower(ind)
    #         OR toLower(ind) CONTAINS toLower(toString(d.primary_disease_area))
    #         OR size([x IN CASE
    #                      WHEN d.primary_disease_area IS NULL THEN []
    #                      WHEN NOT d.primary_disease_area STARTS WITH '[' THEN [d.primary_disease_area]
    #                      ELSE d.primary_disease_area
    #                    END
    #                   WHERE toLower(toString(x)) CONTAINS toLower(ind)
    #                      OR toLower(ind) CONTAINS toLower(toString(x))
    #                 ]) > 0
    #         OR toLower(toString(d.disease_classification)) CONTAINS toLower(ind)
    #         OR toLower(ind) CONTAINS toLower(toString(d.disease_classification))
    #     )
    #     AND t.pro_assessed_in_trial = "Yes"
    #     AND ($phase = "" OR toLower(toString(t.trial_phase)) CONTAINS toLower($phase))
    #     AND ($endpoint = "" OR
    #          toLower(toString(t.trial_primary_endpoint)) CONTAINS toLower($endpoint) OR
    #          toLower(toString(t.trial_secondary_endpoints)) CONTAINS toLower($endpoint))
    #     OPTIONAL MATCH (i:Instrument)
    #     WHERE toLower(toString(i.short_name)) = toLower(ti.instrument_name)
    #        OR toLower(toString(i.full_name)) CONTAINS toLower(ti.instrument_name)
    #     RETURN DISTINCT
    #       ti.instrument_name AS instrument_name,
    #       ti.instrument_domain AS instrument_domain,
    #       ti.instrument_endpoint_role AS endpoint_role,
    #       ti.direction AS direction,
    #       ti.clinically_meaningful_mid_met AS mid_met,
    #       ti.pro_significance_category AS significance,
    #       ti.key_finding_instrument AS key_finding,
    #       ti.effect_size_instrument AS effect_size,
    #       ti.pro_p_value AS p_value,
    #       ti.subscale_results_instrument AS subscale_results,
    #       ti.instrument_subscales_assessed AS instrument_subscales_assessed,
    #       t.trial_id AS trial_id,
    #       t.pivotal_trial_name AS trial_name,
    #       t.trial_nct_id AS nct_id,
    #       t.trial_phase AS phase,
    #       t.trial_patient_population AS patient_population,
    #       t.pro_endpoint_position AS pro_position,
    #       t.pro_prespecified AS prespecified,
    #       t.pro_mid_defined AS mid_defined,
    #       t.pro_compliance_rate AS compliance_rate,
    #       t.pro_assessment_schedule AS assessment_schedule,
    #       t.primary_publication_doi AS publication_doi,
    #       t.primary_publication_year AS publication_year,
    #       d.generic_name AS drug_name,
    #       d.primary_disease_area AS disease_area,
    #       d.disease_classification AS disease_classification,
    #       d.drug_class AS drug_class_name,
    #       d.key_toxicities AS key_toxicities,
    #       d.fda_label_url AS fda_label_url,
    #       d.ema_label_url AS ema_label_url,
    #       i.total_items AS total_items,
    #       i.mcid AS mcid,
    #       i.regulatory_acceptance AS regulatory_acceptance,
    #       i.languages AS languages,
    #       i.mode_options AS mode_options,
    #       i.fda_core_alignment AS fda_alignment,
    #       i.trial_prevalence AS trial_prevalence,
    #       i.source_documents AS source_documents,
    #       i.validation_status AS validation_status,
    #       i.strengths AS strengths,
    #       i.limitations AS limitations,
    #       i.domains_measured AS domains_measured,
    #       i.developer AS developer
    #     ORDER BY
    #       CASE ti.pro_significance_category
    #         WHEN "Significant (p<0.001)" THEN 1
    #         WHEN "Significant (p<0.05)" THEN 2
    #         WHEN "Trend" THEN 3
    #         ELSE 4
    #       END,
    #       CASE t.pro_endpoint_position
    #         WHEN "Primary" THEN 1
    #         WHEN "Secondary" THEN 2
    #         WHEN "Exploratory" THEN 3
    #         ELSE 4
    #       END
    #     LIMIT 50
    #     """
    #     return self.run_query(query, {"indications": indications, "phase": phase, "endpoint": endpoint})

    def get_instruments_by_indication(self, indications=None, phase="", endpoint=""):
        """
        Returns TrialInstrument records joined to Trial, Drug, and Instrument nodes.
        Safely searches across trial_patient_population, primary_disease_area,
        and disease_classification by correctly handling both scalar strings and arrays.
        Supports multiple synonyms via indications list.
        """
        if not indications:
            indications = [""]

        query = """
        MATCH (ti:TrialInstrument)--(t:Trial)--(d:Drug)
        WHERE (size($indications) = 0 OR $indications[0] = "" OR
               ANY(ind IN $indications WHERE
                   ANY(pop IN apoc.convert.toList(t.trial_patient_population) WHERE pop IS NOT NULL AND toLower(toString(pop)) CONTAINS toLower(ind))
                   OR ANY(area IN apoc.convert.toList(d.primary_disease_area) WHERE area IS NOT NULL AND toLower(toString(area)) CONTAINS toLower(ind))
                   OR ANY(class IN apoc.convert.toList(d.disease_classification) WHERE class IS NOT NULL AND toLower(toString(class)) CONTAINS toLower(ind))
               ))
        AND ($phase = "" OR ANY(ph IN apoc.convert.toList(t.trial_phase) WHERE ph IS NOT NULL AND toLower(toString(ph)) CONTAINS toLower($phase)))
        AND ($endpoint = "" OR
             ANY(ep IN apoc.convert.toList(t.trial_primary_endpoint) WHERE ep IS NOT NULL AND toLower(toString(ep)) CONTAINS toLower($endpoint)) OR
             ANY(es IN apoc.convert.toList(t.trial_secondary_endpoints) WHERE es IS NOT NULL AND toLower(toString(es)) CONTAINS toLower($endpoint))
        )
        AND t.pro_assessed_in_trial = "Yes"
        OPTIONAL MATCH (i:Instrument)
        WHERE ANY(sn IN apoc.convert.toList(i.short_name) WHERE sn IS NOT NULL AND toLower(toString(sn)) = toLower(ti.instrument_name))
           OR ANY(fn IN apoc.convert.toList(i.full_name) WHERE fn IS NOT NULL AND toLower(toString(fn)) CONTAINS toLower(ti.instrument_name))
        RETURN
          ti.instrument_name AS instrument_name,
          ti.instrument_domain AS instrument_domain,
          ti.instrument_endpoint_role AS endpoint_role,
          ti.direction AS direction,
          ti.clinically_meaningful_mid_met AS mid_met,
          ti.pro_significance_category AS significance,
          ti.key_finding_instrument AS key_finding,
          ti.effect_size_instrument AS effect_size,
          ti.pro_p_value AS p_value,
          ti.subscale_results_instrument AS subscale_results,
          t.trial_id AS trial_id,
          t.pivotal_trial_name AS trial_name,
          t.trial_nct_id AS nct_id,
          t.trial_phase AS phase,
          t.trial_patient_population AS patient_population,
          t.pro_endpoint_position AS pro_position,
          t.pro_prespecified AS prespecified,
          t.pro_mid_defined AS mid_defined,
          t.pro_compliance_rate AS compliance_rate,
          t.pro_assessment_schedule AS assessment_schedule,
          t.primary_publication_doi AS publication_doi,
          t.primary_publication_year AS publication_year,
          d.generic_name AS drug_name,
          d.primary_disease_area AS disease_area,
          d.disease_classification AS disease_classification,
          d.drug_class AS drug_class_name,
          d.key_toxicities AS key_toxicities,
          d.fda_label_url AS fda_label_url,
          d.ema_label_url AS ema_label_url,
          i.total_items AS total_items,
          i.mcid AS mcid,
          i.regulatory_acceptance AS regulatory_acceptance,
          i.languages AS languages,
          i.mode_options AS mode_options,
          i.fda_core_alignment AS fda_alignment,
          i.trial_prevalence AS trial_prevalence,
          i.source_documents AS source_documents,
          i.validation_status AS validation_status,
          i.strengths AS strengths,
          i.limitations AS limitations
        ORDER BY
          CASE ti.pro_significance_category
            WHEN "Significant (p<0.001)" THEN 1
            WHEN "Significant (p<0.05)" THEN 2
            WHEN "Trend" THEN 3
            ELSE 4
          END,
          CASE t.pro_endpoint_position
            WHEN "Primary" THEN 1
            WHEN "Secondary" THEN 2
            WHEN "Exploratory" THEN 3
            ELSE 4
          END
        LIMIT 100
        """
        return self.run_query(query, {"indications": indications, "phase": phase, "endpoint": endpoint})
    
    # -------------------------------------------------------------------------
    # QUERY 2: Get regulatory evidence by indication
    # KEY FIX: Uses bidirectional scalar/array matching + supports synonym list
    # -------------------------------------------------------------------------
    # def get_regulatory_evidence(self, indications=None, agency=""):
    #     """
    #     Returns RegulatoryReview records joined to Drug nodes.
    #     Searches primary_disease_area and disease_classification with bidirectional
    #     scalar/array matching. Accepts multiple synonym terms.
    #     """
    #     if not indications:
    #         indications = [""]

    #     query = """
    #     UNWIND $indications AS ind
    #     MATCH (rr:RegulatoryReview)--(d:Drug)
    #     WHERE (
    #         ind = ""
    #         OR toLower(toString(d.primary_disease_area)) CONTAINS toLower(ind)
    #         OR toLower(ind) CONTAINS toLower(toString(d.primary_disease_area))
    #         OR size([x IN CASE
    #                      WHEN d.primary_disease_area IS NULL THEN []
    #                      WHEN NOT d.primary_disease_area STARTS WITH '[' THEN [d.primary_disease_area]
    #                      ELSE d.primary_disease_area
    #                    END
    #                   WHERE toLower(toString(x)) CONTAINS toLower(ind)
    #                      OR toLower(ind) CONTAINS toLower(toString(x))
    #                 ]) > 0
    #         OR toLower(toString(d.disease_classification)) CONTAINS toLower(ind)
    #         OR toLower(ind) CONTAINS toLower(toString(d.disease_classification))
    #         OR size([x IN CASE
    #                      WHEN d.disease_classification IS NULL THEN []
    #                      WHEN NOT d.disease_classification STARTS WITH '[' THEN [d.disease_classification]
    #                      ELSE d.disease_classification
    #                    END
    #                   WHERE toLower(toString(x)) CONTAINS toLower(ind)
    #                      OR toLower(ind) CONTAINS toLower(toString(x))
    #                 ]) > 0
    #     )
    #     AND ($agency = "" OR toLower(toString(rr.review_agency)) CONTAINS toLower($agency))
    #     RETURN DISTINCT
    #       rr.review_id AS review_id,
    #       rr.review_agency AS agency,
    #       rr.pro_decision AS decision,
    #       rr.pro_instruments_reviewed AS instruments_reviewed,
    #       rr.pro_instruments_accepted AS instruments_accepted,
    #       rr.pro_claim_type_approved AS claim_type,
    #       rr.pro_approval_reason AS approval_reason,
    #       rr.pro_rejection_reason_primary AS rejection_reason_primary,
    #       rr.pro_rejection_reason_detailed AS rejection_reason_detailed,
    #       rr.pro_missing_data_issue AS missing_data_issue,
    #       rr.pro_alpha_controlled AS alpha_controlled,
    #       rr.pro_prespecified AS prespecified,
    #       rr.pro_label_language_final AS label_language,
    #       rr.pro_label_location AS label_location,
    #       rr.review_date AS review_date,
    #       d.generic_name AS drug_name,
    #       d.primary_disease_area AS disease_area,
    #       d.indication_stage_setting_fda AS indication_fda,
    #       d.fda_label_url AS fda_label_url,
    #       d.ema_label_url AS ema_label_url
    #     ORDER BY rr.review_date DESC
    #     LIMIT 30
    #     """
    #     return self.run_query(query, {"indications": indications, "agency": agency})

    def get_regulatory_evidence(self, indications=None, agency=""):
        """
        Returns RegulatoryReview records joined to Drug nodes.
        Searches primary_disease_area and disease_classification safely handling both strings and arrays.
        """
        if not indications:
            indications = [""]

        query = """
        MATCH (rr:RegulatoryReview)--(d:Drug)
        WHERE (size($indications) = 0 OR $indications[0] = "" OR
               ANY(ind IN $indications WHERE
                   ANY(area IN apoc.convert.toList(d.primary_disease_area) WHERE area IS NOT NULL AND toLower(toString(area)) CONTAINS toLower(ind))
                   OR ANY(class IN apoc.convert.toList(d.disease_classification) WHERE class IS NOT NULL AND toLower(toString(class)) CONTAINS toLower(ind))
               ))
        AND ($agency = "" OR ANY(ag IN apoc.convert.toList(rr.review_agency) WHERE ag IS NOT NULL AND toLower(toString(ag)) CONTAINS toLower($agency)))
        RETURN
          rr.review_id AS review_id,
          rr.review_agency AS agency,
          rr.pro_decision AS decision,
          rr.pro_instruments_reviewed AS instruments_reviewed,
          rr.pro_instruments_accepted AS instruments_accepted,
          rr.pro_claim_type_approved AS claim_type,
          rr.pro_approval_reason AS approval_reason,
          rr.pro_rejection_reason_primary AS rejection_reason_primary,
          rr.pro_rejection_reason_detailed AS rejection_reason_detailed,
          rr.pro_missing_data_issue AS missing_data_issue,
          rr.pro_alpha_controlled AS alpha_controlled,
          rr.pro_prespecified AS prespecified,
          rr.pro_label_language_final AS label_language,
          rr.pro_label_location AS label_location,
          rr.review_date AS review_date,
          d.generic_name AS drug_name,
          d.primary_disease_area AS disease_area,
          d.indication_stage_setting_fda AS indication_fda,
          d.fda_label_url AS fda_label_url,
          d.ema_label_url AS ema_label_url
        ORDER BY rr.review_date DESC
        LIMIT 100
        """
        return self.run_query(query, {"indications": indications, "agency": agency})
    
    # -------------------------------------------------------------------------
    # QUERY 3: Get regulatory reviews mentioning a specific instrument
    # -------------------------------------------------------------------------
    def get_regulatory_evidence_for_instrument(self, instrument_name=""):
        """
        Find regulatory reviews that mention a specific instrument by name.
        Used to build per-instrument regulatory precedent.
        """
        if not instrument_name:
            return []
        query = """
        MATCH (rr:RegulatoryReview)--(d:Drug)
        WHERE ANY(rev IN apoc.convert.toList(rr.pro_instruments_reviewed) WHERE rev IS NOT NULL AND toLower(toString(rev)) CONTAINS toLower($instrument_name))
           OR ANY(acc IN apoc.convert.toList(rr.pro_instruments_accepted) WHERE acc IS NOT NULL AND toLower(toString(acc)) CONTAINS toLower($instrument_name))
        RETURN
          rr.review_id AS review_id,
          rr.review_agency AS agency,
          rr.pro_decision AS decision,
          rr.pro_instruments_reviewed AS instruments_reviewed,
          rr.pro_instruments_accepted AS instruments_accepted,
          rr.pro_claim_type_approved AS claim_type,
          rr.pro_approval_reason AS approval_reason,
          rr.pro_rejection_reason_primary AS rejection_reason_primary,
          rr.pro_rejection_reason_detailed AS rejection_reason_detailed,
          rr.pro_label_language_final AS label_language,
          rr.review_date AS review_date,
          d.generic_name AS drug_name,
          d.primary_disease_area AS disease_area,
          d.fda_label_url AS fda_label_url,
          d.ema_label_url AS ema_label_url
        ORDER BY rr.review_date DESC
        LIMIT 60
        """
        return self.run_query(query, {"instrument_name": instrument_name})

    # -------------------------------------------------------------------------
    # QUERY 4: Get instrument reference data
    # -------------------------------------------------------------------------
    def get_instrument_reference(self, instrument_name=""):
        query = """
        MATCH (i:Instrument)
        WHERE $instrument_name = ""
           OR ANY(sn IN apoc.convert.toList(i.short_name) WHERE sn IS NOT NULL AND toLower(toString(sn)) CONTAINS toLower($instrument_name))
           OR ANY(fn IN apoc.convert.toList(i.full_name) WHERE fn IS NOT NULL AND toLower(toString(fn)) CONTAINS toLower($instrument_name))
        RETURN
          i.short_name AS short_name,
          i.full_name AS full_name,
          i.instrument_type AS type,
          i.domains_measured AS domains,
          i.validation_status AS validation,
          i.mcid AS mcid,
          i.trial_prevalence AS trial_prevalence,
          i.regulatory_acceptance AS regulatory_acceptance,
          i.strengths AS strengths,
          i.limitations AS limitations,
          i.fda_core_alignment AS fda_alignment,
          i.total_items AS total_items,
          i.admin_time AS admin_time
        LIMIT 20
        """
        return self.run_query(query, {"instrument_name": instrument_name})

    # -------------------------------------------------------------------------
    # QUERY 5: Get regulatory rules
    # KEY FIX: lifecycle_stage left empty by default — actual values are
    # "Instrument_Selection", "Protocol_Design", etc., NOT "label claim"
    # -------------------------------------------------------------------------
    def get_regulatory_rules(self, indication="", lifecycle_stage="", decision_type=""):
        query = """
        MATCH (r:RegulatoryRule)
        WHERE ($indication = "" OR
               toLower(toString(r.context)) CONTAINS toLower($indication) OR
               toLower(toString(r.keywords)) CONTAINS toLower($indication))
        AND ($lifecycle_stage = "" OR
             toLower(toString(r.lifecycle_stage)) CONTAINS toLower($lifecycle_stage))
        AND ($decision_type = "" OR
             toLower(toString(r.decision_type)) CONTAINS toLower($decision_type))
        RETURN
          r.rule_id AS rule_id,
          r.source_document AS source_document,
          r.section AS section,
          r.lifecycle_stage AS lifecycle_stage,
          r.decision_type AS decision_type,
          r.stakeholder AS stakeholder,
          r.context AS context,
          r.rule_text AS rule_text,
          r.keywords AS keywords
        ORDER BY r.source_document, r.section
        LIMIT 20
        """
        return self.run_query(query, {
            "indication": indication,
            "lifecycle_stage": lifecycle_stage,
            "decision_type": decision_type
        })


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
    if not all([uri, user, pwd]):
        print("Missing NEO4J credentials in .env")
        exit(1)

    conn = Neo4jConnection(uri, user, pwd)
    try:
        counts = conn.test_connection()
        print("Node counts:", counts)

        print("\n--- Testing instruments: Multiple Myeloma ---")
        instr = conn.get_instruments_by_indication(indications=["Multiple Myeloma", "MM"])
        print(f"Instruments: {len(instr)}")
        if instr:
            print("First:", instr[0].get("instrument_name"), "|", instr[0].get("drug_name"))

        print("\n--- Testing regulatory reviews: MM ---")
        revs = conn.get_regulatory_evidence(indications=["Multiple Myeloma", "MM"])
        print(f"Reviews: {len(revs)}")
        if revs:
            print("First:", revs[0].get("drug_name"), "|", revs[0].get("agency"), "|", revs[0].get("decision"))

        print("\n--- Testing regulatory rules ---")
        rules = conn.get_regulatory_rules(indication="")
        print(f"Rules: {len(rules)}")
        if rules:
            print("First lifecycle_stage:", rules[0].get("lifecycle_stage"))

        print("\n--- Testing instrument precedent: FACT-P ---")
        prec = conn.get_regulatory_evidence_for_instrument("FACT-P")
        print(f"FACT-P precedents: {len(prec)}")

    finally:
        conn.close()