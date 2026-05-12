"""
Neo4j AuraDB connection and query functions for PRO COA AI Agent
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv


class Neo4jConnection:
    """Manages connection to Neo4j AuraDB and provides query functions"""

    def __init__(self, uri, username, password):
        """Initialize the Neo4j driver"""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            print(f"✓ Connected to Neo4j at {uri}")
        except Exception as e:
            print(f"✗ Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            print("✓ Neo4j connection closed")

    def test_connection(self):
        """Test the connection and display node counts by type"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    RETURN labels(n) as label, count(n) as count
                """)

                print("\n=== Database Node Counts ===")
                records = [dict(record) for record in result]
                for record in records:
                    label = record['label'][0] if record['label'] else 'Unknown'
                    count = record['count']
                    print(f"{label}: {count}")
                print("============================\n")
                return records
        except Exception as e:
            print(f"✗ Connection test failed: {e}")
            return []

    def discover_schema(self):
        """Discover and display the database schema"""
        try:
            with self.driver.session() as session:
                result = session.run("CALL db.schema.visualization()")

                print("\n=== Database Schema ===")
                records = [dict(record) for record in result]
                for record in records:
                    print(record)
                print("=======================\n")
                return records
        except Exception as e:
            print(f"✗ Schema discovery failed: {e}")
            return []

    def get_instruments_by_indication(self, indications=None, phase="", endpoint=""):
        """
        Get instruments used in trials filtered by indication, phase, and endpoint

        Args:
            indications: List of disease indications (searches in trial population, disease area, classification)
                        Supports synonyms like ["Multiple Myeloma", "MM"]
            phase: Trial phase (e.g., "Phase 3")
            endpoint: Endpoint type (searches in primary and secondary endpoints)

        Returns:
            List of dictionaries containing instrument and trial information
        """
        if indications is None:
            indications = [""]

        query = """
        MATCH (ti:TrialInstrument)--(t:Trial)--(d:Drug)
        WHERE (size($indications) = 0 OR $indications[0] = "" OR
               ANY(ind IN $indications WHERE
                   ANY(pop IN t.trial_patient_population WHERE toLower(toString(pop)) CONTAINS toLower(ind))
                   OR toLower(toString(d.primary_disease_area)) CONTAINS toLower(ind)
                   OR toLower(ind) CONTAINS toLower(toString(d.primary_disease_area))
                   OR toLower(toString(d.disease_classification)) CONTAINS toLower(ind)
                   OR toLower(ind) CONTAINS toLower(toString(d.disease_classification))
               ))
        AND ($phase = "" OR
             (CASE WHEN t.trial_phase IS NOT NULL
                   THEN ANY(x IN CASE WHEN t.trial_phase IS NULL THEN []
                                      WHEN NOT t.trial_phase IS NULL AND NOT (t.trial_phase STARTS WITH '[')
                                      THEN [t.trial_phase]
                                      ELSE t.trial_phase END
                            WHERE toLower(toString(x)) CONTAINS toLower($phase))
                   ELSE false END))
        AND ($endpoint = "" OR
             (CASE WHEN t.trial_primary_endpoint IS NOT NULL
                   THEN ANY(x IN CASE WHEN t.trial_primary_endpoint IS NULL THEN []
                                      WHEN NOT t.trial_primary_endpoint IS NULL AND NOT (t.trial_primary_endpoint STARTS WITH '[')
                                      THEN [t.trial_primary_endpoint]
                                      ELSE t.trial_primary_endpoint END
                            WHERE toLower(toString(x)) CONTAINS toLower($endpoint))
                   ELSE false END) OR
             (CASE WHEN t.trial_secondary_endpoints IS NOT NULL
                   THEN ANY(x IN CASE WHEN t.trial_secondary_endpoints IS NULL THEN []
                                      WHEN NOT t.trial_secondary_endpoints IS NULL AND NOT (t.trial_secondary_endpoints STARTS WITH '[')
                                      THEN [t.trial_secondary_endpoints]
                                      ELSE t.trial_secondary_endpoints END
                            WHERE toLower(toString(x)) CONTAINS toLower($endpoint))
                   ELSE false END))
        AND t.pro_assessed_in_trial = "Yes"
        OPTIONAL MATCH (i:Instrument)
        WHERE toLower(i.short_name) = toLower(ti.instrument_name)
           OR toLower(i.full_name) CONTAINS toLower(ti.instrument_name)
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
        LIMIT 50
        """

        try:
            with self.driver.session() as session:
                result = session.run(query,
                                    indications=indications,
                                    phase=phase,
                                    endpoint=endpoint)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"✗ Query failed (get_instruments_by_indication): {e}")
            return []

    def get_regulatory_evidence(self, indications=None, agency=""):
        """
        Get regulatory review evidence filtered by indication and agency

        Args:
            indications: List of disease indications (searches in disease area and classification)
                        Supports synonyms like ["Multiple Myeloma", "MM"]
            agency: Regulatory agency (e.g., "FDA", "EMA")

        Returns:
            List of dictionaries containing regulatory review information
        """
        if indications is None:
            indications = [""]

        query = """
        MATCH (rr:RegulatoryReview)--(d:Drug)
        WHERE (size($indications) = 0 OR $indications[0] = "" OR
               ANY(ind IN $indications WHERE
                   toLower(toString(d.primary_disease_area)) CONTAINS toLower(ind)
                   OR toLower(ind) CONTAINS toLower(toString(d.primary_disease_area))
                   OR toLower(toString(d.disease_classification)) CONTAINS toLower(ind)
                   OR toLower(ind) CONTAINS toLower(toString(d.disease_classification))
                   OR (d.primary_disease_area IS NOT NULL AND size([
                         x IN (CASE
                           WHEN d.primary_disease_area IS NULL THEN []
                           WHEN d.primary_disease_area STARTS WITH '[' THEN d.primary_disease_area
                           ELSE [d.primary_disease_area]
                         END)
                         WHERE toLower(toString(x)) CONTAINS toLower(ind)
                            OR toLower(ind) CONTAINS toLower(toString(x))
                       ]) > 0)
               ))
        AND ($agency = "" OR
             (CASE WHEN rr.review_agency IS NOT NULL
                   THEN toLower(toString(rr.review_agency)) CONTAINS toLower($agency)
                   ELSE false END))
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
        LIMIT 20
        """

        try:
            with self.driver.session() as session:
                result = session.run(query,
                                    indications=indications,
                                    agency=agency)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"✗ Query failed (get_regulatory_evidence): {e}")
            return []

    def get_instrument_reference(self, instrument_name=""):
        """
        Get detailed reference information about instruments

        Args:
            instrument_name: Instrument name (searches in short name and full name)

        Returns:
            List of dictionaries containing instrument reference information
        """
        query = """
        MATCH (i:Instrument)
        WHERE $instrument_name = "" OR toLower(toString(i.short_name)) CONTAINS toLower($instrument_name)
           OR toLower(toString(i.full_name)) CONTAINS toLower($instrument_name)
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

        try:
            with self.driver.session() as session:
                result = session.run(query, instrument_name=instrument_name)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"✗ Query failed (get_instrument_reference): {e}")
            return []

    def get_regulatory_evidence_for_instrument(self, instrument_name=""):
        """Find regulatory reviews that mentioned a specific instrument."""
        if not instrument_name:
            return []
        query = """
        MATCH (rr:RegulatoryReview)--(d:Drug)
        WHERE toLower(toString(rr.pro_instruments_reviewed)) CONTAINS toLower($instrument_name)
           OR toLower(toString(rr.pro_instruments_accepted)) CONTAINS toLower($instrument_name)
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
        LIMIT 10
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, instrument_name=instrument_name)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"✗ Query failed (get_regulatory_evidence_for_instrument): {e}")
            return []

    def get_regulatory_rules(self, indication="", lifecycle_stage="", decision_type=""):
        """
        Get regulatory rules filtered by indication, lifecycle stage, and decision type

        Args:
            indication: Disease indication (searches in context and keywords)
            lifecycle_stage: Lifecycle stage (e.g., "label claim", "trial design")
            decision_type: Decision type (e.g., "approval", "rejection")

        Returns:
            List of dictionaries containing regulatory rule information
        """
        query = """
        MATCH (r:RegulatoryRule)
        WHERE ($indication = "" OR toLower(r.context) CONTAINS toLower($indication)
               OR toLower(r.keywords) CONTAINS toLower($indication))
        AND ($lifecycle_stage = "" OR toLower(r.lifecycle_stage) CONTAINS toLower($lifecycle_stage))
        AND ($decision_type = "" OR toLower(r.decision_type) CONTAINS toLower($decision_type))
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

        try:
            with self.driver.session() as session:
                result = session.run(query,
                                    indication=indication,
                                    lifecycle_stage=lifecycle_stage,
                                    decision_type=decision_type)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"✗ Query failed (get_regulatory_rules): {e}")
            return []


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Get Neo4j credentials from environment
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    # Validate credentials
    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        print("✗ Error: Missing Neo4j credentials in .env file")
        print("  Required: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD")
        exit(1)

    # Initialize connection
    print("Initializing Neo4j connection...")
    conn = Neo4jConnection(neo4j_uri, neo4j_username, neo4j_password)

    try:
        # Test the connection
        conn.test_connection()

        # Discover the schema
        conn.discover_schema()

        # Test query with Multiple Myeloma (full name)
        print("\n=== Testing Query with 'Multiple Myeloma' ===")
        instruments = conn.get_instruments_by_indication(indications=["Multiple Myeloma"])
        print(f"Retrieved {len(instruments)} trial instrument records")

        reviews = conn.get_regulatory_evidence(indications=["Multiple Myeloma"])
        print(f"Retrieved {len(reviews)} regulatory review records")

        # Test query with MM (acronym)
        print("\n=== Testing Query with 'MM' (acronym) ===")
        instruments_mm = conn.get_instruments_by_indication(indications=["MM"])
        print(f"Retrieved {len(instruments_mm)} trial instrument records")

        reviews_mm = conn.get_regulatory_evidence(indications=["MM"])
        print(f"Retrieved {len(reviews_mm)} regulatory review records")

        # Test query with both synonyms
        print("\n=== Testing Query with ['Multiple Myeloma', 'MM'] ===")
        instruments_both = conn.get_instruments_by_indication(indications=["Multiple Myeloma", "MM"])
        print(f"Retrieved {len(instruments_both)} trial instrument records")

        reviews_both = conn.get_regulatory_evidence(indications=["Multiple Myeloma", "MM"])
        print(f"Retrieved {len(reviews_both)} regulatory review records")

        refs = conn.get_instrument_reference()
        print(f"\nRetrieved {len(refs)} instrument reference records")
        print("=" * 50)

    finally:
        # Always close the connection
        conn.close()