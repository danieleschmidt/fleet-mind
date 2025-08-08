"""Global compliance and regulatory support for Fleet-Mind."""

import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone


class ComplianceStandard(Enum):
    """Global compliance standards."""
    GDPR = "gdpr"           # General Data Protection Regulation (EU)
    CCPA = "ccpa"           # California Consumer Privacy Act (US)
    PDPA = "pdpa"           # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"           # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"       # Personal Information Protection Act (Canada)
    DPA = "dpa"             # Data Protection Act (UK)
    ITAR = "itar"           # International Traffic in Arms Regulations (US)
    EAR = "ear"             # Export Administration Regulations (US)
    FAA_PART_107 = "faa_part_107"  # FAA Drone Regulations (US)
    EASA = "easa"           # European Union Aviation Safety Agency
    CASA = "casa"           # Civil Aviation Safety Authority (Australia)
    TC = "tc"               # Transport Canada
    CAA_UK = "caa_uk"       # Civil Aviation Authority (UK)
    DGCA = "dgca"           # Directorate General of Civil Aviation (India)


@dataclass
class ComplianceRequirement:
    """A specific compliance requirement."""
    standard: ComplianceStandard
    requirement_id: str
    title: str
    description: str
    mandatory: bool = True
    category: str = "general"
    jurisdiction: str = "global"
    effective_date: Optional[datetime] = None
    compliance_actions: List[str] = field(default_factory=list)


@dataclass
class ComplianceAuditRecord:
    """Record of compliance audit."""
    audit_id: str
    timestamp: float
    standard: ComplianceStandard
    requirements_checked: List[str]
    compliance_status: Dict[str, bool]  # requirement_id -> compliant
    violations: List[str]
    recommendations: List[str]
    auditor: str = "system"
    notes: Optional[str] = None


@dataclass
class DataProcessingRecord:
    """Record of personal data processing for GDPR compliance."""
    record_id: str
    timestamp: float
    data_subject_id: str
    data_types: List[str]  # Types of personal data
    processing_purpose: str
    legal_basis: str  # GDPR Article 6 basis
    retention_period: Optional[int] = None  # days
    third_parties: List[str] = field(default_factory=list)
    consent_obtained: bool = False
    anonymized: bool = False


class ComplianceManager:
    """Global compliance and regulatory management system."""
    
    def __init__(self):
        """Initialize compliance manager."""
        # Compliance requirements database
        self.requirements: Dict[ComplianceStandard, List[ComplianceRequirement]] = {}
        
        # Audit history
        self.audit_history: List[ComplianceAuditRecord] = []
        
        # Data processing records (GDPR)
        self.data_processing_records: List[DataProcessingRecord] = []
        
        # Current compliance status
        self.compliance_status: Dict[ComplianceStandard, Dict[str, bool]] = {}
        
        # Regional configurations
        self.regional_configs: Dict[str, Dict[str, Any]] = {}
        
        # Initialize built-in requirements
        self._initialize_requirements()
        
        print("Compliance manager initialized with global regulatory support")
    
    def _initialize_requirements(self):
        """Initialize built-in compliance requirements."""
        
        # GDPR Requirements
        gdpr_requirements = [
            ComplianceRequirement(
                standard=ComplianceStandard.GDPR,
                requirement_id="gdpr_consent",
                title="Lawful Basis for Processing",
                description="Ensure lawful basis for processing personal data under GDPR Article 6",
                category="data_protection",
                jurisdiction="EU",
                compliance_actions=[
                    "Obtain explicit consent for data processing",
                    "Document legal basis for each processing activity",
                    "Implement consent withdrawal mechanisms"
                ]
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.GDPR,
                requirement_id="gdpr_data_minimization",
                title="Data Minimization Principle",
                description="Process only data necessary for specified purposes",
                category="data_protection",
                jurisdiction="EU",
                compliance_actions=[
                    "Implement data minimization by design",
                    "Regular review of data collection practices",
                    "Automated data purging for expired data"
                ]
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.GDPR,
                requirement_id="gdpr_right_to_erasure",
                title="Right to Erasure (Right to be Forgotten)",
                description="Enable data subjects to request deletion of personal data",
                category="data_subject_rights",
                jurisdiction="EU",
                compliance_actions=[
                    "Implement data deletion mechanisms",
                    "Process erasure requests within 30 days",
                    "Notify third parties of erasure requests"
                ]
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.GDPR,
                requirement_id="gdpr_data_portability",
                title="Right to Data Portability",
                description="Enable data subjects to receive their data in machine-readable format",
                category="data_subject_rights",
                jurisdiction="EU",
                compliance_actions=[
                    "Implement data export functionality",
                    "Support standard data formats (JSON, CSV)",
                    "Process portability requests within 30 days"
                ]
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.GDPR,
                requirement_id="gdpr_breach_notification",
                title="Data Breach Notification",
                description="Notify authorities of data breaches within 72 hours",
                category="incident_response",
                jurisdiction="EU",
                compliance_actions=[
                    "Implement breach detection systems",
                    "Automated notification to supervisory authorities",
                    "Maintain breach notification logs"
                ]
            )
        ]
        
        # CCPA Requirements
        ccpa_requirements = [
            ComplianceRequirement(
                standard=ComplianceStandard.CCPA,
                requirement_id="ccpa_disclosure",
                title="Privacy Policy Disclosure",
                description="Disclose categories of personal information collected and purposes",
                category="transparency",
                jurisdiction="California, US",
                compliance_actions=[
                    "Update privacy policy with CCPA disclosures",
                    "Implement 'Do Not Sell My Personal Information' link",
                    "Provide contact information for privacy requests"
                ]
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.CCPA,
                requirement_id="ccpa_right_to_know",
                title="Consumer Right to Know",
                description="Provide information about personal information collection and use",
                category="consumer_rights",
                jurisdiction="California, US",
                compliance_actions=[
                    "Implement data disclosure request handling",
                    "Provide information about data sources",
                    "Response to requests within 45 days"
                ]
            )
        ]
        
        # Aviation Compliance (FAA Part 107)
        faa_requirements = [
            ComplianceRequirement(
                standard=ComplianceStandard.FAA_PART_107,
                requirement_id="faa_altitude_limit",
                title="Maximum Altitude Restriction",
                description="Maintain drone flight altitude at or below 400 feet AGL",
                category="flight_operations",
                jurisdiction="United States",
                compliance_actions=[
                    "Implement altitude monitoring and enforcement",
                    "Configure maximum altitude limits in flight controller",
                    "Log altitude violations for review"
                ]
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.FAA_PART_107,
                requirement_id="faa_visual_line_of_sight",
                title="Visual Line of Sight Requirements",
                description="Maintain visual line of sight with drone during operation",
                category="flight_operations",
                jurisdiction="United States",
                compliance_actions=[
                    "Implement range limitation systems",
                    "Visual observer coordination protocols",
                    "Emergency return-to-home procedures"
                ]
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.FAA_PART_107,
                requirement_id="faa_no_fly_zones",
                title="Restricted Airspace Compliance",
                description="Avoid flight in restricted airspace without authorization",
                category="flight_operations",
                jurisdiction="United States",
                compliance_actions=[
                    "Implement geofencing for restricted areas",
                    "Real-time airspace status checking",
                    "LAANC authorization integration"
                ]
            )
        ]
        
        # Store requirements
        self.requirements = {
            ComplianceStandard.GDPR: gdpr_requirements,
            ComplianceStandard.CCPA: ccpa_requirements,
            ComplianceStandard.FAA_PART_107: faa_requirements,
        }
    
    def register_data_processing(
        self,
        data_subject_id: str,
        data_types: List[str],
        purpose: str,
        legal_basis: str,
        consent_obtained: bool = False,
        retention_days: Optional[int] = None
    ) -> str:
        """Register data processing activity for GDPR compliance.
        
        Args:
            data_subject_id: ID of data subject (user, drone operator, etc.)
            data_types: Types of personal data being processed
            purpose: Purpose of data processing
            legal_basis: Legal basis under GDPR Article 6
            consent_obtained: Whether explicit consent was obtained
            retention_days: Data retention period in days
            
        Returns:
            Record ID for the processing activity
        """
        record_id = f"dpr_{int(time.time())}_{len(self.data_processing_records):04d}"
        
        record = DataProcessingRecord(
            record_id=record_id,
            timestamp=time.time(),
            data_subject_id=data_subject_id,
            data_types=data_types,
            processing_purpose=purpose,
            legal_basis=legal_basis,
            retention_period=retention_days,
            consent_obtained=consent_obtained
        )
        
        self.data_processing_records.append(record)
        return record_id
    
    def handle_data_subject_request(
        self,
        request_type: str,
        data_subject_id: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle data subject rights requests.
        
        Args:
            request_type: Type of request (access, portability, erasure, etc.)
            data_subject_id: ID of data subject making request
            additional_info: Additional request information
            
        Returns:
            Response with request status and data
        """
        response = {
            "request_id": f"dsr_{int(time.time())}",
            "request_type": request_type,
            "data_subject_id": data_subject_id,
            "status": "processed",
            "timestamp": time.time()
        }
        
        # Find all data processing records for this subject
        subject_records = [
            record for record in self.data_processing_records
            if record.data_subject_id == data_subject_id
        ]
        
        if request_type == "access":
            # Right to access - provide information about processing
            response["data"] = {
                "processing_activities": len(subject_records),
                "data_categories": list(set(
                    dt for record in subject_records for dt in record.data_types
                )),
                "processing_purposes": [record.processing_purpose for record in subject_records],
                "legal_bases": [record.legal_basis for record in subject_records]
            }
        
        elif request_type == "portability":
            # Right to data portability - export data in machine-readable format
            export_data = []
            for record in subject_records:
                export_data.append({
                    "record_id": record.record_id,
                    "processing_date": record.timestamp,
                    "data_types": record.data_types,
                    "purpose": record.processing_purpose,
                    "legal_basis": record.legal_basis
                })
            
            response["data"] = export_data
            response["format"] = "JSON"
        
        elif request_type == "erasure":
            # Right to erasure - delete personal data
            deleted_records = []
            remaining_records = []
            
            for record in self.data_processing_records:
                if record.data_subject_id == data_subject_id:
                    deleted_records.append(record.record_id)
                else:
                    remaining_records.append(record)
            
            self.data_processing_records = remaining_records
            response["data"] = {
                "deleted_records": len(deleted_records),
                "record_ids": deleted_records
            }
        
        return response
    
    def check_compliance(
        self,
        standard: ComplianceStandard,
        system_config: Optional[Dict[str, Any]] = None
    ) -> ComplianceAuditRecord:
        """Perform compliance audit for specified standard.
        
        Args:
            standard: Compliance standard to audit
            system_config: Current system configuration
            
        Returns:
            Audit record with compliance status
        """
        audit_id = f"audit_{standard.value}_{int(time.time())}"
        requirements = self.requirements.get(standard, [])
        
        compliance_status = {}
        violations = []
        recommendations = []
        
        for requirement in requirements:
            # Simplified compliance checking logic
            # In production, this would integrate with actual system checks
            
            is_compliant = True
            
            if standard == ComplianceStandard.GDPR:
                if requirement.requirement_id == "gdpr_consent":
                    # Check if consent mechanism is implemented
                    is_compliant = system_config and system_config.get("consent_enabled", False)
                elif requirement.requirement_id == "gdpr_data_minimization":
                    # Check data minimization practices
                    is_compliant = len(self.data_processing_records) < 1000  # Simplified check
                elif requirement.requirement_id == "gdpr_breach_notification":
                    # Check breach notification system
                    is_compliant = system_config and system_config.get("breach_notification_enabled", False)
            
            elif standard == ComplianceStandard.FAA_PART_107:
                if requirement.requirement_id == "faa_altitude_limit":
                    # Check altitude limits
                    max_altitude = system_config.get("max_altitude", 0) if system_config else 0
                    is_compliant = max_altitude <= 121.92  # 400 feet in meters
                elif requirement.requirement_id == "faa_no_fly_zones":
                    # Check geofencing
                    is_compliant = system_config and system_config.get("geofencing_enabled", False)
            
            compliance_status[requirement.requirement_id] = is_compliant
            
            if not is_compliant:
                violations.append(f"{requirement.title}: {requirement.description}")
                recommendations.extend(requirement.compliance_actions)
        
        # Create audit record
        audit_record = ComplianceAuditRecord(
            audit_id=audit_id,
            timestamp=time.time(),
            standard=standard,
            requirements_checked=[req.requirement_id for req in requirements],
            compliance_status=compliance_status,
            violations=violations,
            recommendations=recommendations
        )
        
        self.audit_history.append(audit_record)
        return audit_record
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get overall compliance summary."""
        summary = {
            "standards_monitored": len(self.requirements),
            "total_requirements": sum(len(reqs) for reqs in self.requirements.values()),
            "data_processing_records": len(self.data_processing_records),
            "audit_history_count": len(self.audit_history),
            "compliance_by_standard": {}
        }
        
        # Calculate compliance rates by standard
        for standard in self.requirements.keys():
            if standard in [audit.standard for audit in self.audit_history]:
                # Get latest audit for this standard
                latest_audit = max(
                    [audit for audit in self.audit_history if audit.standard == standard],
                    key=lambda x: x.timestamp
                )
                
                total_requirements = len(latest_audit.compliance_status)
                compliant_requirements = sum(latest_audit.compliance_status.values())
                compliance_rate = compliant_requirements / total_requirements if total_requirements > 0 else 0
                
                summary["compliance_by_standard"][standard.value] = {
                    "compliance_rate": compliance_rate,
                    "violations_count": len(latest_audit.violations),
                    "last_audit": latest_audit.timestamp
                }
        
        return summary
    
    def export_compliance_report(self, standard: ComplianceStandard) -> Dict[str, Any]:
        """Export compliance report for specific standard.
        
        Args:
            standard: Standard to generate report for
            
        Returns:
            Comprehensive compliance report
        """
        requirements = self.requirements.get(standard, [])
        
        # Get latest audit
        standard_audits = [audit for audit in self.audit_history if audit.standard == standard]
        latest_audit = max(standard_audits, key=lambda x: x.timestamp) if standard_audits else None
        
        # Generate report
        report = {
            "standard": standard.value,
            "report_generated": time.time(),
            "requirements": [
                {
                    "id": req.requirement_id,
                    "title": req.title,
                    "description": req.description,
                    "mandatory": req.mandatory,
                    "category": req.category,
                    "jurisdiction": req.jurisdiction,
                    "compliance_actions": req.compliance_actions
                } for req in requirements
            ],
            "audit_summary": {
                "total_audits": len(standard_audits),
                "latest_audit_date": latest_audit.timestamp if latest_audit else None,
                "current_compliance_status": latest_audit.compliance_status if latest_audit else {},
                "active_violations": latest_audit.violations if latest_audit else [],
                "recommendations": latest_audit.recommendations if latest_audit else []
            }
        }
        
        # Add data processing information for GDPR
        if standard == ComplianceStandard.GDPR:
            report["data_processing"] = {
                "total_records": len(self.data_processing_records),
                "data_subjects": len(set(record.data_subject_id for record in self.data_processing_records)),
                "processing_purposes": list(set(record.processing_purpose for record in self.data_processing_records)),
                "legal_bases": list(set(record.legal_basis for record in self.data_processing_records))
            }
        
        return report
    
    def configure_regional_settings(
        self,
        region: str,
        settings: Dict[str, Any]
    ):
        """Configure region-specific compliance settings.
        
        Args:
            region: Region identifier (e.g., 'EU', 'US', 'APAC')
            settings: Regional configuration settings
        """
        self.regional_configs[region] = settings
    
    def get_applicable_standards(self, region: str) -> List[ComplianceStandard]:
        """Get applicable compliance standards for region.
        
        Args:
            region: Region identifier
            
        Returns:
            List of applicable standards
        """
        # Regional standard mapping
        regional_standards = {
            "EU": [ComplianceStandard.GDPR, ComplianceStandard.EASA],
            "US": [ComplianceStandard.CCPA, ComplianceStandard.ITAR, ComplianceStandard.EAR, ComplianceStandard.FAA_PART_107],
            "UK": [ComplianceStandard.DPA, ComplianceStandard.CAA_UK],
            "CA": [ComplianceStandard.PIPEDA, ComplianceStandard.TC],
            "AU": [ComplianceStandard.CASA],
            "BR": [ComplianceStandard.LGPD],
            "SG": [ComplianceStandard.PDPA],
        }
        
        return regional_standards.get(region, [])


# Global compliance manager
_global_compliance_manager: Optional[ComplianceManager] = None


def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager."""
    global _global_compliance_manager
    
    if _global_compliance_manager is None:
        _global_compliance_manager = ComplianceManager()
    
    return _global_compliance_manager