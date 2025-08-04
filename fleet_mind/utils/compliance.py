"""Compliance and data governance utilities for Fleet-Mind global deployment."""

import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta

from .logging import get_logger
from .i18n import Region, get_i18n_manager


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ProcessingLawfulBasis(Enum):
    """GDPR lawful basis for processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataSubjectRights(Enum):
    """Data subject rights under privacy regulations."""
    ACCESS = "access"
    RECTIFICATION = "rectification"
    ERASURE = "erasure"  # Right to be forgotten
    RESTRICTION = "restriction"
    PORTABILITY = "portability"
    OBJECTION = "objection"
    AUTOMATED_DECISION_MAKING = "automated_decision_making"


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    activity_id: str
    user_id: Optional[str]
    data_subject_id: Optional[str]
    processing_purpose: str
    data_categories: List[str]
    lawful_basis: ProcessingLawfulBasis
    retention_period: int  # days
    timestamp: float = field(default_factory=time.time)
    location: Optional[str] = None
    third_parties: List[str] = field(default_factory=list)
    automated_processing: bool = False
    special_categories: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            'activity_id': self.activity_id,
            'user_id': self.user_id,
            'data_subject_id': self.data_subject_id,
            'processing_purpose': self.processing_purpose,
            'data_categories': self.data_categories,
            'lawful_basis': self.lawful_basis.value,
            'retention_period': self.retention_period,
            'timestamp': self.timestamp,
            'location': self.location,
            'third_parties': self.third_parties,
            'automated_processing': self.automated_processing,
            'special_categories': self.special_categories,
        }


@dataclass
class ConsentRecord:
    """Record of user consent."""
    consent_id: str
    user_id: str
    purpose: str
    granted: bool
    timestamp: float = field(default_factory=time.time)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    version: str = "1.0"
    expires_at: Optional[float] = None
    withdrawn_at: Optional[float] = None
    
    def is_valid(self) -> bool:
        """Check if consent is still valid."""
        current_time = time.time()
        
        # Check if withdrawn
        if self.withdrawn_at and self.withdrawn_at <= current_time:
            return False
        
        # Check if expired
        if self.expires_at and self.expires_at <= current_time:
            return False
        
        return self.granted
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'consent_id': self.consent_id,
            'user_id': self.user_id,
            'purpose': self.purpose,
            'granted': self.granted,
            'timestamp': self.timestamp,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'version': self.version,
            'expires_at': self.expires_at,
            'withdrawn_at': self.withdrawn_at,
        }


@dataclass
class AuditLogEntry:
    """Audit log entry for compliance tracking."""
    event_id: str
    user_id: Optional[str]
    action: str
    resource: str
    result: str  # success, failure, error
    timestamp: float = field(default_factory=time.time)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"  # low, medium, high, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'user_id': self.user_id,
            'action': self.action,
            'resource': self.resource,
            'result': self.result,
            'timestamp': self.timestamp,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'details': self.details,
            'risk_level': self.risk_level,
        }


class ComplianceManager:
    """Comprehensive compliance and data governance manager."""
    
    def __init__(self, region: Optional[Region] = None, data_dir: Optional[Path] = None):
        """Initialize compliance manager.
        
        Args:
            region: Operating region for compliance rules
            data_dir: Directory for compliance data storage
        """
        self.region = region or Region.GLOBAL
        self.data_dir = data_dir or Path.home() / ".fleet-mind" / "compliance"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("compliance_manager", component="compliance")
        self.i18n = get_i18n_manager()
        
        # Storage for compliance records
        self.processing_records: List[DataProcessingRecord] = []
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.audit_log: List[AuditLogEntry] = []
        
        # Load compliance configuration
        self.compliance_config = self._load_compliance_config()
        
        # Initialize data retention policies
        self.retention_policies = self._initialize_retention_policies()
        
        self.logger.info(f"Compliance manager initialized for region {self.region.value}")

    def _load_compliance_config(self) -> Dict[str, Any]:
        """Load compliance configuration for the region."""
        config_file = self.data_dir / f"compliance_{self.region.value}.json"
        
        default_config = {
            "data_protection_officer": {
                "name": "Fleet-Mind DPO",
                "email": "dpo@fleet-mind.com",
                "phone": "+1-555-0123"
            },
            "privacy_policy_version": "1.0",
            "terms_of_service_version": "1.0",
            "cookie_policy_version": "1.0",
            "breach_notification": {
                "internal_notification_hours": 2,
                "authority_notification_hours": 72,
                "data_subject_notification_hours": 72,
                "severity_threshold": "medium"
            },
            "data_subject_request_response_days": 30,
            "consent_renewal_days": 365,
            "audit_log_retention_days": 2555,  # 7 years
            "processing_record_retention_days": 2555
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Failed to load compliance config: {e}")
        else:
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            self.logger.info(f"Created default compliance config at {config_file}")
        
        return default_config

    def _initialize_retention_policies(self) -> Dict[str, int]:
        """Initialize data retention policies by data type."""
        regional_requirements = self.i18n.get_compliance_requirements()
        
        base_policies = {
            "user_profiles": 2555,  # 7 years
            "mission_data": 1095,   # 3 years
            "telemetry_data": 365,  # 1 year
            "communication_logs": 180,  # 6 months
            "audit_logs": 2555,     # 7 years
            "consent_records": 2555, # 7 years
            "processing_records": 2555, # 7 years
            "security_logs": 1095,  # 3 years
            "performance_metrics": 90, # 3 months
            "error_logs": 365,      # 1 year
        }
        
        # Apply regional overrides
        region_policies = regional_requirements.get("data_retention_policies", {})
        base_policies.update(region_policies)
        
        return base_policies

    def record_processing_activity(
        self,
        activity_id: str,
        user_id: Optional[str],
        purpose: str,
        data_categories: List[str],
        lawful_basis: ProcessingLawfulBasis,
        **kwargs
    ) -> None:
        """Record a data processing activity.
        
        Args:
            activity_id: Unique identifier for the activity
            user_id: User performing the activity
            purpose: Purpose of processing
            data_categories: Categories of data being processed
            lawful_basis: Legal basis for processing
            **kwargs: Additional parameters
        """
        try:
            retention_period = kwargs.get('retention_period', 365)
            
            record = DataProcessingRecord(
                activity_id=activity_id,
                user_id=user_id,
                data_subject_id=kwargs.get('data_subject_id'),
                processing_purpose=purpose,
                data_categories=data_categories,
                lawful_basis=lawful_basis,
                retention_period=retention_period,
                location=kwargs.get('location'),
                third_parties=kwargs.get('third_parties', []),
                automated_processing=kwargs.get('automated_processing', False),
                special_categories=kwargs.get('special_categories', False),
            )
            
            self.processing_records.append(record)
            
            # Persist to storage
            self._persist_processing_record(record)
            
            self.logger.info(
                f"Processing activity recorded: {activity_id}",
                activity_id=activity_id,
                purpose=purpose,
                lawful_basis=lawful_basis.value,
            )
            
        except Exception as e:
            self.logger.error(f"Failed to record processing activity: {e}")

    def record_consent(
        self,
        user_id: str,
        purpose: str,
        granted: bool,
        **kwargs
    ) -> str:
        """Record user consent.
        
        Args:
            user_id: User identifier
            purpose: Purpose for which consent is given
            granted: Whether consent was granted
            **kwargs: Additional parameters
            
        Returns:
            Consent record ID
        """
        try:
            consent_id = hashlib.sha256(
                f"{user_id}_{purpose}_{time.time()}".encode()
            ).hexdigest()[:16]
            
            expires_in_days = kwargs.get('expires_in_days', 365)
            expires_at = time.time() + (expires_in_days * 24 * 3600) if expires_in_days else None
            
            consent = ConsentRecord(
                consent_id=consent_id,
                user_id=user_id,
                purpose=purpose,
                granted=granted,
                ip_address=kwargs.get('ip_address'),
                user_agent=kwargs.get('user_agent'),
                version=kwargs.get('version', '1.0'),
                expires_at=expires_at,
            )
            
            # Store consent (keyed by user_id + purpose for easy lookup)
            consent_key = f"{user_id}_{purpose}"
            self.consent_records[consent_key] = consent
            
            # Persist to storage
            self._persist_consent_record(consent)
            
            self.logger.info(
                f"Consent recorded for user {user_id}: {purpose} = {granted}",
                user_id=user_id,
                purpose=purpose,
                granted=granted,
                consent_id=consent_id,
            )
            
            return consent_id
            
        except Exception as e:
            self.logger.error(f"Failed to record consent: {e}")
            return ""

    def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has given valid consent for a purpose.
        
        Args:
            user_id: User identifier
            purpose: Purpose to check
            
        Returns:
            True if valid consent exists
        """
        try:
            consent_key = f"{user_id}_{purpose}"
            consent = self.consent_records.get(consent_key)
            
            if consent and consent.is_valid():
                return True
            
            # Check if consent needs renewal
            if consent and not consent.is_valid():
                self.logger.info(
                    f"Consent expired for user {user_id}: {purpose}",
                    user_id=user_id,
                    purpose=purpose,
                )
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check consent: {e}")
            return False

    def withdraw_consent(self, user_id: str, purpose: str) -> bool:
        """Withdraw user consent.
        
        Args:
            user_id: User identifier
            purpose: Purpose to withdraw consent for
            
        Returns:
            True if consent was withdrawn
        """
        try:
            consent_key = f"{user_id}_{purpose}"
            consent = self.consent_records.get(consent_key)
            
            if consent:
                consent.withdrawn_at = time.time()
                consent.granted = False
                
                # Persist change
                self._persist_consent_record(consent)
                
                self.logger.info(
                    f"Consent withdrawn for user {user_id}: {purpose}",
                    user_id=user_id,
                    purpose=purpose,
                )
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to withdraw consent: {e}")
            return False

    def log_audit_event(
        self,
        user_id: Optional[str],
        action: str,
        resource: str,
        result: str,
        **kwargs
    ) -> None:
        """Log an audit event.
        
        Args:
            user_id: User performing the action
            action: Action performed
            resource: Resource affected
            result: Result of the action
            **kwargs: Additional parameters
        """
        try:
            event_id = hashlib.sha256(
                f"{user_id}_{action}_{resource}_{time.time()}".encode()
            ).hexdigest()[:16]
            
            audit_entry = AuditLogEntry(
                event_id=event_id,
                user_id=user_id,
                action=action,
                resource=resource,
                result=result,
                ip_address=kwargs.get('ip_address'),
                user_agent=kwargs.get('user_agent'),
                details=kwargs.get('details', {}),
                risk_level=kwargs.get('risk_level', 'low'),
            )
            
            self.audit_log.append(audit_entry)
            
            # Persist to storage
            self._persist_audit_entry(audit_entry)
            
            # Log at appropriate level based on risk
            log_level = {
                'low': 'debug',
                'medium': 'info',
                'high': 'warning',
                'critical': 'error'
            }.get(audit_entry.risk_level, 'info')
            
            getattr(self.logger, log_level)(
                f"Audit: {action} on {resource} = {result}",
                user_id=user_id,
                action=action,
                resource=resource,
                result=result,
                event_id=event_id,
                risk_level=audit_entry.risk_level,
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")

    def handle_data_subject_request(
        self,
        user_id: str,
        request_type: DataSubjectRights,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle data subject rights request.
        
        Args:
            user_id: Data subject identifier
            request_type: Type of request
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        try:
            request_id = hashlib.sha256(
                f"{user_id}_{request_type.value}_{time.time()}".encode()
            ).hexdigest()[:16]
            
            self.log_audit_event(
                user_id=user_id,
                action=f"data_subject_request_{request_type.value}",
                resource="personal_data",
                result="initiated",
                details={"request_id": request_id},
                risk_level="medium"
            )
            
            response = {"request_id": request_id, "status": "processing"}
            
            if request_type == DataSubjectRights.ACCESS:
                # Export user data
                response["data"] = self._export_user_data(user_id)
                response["status"] = "completed"
                
            elif request_type == DataSubjectRights.ERASURE:
                # Delete user data (right to be forgotten)
                deleted_records = self._delete_user_data(user_id, kwargs.get('exceptions', []))
                response["deleted_records"] = deleted_records
                response["status"] = "completed"
                
            elif request_type == DataSubjectRights.PORTABILITY:
                # Export data in portable format
                response["export_url"] = self._create_data_export(user_id)
                response["status"] = "completed"
                
            elif request_type == DataSubjectRights.RECTIFICATION:
                # Update user data
                updates = kwargs.get('updates', {})
                updated_fields = self._update_user_data(user_id, updates)
                response["updated_fields"] = updated_fields
                response["status"] = "completed"
                
            else:
                response["status"] = "manual_review_required"
                response["message"] = f"Request type {request_type.value} requires manual review"
            
            self.log_audit_event(
                user_id=user_id,
                action=f"data_subject_request_{request_type.value}",
                resource="personal_data",
                result=response["status"],
                details={"request_id": request_id, "response": response},
                risk_level="medium"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to handle data subject request: {e}")
            return {"status": "error", "message": str(e)}

    def check_data_retention(self) -> Dict[str, Any]:
        """Check data retention policies and identify data for deletion.
        
        Returns:
            Summary of retention check
        """
        try:
            current_time = time.time()
            retention_summary = {
                "checked_at": current_time,
                "records_to_delete": [],
                "policies_applied": self.retention_policies,
                "total_records_checked": 0,
                "total_records_for_deletion": 0
            }
            
            # Check processing records
            for record in self.processing_records:
                if self._should_delete_record(record.timestamp, record.retention_period):
                    retention_summary["records_to_delete"].append({
                        "type": "processing_record",
                        "id": record.activity_id,
                        "age_days": (current_time - record.timestamp) / (24 * 3600)
                    })
            
            # Check audit log entries
            audit_retention_days = self.retention_policies.get("audit_logs", 2555)
            for entry in self.audit_log:
                if self._should_delete_record(entry.timestamp, audit_retention_days):
                    retention_summary["records_to_delete"].append({
                        "type": "audit_entry",
                        "id": entry.event_id,
                        "age_days": (current_time - entry.timestamp) / (24 * 3600)
                    })
            
            retention_summary["total_records_checked"] = len(self.processing_records) + len(self.audit_log)
            retention_summary["total_records_for_deletion"] = len(retention_summary["records_to_delete"])
            
            self.logger.info(
                f"Data retention check completed: {retention_summary['total_records_for_deletion']} records for deletion",
                records_checked=retention_summary["total_records_checked"],
                records_for_deletion=retention_summary["total_records_for_deletion"]
            )
            
            return retention_summary
            
        except Exception as e:
            self.logger.error(f"Data retention check failed: {e}")
            return {"status": "error", "message": str(e)}

    def _should_delete_record(self, timestamp: float, retention_days: int) -> bool:
        """Check if a record should be deleted based on retention policy."""
        age_seconds = time.time() - timestamp
        age_days = age_seconds / (24 * 3600)
        return age_days > retention_days

    def _export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data for a user (GDPR Article 15)."""
        # This is a simplified implementation
        # In production, this would query all systems
        user_data = {
            "user_id": user_id,
            "export_timestamp": time.time(),
            "processing_records": [
                record.to_dict() for record in self.processing_records
                if record.user_id == user_id or record.data_subject_id == user_id
            ],
            "consent_records": [
                consent.to_dict() for consent in self.consent_records.values()
                if consent.user_id == user_id
            ],
            "audit_log": [
                entry.to_dict() for entry in self.audit_log
                if entry.user_id == user_id
            ]
        }
        return user_data

    def _delete_user_data(self, user_id: str, exceptions: List[str]) -> int:
        """Delete user data (GDPR Article 17 - Right to erasure)."""
        deleted_count = 0
        
        # Delete processing records (unless in exceptions)
        if "processing_records" not in exceptions:
            initial_count = len(self.processing_records)
            self.processing_records = [
                record for record in self.processing_records
                if record.user_id != user_id and record.data_subject_id != user_id
            ]
            deleted_count += initial_count - len(self.processing_records)
        
        # Delete consent records (unless in exceptions)  
        if "consent_records" not in exceptions:
            initial_count = len(self.consent_records)
            self.consent_records = {
                key: consent for key, consent in self.consent_records.items()
                if consent.user_id != user_id
            }
            deleted_count += initial_count - len(self.consent_records)
        
        # Anonymize audit log entries (don't delete for compliance)
        for entry in self.audit_log:
            if entry.user_id == user_id:
                entry.user_id = "[deleted]"
        
        return deleted_count

    def _update_user_data(self, user_id: str, updates: Dict[str, Any]) -> List[str]:
        """Update user data (GDPR Article 16 - Right to rectification)."""
        updated_fields = []
        
        # Update consent records
        for consent in self.consent_records.values():
            if consent.user_id == user_id:
                for field, value in updates.items():
                    if hasattr(consent, field):
                        setattr(consent, field, value)
                        updated_fields.append(f"consent.{field}")
        
        return updated_fields

    def _create_data_export(self, user_id: str) -> str:
        """Create data export for portability (GDPR Article 20)."""
        # In production, this would create a secure download link
        export_id = hashlib.sha256(f"{user_id}_export_{time.time()}".encode()).hexdigest()[:16]
        export_url = f"/api/compliance/export/{export_id}"
        
        # Store export data temporarily
        export_data = self._export_user_data(user_id)
        # Store export_data with export_id for later retrieval
        
        return export_url

    def _persist_processing_record(self, record: DataProcessingRecord) -> None:
        """Persist processing record to storage."""
        # In production, this would write to secure database
        pass

    def _persist_consent_record(self, consent: ConsentRecord) -> None:
        """Persist consent record to storage."""
        # In production, this would write to secure database  
        pass

    def _persist_audit_entry(self, entry: AuditLogEntry) -> None:
        """Persist audit entry to storage."""
        # In production, this would write to secure audit database
        pass


# Global compliance manager instance
_global_compliance_manager = None


def get_compliance_manager(region: Optional[Region] = None) -> ComplianceManager:
    """Get global compliance manager instance."""
    global _global_compliance_manager
    if _global_compliance_manager is None:
        _global_compliance_manager = ComplianceManager(region)
    return _global_compliance_manager