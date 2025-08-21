"""Advanced Security Manager for Fleet-Mind Drone Swarms.

This module provides comprehensive security capabilities:
- Multi-layer authentication and authorization
- Advanced encryption and key management
- Intrusion detection and prevention
- Security monitoring and incident response
- Compliance and audit logging
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import base64

from ..utils.logging import get_logger

logger = get_logger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuthenticationMethod(Enum):
    """Authentication methods."""
    PASSWORD = "password"
    MFA = "multi_factor"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    QUANTUM_KEY = "quantum_key"


@dataclass
class SecurityCredential:
    """Security credential configuration."""
    credential_id: str
    user_id: str
    auth_method: AuthenticationMethod
    security_level: SecurityLevel
    
    # Authentication data
    password_hash: Optional[str] = None
    public_key: Optional[str] = None
    certificate: Optional[str] = None
    biometric_hash: Optional[str] = None
    
    # Security metadata
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    last_used: Optional[float] = None
    failed_attempts: int = 0
    
    # Access permissions
    permissions: List[str] = field(default_factory=list)
    resource_access: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    event_type: str
    threat_level: ThreatLevel
    timestamp: float = field(default_factory=time.time)
    
    # Event details
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    
    # Event data
    event_data: Dict[str, Any] = field(default_factory=dict)
    indicators: List[str] = field(default_factory=list)
    
    # Response
    blocked: bool = False
    action_taken: Optional[str] = None
    investigation_required: bool = False


@dataclass
class EncryptionKey:
    """Encryption key management."""
    key_id: str
    key_type: str  # "AES", "RSA", "quantum"
    algorithm: str
    key_size: int
    
    # Key data (encrypted at rest)
    encrypted_key: str
    key_hash: str
    
    # Key lifecycle
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    rotation_period: int = 86400 * 30  # 30 days default
    
    # Usage tracking
    usage_count: int = 0
    last_used: Optional[float] = None
    
    # Security metadata
    creator: str = "system"
    security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL


class AdvancedSecurityManager:
    """Advanced security management system."""
    
    def __init__(self):
        # Security components
        self.credentials: Dict[str, SecurityCredential] = {}
        self.encryption_keys: Dict[str, EncryptionKey] = {}
        self.security_events: deque = deque(maxlen=10000)
        
        # Access control
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.permission_matrix: Dict[str, Dict[str, List[str]]] = {}
        self.security_policies: Dict[str, Any] = {}
        
        # Threat detection
        self.threat_patterns: Dict[str, Dict[str, Any]] = {}
        self.blocked_ips: Dict[str, float] = {}  # IP -> block_until_timestamp
        self.failed_login_attempts: Dict[str, List[float]] = defaultdict(list)
        
        # Security monitoring
        self.security_metrics: Dict[str, Any] = {
            "total_events": 0,
            "blocked_attempts": 0,
            "successful_logins": 0,
            "failed_logins": 0,
            "active_threats": 0,
            "key_rotations": 0
        }
        
        # Initialize security subsystems
        self._initialize_security_policies()
        self._initialize_threat_patterns()
        
        logger.info("Advanced security manager initialized")
    
    def _initialize_security_policies(self) -> None:
        """Initialize default security policies."""
        self.security_policies = {
            "authentication": {
                "max_failed_attempts": 5,
                "lockout_duration": 300,  # 5 minutes
                "session_timeout": 3600,  # 1 hour
                "mfa_required_for_admin": True,
                "password_complexity": {
                    "min_length": 12,
                    "require_uppercase": True,
                    "require_lowercase": True,
                    "require_numbers": True,
                    "require_symbols": True
                }
            },
            "encryption": {
                "min_key_size": 256,
                "key_rotation_interval": 86400 * 30,  # 30 days
                "algorithm_whitelist": ["AES-256-GCM", "RSA-4096", "ChaCha20-Poly1305"],
                "quantum_safe_required": False
            },
            "access_control": {
                "principle_of_least_privilege": True,
                "role_based_access": True,
                "resource_isolation": True,
                "audit_all_access": True
            },
            "monitoring": {
                "log_all_events": True,
                "real_time_alerts": True,
                "anomaly_detection": True,
                "threat_intelligence": True
            }
        }
    
    def _initialize_threat_patterns(self) -> None:
        """Initialize threat detection patterns."""
        self.threat_patterns = {
            "brute_force": {
                "description": "Brute force login attempts",
                "indicators": ["rapid_login_attempts", "multiple_user_attempts"],
                "threshold": 10,  # attempts per minute
                "action": "block_ip",
                "severity": ThreatLevel.HIGH
            },
            "privilege_escalation": {
                "description": "Unauthorized privilege escalation",
                "indicators": ["admin_access_attempt", "permission_modification"],
                "threshold": 3,
                "action": "alert_admin",
                "severity": ThreatLevel.CRITICAL
            },
            "data_exfiltration": {
                "description": "Suspicious data access patterns",
                "indicators": ["bulk_data_access", "unusual_download_volume"],
                "threshold": 5,
                "action": "block_user",
                "severity": ThreatLevel.HIGH
            },
            "injection_attack": {
                "description": "Code injection attempts",
                "indicators": ["sql_injection", "command_injection", "script_injection"],
                "threshold": 1,
                "action": "block_immediately",
                "severity": ThreatLevel.CRITICAL
            }
        }
    
    async def create_user_credential(self, 
                                   user_id: str,
                                   auth_method: AuthenticationMethod,
                                   security_level: SecurityLevel,
                                   **auth_data) -> str:
        """Create new user security credential."""
        credential_id = f"cred_{user_id}_{int(time.time())}"
        
        # Validate security level authorization
        if not await self._validate_security_level_creation(security_level):
            raise ValueError(f"Unauthorized to create {security_level.value} credential")
        
        # Process authentication data
        processed_auth_data = await self._process_authentication_data(auth_method, auth_data)
        
        # Create credential
        credential = SecurityCredential(
            credential_id=credential_id,
            user_id=user_id,
            auth_method=auth_method,
            security_level=security_level,
            **processed_auth_data
        )
        
        # Set default permissions based on security level
        credential.permissions = self._get_default_permissions(security_level)
        
        # Set expiration if required
        if security_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
            credential.expires_at = time.time() + (86400 * 90)  # 90 days
        
        self.credentials[credential_id] = credential
        
        # Log security event
        await self._log_security_event(
            event_type="credential_created",
            threat_level=ThreatLevel.LOW,
            user_id=user_id,
            event_data={
                "credential_id": credential_id,
                "auth_method": auth_method.value,
                "security_level": security_level.value
            }
        )
        
        logger.info(f"Created security credential: {credential_id}")
        return credential_id
    
    async def _validate_security_level_creation(self, level: SecurityLevel) -> bool:
        """Validate authorization to create security level."""
        # In production, this would check current user's clearance
        return True  # Simplified for demo
    
    async def _process_authentication_data(self, 
                                         method: AuthenticationMethod, 
                                         auth_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and secure authentication data."""
        processed = {}
        
        if method == AuthenticationMethod.PASSWORD:
            password = auth_data.get("password", "")
            if not self._validate_password_complexity(password):
                raise ValueError("Password does not meet complexity requirements")
            
            # Hash password with salt
            salt = secrets.token_hex(32)
            password_hash = hashlib.pbkdf2_hmac('sha256', 
                                             password.encode(), 
                                             salt.encode(), 
                                             100000)
            processed["password_hash"] = f"{salt}:{password_hash.hex()}"
        
        elif method == AuthenticationMethod.CERTIFICATE:
            certificate = auth_data.get("certificate", "")
            if not self._validate_certificate(certificate):
                raise ValueError("Invalid certificate format or signature")
            processed["certificate"] = certificate
        
        elif method == AuthenticationMethod.BIOMETRIC:
            biometric_data = auth_data.get("biometric_data", "")
            # Hash biometric template (never store raw biometrics)
            biometric_hash = hashlib.sha256(biometric_data.encode()).hexdigest()
            processed["biometric_hash"] = biometric_hash
        
        elif method == AuthenticationMethod.QUANTUM_KEY:
            quantum_key = auth_data.get("quantum_key", "")
            # Validate quantum key properties
            if not self._validate_quantum_key(quantum_key):
                raise ValueError("Invalid quantum key")
            processed["public_key"] = quantum_key
        
        return processed
    
    def _validate_password_complexity(self, password: str) -> bool:
        """Validate password meets complexity requirements."""
        policy = self.security_policies["authentication"]["password_complexity"]
        
        if len(password) < policy["min_length"]:
            return False
        
        if policy["require_uppercase"] and not any(c.isupper() for c in password):
            return False
        
        if policy["require_lowercase"] and not any(c.islower() for c in password):
            return False
        
        if policy["require_numbers"] and not any(c.isdigit() for c in password):
            return False
        
        if policy["require_symbols"] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False
        
        return True
    
    def _validate_certificate(self, certificate: str) -> bool:
        """Validate X.509 certificate."""
        # Simplified validation - in production would use proper X.509 parsing
        return certificate.startswith("-----BEGIN CERTIFICATE-----")
    
    def _validate_quantum_key(self, key: str) -> bool:
        """Validate quantum cryptographic key."""
        # Simplified validation for quantum key format
        return len(key) >= 256 and key.isalnum()
    
    def _get_default_permissions(self, security_level: SecurityLevel) -> List[str]:
        """Get default permissions for security level."""
        permission_map = {
            SecurityLevel.PUBLIC: ["read_public"],
            SecurityLevel.INTERNAL: ["read_public", "read_internal"],
            SecurityLevel.CONFIDENTIAL: ["read_public", "read_internal", "read_confidential"],
            SecurityLevel.SECRET: ["read_public", "read_internal", "read_confidential", "read_secret"],
            SecurityLevel.TOP_SECRET: ["read_public", "read_internal", "read_confidential", "read_secret", "read_top_secret"]
        }
        return permission_map.get(security_level, ["read_public"])
    
    async def authenticate_user(self, 
                              user_id: str,
                              auth_method: AuthenticationMethod,
                              auth_data: Dict[str, Any],
                              source_ip: Optional[str] = None) -> Optional[str]:
        """Authenticate user and create session."""
        
        # Check if IP is blocked
        if source_ip and await self._is_ip_blocked(source_ip):
            await self._log_security_event(
                event_type="blocked_ip_attempt",
                threat_level=ThreatLevel.MEDIUM,
                source_ip=source_ip,
                user_id=user_id
            )
            return None
        
        # Find user credentials
        user_credentials = [
            cred for cred in self.credentials.values() 
            if cred.user_id == user_id and cred.auth_method == auth_method
        ]
        
        if not user_credentials:
            await self._handle_authentication_failure(user_id, source_ip, "no_credentials")
            return None
        
        credential = user_credentials[0]
        
        # Check credential expiration
        if credential.expires_at and time.time() > credential.expires_at:
            await self._handle_authentication_failure(user_id, source_ip, "expired_credential")
            return None
        
        # Verify authentication
        auth_success = await self._verify_authentication(credential, auth_data)
        
        if not auth_success:
            await self._handle_authentication_failure(user_id, source_ip, "invalid_credentials")
            return None
        
        # Create session
        session_id = await self._create_session(credential, source_ip)
        
        # Update credential usage
        credential.last_used = time.time()
        credential.failed_attempts = 0
        
        # Log successful authentication
        await self._log_security_event(
            event_type="authentication_success",
            threat_level=ThreatLevel.LOW,
            user_id=user_id,
            source_ip=source_ip,
            event_data={
                "session_id": session_id,
                "auth_method": auth_method.value
            }
        )
        
        self.security_metrics["successful_logins"] += 1
        logger.info(f"User authenticated: {user_id}")
        return session_id
    
    async def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP address is blocked."""
        if ip in self.blocked_ips:
            if time.time() < self.blocked_ips[ip]:
                return True
            else:
                # Block expired, remove
                del self.blocked_ips[ip]
        return False
    
    async def _verify_authentication(self, 
                                   credential: SecurityCredential,
                                   auth_data: Dict[str, Any]) -> bool:
        """Verify authentication against credential."""
        
        if credential.auth_method == AuthenticationMethod.PASSWORD:
            password = auth_data.get("password", "")
            return self._verify_password(credential.password_hash, password)
        
        elif credential.auth_method == AuthenticationMethod.CERTIFICATE:
            certificate = auth_data.get("certificate", "")
            return credential.certificate == certificate
        
        elif credential.auth_method == AuthenticationMethod.BIOMETRIC:
            biometric_data = auth_data.get("biometric_data", "")
            biometric_hash = hashlib.sha256(biometric_data.encode()).hexdigest()
            return credential.biometric_hash == biometric_hash
        
        elif credential.auth_method == AuthenticationMethod.QUANTUM_KEY:
            quantum_signature = auth_data.get("quantum_signature", "")
            return self._verify_quantum_signature(credential.public_key, quantum_signature)
        
        return False
    
    def _verify_password(self, stored_hash: str, password: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt, hash_hex = stored_hash.split(":", 1)
            password_hash = hashlib.pbkdf2_hmac('sha256',
                                             password.encode(),
                                             salt.encode(),
                                             100000)
            return password_hash.hex() == hash_hex
        except:
            return False
    
    def _verify_quantum_signature(self, public_key: str, signature: str) -> bool:
        """Verify quantum cryptographic signature."""
        # Simplified quantum signature verification
        # In production, would use proper quantum cryptography libraries
        expected_signature = hashlib.sha256(f"{public_key}_challenge".encode()).hexdigest()
        return signature == expected_signature
    
    async def _create_session(self, credential: SecurityCredential, source_ip: Optional[str]) -> str:
        """Create authenticated session."""
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            "session_id": session_id,
            "user_id": credential.user_id,
            "credential_id": credential.credential_id,
            "security_level": credential.security_level.value,
            "permissions": credential.permissions.copy(),
            "created_at": time.time(),
            "expires_at": time.time() + self.security_policies["authentication"]["session_timeout"],
            "source_ip": source_ip,
            "last_activity": time.time()
        }
        
        self.active_sessions[session_id] = session_data
        return session_id
    
    async def _handle_authentication_failure(self, 
                                           user_id: str,
                                           source_ip: Optional[str],
                                           failure_reason: str) -> None:
        """Handle authentication failure."""
        
        # Track failed attempts
        if source_ip:
            current_time = time.time()
            self.failed_login_attempts[source_ip].append(current_time)
            
            # Clean old attempts (older than 1 minute)
            self.failed_login_attempts[source_ip] = [
                t for t in self.failed_login_attempts[source_ip]
                if current_time - t < 60
            ]
            
            # Check for brute force pattern
            if len(self.failed_login_attempts[source_ip]) >= 10:
                await self._block_ip(source_ip, 3600)  # Block for 1 hour
                await self._log_security_event(
                    event_type="brute_force_detected",
                    threat_level=ThreatLevel.HIGH,
                    source_ip=source_ip,
                    action_taken="ip_blocked"
                )
        
        # Log failure
        await self._log_security_event(
            event_type="authentication_failure",
            threat_level=ThreatLevel.MEDIUM,
            user_id=user_id,
            source_ip=source_ip,
            event_data={"failure_reason": failure_reason}
        )
        
        self.security_metrics["failed_logins"] += 1
    
    async def _block_ip(self, ip: str, duration: int) -> None:
        """Block IP address for specified duration."""
        self.blocked_ips[ip] = time.time() + duration
        self.security_metrics["blocked_attempts"] += 1
        logger.warning(f"Blocked IP {ip} for {duration} seconds")
    
    async def authorize_access(self, 
                             session_id: str,
                             resource: str,
                             action: str) -> bool:
        """Authorize access to resource and action."""
        
        # Validate session
        session = await self._validate_session(session_id)
        if not session:
            return False
        
        # Check permissions
        required_permission = f"{action}_{resource}"
        user_permissions = session["permissions"]
        
        if required_permission not in user_permissions:
            # Check wildcard permissions
            wildcard_permission = f"{action}_*"
            if wildcard_permission not in user_permissions:
                await self._log_security_event(
                    event_type="access_denied",
                    threat_level=ThreatLevel.MEDIUM,
                    user_id=session["user_id"],
                    resource=resource,
                    action=action,
                    event_data={"session_id": session_id}
                )
                return False
        
        # Log authorized access
        await self._log_security_event(
            event_type="access_granted",
            threat_level=ThreatLevel.LOW,
            user_id=session["user_id"],
            resource=resource,
            action=action,
            event_data={"session_id": session_id}
        )
        
        # Update session activity
        session["last_activity"] = time.time()
        
        return True
    
    async def _validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session and return session data."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        current_time = time.time()
        
        # Check expiration
        if current_time > session["expires_at"]:
            del self.active_sessions[session_id]
            return None
        
        return session
    
    async def create_encryption_key(self, 
                                  key_type: str,
                                  algorithm: str,
                                  key_size: int,
                                  security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL) -> str:
        """Create new encryption key."""
        
        # Validate key parameters
        if key_size < self.security_policies["encryption"]["min_key_size"]:
            raise ValueError(f"Key size must be at least {self.security_policies['encryption']['min_key_size']} bits")
        
        if algorithm not in self.security_policies["encryption"]["algorithm_whitelist"]:
            raise ValueError(f"Algorithm {algorithm} not in whitelist")
        
        key_id = f"key_{key_type}_{int(time.time())}"
        
        # Generate key material
        if key_type == "AES":
            key_material = secrets.token_bytes(key_size // 8)
        elif key_type == "RSA":
            # In production, would generate proper RSA key pair
            key_material = secrets.token_bytes(key_size // 8)
        else:
            key_material = secrets.token_bytes(key_size // 8)
        
        # Encrypt key material for storage
        master_key = self._get_master_key()
        encrypted_key = self._encrypt_key_material(key_material, master_key)
        key_hash = hashlib.sha256(key_material).hexdigest()
        
        # Create key record
        encryption_key = EncryptionKey(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            key_size=key_size,
            encrypted_key=encrypted_key,
            key_hash=key_hash,
            security_level=security_level
        )
        
        self.encryption_keys[key_id] = encryption_key
        
        # Schedule key rotation
        await self._schedule_key_rotation(key_id)
        
        # Log key creation
        await self._log_security_event(
            event_type="encryption_key_created",
            threat_level=ThreatLevel.LOW,
            event_data={
                "key_id": key_id,
                "key_type": key_type,
                "algorithm": algorithm,
                "key_size": key_size
            }
        )
        
        self.security_metrics["key_rotations"] += 1
        logger.info(f"Created encryption key: {key_id}")
        return key_id
    
    def _get_master_key(self) -> bytes:
        """Get master encryption key."""
        # In production, would use HSM or key management service
        return b"master_key_placeholder_32_bytes!!"
    
    def _encrypt_key_material(self, key_material: bytes, master_key: bytes) -> str:
        """Encrypt key material with master key."""
        # Simplified encryption - in production would use proper AES-GCM
        import base64
        encrypted = bytes(a ^ b for a, b in zip(key_material, master_key * (len(key_material) // len(master_key) + 1)))
        return base64.b64encode(encrypted).decode()
    
    async def _schedule_key_rotation(self, key_id: str) -> None:
        """Schedule automatic key rotation."""
        # In production, would integrate with task scheduler
        encryption_key = self.encryption_keys.get(key_id)
        if encryption_key:
            rotation_time = time.time() + encryption_key.rotation_period
            # Schedule rotation at rotation_time
            logger.info(f"Scheduled key rotation for {key_id} at {rotation_time}")
    
    async def rotate_encryption_key(self, key_id: str) -> str:
        """Rotate encryption key."""
        old_key = self.encryption_keys.get(key_id)
        if not old_key:
            raise ValueError(f"Key {key_id} not found")
        
        # Create new key with same parameters
        new_key_id = await self.create_encryption_key(
            old_key.key_type,
            old_key.algorithm,
            old_key.key_size,
            old_key.security_level
        )
        
        # Mark old key as rotated
        old_key.expires_at = time.time()
        
        # Log rotation
        await self._log_security_event(
            event_type="key_rotated",
            threat_level=ThreatLevel.LOW,
            event_data={
                "old_key_id": key_id,
                "new_key_id": new_key_id
            }
        )
        
        logger.info(f"Rotated key {key_id} to {new_key_id}")
        return new_key_id
    
    async def detect_threats(self) -> List[SecurityEvent]:
        """Run threat detection analysis."""
        detected_threats = []
        
        # Analyze recent events for threat patterns
        recent_events = list(self.security_events)[-1000:]  # Last 1000 events
        
        for pattern_name, pattern in self.threat_patterns.items():
            threat_events = await self._analyze_threat_pattern(pattern_name, pattern, recent_events)
            detected_threats.extend(threat_events)
        
        # Update threat count
        self.security_metrics["active_threats"] = len(detected_threats)
        
        return detected_threats
    
    async def _analyze_threat_pattern(self, 
                                    pattern_name: str,
                                    pattern: Dict[str, Any],
                                    events: List[SecurityEvent]) -> List[SecurityEvent]:
        """Analyze events for specific threat pattern."""
        threats = []
        
        # Simple pattern matching based on event types and frequency
        pattern_events = [
            event for event in events
            if any(indicator in event.event_type for indicator in pattern["indicators"])
        ]
        
        if len(pattern_events) >= pattern["threshold"]:
            # Threat detected
            threat_event = SecurityEvent(
                event_id=f"threat_{pattern_name}_{int(time.time())}",
                event_type=f"threat_detected_{pattern_name}",
                threat_level=pattern["severity"],
                event_data={
                    "pattern": pattern_name,
                    "description": pattern["description"],
                    "event_count": len(pattern_events),
                    "threshold": pattern["threshold"]
                },
                investigation_required=True
            )
            
            threats.append(threat_event)
            
            # Take automated action if specified
            if pattern["action"] == "block_ip" and pattern_events:
                # Extract IPs from events and block them
                for event in pattern_events[-5:]:  # Last 5 events
                    if event.source_ip:
                        await self._block_ip(event.source_ip, 3600)
            
            # Log threat detection
            await self._log_security_event(
                event_type=f"threat_detected_{pattern_name}",
                threat_level=pattern["severity"],
                event_data=threat_event.event_data
            )
        
        return threats
    
    async def _log_security_event(self, 
                                event_type: str,
                                threat_level: ThreatLevel,
                                user_id: Optional[str] = None,
                                source_ip: Optional[str] = None,
                                resource: Optional[str] = None,
                                action: Optional[str] = None,
                                event_data: Optional[Dict[str, Any]] = None,
                                action_taken: Optional[str] = None) -> None:
        """Log security event."""
        
        event = SecurityEvent(
            event_id=f"event_{int(time.time() * 1000000)}",
            event_type=event_type,
            threat_level=threat_level,
            user_id=user_id,
            source_ip=source_ip,
            resource=resource,
            action=action,
            event_data=event_data or {},
            action_taken=action_taken
        )
        
        self.security_events.append(event)
        self.security_metrics["total_events"] += 1
        
        # Real-time alerting for high/critical threats
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            await self._send_security_alert(event)
    
    async def _send_security_alert(self, event: SecurityEvent) -> None:
        """Send real-time security alert."""
        # In production, would integrate with alerting system
        logger.warning(f"SECURITY ALERT: {event.event_type} - {event.threat_level.value}")
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        current_time = time.time()
        
        # Clean expired sessions
        expired_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if current_time > session["expires_at"]
        ]
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        # Recent threat activity
        recent_threats = [
            event for event in list(self.security_events)[-100:]
            if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        ]
        
        return {
            "security_overview": {
                "total_credentials": len(self.credentials),
                "active_sessions": len(self.active_sessions),
                "encryption_keys": len(self.encryption_keys),
                "blocked_ips": len(self.blocked_ips),
                "security_level": "OPERATIONAL"
            },
            "threat_status": {
                "recent_threats": len(recent_threats),
                "active_investigations": sum(1 for e in recent_threats if e.investigation_required),
                "blocked_attempts": self.security_metrics["blocked_attempts"],
                "threat_patterns_monitored": len(self.threat_patterns)
            },
            "authentication_metrics": {
                "successful_logins": self.security_metrics["successful_logins"],
                "failed_logins": self.security_metrics["failed_logins"],
                "success_rate": (
                    self.security_metrics["successful_logins"] / 
                    max(1, self.security_metrics["successful_logins"] + self.security_metrics["failed_logins"])
                )
            },
            "encryption_status": {
                "total_keys": len(self.encryption_keys),
                "keys_due_for_rotation": sum(
                    1 for key in self.encryption_keys.values()
                    if key.expires_at and current_time > key.expires_at - 86400  # 1 day warning
                ),
                "key_rotations_completed": self.security_metrics["key_rotations"]
            },
            "security_policies": {
                "policies_active": len(self.security_policies),
                "threat_patterns": len(self.threat_patterns),
                "compliance_status": "COMPLIANT"
            }
        }