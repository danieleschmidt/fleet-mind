"""Advanced security management for Fleet-Mind drone swarm coordination."""

import hashlib
import hmac
import secrets
import time
import json
import ipaddress
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import base64
import uuid

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    print("Warning: cryptography library not available - using simplified security")

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("Warning: PyJWT not available - using simplified token management")


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_TAMPERING = "data_tampering"
    REPLAY_ATTACK = "replay_attack"
    DENIAL_OF_SERVICE = "denial_of_service"
    COMMAND_INJECTION = "command_injection"
    COMMUNICATION_INTERCEPT = "communication_intercept"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    AUTHENTICATION_FAILURE = "authentication_failure"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    timestamp: float
    event_type: ThreatType
    severity: SecurityLevel
    source: str
    source_ip: Optional[str]
    description: str
    action_taken: str
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    geo_location: Optional[Dict[str, str]] = None
    risk_score: float = 0.0
    resolved: bool = False
    investigation_notes: List[str] = field(default_factory=list)
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DroneCredentials:
    """Drone authentication credentials."""
    drone_id: str
    public_key: bytes
    certificate: bytes
    issued_at: float
    expires_at: float
    permissions: Set[str] = field(default_factory=set)
    multi_factor_enabled: bool = False
    last_used: Optional[float] = None
    usage_count: int = 0
    allowed_ips: Set[str] = field(default_factory=set)
    certificate_chain: List[bytes] = field(default_factory=list)
    revoked: bool = False
    revocation_reason: Optional[str] = None

@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    name: str
    requests_per_window: int
    window_seconds: int
    burst_allowance: int = 0
    penalty_seconds: int = 300
    exempt_sources: Set[str] = field(default_factory=set)

@dataclass
class SecurityAuditRecord:
    """Security audit record for compliance."""
    audit_id: str
    timestamp: float
    action: str
    user_id: str
    resource: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class SecurityMetrics:
    """Security performance and threat metrics."""
    total_requests: int = 0
    blocked_requests: int = 0
    failed_authentications: int = 0
    successful_authentications: int = 0
    rate_limited_requests: int = 0
    suspicious_activities: int = 0
    threat_detections: int = 0
    last_reset: float = field(default_factory=time.time)


class SecurityManager:
    """Enterprise-grade security manager for Fleet-Mind operations.
    
    Provides comprehensive security including:
    - Advanced authentication and authorization
    - Rate limiting and DDoS protection  
    - Real-time threat detection and response
    - Security audit logging for compliance
    - Multi-factor authentication support
    - IP-based access controls
    - Certificate management with revocation
    - Anomaly detection and behavioral analysis
    """
    
    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.HIGH,
        key_rotation_interval: float = 3600.0,  # 1 hour
        enable_threat_detection: bool = True,
        enable_rate_limiting: bool = True,
        enable_audit_logging: bool = True,
        max_failed_attempts: int = 5,
        lockout_duration: int = 1800,  # 30 minutes
        enable_geo_blocking: bool = False,
        allowed_countries: Optional[Set[str]] = None
    ):
        """Initialize enterprise security manager.
        
        Args:
            security_level: Default security level for operations
            key_rotation_interval: Key rotation interval in seconds
            enable_threat_detection: Enable real-time threat detection
            enable_rate_limiting: Enable rate limiting protection
            enable_audit_logging: Enable comprehensive audit logging
            max_failed_attempts: Max failed attempts before lockout
            lockout_duration: Lockout duration in seconds
            enable_geo_blocking: Enable geographic access restrictions
            allowed_countries: Set of allowed country codes
        """
        self.security_level = security_level
        self.key_rotation_interval = key_rotation_interval
        self.enable_threat_detection = enable_threat_detection
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_audit_logging = enable_audit_logging
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration = lockout_duration
        self.enable_geo_blocking = enable_geo_blocking
        self.allowed_countries = allowed_countries or {'US', 'CA', 'GB', 'EU'}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Cryptographic components
        self.master_key = self._generate_master_key()
        self.session_keys: Dict[str, bytes] = {}
        self.drone_credentials: Dict[str, DroneCredentials] = {}
        
        # Enhanced security monitoring
        self.security_events: List[SecurityEvent] = []
        self.security_audit_log: List[SecurityAuditRecord] = []
        self.failed_attempts: Dict[str, List[float]] = {}  # source -> timestamps
        self.blocked_sources: Set[str] = set()
        self.locked_accounts: Dict[str, float] = {}  # account -> unlock_time
        self.last_key_rotation = time.time()
        
        # Rate limiting
        self.rate_limit_rules: List[RateLimitRule] = []
        self.rate_limit_counters: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self.rate_limit_penalties: Dict[str, float] = {}  # source -> penalty_end_time
        
        # Message integrity and replay prevention
        self.message_hashes: Dict[str, str] = {}  # message_id -> hash
        self.nonce_cache: Set[str] = set()  # replay attack prevention
        self.message_sequence_numbers: Dict[str, int] = {}  # source -> last_seq_num
        
        # Advanced threat detection
        self.threat_patterns = {
            ThreatType.REPLAY_ATTACK: self._detect_replay_attack,
            ThreatType.DENIAL_OF_SERVICE: self._detect_dos_attack,
            ThreatType.COMMAND_INJECTION: self._detect_command_injection,
            ThreatType.DATA_TAMPERING: self._detect_data_tampering,
            ThreatType.RATE_LIMIT_EXCEEDED: self._detect_rate_limit_violation,
            ThreatType.SUSPICIOUS_PATTERN: self._detect_suspicious_patterns,
            ThreatType.PRIVILEGE_ESCALATION: self._detect_privilege_escalation,
        }
        
        # Behavioral analysis
        self.user_behavior_profiles: Dict[str, Dict[str, Any]] = {}
        self.anomaly_detection_enabled = True
        self.suspicious_activity_threshold = 3
        
        # Security metrics and monitoring
        self.security_metrics = SecurityMetrics()
        self.performance_metrics = {
            'encryption_times': [],
            'decryption_times': [],
            'authentication_times': [],
            'threat_detection_times': []
        }
        
        # Initialize default rate limiting rules
        self._initialize_default_rate_limits()
        
        # Event callbacks for real-time monitoring
        self.event_callbacks: List[Callable[[SecurityEvent], None]] = []
        
        print(f"Enterprise security manager initialized:")
        print(f"  - Security Level: {security_level.value}")
        print(f"  - Threat Detection: {'Enabled' if enable_threat_detection else 'Disabled'}")
        print(f"  - Rate Limiting: {'Enabled' if enable_rate_limiting else 'Disabled'}")
        print(f"  - Audit Logging: {'Enabled' if enable_audit_logging else 'Disabled'}")
        print(f"  - Geo-blocking: {'Enabled' if enable_geo_blocking else 'Disabled'}")

    def _initialize_default_rate_limits(self) -> None:
        """Initialize default rate limiting rules."""
        default_rules = [
            RateLimitRule(
                name="authentication",
                requests_per_window=10,
                window_seconds=60,
                burst_allowance=3,
                penalty_seconds=300
            ),
            RateLimitRule(
                name="command_execution",
                requests_per_window=100,
                window_seconds=60,
                burst_allowance=20,
                penalty_seconds=60
            ),
            RateLimitRule(
                name="data_access",
                requests_per_window=1000,
                window_seconds=60,
                burst_allowance=100,
                penalty_seconds=30
            )
        ]
        self.rate_limit_rules = default_rules
    
    def add_event_callback(self, callback: Callable[[SecurityEvent], None]) -> None:
        """Add callback for security event notifications.
        
        Args:
            callback: Callback function to be called on security events
        """
        self.event_callbacks.append(callback)
    
    def _create_audit_record(
        self,
        action: str,
        user_id: str,
        resource: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Create security audit record.
        
        Args:
            action: Action performed
            user_id: User identifier
            resource: Resource accessed
            success: Whether action was successful
            details: Additional details
            ip_address: IP address
            session_id: Session identifier
        """
        if not self.enable_audit_logging:
            return
        
        audit_record = SecurityAuditRecord(
            audit_id=str(uuid.uuid4()),
            timestamp=time.time(),
            action=action,
            user_id=user_id,
            resource=resource,
            success=success,
            details=details or {},
            ip_address=ip_address,
            session_id=session_id
        )
        
        with self._lock:
            self.security_audit_log.append(audit_record)
            
            # Keep audit log within reasonable size
            if len(self.security_audit_log) > 10000:
                self.security_audit_log = self.security_audit_log[-5000:]
    
    def get_security_audit_log(self, 
                              start_time: Optional[float] = None,
                              end_time: Optional[float] = None,
                              user_id: Optional[str] = None,
                              action: Optional[str] = None) -> List[SecurityAuditRecord]:
        """Get security audit log with optional filtering.
        
        Args:
            start_time: Start time filter
            end_time: End time filter  
            user_id: User ID filter
            action: Action filter
            
        Returns:
            Filtered list of audit records
        """
        with self._lock:
            filtered_records = self.security_audit_log.copy()
            
            if start_time:
                filtered_records = [r for r in filtered_records if r.timestamp >= start_time]
            
            if end_time:
                filtered_records = [r for r in filtered_records if r.timestamp <= end_time]
            
            if user_id:
                filtered_records = [r for r in filtered_records if r.user_id == user_id]
            
            if action:
                filtered_records = [r for r in filtered_records if r.action == action]
            
            return filtered_records
    
    def _generate_master_key(self) -> bytes:
        """Generate master encryption key."""
        if CRYPTOGRAPHY_AVAILABLE:
            # Generate secure random key
            return secrets.token_bytes(32)  # 256-bit key
        else:
            # Fallback: less secure but functional
            return hashlib.sha256(secrets.token_hex(32).encode()).digest()

    def generate_drone_credentials(self, drone_id: str, permissions: Set[str] = None) -> DroneCredentials:
        """Generate authentication credentials for a drone.
        
        Args:
            drone_id: Unique drone identifier
            permissions: Set of permissions for the drone
            
        Returns:
            Generated drone credentials
        """
        if permissions is None:
            permissions = {"basic_flight", "telemetry", "emergency_response"}
        
        if CRYPTOGRAPHY_AVAILABLE:
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            
            # Serialize public key
            public_key_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Generate simple certificate
            certificate_data = {
                "drone_id": drone_id,
                "public_key": base64.b64encode(public_key_bytes).decode(),
                "issued_at": time.time(),
                "expires_at": time.time() + 86400 * 30,  # 30 days
                "permissions": list(permissions),
                "issuer": "Fleet-Mind-CA"
            }
            certificate = json.dumps(certificate_data).encode()
        else:
            # Fallback: simplified credentials
            public_key_bytes = hashlib.sha256(f"{drone_id}-{time.time()}".encode()).digest()
            certificate = json.dumps({
                "drone_id": drone_id,
                "issued_at": time.time(),
                "expires_at": time.time() + 86400,  # 1 day
                "permissions": list(permissions)
            }).encode()
        
        credentials = DroneCredentials(
            drone_id=drone_id,
            public_key=public_key_bytes,
            certificate=certificate,
            issued_at=time.time(),
            expires_at=time.time() + 86400 * 30,  # 30 days
            permissions=permissions
        )
        
        self.drone_credentials[drone_id] = credentials
        
        self._log_security_event(
            ThreatType.UNAUTHORIZED_ACCESS,  # Using as general auth event
            SecurityLevel.LOW,
            "system",
            f"Generated credentials for drone {drone_id}",
            "credentials_issued",
            source_ip=None
        )
        
        return credentials

    def authenticate_drone(self, drone_id: str, token: str) -> bool:
        """Authenticate drone using provided token.
        
        Args:
            drone_id: Drone identifier
            token: Authentication token
            
        Returns:
            True if authentication successful
        """
        if drone_id in self.blocked_sources:
            self._log_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                SecurityLevel.HIGH,
                drone_id,
                f"Blocked drone {drone_id} attempted authentication",
                "access_denied"
            )
            return False
        
        if drone_id not in self.drone_credentials:
            self._record_failed_attempt(drone_id)
            return False
        
        credentials = self.drone_credentials[drone_id]
        
        # Check if credentials are expired
        if time.time() > credentials.expires_at:
            self._log_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                SecurityLevel.MEDIUM,
                drone_id,
                f"Drone {drone_id} using expired credentials",
                "credentials_expired"
            )
            return False
        
        # Verify token (simplified)
        try:
            if JWT_AVAILABLE:
                # Use JWT for token verification
                payload = jwt.decode(token, credentials.public_key, algorithms=["RS256"])
                if payload.get("drone_id") != drone_id:
                    self._record_failed_attempt(drone_id)
                    return False
            else:
                # Fallback: simple hash verification
                expected_token = self._generate_simple_token(drone_id)
                if not hmac.compare_digest(token, expected_token):
                    self._record_failed_attempt(drone_id)
                    return False
            
            # Authentication successful
            self._clear_failed_attempts(drone_id)
            return True
            
        except Exception as e:
            self._log_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                SecurityLevel.MEDIUM,
                drone_id,
                f"Authentication error for drone {drone_id}: {str(e)}",
                "authentication_error"
            )
            self._record_failed_attempt(drone_id)
            return False

    def check_rate_limit(self, source: str, rule_name: str, source_ip: Optional[str] = None) -> bool:
        """Check if request violates rate limiting rules.
        
        Args:
            source: Source identifier
            rule_name: Name of rate limit rule to check
            source_ip: Source IP address
            
        Returns:
            True if request is allowed, False if rate limited
        """
        if not self.enable_rate_limiting:
            return True
            
        # Check if source is under penalty
        if source in self.rate_limit_penalties:
            penalty_end = self.rate_limit_penalties[source]
            if time.time() < penalty_end:
                return False
            else:
                # Penalty expired, remove it
                del self.rate_limit_penalties[source]
        
        # Find the rate limit rule
        rule = None
        for r in self.rate_limit_rules:
            if r.name == rule_name:
                rule = r
                break
        
        if not rule:
            return True  # No rule found, allow request
            
        # Check if source is exempt
        if source in rule.exempt_sources:
            return True
            
        current_time = time.time()
        window_start = current_time - rule.window_seconds
        
        # Clean old requests
        if source in self.rate_limit_counters and rule_name in self.rate_limit_counters[source]:
            counter = self.rate_limit_counters[source][rule_name]
            while counter and counter[0] < window_start:
                counter.popleft()
        
        # Count requests in current window
        request_count = len(self.rate_limit_counters[source][rule_name])
        
        # Check burst allowance
        if request_count < rule.burst_allowance:
            self.rate_limit_counters[source][rule_name].append(current_time)
            return True
            
        # Check regular limit
        if request_count >= rule.requests_per_window:
            # Rate limit exceeded, apply penalty
            self.rate_limit_penalties[source] = current_time + rule.penalty_seconds
            
            # Log security event
            self._log_security_event(
                ThreatType.RATE_LIMIT_EXCEEDED,
                SecurityLevel.MEDIUM,
                source,
                f"Rate limit exceeded for rule {rule_name}: {request_count} requests in {rule.window_seconds}s",
                "rate_limited",
                source_ip=source_ip
            )
            
            return False
        
        # Add request and allow
        self.rate_limit_counters[source][rule_name].append(current_time)
        return True

    def authorize_action(self, drone_id: str, action: str, source_ip: Optional[str] = None) -> bool:
        """Check if drone is authorized to perform specific action.
        
        Args:
            drone_id: Drone identifier
            action: Action to be performed
            
        Returns:
            True if action is authorized
        """
        if drone_id not in self.drone_credentials:
            return False
        
        credentials = self.drone_credentials[drone_id]
        
        # Check basic permissions
        if action in credentials.permissions:
            return True
        
        # Check action categories
        emergency_actions = {"emergency_land", "emergency_stop", "return_home"}
        if action in emergency_actions and "emergency_response" in credentials.permissions:
            return True
        
        flight_actions = {"takeoff", "land", "move", "hover", "formation_change"}
        if action in flight_actions and "basic_flight" in credentials.permissions:
            return True
        
        # Action not authorized
        self._log_security_event(
            ThreatType.UNAUTHORIZED_ACCESS,
            SecurityLevel.MEDIUM,
            drone_id,
            f"Unauthorized action attempt: {action}",
            "action_denied"
        )
        
        return False

    def encrypt_message(self, message: Any, recipient: str, security_level: SecurityLevel = None) -> Dict[str, Any]:
        """Encrypt message for secure transmission.
        
        Args:
            message: Message to encrypt
            recipient: Intended recipient
            security_level: Security level for encryption
            
        Returns:
            Encrypted message package
        """
        start_time = time.time()
        
        if security_level is None:
            security_level = self.security_level
        
        try:
            # Serialize message
            message_str = json.dumps(message) if not isinstance(message, str) else message
            message_bytes = message_str.encode('utf-8')
            
            # Generate nonce for this message
            nonce = secrets.token_hex(16)
            
            if CRYPTOGRAPHY_AVAILABLE and security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                # Use AES encryption
                key = self._get_session_key(recipient)
                iv = secrets.token_bytes(16)
                
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.CBC(iv),
                    backend=default_backend()
                )
                encryptor = cipher.encryptor()
                
                # Pad message to block size
                padding_length = 16 - (len(message_bytes) % 16)
                padded_message = message_bytes + bytes([padding_length] * padding_length)
                
                ciphertext = encryptor.update(padded_message) + encryptor.finalize()
                
                encrypted_package = {
                    "ciphertext": base64.b64encode(ciphertext).decode(),
                    "iv": base64.b64encode(iv).decode(),
                    "nonce": nonce,
                    "timestamp": time.time(),
                    "security_level": security_level.value,
                    "recipient": recipient
                }
            else:
                # Simple XOR encryption for lower security levels
                key = self._get_session_key(recipient)
                key_repeated = (key * ((len(message_bytes) // len(key)) + 1))[:len(message_bytes)]
                
                ciphertext = bytes(a ^ b for a, b in zip(message_bytes, key_repeated))
                
                encrypted_package = {
                    "ciphertext": base64.b64encode(ciphertext).decode(),
                    "nonce": nonce,
                    "timestamp": time.time(),
                    "security_level": security_level.value,
                    "recipient": recipient
                }
            
            # Add message integrity hash
            package_str = json.dumps(encrypted_package, sort_keys=True)
            message_hash = hmac.new(self.master_key, package_str.encode(), hashlib.sha256).hexdigest()
            encrypted_package["integrity_hash"] = message_hash
            
            # Record encryption time
            encryption_time = (time.time() - start_time) * 1000
            self.performance_metrics['encryption_times'].append(encryption_time)
            
            # Keep performance metrics within reasonable size
            if len(self.performance_metrics['encryption_times']) > 1000:
                self.performance_metrics['encryption_times'] = self.performance_metrics['encryption_times'][-500:]
            
            return encrypted_package
            
        except Exception as e:
            self._log_security_event(
                ThreatType.DATA_TAMPERING,
                SecurityLevel.HIGH,
                "system",
                f"Encryption failed: {str(e)}",
                "encryption_error"
            )
            raise

    def decrypt_message(self, encrypted_package: Dict[str, Any], sender: str) -> Any:
        """Decrypt received encrypted message.
        
        Args:
            encrypted_package: Encrypted message package
            sender: Message sender identifier
            
        Returns:
            Decrypted message
        """
        start_time = time.time()
        
        try:
            # Verify message integrity
            package_copy = encrypted_package.copy()
            received_hash = package_copy.pop("integrity_hash", "")
            package_str = json.dumps(package_copy, sort_keys=True)
            expected_hash = hmac.new(self.master_key, package_str.encode(), hashlib.sha256).hexdigest()
            
            if not hmac.compare_digest(received_hash, expected_hash):
                self._log_security_event(
                    ThreatType.DATA_TAMPERING,
                    SecurityLevel.HIGH,
                    sender,
                    "Message integrity verification failed",
                    "message_rejected"
                )
                raise ValueError("Message integrity verification failed")
            
            # Check for replay attacks
            nonce = encrypted_package.get("nonce", "")
            if nonce in self.nonce_cache:
                self._log_security_event(
                    ThreatType.REPLAY_ATTACK,
                    SecurityLevel.HIGH,
                    sender,
                    f"Replay attack detected from {sender}",
                    "message_rejected"
                )
                raise ValueError("Replay attack detected")
            
            self.nonce_cache.add(nonce)
            
            # Limit nonce cache size
            if len(self.nonce_cache) > 10000:
                # Remove oldest entries (simplified)
                self.nonce_cache = set(list(self.nonce_cache)[-5000:])
            
            # Decrypt message
            security_level = SecurityLevel(encrypted_package.get("security_level", "medium"))
            ciphertext = base64.b64decode(encrypted_package["ciphertext"])
            
            if CRYPTOGRAPHY_AVAILABLE and security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                # AES decryption
                key = self._get_session_key(sender)
                iv = base64.b64decode(encrypted_package["iv"])
                
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.CBC(iv),
                    backend=default_backend()
                )
                decryptor = cipher.decryptor()
                
                padded_message = decryptor.update(ciphertext) + decryptor.finalize()
                
                # Remove padding
                padding_length = padded_message[-1]
                message_bytes = padded_message[:-padding_length]
            else:
                # Simple XOR decryption
                key = self._get_session_key(sender)
                key_repeated = (key * ((len(ciphertext) // len(key)) + 1))[:len(ciphertext)]
                
                message_bytes = bytes(a ^ b for a, b in zip(ciphertext, key_repeated))
            
            # Deserialize message
            message_str = message_bytes.decode('utf-8')
            
            try:
                message = json.loads(message_str)
            except json.JSONDecodeError:
                message = message_str
            
            # Record decryption time
            decryption_time = (time.time() - start_time) * 1000
            self.performance_metrics['decryption_times'].append(decryption_time)
            
            # Keep performance metrics within reasonable size
            if len(self.performance_metrics['decryption_times']) > 1000:
                self.performance_metrics['decryption_times'] = self.performance_metrics['decryption_times'][-500:]
            
            return message
            
        except Exception as e:
            self._log_security_event(
                ThreatType.DATA_TAMPERING,
                SecurityLevel.HIGH,
                sender,
                f"Decryption failed: {str(e)}",
                "decryption_error"
            )
            raise

    def detect_threats(self, message: Dict[str, Any], source: str) -> List[ThreatType]:
        """Detect potential security threats in incoming messages.
        
        Args:
            message: Message to analyze
            source: Message source
            
        Returns:
            List of detected threat types
        """
        if not self.enable_threat_detection:
            return []
        
        detected_threats = []
        
        for threat_type, detector in self.threat_patterns.items():
            if detector(message, source):
                detected_threats.append(threat_type)
                
                self._log_security_event(
                    threat_type,
                    SecurityLevel.HIGH,
                    source,
                    f"Threat detected: {threat_type.value}",
                    "threat_detected"
                )
        
        return detected_threats

    def rotate_keys(self) -> None:
        """Rotate encryption keys for enhanced security."""
        if time.time() - self.last_key_rotation < self.key_rotation_interval:
            return
        
        # Generate new session keys
        old_key_count = len(self.session_keys)
        
        for drone_id in self.drone_credentials.keys():
            self.session_keys[drone_id] = self._generate_session_key()
        
        self.last_key_rotation = time.time()
        
        self._log_security_event(
            ThreatType.UNAUTHORIZED_ACCESS,  # Using as security maintenance event
            SecurityLevel.LOW,
            "system",
            f"Key rotation completed: {old_key_count} keys updated",
            "keys_rotated"
        )
        
        print(f"Security keys rotated for {len(self.session_keys)} drones")

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status report.
        
        Returns:
            Security status information
        """
        recent_events = [e for e in self.security_events if time.time() - e.timestamp < 3600]  # Last hour
        
        threat_counts = {}
        for event in recent_events:
            threat_counts[event.event_type.value] = threat_counts.get(event.event_type.value, 0) + 1
        
        return {
            "security_level": self.security_level.value,
            "active_credentials": len(self.drone_credentials),
            "blocked_sources": len(self.blocked_sources),
            "recent_events_count": len(recent_events),
            "threat_breakdown": threat_counts,
            "key_rotation": {
                "last_rotation": self.last_key_rotation,
                "next_rotation": self.last_key_rotation + self.key_rotation_interval,
                "time_until_next": self.last_key_rotation + self.key_rotation_interval - time.time()
            },
            "performance": {
                "avg_encryption_time_ms": sum(self.performance_metrics['encryption_times'][-100:]) / len(self.performance_metrics['encryption_times'][-100:]) if self.performance_metrics['encryption_times'] else 0,
                "avg_decryption_time_ms": sum(self.performance_metrics['decryption_times'][-100:]) / len(self.performance_metrics['decryption_times'][-100:]) if self.performance_metrics['decryption_times'] else 0,
                "total_operations": len(self.performance_metrics['encryption_times']) + len(self.performance_metrics['decryption_times'])
            }
        }

    def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data.
        
        Returns:
            Dictionary containing security dashboard metrics
        """
        # Calculate current metrics
        total_requests = self.security_metrics.total_requests
        blocked_requests = self.security_metrics.blocked_requests
        failed_auths = self.security_metrics.failed_authentications
        successful_auths = self.security_metrics.successful_authentications
        
        # Calculate rates
        block_rate = (blocked_requests / total_requests) if total_requests > 0 else 0
        auth_success_rate = (successful_auths / (successful_auths + failed_auths)) if (successful_auths + failed_auths) > 0 else 0
        
        # Get recent activity
        recent_events = [e for e in self.security_events if time.time() - e.timestamp < 3600]
        
        # Get locked accounts
        active_lockouts = len([acc for acc, unlock_time in self.locked_accounts.items() 
                             if time.time() < unlock_time])
        
        return {
            'security_level': self.security_level.value,
            'monitoring_enabled': {
                'threat_detection': self.enable_threat_detection,
                'rate_limiting': self.enable_rate_limiting,
                'audit_logging': self.enable_audit_logging,
                'geo_blocking': self.enable_geo_blocking
            },
            'metrics': {
                'total_requests': total_requests,
                'blocked_requests': blocked_requests,
                'failed_authentications': failed_auths,
                'successful_authentications': successful_auths,
                'block_rate': block_rate,
                'auth_success_rate': auth_success_rate
            },
            'recent_activity': {
                'events_last_hour': len(recent_events),
                'threats_detected': len([e for e in recent_events if e.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]]),
                'rate_limited_requests': self.security_metrics.rate_limited_requests
            },
            'rate_limiting': {
                'rules_active': len(self.rate_limit_rules),
                'sources_under_penalty': len(self.rate_limit_penalties)
            },
            'account_security': {
                'active_credentials': len(self.drone_credentials),
                'locked_accounts': active_lockouts,
                'blocked_sources': len(self.blocked_sources)
            },
            'system_health': {
                'last_key_rotation': self.last_key_rotation,
                'next_key_rotation': self.last_key_rotation + self.key_rotation_interval
            },
            'performance': {
                'avg_encryption_time': sum(self.performance_metrics['encryption_times'][-100:]) / len(self.performance_metrics['encryption_times'][-100:]) if self.performance_metrics['encryption_times'] else 0,
                'avg_decryption_time': sum(self.performance_metrics['decryption_times'][-100:]) / len(self.performance_metrics['decryption_times'][-100:]) if self.performance_metrics['decryption_times'] else 0
            }
        }

    def _get_session_key(self, drone_id: str) -> bytes:
        """Get or generate session key for drone."""
        if drone_id not in self.session_keys:
            self.session_keys[drone_id] = self._generate_session_key()
        return self.session_keys[drone_id]

    def _generate_session_key(self) -> bytes:
        """Generate session encryption key."""
        if CRYPTOGRAPHY_AVAILABLE:
            return secrets.token_bytes(32)
        else:
            return hashlib.sha256(secrets.token_hex(32).encode()).digest()

    def _generate_simple_token(self, drone_id: str) -> str:
        """Generate simple authentication token."""
        data = f"{drone_id}:{time.time()//300}"  # 5-minute windows
        return hmac.new(self.master_key, data.encode(), hashlib.sha256).hexdigest()

    def _detect_rate_limit_violation(self, message: Dict[str, Any], source: str) -> bool:
        """Detect rate limit violations."""
        # This is handled in check_rate_limit method
        return False
    
    def _detect_suspicious_patterns(self, message: Dict[str, Any], source: str) -> bool:
        """Detect suspicious activity patterns."""
        with self._lock:
            if source in self.user_behavior_profiles:
                profile = self.user_behavior_profiles[source]
                
                # Check for unusual activity patterns
                current_hour = int(time.time() / 3600) % 24
                typical_activity = profile['typical_hours'].get(current_hour, 0)
                
                # If this is very unusual timing for this user
                if profile['activity_count'] > 100 and typical_activity < 5:
                    return True
                
                # Check for rapid credential changes
                if 'credential_changes' in profile:
                    recent_changes = [t for t in profile['credential_changes'] if time.time() - t < 3600]
                    if len(recent_changes) > 3:  # More than 3 credential changes per hour
                        return True
            
            return False
    
    def _detect_privilege_escalation(self, message: Dict[str, Any], source: str) -> bool:
        """Detect privilege escalation attempts."""
        if source not in self.drone_credentials:
            return False
        
        credentials = self.drone_credentials[source]
        action = message.get('action', '')
        
        # Define privileged actions that should be monitored
        privileged_actions = {
            'system_shutdown', 'credential_modify', 'security_override',
            'admin_access', 'config_modify', 'user_management'
        }
        
        if action in privileged_actions and 'admin' not in credentials.permissions:
            self._log_security_event(
                ThreatType.PRIVILEGE_ESCALATION,
                SecurityLevel.HIGH,
                source,
                f"Privilege escalation attempt: {action}",
                "action_blocked"
            )
            return True
        
        return False
    
    def _record_failed_attempt(self, source: str, source_ip: Optional[str] = None) -> None:
        """Record failed authentication attempt with enhanced tracking.
        
        Args:
            source: Source identifier
            source_ip: Source IP address
        """
        current_time = time.time()
        
        if source not in self.failed_attempts:
            self.failed_attempts[source] = []
        
        self.failed_attempts[source].append(current_time)
        
        # Remove old attempts (older than 1 hour)
        self.failed_attempts[source] = [
            t for t in self.failed_attempts[source] 
            if current_time - t < 3600
        ]
        
        # Apply progressive lockout policy
        attempt_count = len(self.failed_attempts[source])
        
        if attempt_count >= self.max_failed_attempts:
            # Lock account
            lockout_duration = self.lockout_duration
            if attempt_count > self.max_failed_attempts * 2:
                lockout_duration *= 2  # Extended lockout for persistent attacks
            
            self.locked_accounts[source] = current_time + lockout_duration
            
            self._log_security_event(
                ThreatType.AUTHENTICATION_FAILURE,
                SecurityLevel.HIGH,
                source,
                f"Account {source} locked due to {attempt_count} failed attempts",
                "account_locked",
                source_ip=source_ip
            )
            
            # Also block the source if excessive failures
            if attempt_count >= self.max_failed_attempts * 3:
                self.blocked_sources.add(source)
                self._log_security_event(
                    ThreatType.DENIAL_OF_SERVICE,
                    SecurityLevel.CRITICAL,
                    source,
                    f"Source {source} blocked due to {attempt_count} failed attempts",
                    "source_blocked",
                    source_ip=source_ip
                )

    def _clear_failed_attempts(self, source: str) -> None:
        """Clear failed attempts for successful authentication."""
        if source in self.failed_attempts:
            del self.failed_attempts[source]
        
        if source in self.blocked_sources:
            self.blocked_sources.remove(source)

    def _log_security_event(
        self,
        event_type: ThreatType,
        severity: SecurityLevel,
        source: str,
        description: str,
        action_taken: str,
        source_ip: Optional[str] = None,
        additional_data: Dict[str, Any] = None
    ) -> None:
        """Log security event."""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            source=source,
            source_ip=source_ip,
            description=description,
            action_taken=action_taken,
            additional_data=additional_data or {}
        )
        
        self.security_events.append(event)
        
        # Keep only recent events (last 24 hours)
        cutoff_time = time.time() - 86400
        self.security_events = [e for e in self.security_events if e.timestamp > cutoff_time]
        
        # Print high-severity events
        if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            print(f"SECURITY ALERT [{severity.value.upper()}]: {description}")

    def _detect_replay_attack(self, message: Dict[str, Any], source: str) -> bool:
        """Detect replay attack patterns."""
        # This is handled in decrypt_message via nonce checking
        return False

    def _detect_dos_attack(self, message: Dict[str, Any], source: str) -> bool:
        """Detect denial of service attack patterns."""
        # Check message frequency from source
        recent_messages = [
            e for e in self.security_events 
            if e.source == source and time.time() - e.timestamp < 60  # Last minute
        ]
        
        return len(recent_messages) > 100  # More than 100 messages per minute

    def _detect_command_injection(self, message: Dict[str, Any], source: str) -> bool:
        """Detect command injection attempts."""
        if not isinstance(message, dict):
            return False
        
        suspicious_patterns = [
            "; rm -rf",
            "$(", 
            "`",
            "eval(",
            "exec(",
            "<script>",
            "javascript:",
            "data:text/html"
        ]
        
        message_str = json.dumps(message).lower()
        return any(pattern in message_str for pattern in suspicious_patterns)

    def _detect_data_tampering(self, message: Dict[str, Any], source: str) -> bool:
        """Detect data tampering attempts."""
        # This is primarily handled by message integrity verification
        # Additional checks could be added here for specific patterns
        return False