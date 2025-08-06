"""Advanced security management for Fleet-Mind drone swarm coordination."""

import hashlib
import hmac
import secrets
import time
import json
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import base64

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


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_type: ThreatType
    severity: SecurityLevel
    source: str
    description: str
    action_taken: str
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


class SecurityManager:
    """Comprehensive security manager for Fleet-Mind operations.
    
    Provides encryption, authentication, authorization, threat detection,
    and security monitoring for drone swarm communications.
    """
    
    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.HIGH,
        key_rotation_interval: float = 3600.0,  # 1 hour
        enable_threat_detection: bool = True
    ):
        """Initialize security manager.
        
        Args:
            security_level: Default security level for operations
            key_rotation_interval: Key rotation interval in seconds
            enable_threat_detection: Enable real-time threat detection
        """
        self.security_level = security_level
        self.key_rotation_interval = key_rotation_interval
        self.enable_threat_detection = enable_threat_detection
        
        # Cryptographic components
        self.master_key = self._generate_master_key()
        self.session_keys: Dict[str, bytes] = {}
        self.drone_credentials: Dict[str, DroneCredentials] = {}
        
        # Security monitoring
        self.security_events: List[SecurityEvent] = []
        self.failed_attempts: Dict[str, List[float]] = {}  # source -> timestamps
        self.blocked_sources: Set[str] = set()
        self.last_key_rotation = time.time()
        
        # Message integrity
        self.message_hashes: Dict[str, str] = {}  # message_id -> hash
        self.nonce_cache: Set[str] = set()  # replay attack prevention
        
        # Threat detection patterns
        self.threat_patterns = {
            ThreatType.REPLAY_ATTACK: self._detect_replay_attack,
            ThreatType.DENIAL_OF_SERVICE: self._detect_dos_attack,
            ThreatType.COMMAND_INJECTION: self._detect_command_injection,
            ThreatType.DATA_TAMPERING: self._detect_data_tampering,
        }
        
        # Performance metrics
        self.encryption_times: List[float] = []
        self.decryption_times: List[float] = []
        
        print(f"Security manager initialized with {security_level.value} security level")

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
            "credentials_issued"
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

    def authorize_action(self, drone_id: str, action: str) -> bool:
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
            self.encryption_times.append(encryption_time)
            
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
            self.decryption_times.append(decryption_time)
            
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
                "avg_encryption_time_ms": sum(self.encryption_times[-100:]) / len(self.encryption_times[-100:]) if self.encryption_times else 0,
                "avg_decryption_time_ms": sum(self.decryption_times[-100:]) / len(self.decryption_times[-100:]) if self.decryption_times else 0,
                "total_operations": len(self.encryption_times) + len(self.decryption_times)
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

    def _record_failed_attempt(self, source: str) -> None:
        """Record failed authentication attempt."""
        current_time = time.time()
        
        if source not in self.failed_attempts:
            self.failed_attempts[source] = []
        
        self.failed_attempts[source].append(current_time)
        
        # Remove old attempts (older than 1 hour)
        self.failed_attempts[source] = [
            t for t in self.failed_attempts[source] 
            if current_time - t < 3600
        ]
        
        # Block source if too many failed attempts
        if len(self.failed_attempts[source]) >= 5:  # 5 attempts in 1 hour
            self.blocked_sources.add(source)
            self._log_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                SecurityLevel.HIGH,
                source,
                f"Source {source} blocked due to repeated failed attempts",
                "source_blocked"
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
        additional_data: Dict[str, Any] = None
    ) -> None:
        """Log security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            source=source,
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