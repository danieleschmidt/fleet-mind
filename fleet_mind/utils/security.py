"""Comprehensive security system for Fleet-Mind."""

import hashlib
import hmac
import secrets
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Cryptography and JWT imports with fallback handling
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    # Fallback implementations for when cryptography is not available
    class Fernet:
        @staticmethod
        def generate_key(): return b'fallback_key_32_bytes_long_____'
        def __init__(self, key): self.key = key
        def encrypt(self, data): return b'encrypted_' + data
        def decrypt(self, data): return data[10:] if data.startswith(b'encrypted_') else data
    
    class MockRSAKey:
        def sign(self, *args, **kwargs): return b'mock_signature'
        def verify(self, *args, **kwargs): pass
        def public_key(self): return self
        def private_bytes(self, *args, **kwargs): return b'mock_private_key'
        def public_bytes(self, *args, **kwargs): return b'mock_public_key'
    
    class rsa:
        @staticmethod
        def generate_private_key(*args, **kwargs): return MockRSAKey()
    
    class hashes:
        class SHA256: pass
    
    class padding:
        class PSS:
            def __init__(self, *args, **kwargs): pass
            MAX_LENGTH = 0
        class MGF1:
            def __init__(self, *args, **kwargs): pass
    
    class serialization:
        class Encoding:
            PEM = 'PEM'
        class PrivateFormat:
            PKCS8 = 'PKCS8'
        class PublicFormat:
            SubjectPublicKeyInfo = 'SubjectPublicKeyInfo'
        class NoEncryption: pass
    
    def load_pem_private_key(data, password=None): return MockRSAKey()
    def load_pem_public_key(data): return MockRSAKey()
    
    CRYPTOGRAPHY_AVAILABLE = False
    print("Warning: cryptography not available, using mock security functions")

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    # Fallback JWT implementation
    class jwt:
        @staticmethod
        def encode(payload, secret, algorithm):
            return f"mock_jwt_token_{payload.get('user_id', 'unknown')}"
        
        @staticmethod
        def decode(token, secret, algorithms):
            if token.startswith('mock_jwt_token_'):
                user_id = token.replace('mock_jwt_token_', '')
                return {'user_id': user_id, 'exp': time.time() + 3600}
            raise Exception("Invalid token")
        
        class ExpiredSignatureError(Exception): pass
        class InvalidTokenError(Exception): pass
    
    JWT_AVAILABLE = False
    print("Warning: PyJWT not available, using mock JWT implementation")

from .logging import get_logger


class SecurityLevel(Enum):
    """Security level classifications."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """Threat level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: Optional[str] = None
    role: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ThreatDetection:
    """Threat detection result."""
    threat_type: str
    level: ThreatLevel
    description: str
    source: str
    timestamp: float = field(default_factory=time.time)
    mitigation: Optional[str] = None
    blocked: bool = False


class SecurityManager:
    """Comprehensive security manager for Fleet-Mind operations."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize security manager.
        
        Args:
            config_dir: Directory for security configuration and keys
        """
        self.config_dir = config_dir or Path.home() / ".fleet-mind" / "security"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("security_manager", component="security")
        
        # Initialize encryption keys
        self._symmetric_key = self._load_or_generate_symmetric_key()
        self._private_key, self._public_key = self._load_or_generate_key_pair()
        
        # JWT settings
        self._jwt_secret = self._load_or_generate_jwt_secret()
        self._jwt_algorithm = "HS256"
        self._jwt_expiry = 3600  # 1 hour
        
        # Security policies
        self.policies = self._load_security_policies()
        
        # Threat detection
        self.threat_patterns = self._initialize_threat_patterns()
        self.failed_attempts = {}  # Track failed authentication attempts
        self.rate_limits = {}  # Track rate limiting
        
        # Active sessions
        self.active_sessions = {}
        
        self.logger.info("Security manager initialized")

    def _load_or_generate_symmetric_key(self) -> bytes:
        """Load or generate symmetric encryption key."""
        key_file = self.config_dir / "symmetric.key"
        
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                self.logger.warning(f"Could not load symmetric key: {e}")
        
        # Generate new key
        key = Fernet.generate_key()
        try:
            with open(key_file, 'wb') as f:
                f.write(key)
            key_file.chmod(0o600)  # Restrict permissions
            self.logger.info("Generated new symmetric encryption key")
        except Exception as e:
            self.logger.error(f"Could not save symmetric key: {e}")
        
        return key

    def _load_or_generate_key_pair(self) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """Load or generate RSA key pair."""
        private_key_file = self.config_dir / "private.pem"
        public_key_file = self.config_dir / "public.pem"
        
        if private_key_file.exists() and public_key_file.exists():
            try:
                with open(private_key_file, 'rb') as f:
                    private_key = load_pem_private_key(f.read(), password=None)
                
                with open(public_key_file, 'rb') as f:
                    public_key = load_pem_public_key(f.read())
                
                return private_key, public_key
                
            except Exception as e:
                self.logger.warning(f"Could not load key pair: {e}")
        
        # Generate new key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()
        
        try:
            # Save private key
            with open(private_key_file, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            private_key_file.chmod(0o600)
            
            # Save public key
            with open(public_key_file, 'wb') as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
            
            self.logger.info("Generated new RSA key pair")
            
        except Exception as e:
            self.logger.error(f"Could not save key pair: {e}")
        
        return private_key, public_key

    def _load_or_generate_jwt_secret(self) -> str:
        """Load or generate JWT secret."""
        secret_file = self.config_dir / "jwt.secret"
        
        if secret_file.exists():
            try:
                with open(secret_file, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                self.logger.warning(f"Could not load JWT secret: {e}")
        
        # Generate new secret
        secret = secrets.token_urlsafe(64)
        try:
            with open(secret_file, 'w') as f:
                f.write(secret)
            secret_file.chmod(0o600)
            self.logger.info("Generated new JWT secret")
        except Exception as e:
            self.logger.error(f"Could not save JWT secret: {e}")
        
        return secret

    def _load_security_policies(self) -> Dict[str, Any]:
        """Load security policies configuration."""
        default_policies = {
            'authentication': {
                'max_failed_attempts': 5,
                'lockout_duration': 300,  # 5 minutes
                'session_timeout': 3600,  # 1 hour
                'require_mfa': False,
            },
            'authorization': {
                'default_role': 'operator',
                'admin_role': 'admin',
                'roles': {
                    'operator': ['read', 'execute_mission'],
                    'supervisor': ['read', 'write', 'execute_mission', 'emergency_stop'],
                    'admin': ['read', 'write', 'execute_mission', 'emergency_stop', 'configure'],
                },
            },
            'communication': {
                'require_encryption': True,
                'message_ttl': 300,  # 5 minutes
                'max_message_size': 1048576,  # 1MB
            },
            'data': {
                'classification_levels': ['public', 'internal', 'confidential'],
                'retention_periods': {
                    'logs': 90,  # days
                    'telemetry': 30,
                    'missions': 365,
                },
            },
        }
        
        policy_file = self.config_dir / "policies.json"
        if policy_file.exists():
            try:
                with open(policy_file, 'r') as f:
                    user_policies = json.load(f)
                # Merge with defaults
                default_policies.update(user_policies)
            except Exception as e:
                self.logger.warning(f"Could not load security policies: {e}")
        
        return default_policies

    def _initialize_threat_patterns(self) -> Dict[str, Any]:
        """Initialize threat detection patterns."""
        return {
            'brute_force': {
                'pattern': r'failed.*authentication',
                'threshold': 5,
                'window': 300,  # 5 minutes
                'action': 'block_ip',
            },
            'injection': {
                'patterns': [
                    r'(\bOR\b|\bAND\b).*=.*\d',  # SQL injection
                    r'<script.*?>',  # XSS
                    r'javascript:',  # JavaScript injection
                    r'\$\{.*\}',  # Template injection
                ],
                'action': 'sanitize_input',
            },
            'anomalous_behavior': {
                'unusual_locations': True,
                'unusual_times': True,
                'unusual_patterns': True,
                'threshold': 0.7,  # Anomaly score threshold
            },
            'resource_exhaustion': {
                'max_requests_per_minute': 100,
                'max_concurrent_sessions': 10,
                'max_data_rate_mbps': 10,
            },
        }

    def encrypt_message(self, message: Union[str, bytes, Dict[str, Any]]) -> bytes:
        """Encrypt message using symmetric encryption.
        
        Args:
            message: Message to encrypt
            
        Returns:
            Encrypted message bytes
        """
        try:
            # Convert to bytes if needed
            if isinstance(message, dict):
                message_bytes = json.dumps(message).encode('utf-8')
            elif isinstance(message, str):
                message_bytes = message.encode('utf-8')
            else:
                message_bytes = message
            
            # Encrypt
            fernet = Fernet(self._symmetric_key)
            encrypted = fernet.encrypt(message_bytes)
            
            self.logger.debug("Message encrypted successfully")
            return encrypted
            
        except Exception as e:
            self.logger.error(f"Message encryption failed: {e}")
            raise

    def decrypt_message(self, encrypted_message: bytes) -> bytes:
        """Decrypt message using symmetric encryption.
        
        Args:
            encrypted_message: Encrypted message bytes
            
        Returns:
            Decrypted message bytes
        """
        try:
            fernet = Fernet(self._symmetric_key)
            decrypted = fernet.decrypt(encrypted_message)
            
            self.logger.debug("Message decrypted successfully")
            return decrypted
            
        except Exception as e:
            self.logger.error(f"Message decryption failed: {e}")
            raise

    def sign_message(self, message: bytes) -> bytes:
        """Sign message using private key.
        
        Args:
            message: Message to sign
            
        Returns:
            Digital signature
        """
        try:
            signature = self._private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            self.logger.debug("Message signed successfully")
            return signature
            
        except Exception as e:
            self.logger.error(f"Message signing failed: {e}")
            raise

    def verify_signature(self, message: bytes, signature: bytes, public_key: Optional[rsa.RSAPublicKey] = None) -> bool:
        """Verify message signature.
        
        Args:
            message: Original message
            signature: Digital signature
            public_key: Public key to use (defaults to own public key)
            
        Returns:
            True if signature is valid
        """
        try:
            key = public_key or self._public_key
            
            key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            self.logger.debug("Signature verified successfully")
            return True
            
        except Exception as e:
            self.logger.warning(f"Signature verification failed: {e}")
            return False

    def create_secure_token(self, payload: Dict[str, Any], expires_in: Optional[int] = None) -> str:
        """Create secure JWT token.
        
        Args:
            payload: Token payload
            expires_in: Expiry time in seconds (defaults to configured JWT expiry)
            
        Returns:
            JWT token string
        """
        try:
            expiry = expires_in or self._jwt_expiry
            
            token_payload = {
                **payload,
                'iat': int(time.time()),
                'exp': int(time.time() + expiry),
                'jti': secrets.token_urlsafe(16),  # JWT ID for uniqueness
            }
            
            token = jwt.encode(
                token_payload,
                self._jwt_secret,
                algorithm=self._jwt_algorithm
            )
            
            self.logger.debug("Secure token created")
            return token
            
        except Exception as e:
            self.logger.error(f"Token creation failed: {e}")
            raise

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                self._jwt_secret,
                algorithms=[self._jwt_algorithm]
            )
            
            self.logger.debug("Token verified successfully")
            return payload
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Token verification failed: {e}")
            return None

    def authenticate_user(self, username: str, password: str, ip_address: Optional[str] = None) -> Optional[SecurityContext]:
        """Authenticate user credentials.
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address
            
        Returns:
            SecurityContext if authentication successful, None otherwise
        """
        try:
            # Check for rate limiting
            if self._is_rate_limited(username, ip_address):
                self.logger.warning(f"Authentication rate limited for {username}")
                return None
            
            # Simple authentication (in production, use proper user database)
            if self._verify_credentials(username, password):
                # Reset failed attempts
                self._reset_failed_attempts(username, ip_address)
                
                # Create security context
                context = SecurityContext(
                    user_id=username,
                    role=self._get_user_role(username),
                    permissions=self._get_user_permissions(username),
                    session_id=secrets.token_urlsafe(32),
                    ip_address=ip_address,
                )
                
                # Store active session
                self.active_sessions[context.session_id] = context
                
                self.logger.info(f"User {username} authenticated successfully", user=username)
                return context
            else:
                # Record failed attempt
                self._record_failed_attempt(username, ip_address)
                self.logger.warning(f"Authentication failed for {username}", user=username)
                return None
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return None

    def authorize_action(self, context: SecurityContext, resource: str, action: str) -> bool:
        """Authorize user action on resource.
        
        Args:
            context: Security context
            resource: Resource identifier
            action: Action to perform
            
        Returns:
            True if authorized
        """
        try:
            # Check if user has required permission
            required_permission = f"{action}_{resource}"
            
            if required_permission in context.permissions or action in context.permissions:
                self.logger.debug(f"Action authorized: {action} on {resource}", user=context.user_id)
                return True
            
            # Check role-based permissions
            role_permissions = self.policies['authorization']['roles'].get(context.role, [])
            if action in role_permissions:
                self.logger.debug(f"Action authorized by role: {action} on {resource}", user=context.user_id)
                return True
            
            self.logger.warning(f"Action not authorized: {action} on {resource}", user=context.user_id)
            return False
            
        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            return False

    def detect_threats(self, data: Dict[str, Any], context: Optional[SecurityContext] = None) -> List[ThreatDetection]:
        """Detect security threats in data.
        
        Args:
            data: Data to analyze
            context: Security context
            
        Returns:
            List of detected threats
        """
        threats = []
        
        try:
            # Check for injection patterns
            threats.extend(self._detect_injection_threats(data))
            
            # Check for anomalous behavior
            if context:
                threats.extend(self._detect_anomalous_behavior(data, context))
            
            # Check for resource exhaustion
            threats.extend(self._detect_resource_exhaustion(data))
            
            # Log detected threats
            for threat in threats:
                self.logger.warning(
                    f"Threat detected: {threat.threat_type}",
                    threat_type=threat.threat_type,
                    level=threat.level.value,
                    description=threat.description,
                    source=threat.source,
                )
            
            return threats
            
        except Exception as e:
            self.logger.error(f"Threat detection error: {e}")
            return []

    def sanitize_input(self, input_data: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """Sanitize input data to prevent security issues.
        
        Args:
            input_data: Input data to sanitize
            
        Returns:
            Sanitized data
        """
        try:
            if isinstance(input_data, str):
                return self._sanitize_string(input_data)
            elif isinstance(input_data, dict):
                return {k: self.sanitize_input(v) for k, v in input_data.items()}
            elif isinstance(input_data, list):
                return [self.sanitize_input(item) for item in input_data]
            else:
                return input_data
                
        except Exception as e:
            self.logger.error(f"Input sanitization error: {e}")
            return input_data

    def _verify_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials (simplified for MVP)."""
        # In production, use proper password hashing and user database
        default_users = {
            'admin': 'admin123',
            'operator': 'operator123',
            'supervisor': 'supervisor123',
        }
        
        return default_users.get(username) == password

    def _get_user_role(self, username: str) -> str:
        """Get user role (simplified for MVP)."""
        role_mapping = {
            'admin': 'admin',
            'supervisor': 'supervisor',
            'operator': 'operator',
        }
        
        return role_mapping.get(username, self.policies['authorization']['default_role'])

    def _get_user_permissions(self, username: str) -> List[str]:
        """Get user permissions based on role."""
        role = self._get_user_role(username)
        return self.policies['authorization']['roles'].get(role, [])

    def _is_rate_limited(self, username: str, ip_address: Optional[str]) -> bool:
        """Check if user/IP is rate limited."""
        current_time = time.time()
        
        # Check failed attempts for user
        user_key = f"user:{username}"
        if user_key in self.failed_attempts:
            attempts = self.failed_attempts[user_key]
            if (len(attempts) >= self.policies['authentication']['max_failed_attempts'] and
                current_time - attempts[-1] < self.policies['authentication']['lockout_duration']):
                return True
        
        # Check failed attempts for IP
        if ip_address:
            ip_key = f"ip:{ip_address}"
            if ip_key in self.failed_attempts:
                attempts = self.failed_attempts[ip_key]
                if (len(attempts) >= self.policies['authentication']['max_failed_attempts'] * 2 and
                    current_time - attempts[-1] < self.policies['authentication']['lockout_duration']):
                    return True
        
        return False

    def _record_failed_attempt(self, username: str, ip_address: Optional[str]) -> None:
        """Record failed authentication attempt."""
        current_time = time.time()
        
        # Record for user
        user_key = f"user:{username}"
        if user_key not in self.failed_attempts:
            self.failed_attempts[user_key] = []
        self.failed_attempts[user_key].append(current_time)
        
        # Record for IP
        if ip_address:
            ip_key = f"ip:{ip_address}"
            if ip_key not in self.failed_attempts:
                self.failed_attempts[ip_key] = []
            self.failed_attempts[ip_key].append(current_time)
        
        # Cleanup old attempts
        self._cleanup_failed_attempts()

    def _reset_failed_attempts(self, username: str, ip_address: Optional[str]) -> None:
        """Reset failed attempts after successful authentication."""
        user_key = f"user:{username}"
        self.failed_attempts.pop(user_key, None)
        
        if ip_address:
            ip_key = f"ip:{ip_address}"
            self.failed_attempts.pop(ip_key, None)

    def _cleanup_failed_attempts(self) -> None:
        """Cleanup old failed attempts."""
        current_time = time.time()
        cleanup_threshold = self.policies['authentication']['lockout_duration']
        
        for key in list(self.failed_attempts.keys()):
            attempts = self.failed_attempts[key]
            # Keep only recent attempts
            recent_attempts = [t for t in attempts if current_time - t < cleanup_threshold]
            
            if recent_attempts:
                self.failed_attempts[key] = recent_attempts
            else:
                del self.failed_attempts[key]

    def _detect_injection_threats(self, data: Dict[str, Any]) -> List[ThreatDetection]:
        """Detect injection attack patterns."""
        threats = []
        patterns = self.threat_patterns['injection']['patterns']
        
        def check_value(value: str, path: str) -> None:
            for pattern in patterns:
                import re
                if re.search(pattern, value, re.IGNORECASE):
                    threats.append(ThreatDetection(
                        threat_type="injection",
                        level=ThreatLevel.HIGH,
                        description=f"Potential injection pattern detected in {path}",
                        source=path,
                        mitigation="Input sanitized",
                    ))
        
        def traverse_data(obj: Any, path: str = "root") -> None:
            if isinstance(obj, str):
                check_value(obj, path)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    traverse_data(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    traverse_data(item, f"{path}[{i}]")
        
        traverse_data(data)
        return threats

    def _detect_anomalous_behavior(self, data: Dict[str, Any], context: SecurityContext) -> List[ThreatDetection]:
        """Detect anomalous behavior patterns."""
        threats = []
        
        # Simple anomaly detection (in production, use ML models)
        # Check for unusual timing
        current_hour = time.localtime().tm_hour
        if current_hour < 6 or current_hour > 22:  # Outside normal hours
            threats.append(ThreatDetection(
                threat_type="anomalous_timing",
                level=ThreatLevel.MEDIUM,
                description="Activity detected outside normal business hours",
                source=context.user_id or "unknown",
            ))
        
        return threats

    def _detect_resource_exhaustion(self, data: Dict[str, Any]) -> List[ThreatDetection]:
        """Detect resource exhaustion attacks."""
        threats = []
        
        # Check message size
        try:
            data_size = len(json.dumps(data))
            max_size = self.policies['communication']['max_message_size']
            
            if data_size > max_size:
                threats.append(ThreatDetection(
                    threat_type="resource_exhaustion",
                    level=ThreatLevel.MEDIUM,
                    description=f"Message size ({data_size}) exceeds limit ({max_size})",
                    source="message_size",
                    mitigation="Message rejected",
                    blocked=True,
                ))
        except (TypeError, ValueError):
            pass
        
        return threats

    def _sanitize_string(self, input_str: str) -> str:
        """Sanitize string input."""
        # Remove potentially dangerous characters
        import re
        
        # Remove HTML tags
        clean_str = re.sub(r'<[^>]+>', '', input_str)
        
        # Remove SQL injection patterns
        sql_patterns = [
            r'(\bOR\b|\bAND\b)\s*=\s*\d',
            r'\bUNION\b.*\bSELECT\b',
            r'\bDROP\b.*\bTABLE\b',
            r'\bINSERT\b.*\bINTO\b',
            r'\bUPDATE\b.*\bSET\b',
            r'\bDELETE\b.*\bFROM\b',
        ]
        
        for pattern in sql_patterns:
            clean_str = re.sub(pattern, '', clean_str, flags=re.IGNORECASE)
        
        # Remove JavaScript injection
        js_patterns = [
            r'javascript:',
            r'<script.*?>.*?</script>',
            r'on\w+\s*=',
        ]
        
        for pattern in js_patterns:
            clean_str = re.sub(pattern, '', clean_str, flags=re.IGNORECASE)
        
        return clean_str.strip()


# Convenience functions
def encrypt_message(message: Union[str, bytes, Dict[str, Any]], security_manager: Optional[SecurityManager] = None) -> bytes:
    """Encrypt message using default security manager."""
    if security_manager is None:
        security_manager = SecurityManager()
    return security_manager.encrypt_message(message)


def decrypt_message(encrypted_message: bytes, security_manager: Optional[SecurityManager] = None) -> bytes:
    """Decrypt message using default security manager."""
    if security_manager is None:
        security_manager = SecurityManager()
    return security_manager.decrypt_message(encrypted_message)


# Global security manager instance
_global_security_manager = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    return _global_security_manager