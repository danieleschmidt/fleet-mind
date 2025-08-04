"""Internationalization (i18n) support for Fleet-Mind global deployment."""

import json
import os
from typing import Dict, Optional, Any, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from .logging import get_logger


class SupportedLanguage(Enum):
    """Supported languages for Fleet-Mind."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    ARABIC = "ar"
    RUSSIAN = "ru"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    HINDI = "hi"
    TURKISH = "tr"


class Region(Enum):
    """Supported regions for compliance and localization."""
    NORTH_AMERICA = "na"
    EUROPE = "eu"
    ASIA_PACIFIC = "apac"
    LATIN_AMERICA = "latam"
    MIDDLE_EAST_AFRICA = "mea"
    GLOBAL = "global"


@dataclass
class LocalizationConfig:
    """Configuration for localization and regional compliance."""
    language: SupportedLanguage = SupportedLanguage.ENGLISH
    region: Region = Region.GLOBAL
    timezone: str = "UTC"
    date_format: str = "YYYY-MM-DD"
    time_format: str = "24h"
    currency: str = "USD"
    distance_unit: str = "meters"  # meters, feet
    coordinate_system: str = "WGS84"
    data_residency: Optional[str] = None  # Country code for data residency
    compliance_frameworks: List[str] = field(default_factory=lambda: ["GDPR", "SOC2"])
    

class I18nManager:
    """Internationalization and localization manager."""
    
    def __init__(self, config: Optional[LocalizationConfig] = None):
        """Initialize i18n manager.
        
        Args:
            config: Localization configuration
        """
        self.config = config or LocalizationConfig()
        self.logger = get_logger("i18n_manager", component="i18n")
        
        # Initialize translation dictionaries
        self.translations: Dict[str, Dict[str, str]] = {}
        self.fallback_language = SupportedLanguage.ENGLISH
        
        # Load translations
        self._load_translations()
        
        # Regional settings
        self.regional_settings = self._load_regional_settings()
        
        self.logger.info(
            f"i18n manager initialized for {self.config.language.value} in {self.config.region.value}"
        )

    def _load_translations(self) -> None:
        """Load translation files."""
        translations_dir = Path(__file__).parent.parent / "translations"
        translations_dir.mkdir(exist_ok=True)
        
        # Create default translations if they don't exist
        self._create_default_translations(translations_dir)
        
        # Load all available translations
        for language in SupportedLanguage:
            translation_file = translations_dir / f"{language.value}.json"
            if translation_file.exists():
                try:
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        self.translations[language.value] = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load translations for {language.value}: {e}")
            else:
                self.logger.debug(f"Translation file not found for {language.value}")

    def _create_default_translations(self, translations_dir: Path) -> None:
        """Create default translation files."""
        default_translations = {
            "common": {
                "error": "Error",
                "warning": "Warning", 
                "info": "Information",
                "success": "Success",
                "cancel": "Cancel",
                "confirm": "Confirm",
                "yes": "Yes",
                "no": "No",
                "ok": "OK",
                "loading": "Loading...",
                "retry": "Retry",
                "close": "Close"
            },
            "fleet": {
                "mission_started": "Mission started successfully",
                "mission_completed": "Mission completed",
                "mission_failed": "Mission failed",
                "drone_connected": "Drone connected",
                "drone_disconnected": "Drone disconnected",
                "swarm_ready": "Swarm ready for operations",
                "emergency_stop": "Emergency stop activated",
                "coordinates_invalid": "Invalid coordinates provided",
                "altitude_limit_exceeded": "Altitude limit exceeded",
                "weather_conditions_unsafe": "Weather conditions unsafe for flight"
            },
            "security": {
                "authentication_required": "Authentication required",
                "access_denied": "Access denied",
                "session_expired": "Session expired",
                "invalid_credentials": "Invalid credentials",
                "account_locked": "Account locked due to multiple failed attempts",
                "permission_denied": "Permission denied for this operation",
                "security_threat_detected": "Security threat detected"
            },
            "system": {
                "system_ready": "System ready",
                "maintenance_mode": "System in maintenance mode",
                "high_load": "System under high load",
                "service_unavailable": "Service temporarily unavailable",
                "database_error": "Database connection error",
                "network_error": "Network connection error",
                "configuration_error": "Configuration error"
            },
            "compliance": {
                "data_processing_consent": "Data processing requires your consent",
                "data_retention_policy": "Data will be retained according to policy",
                "privacy_notice": "Please review our privacy notice",
                "right_to_deletion": "You have the right to request data deletion",
                "data_export_ready": "Your data export is ready for download",
                "audit_log_entry": "Action recorded in audit log"
            }
        }
        
        # Create English translations
        en_file = translations_dir / "en.json"
        if not en_file.exists():
            with open(en_file, 'w', encoding='utf-8') as f:
                json.dump(default_translations, f, indent=2, ensure_ascii=False)
        
        # Create sample translations for other languages
        sample_translations = {
            "es": {  # Spanish
                "common": {
                    "error": "Error",
                    "warning": "Advertencia",
                    "info": "Información",
                    "success": "Éxito",
                    "cancel": "Cancelar",
                    "confirm": "Confirmar",
                    "yes": "Sí",
                    "no": "No",
                    "ok": "Aceptar",
                    "loading": "Cargando...",
                    "retry": "Reintentar",
                    "close": "Cerrar"
                },
                "fleet": {
                    "mission_started": "Misión iniciada exitosamente",
                    "mission_completed": "Misión completada",
                    "mission_failed": "Misión fallida",
                    "drone_connected": "Drone conectado",
                    "drone_disconnected": "Drone desconectado",
                    "swarm_ready": "Enjambre listo para operaciones",
                    "emergency_stop": "Parada de emergencia activada"
                }
            },
            "fr": {  # French
                "common": {
                    "error": "Erreur",
                    "warning": "Avertissement",
                    "info": "Information",
                    "success": "Succès",
                    "cancel": "Annuler",
                    "confirm": "Confirmer",
                    "yes": "Oui",
                    "no": "Non",
                    "ok": "OK",
                    "loading": "Chargement...",
                    "retry": "Réessayer",
                    "close": "Fermer"
                },
                "fleet": {
                    "mission_started": "Mission démarrée avec succès",
                    "mission_completed": "Mission terminée",
                    "mission_failed": "Mission échouée",
                    "drone_connected": "Drone connecté",
                    "drone_disconnected": "Drone déconnecté",
                    "swarm_ready": "Essaim prêt pour les opérations",
                    "emergency_stop": "Arrêt d'urgence activé"
                }
            }
        }
        
        for lang_code, translations in sample_translations.items():
            lang_file = translations_dir / f"{lang_code}.json"
            if not lang_file.exists():
                with open(lang_file, 'w', encoding='utf-8') as f:
                    json.dump(translations, f, indent=2, ensure_ascii=False)

    def _load_regional_settings(self) -> Dict[str, Any]:
        """Load regional settings and compliance requirements."""
        return {
            Region.EUROPE: {
                "compliance_frameworks": ["GDPR", "ePrivacy", "NIS2"],
                "data_protection_authority": "relevant_dpa",
                "data_residency_required": True,
                "cookie_consent_required": True,
                "right_to_be_forgotten": True,
                "data_portability": True,
                "breach_notification_hours": 72,
                "privacy_by_design": True
            },
            Region.NORTH_AMERICA: {
                "compliance_frameworks": ["SOC2", "CCPA", "PIPEDA"],
                "data_residency_required": False,
                "cookie_consent_required": False,
                "right_to_be_forgotten": True,  # CCPA
                "data_portability": True,
                "breach_notification_hours": 72,
                "privacy_by_design": False
            },
            Region.ASIA_PACIFIC: {
                "compliance_frameworks": ["PDPA", "Privacy Act", "PIPL"],
                "data_residency_required": True,  # Some countries
                "cookie_consent_required": True,
                "right_to_be_forgotten": True,
                "data_portability": True,
                "breach_notification_hours": 72,
                "privacy_by_design": True
            },
            Region.GLOBAL: {
                "compliance_frameworks": ["GDPR", "SOC2", "ISO27001"],
                "data_residency_required": True,
                "cookie_consent_required": True,
                "right_to_be_forgotten": True,
                "data_portability": True,
                "breach_notification_hours": 24,  # Most stringent
                "privacy_by_design": True
            }
        }

    def translate(self, key: str, category: str = "common", **kwargs) -> str:
        """Translate a key to the current language.
        
        Args:
            key: Translation key
            category: Translation category
            **kwargs: Format arguments
            
        Returns:
            Translated string
        """
        try:
            # Get translation for current language
            lang_translations = self.translations.get(self.config.language.value, {})
            category_translations = lang_translations.get(category, {})
            
            if key in category_translations:
                translation = category_translations[key]
            else:
                # Fallback to English
                fallback_translations = self.translations.get(self.fallback_language.value, {})
                fallback_category = fallback_translations.get(category, {})
                translation = fallback_category.get(key, f"[{category}.{key}]")
                
                self.logger.debug(f"Translation not found for {key} in {self.config.language.value}")
            
            # Format with provided arguments
            if kwargs:
                translation = translation.format(**kwargs)
            
            return translation
            
        except Exception as e:
            self.logger.error(f"Translation error for {key}: {e}")
            return f"[{category}.{key}]"

    def format_datetime(self, dt: Any, include_time: bool = True) -> str:
        """Format datetime according to regional preferences.
        
        Args:
            dt: Datetime object
            include_time: Whether to include time
            
        Returns:
            Formatted datetime string
        """
        try:
            # This is a simplified implementation
            # In production, use libraries like babel or moment.js
            
            if self.config.date_format == "DD/MM/YYYY":
                date_str = dt.strftime("%d/%m/%Y")
            elif self.config.date_format == "MM/DD/YYYY":
                date_str = dt.strftime("%m/%d/%Y")
            else:  # ISO format
                date_str = dt.strftime("%Y-%m-%d")
            
            if include_time:
                if self.config.time_format == "12h":
                    time_str = dt.strftime("%I:%M %p")
                else:
                    time_str = dt.strftime("%H:%M")
                
                return f"{date_str} {time_str}"
            
            return date_str
            
        except Exception as e:
            self.logger.error(f"Datetime formatting error: {e}")
            return str(dt)

    def format_distance(self, distance_meters: float) -> str:
        """Format distance according to regional preferences.
        
        Args:
            distance_meters: Distance in meters
            
        Returns:
            Formatted distance string
        """
        try:
            if self.config.distance_unit == "feet":
                distance_feet = distance_meters * 3.28084
                if distance_feet > 5280:  # Convert to miles
                    distance_miles = distance_feet / 5280
                    return f"{distance_miles:.2f} miles"
                else:
                    return f"{distance_feet:.1f} ft"
            else:
                if distance_meters > 1000:  # Convert to kilometers
                    distance_km = distance_meters / 1000
                    return f"{distance_km:.2f} km"
                else:
                    return f"{distance_meters:.1f} m"
                    
        except Exception as e:
            self.logger.error(f"Distance formatting error: {e}")
            return f"{distance_meters} m"

    def get_compliance_requirements(self) -> Dict[str, Any]:
        """Get compliance requirements for the current region.
        
        Returns:
            Compliance requirements dictionary
        """
        return self.regional_settings.get(self.config.region, {})

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a compliance feature is enabled for the current region.
        
        Args:
            feature: Feature name (e.g., 'right_to_be_forgotten')
            
        Returns:
            True if feature is enabled
        """
        requirements = self.get_compliance_requirements()
        return requirements.get(feature, False)

    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages.
        
        Returns:
            List of language info dictionaries
        """
        return [
            {"code": lang.value, "name": self._get_language_name(lang)}
            for lang in SupportedLanguage
        ]

    def _get_language_name(self, language: SupportedLanguage) -> str:
        """Get display name for a language."""
        names = {
            SupportedLanguage.ENGLISH: "English",
            SupportedLanguage.SPANISH: "Español",
            SupportedLanguage.FRENCH: "Français",
            SupportedLanguage.GERMAN: "Deutsch",
            SupportedLanguage.JAPANESE: "日本語",
            SupportedLanguage.CHINESE_SIMPLIFIED: "简体中文",
            SupportedLanguage.CHINESE_TRADITIONAL: "繁體中文",
            SupportedLanguage.KOREAN: "한국어",
            SupportedLanguage.ARABIC: "العربية",
            SupportedLanguage.RUSSIAN: "Русский",
            SupportedLanguage.PORTUGUESE: "Português",
            SupportedLanguage.ITALIAN: "Italiano",
            SupportedLanguage.DUTCH: "Nederlands",
            SupportedLanguage.HINDI: "हिन्दी",
            SupportedLanguage.TURKISH: "Türkçe"
        }
        return names.get(language, language.value)

    def change_language(self, language_code: str) -> bool:
        """Change the current language.
        
        Args:
            language_code: ISO language code
            
        Returns:
            True if language was changed successfully
        """
        try:
            new_language = SupportedLanguage(language_code)
            self.config.language = new_language
            self.logger.info(f"Language changed to {language_code}")
            return True
        except ValueError:
            self.logger.error(f"Unsupported language code: {language_code}")
            return False

    def export_translations(self, language_code: str) -> Dict[str, Any]:
        """Export translations for a specific language.
        
        Args:
            language_code: ISO language code
            
        Returns:
            Translation dictionary
        """
        return self.translations.get(language_code, {})


# Global i18n manager instance
_global_i18n_manager = None


def get_i18n_manager(config: Optional[LocalizationConfig] = None) -> I18nManager:
    """Get global i18n manager instance."""
    global _global_i18n_manager
    if _global_i18n_manager is None:
        _global_i18n_manager = I18nManager(config)
    return _global_i18n_manager


def translate(key: str, category: str = "common", **kwargs) -> str:
    """Convenience function for translation."""
    return get_i18n_manager().translate(key, category, **kwargs)


def format_datetime(dt: Any, include_time: bool = True) -> str:
    """Convenience function for datetime formatting."""
    return get_i18n_manager().format_datetime(dt, include_time)


def format_distance(distance_meters: float) -> str:
    """Convenience function for distance formatting."""
    return get_i18n_manager().format_distance(distance_meters)