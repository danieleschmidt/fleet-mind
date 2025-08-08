"""Advanced internationalization and localization system for Fleet-Mind."""

import json
import time
from typing import Dict, Optional, Any, List, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Locale(Enum):
    """Supported locales."""
    EN_US = "en_US"  # English (United States)
    EN_GB = "en_GB"  # English (United Kingdom)
    ES_ES = "es_ES"  # Spanish (Spain)
    ES_MX = "es_MX"  # Spanish (Mexico)
    FR_FR = "fr_FR"  # French (France)
    DE_DE = "de_DE"  # German (Germany)
    JA_JP = "ja_JP"  # Japanese (Japan)
    ZH_CN = "zh_CN"  # Chinese (Simplified)
    ZH_TW = "zh_TW"  # Chinese (Traditional)
    PT_BR = "pt_BR"  # Portuguese (Brazil)
    RU_RU = "ru_RU"  # Russian (Russia)
    AR_SA = "ar_SA"  # Arabic (Saudi Arabia)


@dataclass
class LocalizationContext:
    """Context for localization operations."""
    locale: Locale
    timezone: str = "UTC"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "1,000.00"
    currency_code: str = "USD"
    rtl_text: bool = False


class LocalizationManager:
    """Advanced localization manager with dynamic translation support."""
    
    def __init__(self, default_locale: Locale = Locale.EN_US):
        """Initialize localization manager.
        
        Args:
            default_locale: Default locale to use
        """
        self.default_locale = default_locale
        self.current_locale = default_locale
        
        # Translation storage
        self.translations: Dict[Locale, Dict[str, str]] = {}
        self.contexts: Dict[Locale, LocalizationContext] = {}
        
        # Initialize built-in translations
        self._initialize_translations()
        self._initialize_contexts()
        
        print(f"Localization manager initialized with {default_locale.value}")
    
    def _initialize_translations(self):
        """Initialize built-in translations."""
        # Fleet-Mind core translations
        translations = {
            Locale.EN_US: {
                # System messages
                "system.startup": "Fleet-Mind system starting",
                "system.shutdown": "Fleet-Mind system shutting down",
                "system.ready": "System ready",
                "system.error": "System error occurred",
                
                # Mission messages
                "mission.starting": "Mission starting: {mission_name}",
                "mission.completed": "Mission completed successfully",
                "mission.failed": "Mission failed: {error}",
                "mission.paused": "Mission paused",
                "mission.resumed": "Mission resumed",
                
                # Drone messages
                "drone.connected": "Drone {drone_id} connected",
                "drone.disconnected": "Drone {drone_id} disconnected",
                "drone.low_battery": "Drone {drone_id} low battery: {battery}%",
                "drone.emergency": "Emergency: Drone {drone_id}",
                
                # Communication messages
                "comm.connection_lost": "Connection lost to {target}",
                "comm.connection_restored": "Connection restored to {target}",
                "comm.high_latency": "High latency detected: {latency}ms",
                
                # Security messages
                "security.auth_failed": "Authentication failed for {user}",
                "security.access_denied": "Access denied",
                "security.threat_detected": "Security threat detected",
                
                # Performance messages
                "perf.high_load": "High system load: {load}%",
                "perf.memory_warning": "Memory usage high: {memory}MB",
                "perf.optimization_applied": "Performance optimization applied",
                
                # User interface
                "ui.welcome": "Welcome to Fleet-Mind",
                "ui.goodbye": "Thank you for using Fleet-Mind",
                "ui.confirm": "Confirm",
                "ui.cancel": "Cancel",
                "ui.save": "Save",
                "ui.load": "Load",
            },
            
            Locale.ES_ES: {
                # Spanish translations
                "system.startup": "Iniciando sistema Fleet-Mind",
                "system.shutdown": "Cerrando sistema Fleet-Mind",
                "system.ready": "Sistema listo",
                "system.error": "Error del sistema",
                
                "mission.starting": "Iniciando misión: {mission_name}",
                "mission.completed": "Misión completada exitosamente",
                "mission.failed": "Misión fallida: {error}",
                "mission.paused": "Misión pausada",
                "mission.resumed": "Misión reanudada",
                
                "drone.connected": "Dron {drone_id} conectado",
                "drone.disconnected": "Dron {drone_id} desconectado",
                "drone.low_battery": "Dron {drone_id} batería baja: {battery}%",
                "drone.emergency": "Emergencia: Dron {drone_id}",
                
                "ui.welcome": "Bienvenido a Fleet-Mind",
                "ui.goodbye": "Gracias por usar Fleet-Mind",
                "ui.confirm": "Confirmar",
                "ui.cancel": "Cancelar",
                "ui.save": "Guardar",
                "ui.load": "Cargar",
            },
            
            Locale.FR_FR: {
                # French translations
                "system.startup": "Démarrage du système Fleet-Mind",
                "system.shutdown": "Arrêt du système Fleet-Mind",
                "system.ready": "Système prêt",
                "system.error": "Erreur système",
                
                "mission.starting": "Démarrage de la mission: {mission_name}",
                "mission.completed": "Mission terminée avec succès",
                "mission.failed": "Mission échouée: {error}",
                "mission.paused": "Mission en pause",
                "mission.resumed": "Mission reprise",
                
                "drone.connected": "Drone {drone_id} connecté",
                "drone.disconnected": "Drone {drone_id} déconnecté",
                "drone.low_battery": "Drone {drone_id} batterie faible: {battery}%",
                "drone.emergency": "Urgence: Drone {drone_id}",
                
                "ui.welcome": "Bienvenue dans Fleet-Mind",
                "ui.goodbye": "Merci d'utiliser Fleet-Mind",
                "ui.confirm": "Confirmer",
                "ui.cancel": "Annuler",
                "ui.save": "Enregistrer",
                "ui.load": "Charger",
            },
            
            Locale.DE_DE: {
                # German translations
                "system.startup": "Fleet-Mind System startet",
                "system.shutdown": "Fleet-Mind System wird heruntergefahren",
                "system.ready": "System bereit",
                "system.error": "Systemfehler aufgetreten",
                
                "mission.starting": "Mission startet: {mission_name}",
                "mission.completed": "Mission erfolgreich abgeschlossen",
                "mission.failed": "Mission fehlgeschlagen: {error}",
                "mission.paused": "Mission pausiert",
                "mission.resumed": "Mission fortgesetzt",
                
                "drone.connected": "Drohne {drone_id} verbunden",
                "drone.disconnected": "Drohne {drone_id} getrennt",
                "drone.low_battery": "Drohne {drone_id} niedrige Batterie: {battery}%",
                "drone.emergency": "Notfall: Drohne {drone_id}",
                
                "ui.welcome": "Willkommen bei Fleet-Mind",
                "ui.goodbye": "Vielen Dank für die Nutzung von Fleet-Mind",
                "ui.confirm": "Bestätigen",
                "ui.cancel": "Abbrechen",
                "ui.save": "Speichern",
                "ui.load": "Laden",
            },
            
            Locale.JA_JP: {
                # Japanese translations
                "system.startup": "Fleet-Mindシステムが開始されています",
                "system.shutdown": "Fleet-Mindシステムがシャットダウンされています",
                "system.ready": "システム準備完了",
                "system.error": "システムエラーが発生しました",
                
                "mission.starting": "ミッション開始: {mission_name}",
                "mission.completed": "ミッションが正常に完了しました",
                "mission.failed": "ミッション失敗: {error}",
                "mission.paused": "ミッション一時停止",
                "mission.resumed": "ミッション再開",
                
                "drone.connected": "ドローン{drone_id}が接続されました",
                "drone.disconnected": "ドローン{drone_id}が切断されました",
                "drone.low_battery": "ドローン{drone_id}のバッテリーが低下: {battery}%",
                "drone.emergency": "緊急事態: ドローン{drone_id}",
                
                "ui.welcome": "Fleet-Mindへようこそ",
                "ui.goodbye": "Fleet-Mindをご利用いただきありがとうございました",
                "ui.confirm": "確認",
                "ui.cancel": "キャンセル",
                "ui.save": "保存",
                "ui.load": "読み込み",
            },
            
            Locale.ZH_CN: {
                # Chinese (Simplified) translations
                "system.startup": "Fleet-Mind系统正在启动",
                "system.shutdown": "Fleet-Mind系统正在关闭",
                "system.ready": "系统就绪",
                "system.error": "系统发生错误",
                
                "mission.starting": "任务开始: {mission_name}",
                "mission.completed": "任务成功完成",
                "mission.failed": "任务失败: {error}",
                "mission.paused": "任务已暂停",
                "mission.resumed": "任务已恢复",
                
                "drone.connected": "无人机{drone_id}已连接",
                "drone.disconnected": "无人机{drone_id}已断开连接",
                "drone.low_battery": "无人机{drone_id}电量不足: {battery}%",
                "drone.emergency": "紧急情况: 无人机{drone_id}",
                
                "ui.welcome": "欢迎使用Fleet-Mind",
                "ui.goodbye": "感谢使用Fleet-Mind",
                "ui.confirm": "确认",
                "ui.cancel": "取消",
                "ui.save": "保存",
                "ui.load": "加载",
            },
        }
        
        self.translations = translations
    
    def _initialize_contexts(self):
        """Initialize localization contexts."""
        self.contexts = {
            Locale.EN_US: LocalizationContext(
                locale=Locale.EN_US,
                timezone="America/New_York",
                date_format="%m/%d/%Y",
                time_format="%I:%M:%S %p",
                currency_code="USD"
            ),
            
            Locale.EN_GB: LocalizationContext(
                locale=Locale.EN_GB,
                timezone="Europe/London",
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                currency_code="GBP"
            ),
            
            Locale.ES_ES: LocalizationContext(
                locale=Locale.ES_ES,
                timezone="Europe/Madrid",
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                currency_code="EUR"
            ),
            
            Locale.FR_FR: LocalizationContext(
                locale=Locale.FR_FR,
                timezone="Europe/Paris",
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                currency_code="EUR"
            ),
            
            Locale.DE_DE: LocalizationContext(
                locale=Locale.DE_DE,
                timezone="Europe/Berlin",
                date_format="%d.%m.%Y",
                time_format="%H:%M:%S",
                currency_code="EUR"
            ),
            
            Locale.JA_JP: LocalizationContext(
                locale=Locale.JA_JP,
                timezone="Asia/Tokyo",
                date_format="%Y/%m/%d",
                time_format="%H:%M:%S",
                currency_code="JPY"
            ),
            
            Locale.ZH_CN: LocalizationContext(
                locale=Locale.ZH_CN,
                timezone="Asia/Shanghai",
                date_format="%Y-%m-%d",
                time_format="%H:%M:%S",
                currency_code="CNY"
            ),
        }
    
    def set_locale(self, locale: Locale):
        """Set current locale."""
        self.current_locale = locale
        print(f"Locale changed to {locale.value}")
    
    def get_current_locale(self) -> Locale:
        """Get current locale."""
        return self.current_locale
    
    def translate(
        self,
        key: str,
        locale: Optional[Locale] = None,
        **kwargs
    ) -> str:
        """Translate a message key.
        
        Args:
            key: Translation key
            locale: Target locale (current if None)
            **kwargs: Format parameters
            
        Returns:
            Translated message
        """
        target_locale = locale or self.current_locale
        
        # Get translation
        if target_locale in self.translations:
            translation = self.translations[target_locale].get(key)
            
            if translation:
                try:
                    # Format with parameters
                    return translation.format(**kwargs)
                except KeyError as e:
                    print(f"Missing format parameter {e} for key {key}")
                    return translation
        
        # Fallback to default locale
        if (target_locale != self.default_locale and 
            self.default_locale in self.translations):
            
            translation = self.translations[self.default_locale].get(key)
            if translation:
                try:
                    return translation.format(**kwargs)
                except KeyError:
                    return translation
        
        # Final fallback - return key
        return key
    
    def get_context(self, locale: Optional[Locale] = None) -> LocalizationContext:
        """Get localization context.
        
        Args:
            locale: Target locale (current if None)
            
        Returns:
            Localization context
        """
        target_locale = locale or self.current_locale
        return self.contexts.get(target_locale, self.contexts[self.default_locale])
    
    def format_datetime(
        self,
        timestamp: float,
        locale: Optional[Locale] = None,
        include_time: bool = True
    ) -> str:
        """Format datetime according to locale.
        
        Args:
            timestamp: Unix timestamp
            locale: Target locale
            include_time: Include time in format
            
        Returns:
            Formatted datetime string
        """
        context = self.get_context(locale)
        dt = time.localtime(timestamp)
        
        if include_time:
            format_str = f"{context.date_format} {context.time_format}"
        else:
            format_str = context.date_format
        
        return time.strftime(format_str, dt)
    
    def format_number(
        self,
        number: Union[int, float],
        locale: Optional[Locale] = None,
        decimal_places: int = 2
    ) -> str:
        """Format number according to locale.
        
        Args:
            number: Number to format
            locale: Target locale
            decimal_places: Number of decimal places
            
        Returns:
            Formatted number string
        """
        context = self.get_context(locale)
        
        # Simple formatting - in production would use locale.format()
        if context.locale in [Locale.EN_US, Locale.EN_GB]:
            # English - comma thousands separator, period decimal
            return f"{number:,.{decimal_places}f}"
        elif context.locale in [Locale.FR_FR, Locale.ES_ES]:
            # French/Spanish - space thousands separator, comma decimal
            formatted = f"{number:,.{decimal_places}f}"
            return formatted.replace(",", " ").replace(".", ",")
        elif context.locale == Locale.DE_DE:
            # German - period thousands separator, comma decimal
            formatted = f"{number:,.{decimal_places}f}"
            return formatted.replace(",", ".").replace(".", ",", 1)
        else:
            # Default formatting
            return f"{number:,.{decimal_places}f}"
    
    def format_currency(
        self,
        amount: float,
        locale: Optional[Locale] = None
    ) -> str:
        """Format currency according to locale.
        
        Args:
            amount: Currency amount
            locale: Target locale
            
        Returns:
            Formatted currency string
        """
        context = self.get_context(locale)
        number_str = self.format_number(amount, locale, 2)
        
        # Currency symbols and positions
        currency_symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "CNY": "¥",
        }
        
        symbol = currency_symbols.get(context.currency_code, context.currency_code)
        
        if context.currency_code in ["USD", "GBP"]:
            return f"{symbol}{number_str}"
        else:
            return f"{number_str} {symbol}"
    
    def add_translations(
        self,
        locale: Locale,
        translations: Dict[str, str]
    ):
        """Add translations for a locale.
        
        Args:
            locale: Target locale
            translations: Dictionary of key-value translations
        """
        if locale not in self.translations:
            self.translations[locale] = {}
        
        self.translations[locale].update(translations)
    
    def load_translations_from_file(
        self,
        locale: Locale,
        file_path: Union[str, Path]
    ):
        """Load translations from JSON file.
        
        Args:
            locale: Target locale
            file_path: Path to JSON translation file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                self.add_translations(locale, translations)
        except Exception as e:
            print(f"Failed to load translations from {file_path}: {e}")
    
    def get_available_locales(self) -> List[Locale]:
        """Get list of available locales."""
        return list(self.translations.keys())
    
    def is_rtl_language(self, locale: Optional[Locale] = None) -> bool:
        """Check if language uses right-to-left text.
        
        Args:
            locale: Target locale
            
        Returns:
            True if RTL language
        """
        target_locale = locale or self.current_locale
        return target_locale in [Locale.AR_SA]  # Add more RTL locales as needed
    
    def get_language_name(self, locale: Locale, in_locale: Optional[Locale] = None) -> str:
        """Get language name in specified locale.
        
        Args:
            locale: Language to get name for
            in_locale: Locale to show name in
            
        Returns:
            Language name
        """
        target_locale = in_locale or self.current_locale
        
        # Language names in different locales
        language_names = {
            Locale.EN_US: {
                Locale.EN_US: "English (US)",
                Locale.EN_GB: "English (UK)",
                Locale.ES_ES: "Spanish",
                Locale.FR_FR: "French",
                Locale.DE_DE: "German",
                Locale.JA_JP: "Japanese",
                Locale.ZH_CN: "Chinese (Simplified)",
            },
            Locale.ES_ES: {
                Locale.EN_US: "Inglés (EE.UU.)",
                Locale.EN_GB: "Inglés (Reino Unido)",
                Locale.ES_ES: "Español",
                Locale.FR_FR: "Francés",
                Locale.DE_DE: "Alemán",
                Locale.JA_JP: "Japonés",
                Locale.ZH_CN: "Chino (Simplificado)",
            },
            # Add more as needed
        }
        
        if target_locale in language_names and locale in language_names[target_locale]:
            return language_names[target_locale][locale]
        
        # Fallback
        return locale.value


# Global localization manager
_global_manager: Optional[LocalizationManager] = None


def get_translator(default_locale: Locale = Locale.EN_US) -> LocalizationManager:
    """Get global localization manager."""
    global _global_manager
    
    if _global_manager is None:
        _global_manager = LocalizationManager(default_locale)
    
    return _global_manager


# Convenience function for translation
def t(key: str, **kwargs) -> str:
    """Translate message using global manager."""
    manager = get_translator()
    return manager.translate(key, **kwargs)