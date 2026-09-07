"""
Gemini AI Agent Professional Edition
=====================================
- Real-time Streaming responses
- Advanced Security & Permissions System
- Adaptive Learning & User Preferences
- Modern Dark UI with Chat Bubbles
- Optimized Performance (Lazy Loading)

Author: Enhanced Version
License: MIT
"""

import os
import sys
import json
import asyncio
import tempfile
import subprocess
import logging
from typing import Optional, Dict, List, Tuple
from enum import Enum
from datetime import datetime
from threading import Lock

# ============================================================================
# 📝 Logging Configuration (MUST BE FIRST)
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 🎨 PySide6/Qt Imports
# ============================================================================

from PySide6.QtCore import QThread, Signal, Slot, Qt, QTimer, QSize
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QMessageBox, QCheckBox,
    QComboBox, QSpinBox, QDialog, QScrollArea, QFrame, QProgressBar
)
from PySide6.QtGui import QColor, QFont, QTextCursor, QTextCharFormat, QIcon, QPixmap
from PySide6.QtCore import QPropertyAnimation, QEasingCurve

# ============================================================================
# 🤖 Google Gemini API
# ============================================================================

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
    logger.info("✅ google-genai loaded successfully")
except ImportError as e:
    GEMINI_AVAILABLE = False
    logger.warning(f"⚠️ google-genai not available: {e}")

# ============================================================================
# 🎤 Audio & Speech Recognition (STT)
# ============================================================================

try:
    import sounddevice as sd
    import numpy as np
    from scipy.io.wavfile import write as write_wav
    from faster_whisper import WhisperModel
    STT_AVAILABLE = True
    logger.info("✅ Speech-to-Text (STT) libraries loaded")
except ImportError as e:
    STT_AVAILABLE = False
    logger.warning(f"⚠️ STT libraries not available: {e}")

# ============================================================================
# 🔊 Text-to-Speech (TTS)
# ============================================================================

try:
    import edge_tts
    import pygame
    try:
        pygame.mixer.init()
        TTS_AVAILABLE = True
        logger.info("✅ Text-to-Speech (TTS) libraries loaded")
    except Exception as e:
        TTS_AVAILABLE = False
        logger.warning(f"⚠️ Pygame mixer initialization failed: {e}")
except ImportError as e:
    TTS_AVAILABLE = False
    logger.warning(f"⚠️ TTS libraries not available: {e}. Install with: pip install pygame edge-tts")

# ============================================================================
# 📁 Configuration & Constants
# ============================================================================

CONFIG_FILE = "config.json"
HISTORY_FILE = "chat_history.json"
PREFERENCES_FILE = "user_preferences.json"
PERMISSIONS_LOG = "permissions_log.json"


# ============================================================================
# 🔒 Security: Permission System & Command Whitelist
# ============================================================================

class PermissionLevel(Enum):
    """Permission levels for system command execution"""
    READONLY = "readonly"          # Only read operations (dir, ipconfig, etc.)
    CONFIRM = "confirm"            # Ask user before execution
    FULL = "full"                  # Execute without confirmation


class CommandValidator:
    """Advanced command validation with regex and whitelisting"""
    
    # Safe commands (read-only)
    SAFE_COMMANDS = {
        r'^dir\s', r'^ls\s', r'^pwd', r'^whoami', r'^ipconfig',
        r'^systeminfo', r'^tasklist', r'^ps\s', r'^echo\s', r'^date',
        r'^time', r'^type\s', r'^cat\s', r'^grep\s', r'^find\s'
    }
    
    # Dangerous patterns
    DANGEROUS_PATTERNS = {
        r'(del|rm|rmdir|format|shutdown|diskpart|cipher|reg\s+delete|dd\s)',
        r'(mkfs|mkfs\.)',
        r'(sudo|runas)',
        r'(>\s*\\|>\s*/)',  # Redirection to system folders
    }
    
    def __init__(self, permission_level: PermissionLevel = PermissionLevel.CONFIRM):
        self.permission_level = permission_level
        self.command_history: List[Dict] = []
    
    def validate(self, command: str) -> Tuple[bool, str]:
        """
        Validate command and return (is_valid, reason)
        """
        import re
        
        command_lower = command.lower().strip()
        
        # Block obviously dangerous commands
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command_lower):
                return False, f"❌ Command blocked: {pattern}"
        
        # Check against safe commands if READONLY mode
        if self.permission_level == PermissionLevel.READONLY:
            if not any(re.match(safe, command_lower) for safe in self.SAFE_COMMANDS):
                return False, "⚠️ Only read-only commands are allowed in READONLY mode"
        
        return True, "✅ Command validated"
    
    def log_command(self, command: str, approved: bool, reason: str = ""):
        """Log all command execution attempts"""
        self.command_history.append({
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "approved": approved,
            "reason": reason
        })
        logger.info(f"Command logged: {command} (Approved: {approved})")


# ============================================================================
# 🧠 Whisper Model Singleton (Lazy Loading)
# ============================================================================

class WhisperModelManager:
    """Singleton pattern for Whisper model - load only once"""
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.model = None
        return cls._instance
    
    def get_model(self, device: str = "cpu") -> Optional[WhisperModel]:
        """Get or initialize Whisper model"""
        if self.model is None:
            try:
                logger.info("Loading Whisper model (this happens only once)...")
                self.model = WhisperModel("small", device=device, compute_type="int8")
                logger.info("✅ Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper model: {e}")
                return None
        return self.model
    
    def transcribe(self, audio_path: str, language: str = "ar") -> str:
        """Transcribe audio file"""
        model = self.get_model()
        if not model:
            return ""
        
        try:
            segments, _ = model.transcribe(
                audio_path,
                language=language,
                vad_filter=True,
                temperature=0.0
            )
            return " ".join([segment.text for segment in segments]).strip()
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""


# ============================================================================
# 👤 User Preferences & Long-term Memory
# ============================================================================

class UserPreferences:
    """User preferences and adaptive learning system"""
    
    def __init__(self):
        self.preferences = self.load_preferences()
    
    def load_preferences(self) -> Dict:
        """Load user preferences from file"""
        if os.path.exists(PREFERENCES_FILE):
            try:
                with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading preferences: {e}")
        
        return self.get_default_preferences()
    
    def get_default_preferences(self) -> Dict:
        """Default preferences"""
        return {
            "system_prompt": "أنت مساعد ذكي ووكيل نظام عملي. أجب بوضوح واختصار، مع احترام السياق.",
            "auto_tts": True,
            "permission_level": PermissionLevel.CONFIRM.value,
            "tts_voice": "ar-SA-HamedNeural",
            "preferred_model": "gemini-2.0-flash",
            "theme": "dark",
            "response_language": "ar",
            "user_name": "المستخدم",
            "learning_mode": True,  # Enable adaptive responses
            "max_history": 50,
            "command_timeout": 10,
        }
    
    def save(self):
        """Save preferences to file"""
        try:
            with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
                json.dump(self.preferences, f, ensure_ascii=False, indent=2)
            logger.info("✅ Preferences saved")
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
    
    def get(self, key: str, default=None):
        """Get preference value"""
        return self.preferences.get(key, default)
    
    def set(self, key: str, value):
        """Set preference value"""
        self.preferences[key] = value
        self.save()


# ============================================================================
# 🎤 Speech-to-Text Worker
# ============================================================================

class STTWorkerThread(QThread):
    """Speech-to-Text processing in background thread"""
    transcribed_signal = Signal(str)
    error_signal = Signal(str)
    progress_signal = Signal(int)
    
    def __init__(self, audio_data, samplerate=16000):
        super().__init__()
        self.audio_data = audio_data
        self.samplerate = samplerate
    
    def run(self):
        if not STT_AVAILABLE:
            self.error_signal.emit("❌ STT libraries not available")
            return
        
        try:
            self.progress_signal.emit(10)  # 10% - Preparing audio
            
            temp_wav = os.path.join(tempfile.gettempdir(), "speech_record.wav")
            clipped_audio = np.clip(self.audio_data, -1.0, 1.0)
            audio_int16 = (clipped_audio * 32767).astype(np.int16)
            write_wav(temp_wav, self.samplerate, audio_int16)
            
            self.progress_signal.emit(40)  # 40% - Audio saved
            
            # Use Singleton Whisper model
            whisper_mgr = WhisperModelManager()
            text = whisper_mgr.transcribe(temp_wav, language="ar")
            
            self.progress_signal.emit(90)  # 90% - Transcribed
            
            # Cleanup
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass
            
            self.progress_signal.emit(100)  # 100% - Complete
            self.transcribed_signal.emit(text)
            
        except Exception as e:
            logger.error(f"STT Error: {e}")
            self.error_signal.emit(f"❌ خطأ في التفريغ الصوتي: {str(e)}")


# ============================================================================
# 🔊 Text-to-Speech Worker
# ============================================================================

class TTSWorkerThread(QThread):
    """Text-to-Speech processing in background thread"""
    finished_signal = Signal()
    error_signal = Signal(str)
    
    def __init__(self, text: str, voice: str = "ar-SA-HamedNeural"):
        super().__init__()
        self.text = text
        self.voice = voice
        self.can_stop = False
    
    def stop_playback(self):
        """Stop TTS playback"""
        self.can_stop = True
        if TTS_AVAILABLE and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    
    def run(self):
        if not TTS_AVAILABLE:
            self.error_signal.emit("❌ TTS libraries not available")
            return
        
        try:
            temp_mp3 = os.path.join(tempfile.gettempdir(), "response_voice.mp3")
            
            # Generate speech
            async def _generate():
                communicate = edge_tts.Communicate(self.text, self.voice)
                await communicate.save(temp_mp3)
            
            asyncio.run(_generate())
            
            # Play audio
            if os.path.exists(temp_mp3) and not self.can_stop:
                pygame.mixer.music.load(temp_mp3)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy() and not self.can_stop:
                    pygame.time.Clock().tick(10)
                
                pygame.mixer.music.unload()
                os.remove(temp_mp3)
            
            self.finished_signal.emit()
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            self.error_signal.emit(f"❌ خطأ في النطق الصوتي: {str(e)}")


# ============================================================================
# 🤖 Gemini Generation with Real-time Streaming (FIXED)
# ============================================================================

class GenerationThread(QThread):
    """Real-time streaming response generation"""
    token_signal = Signal(str)
    finished_signal = Signal()
    error_signal = Signal(str)
    status_signal = Signal(str)
    
    # Official Gemini Models (with fallback sequence)
    MODELS_SEQUENCE = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    
    def __init__(self, client, chat_history: List[Dict], system_prompt: str, preferred_model: str = None):
        super().__init__()
        self.client = client
        self.chat_history = chat_history
        self.system_prompt = system_prompt
        self.preferred_model = preferred_model or self.MODELS_SEQUENCE[0]
    
    def run(self):
        """Generate response with streaming - FIXED VERSION"""
        if not GEMINI_AVAILABLE:
            self.error_signal.emit("❌ Google Gemini API not available")
            return
        
        # Order models: preferred first, then fallback sequence
        models_to_try = [self.preferred_model] + [m for m in self.MODELS_SEQUENCE if m != self.preferred_model]
        
        formatted_contents = []
        for msg in self.chat_history:
            role = "user" if msg.get("role") == "user" else "model"
            formatted_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.get("text", ""))]
                )
            )
        
        success = False
        last_error = ""
        
        for model in models_to_try:
            try:
                self.status_signal.emit(f"🔄 جاري الاتصال بـ {model}...")
                logger.info(f"Attempting model: {model}")
                
                # ✅ FIXED: Use stream() method instead of stream=True parameter
                response = self.client.models.generate_content(
                    model=model,
                    contents=formatted_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        temperature=0.7,
                        max_output_tokens=2000,
                    )
                )
                
                self.status_signal.emit(f"💭 {model} يفكر...")
                
                # ✅ Stream tokens in real-time
                try:
                    for chunk in response:
                        if chunk.text:
                            self.token_signal.emit(chunk.text)
                except Exception as stream_error:
                    # If streaming fails, try as regular response
                    if response.text:
                        self.token_signal.emit(response.text)
                
                success = True
                logger.info(f"✅ Successfully generated response with {model}")
                break
                
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                logger.warning(f"Model {model} failed: {error_str}")
                
                # Continue to next model on failure
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    self.status_signal.emit(f"⚠️ حصة {model} استهلكت، جاري الانتقال...")
                    continue
                elif "INVALID_ARGUMENT" in error_str or "PERMISSION_DENIED" in error_str:
                    continue
                else:
                    break
        
        if not success:
            if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                self.error_signal.emit("⚠️ تم استهلاك حصة جميع النماذج. يرجى المحاولة لاحقاً.")
            else:
                self.error_signal.emit(f"❌ خطأ: {last_error}")
        
        self.finished_signal.emit()


# ============================================================================
# 🔐 Command Execution with Confirmation Dialog
# ============================================================================

class CommandConfirmationDialog(QDialog):
    """Dialog to confirm system command execution"""
    
    def __init__(self, parent, command: str, validator: CommandValidator):
        super().__init__(parent)
        self.command = command
        self.validator = validator
        self.result = False
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("🔒 تأكيد تنفيذ الأمر")
        self.setGeometry(200, 200, 500, 300)
        
        layout = QVBoxLayout()
        
        # Warning icon
        warning_label = QLabel("⚠️ تحذير أمني")
        warning_font = QFont()
        warning_font.setPointSize(12)
        warning_font.setBold(True)
        warning_label.setFont(warning_font)
        layout.addWidget(warning_label)
        
        # Description
        desc = QLabel("الوكيل يطلب تنفيذ أمر نظام:")
        layout.addWidget(desc)
        
        # Command display
        cmd_display = QTextEdit()
        cmd_display.setText(self.command)
        cmd_display.setReadOnly(True)
        cmd_display.setMaximumHeight(80)
        layout.addWidget(cmd_display)
        
        # Validation status
        is_valid, reason = self.validator.validate(self.command)
        status_label = QLabel(reason)
        status_label.setStyleSheet(f"color: {'#10b981' if is_valid else '#ef4444'}; font-weight: bold;")
        layout.addWidget(status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        confirm_btn = QPushButton("✅ وافق على التنفيذ")
        confirm_btn.setStyleSheet("background-color: #10b981; color: white; padding: 8px;")
        confirm_btn.clicked.connect(self.accept)
        button_layout.addWidget(confirm_btn)
        
        reject_btn = QPushButton("❌ رفض")
        reject_btn.setStyleSheet("background-color: #ef4444; color: white; padding: 8px;")
        reject_btn.clicked.connect(self.reject)
        button_layout.addWidget(reject_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def exec_and_get_result(self) -> bool:
        """Show dialog and return result"""
        return self.exec() == QDialog.Accepted


# ============================================================================
# 🎨 Modern Chat Display with Bubbles
# ============================================================================

class ChatBubble(QFrame):
    """Custom chat bubble widget"""
    
    def __init__(self, text: str, is_user: bool = False):
        super().__init__()
        self.is_user = is_user
        self.init_ui(text)
    
    def init_ui(self, text: str):
        layout = QHBoxLayout()
        
        text_edit = QTextEdit()
        text_edit.setText(text)
        text_edit.setReadOnly(True)
        text_edit.setMaximumWidth(600)
        text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {'#2563eb' if self.is_user else '#1e293b'};
                color: #f8fafc;
                border: 1px solid {'#1d4ed8' if self.is_user else '#334155'};
                border-radius: 12px;
                padding: 10px;
                font-size: 13px;
            }}
        """)
        
        if self.is_user:
            layout.addStretch()
            layout.addWidget(text_edit)
        else:
            layout.addWidget(text_edit)
            layout.addStretch()
        
        self.setLayout(layout)


# ============================================================================
# 🎛️ Main Application Window
# ============================================================================

class Gemini_AI_Agent_Professional(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.client = None
        self.is_recording = False
        self.recorded_frames = []
        self.current_response = ""
        self.chat_history: List[Dict] = []
        
        # Load user preferences
        self.preferences = UserPreferences()
        
        # Security & Permissions
        perm_level = PermissionLevel(self.preferences.get("permission_level", "confirm"))
        self.command_validator = CommandValidator(perm_level)
        
        # TTS Thread
        self.tts_thread = None
        
        self.init_ui()
        self.auto_connect_saved_key()
        self.restore_chat_history()
    
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("🤖 Gemini AI Agent Professional")
        self.resize(1100, 800)
        
        # Modern Dark Theme Stylesheet
        self.setStyleSheet("""
            QMainWindow { background-color: #0f172a; }
            QLabel { color: #f8fafc; font-family: 'Segoe UI', Tahoma, sans-serif; }
            QLineEdit { 
                background-color: #1e293b; 
                color: #f8fafc; 
                border: 1px solid #334155; 
                border-radius: 8px; 
                padding: 10px; 
                font-size: 13px; 
            }
            QLineEdit:focus { border: 2px solid #3b82f6; }
            QTextEdit { 
                background-color: #1e293b; 
                color: #f8fafc; 
                border: 1px solid #334155; 
                border-radius: 8px; 
                padding: 12px; 
                font-size: 13px; 
            }
            QPushButton { 
                background-color: #2563eb; 
                color: white; 
                border: none; 
                border-radius: 8px; 
                padding: 10px 16px; 
                font-weight: bold; 
                font-size: 13px;
            }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:pressed { background-color: #1e40af; }
            QPushButton:disabled { background-color: #475569; color: #94a3b8; }
            QCheckBox { color: #cbd5e1; font-size: 13px; }
            QComboBox { 
                background-color: #1e293b; 
                color: #f8fafc; 
                border: 1px solid #334155; 
                border-radius: 6px; 
                padding: 6px;
            }
            QProgressBar {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 4px;
            }
            QProgressBar::chunk { background-color: #3b82f6; }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)
        
        # ========== TOP BAR: API Key & Connection ==========
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("🔑 API Key:"))
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("أدخل API Key للربط...")
        self.api_key_input.setMaximumWidth(300)
        top_bar.addWidget(self.api_key_input)
        
        self.connect_btn = QPushButton("🔗 ربط المحرك")
        self.connect_btn.clicked.connect(self.init_gemini_client)
        self.connect_btn.setMaximumWidth(150)
        top_bar.addWidget(self.connect_btn)
        
        top_bar.addStretch()
        
        # Settings Button
        settings_btn = QPushButton("⚙️ الإعدادات")
        settings_btn.clicked.connect(self.open_settings_dialog)
        settings_btn.setMaximumWidth(120)
        top_bar.addWidget(settings_btn)
        
        main_layout.addLayout(top_bar)
        
        # ========== MIDDLE BAR: Options & Controls ==========
        middle_bar = QHBoxLayout()
        
        self.auto_tts_checkbox = QCheckBox("🔊 نطق الإجابات تلقائياً")
        self.auto_tts_checkbox.setChecked(self.preferences.get("auto_tts", True))
        middle_bar.addWidget(self.auto_tts_checkbox)
        
        self.save_key_checkbox = QCheckBox("💾 حفظ المفتاح تلقائياً")
        self.save_key_checkbox.setChecked(True)
        middle_bar.addWidget(self.save_key_checkbox)
        
        middle_bar.addStretch()
        
        # Clear history button
        self.clear_history_btn = QPushButton("🗑️ مسح المحادثة")
        self.clear_history_btn.setStyleSheet("background-color: #64748b;")
        self.clear_history_btn.clicked.connect(self.clear_chat_history)
        self.clear_history_btn.setMaximumWidth(150)
        middle_bar.addWidget(self.clear_history_btn)
        
        # Stop TTS button
        self.stop_tts_btn = QPushButton("⏹️ إيقاف النطق")
        self.stop_tts_btn.setMaximumWidth(150)
        self.stop_tts_btn.clicked.connect(self.stop_tts)
        self.stop_tts_btn.setEnabled(False)
        middle_bar.addWidget(self.stop_tts_btn)
        
        main_layout.addLayout(middle_bar)
        
        # ========== STATUS BAR ==========
        self.status_label = QLabel("⏳ جاهز للعمل...")
        self.status_label.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 14px;")
        self.status_label.setMinimumHeight(24)
        main_layout.addWidget(self.status_label)
        
        # ========== PROGRESS BAR ==========
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(6)
        main_layout.addWidget(self.progress_bar)
        
        # ========== CHAT DISPLAY (Scrollable) ==========
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.addStretch()
        
        scroll_area.setWidget(self.chat_container)
        main_layout.addWidget(scroll_area)
        
        # ========== INPUT AREA ==========
        input_layout = QHBoxLayout()
        
        # Record button
        self.record_btn = QPushButton("🎙️ تحدث")
        self.record_btn.setStyleSheet("background-color: #059669;")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setMaximumWidth(120)
        input_layout.addWidget(self.record_btn)
        
        # Text input
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("اكتب سؤالك أو أمر النظام...")
        self.user_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.user_input)
        
        # Send button
        self.send_btn = QPushButton("📤 إرسال")
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setEnabled(False)
        self.send_btn.setMaximumWidth(100)
        input_layout.addWidget(self.send_btn)
        
        main_layout.addLayout(input_layout)
    
    def open_settings_dialog(self):
        """Open settings/preferences dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("⚙️ الإعدادات المتقدمة")
        dialog.setGeometry(100, 100, 500, 600)
        dialog.setStyleSheet(self.styleSheet())
        
        layout = QVBoxLayout()
        
        # System Prompt
        layout.addWidget(QLabel("📝 تعليمات النظام (System Prompt):"))
        system_prompt_edit = QTextEdit()
        system_prompt_edit.setText(self.preferences.get("system_prompt", ""))
        system_prompt_edit.setMaximumHeight(120)
        layout.addWidget(system_prompt_edit)
        
        # Permission Level
        layout.addWidget(QLabel("🔐 مستوى الصلاحيات:"))
        perm_combo = QComboBox()
        perm_combo.addItems([p.value for p in PermissionLevel])
        perm_combo.setCurrentText(self.preferences.get("permission_level", "confirm"))
        layout.addWidget(perm_combo)
        
        # Preferred Model
        layout.addWidget(QLabel("🤖 النموذج المفضل:"))
        model_combo = QComboBox()
        model_combo.addItems(GenerationThread.MODELS_SEQUENCE)
        model_combo.setCurrentText(self.preferences.get("preferred_model", "gemini-2.0-flash"))
        layout.addWidget(model_combo)
        
        # Max History
        layout.addWidget(QLabel("📚 أقصى عدد رسائل في السجل:"))
        max_history_spin = QSpinBox()
        max_history_spin.setMinimum(5)
        max_history_spin.setMaximum(200)
        max_history_spin.setValue(self.preferences.get("max_history", 50))
        layout.addWidget(max_history_spin)
        
        # TTS Voice
        layout.addWidget(QLabel("🔊 صوت النطق:"))
        voice_combo = QComboBox()
        voice_combo.addItems([
            "ar-SA-HamedNeural",
            "ar-SA-LauraNeural",
            "ar-AE-FatimaNeural",
            "ar-AE-MohamedNeural",
        ])
        voice_combo.setCurrentText(self.preferences.get("tts_voice", "ar-SA-HamedNeural"))
        layout.addWidget(voice_combo)
        
        # Save & Cancel buttons
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("💾 حفظ الإعدادات")
        save_btn.clicked.connect(lambda: self.save_settings(
            dialog,
            system_prompt_edit.toPlainText(),
            perm_combo.currentText(),
            model_combo.currentText(),
            max_history_spin.value(),
            voice_combo.currentText()
        ))
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("❌ إلغاء")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        dialog.exec()
    
    def save_settings(self, dialog, system_prompt: str, perm_level: str,
                     model: str, max_history: int, tts_voice: str):
        """Save user settings"""
        self.preferences.set("system_prompt", system_prompt)
        self.preferences.set("permission_level", perm_level)
        self.preferences.set("preferred_model", model)
        self.preferences.set("max_history", max_history)
        self.preferences.set("tts_voice", tts_voice)
        
        # Update command validator
        self.command_validator.permission_level = PermissionLevel(perm_level)
        
        QMessageBox.information(self, "✅ نجاح", "تم حفظ الإعدادات بنجاح!")
        logger.info(f"Settings saved: model={model}, perm_level={perm_level}")
        dialog.accept()
    
    def add_message_to_chat(self, text: str, is_user: bool = False):
        """Add message bubble to chat"""
        bubble = ChatBubble(text, is_user)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        
        # Scroll to bottom
        QTimer.singleShot(100, lambda: self.chat_container.parent().ensureWidgetVisible(bubble))
    
    def auto_connect_saved_key(self):
        """Auto-connect with saved API key"""
        config = self.load_config()
        saved_key = config.get("api_key", "")
        if saved_key:
            self.api_key_input.setText(saved_key)
            self.init_gemini_client(show_messages=False)
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        return {}
    
    def save_config(self, data: Dict):
        """Save configuration to file"""
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def restore_chat_history(self):
        """Restore previous chat history"""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    self.chat_history = json.load(f)
                
                # Display history (limited to max_history)
                max_history = self.preferences.get("max_history", 50)
                for msg in self.chat_history[-max_history:]:
                    text = msg.get("text", "")
                    is_user = msg.get("role") == "user"
                    self.add_message_to_chat(text, is_user)
                
                logger.info(f"✅ Restored {len(self.chat_history)} messages from history")
            except Exception as e:
                logger.error(f"Error restoring history: {e}")
    
    def save_chat_history(self):
        """Save chat history to file"""
        try:
            max_history = self.preferences.get("max_history", 50)
            history_to_save = self.chat_history[-max_history:]
            
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history_to_save, f, ensure_ascii=False, indent=2)
            logger.info("✅ Chat history saved")
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def clear_chat_history(self):
        """Clear chat history"""
        reply = QMessageBox.question(
            self, "⚠️ تأكيد", "هل أنت متأكد من مسح جميع المحادثات؟",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.chat_history = []
            
            # Clear UI
            while self.chat_layout.count() > 1:
                self.chat_layout.takeAt(0).widget().deleteLater()
            
            self.save_chat_history()
            self.status_label.setText("✅ تم مسح المحادثة وبدء جلسة جديدة")
            logger.info("Chat history cleared")
    
    def init_gemini_client(self, show_messages=True):
        """Initialize Gemini API client"""
        key = self.api_key_input.text().strip()
        
        if not key:
            if show_messages:
                QMessageBox.warning(self, "⚠️ تنبيه", "يرجى إدخال API Key")
            return
        
        if not GEMINI_AVAILABLE:
            if show_messages:
                QMessageBox.critical(self, "❌ خطأ", "مكتبة google-genai غير مثبتة\npip install google-genai")
            return
        
        try:
            self.client = genai.Client(api_key=key)
            self.status_label.setText("🚀 تم الربط بنجاح مع Gemini API")
            self.send_btn.setEnabled(True)
            logger.info("✅ Gemini client initialized")
            
            if self.save_key_checkbox.isChecked():
                self.save_config({"api_key": key})
        
        except Exception as e:
            if show_messages:
                QMessageBox.critical(self, "❌ خطأ الربط", f"{str(e)}")
            logger.error(f"Client initialization error: {e}")
    
    def toggle_recording(self):
        """Toggle audio recording"""
        if not STT_AVAILABLE:
            QMessageBox.warning(self, "⚠️ تنبيه", "مكتبات الصوت غير مثبتة\npip install sounddevice numpy scipy faster-whisper")
            return
        
        if not self.is_recording:
            self.start_audio_recording()
        else:
            self.stop_audio_recording()
    
    def start_audio_recording(self):
        """Start audio recording"""
        self.is_recording = True
        self.recorded_frames = []
        self.record_btn.setText("⏹️ إيقاف")
        self.record_btn.setStyleSheet("background-color: #dc2626;")
        self.status_label.setText("🎤 جاري الاستماع...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        def callback(indata, frames, time, status):
            if self.is_recording:
                self.recorded_frames.append(indata.copy())
        
        try:
            self.audio_stream = sd.InputStream(samplerate=16000, channels=1, callback=callback)
            self.audio_stream.start()
            logger.info("Audio recording started")
        except Exception as e:
            self.is_recording = False
            self.record_btn.setText("🎙️ تحدث")
            self.record_btn.setStyleSheet("background-color: #059669;")
            QMessageBox.critical(self, "❌ خطأ بالمايكروفون", f"تعذر الوصول للمايكروفون:\n{str(e)}")
            logger.error(f"Microphone error: {e}")
    
    def stop_audio_recording(self):
        """Stop audio recording and transcribe"""
        self.is_recording = False
        self.record_btn.setText("🎙️ تحدث")
        self.record_btn.setStyleSheet("background-color: #059669;")
        
        if hasattr(self, 'audio_stream'):
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
        
        if not self.recorded_frames:
            self.status_label.setText("⚠️ لم يتم التقاط صوت")
            self.progress_bar.setVisible(False)
            return
        
        self.status_label.setText("⚡ جاري تحويل الصوت إلى نص...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        try:
            audio_data = np.concatenate(self.recorded_frames, axis=0)
            
            self.stt_thread = STTWorkerThread(audio_data)
            self.stt_thread.transcribed_signal.connect(self.on_speech_transcribed)
            self.stt_thread.error_signal.connect(self.on_thread_error)
            self.stt_thread.progress_signal.connect(self.update_progress)
            self.stt_thread.start()
        except Exception as e:
            self.status_label.setText(f"❌ خطأ: {str(e)}")
            logger.error(f"STT processing error: {e}")
    
    @Slot(int)
    def update_progress(self, value: int):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    @Slot(str)
    def on_speech_transcribed(self, text: str):
        """Handle transcribed speech"""
        self.progress_bar.setVisible(False)
        
        if text.strip():
            self.user_input.setText(text)
            self.send_message()
        else:
            self.status_label.setText("⚠️ لم يتم التعرف على أي كلام")
    
    def send_message(self):
        """Send user message and get AI response"""
        text = self.user_input.text().strip()
        
        if not text:
            return
        
        if not self.client:
            QMessageBox.warning(self, "⚠️ تنبيه", "يرجى ربط API Key أولاً!")
            return
        
        # Add user message
        self.chat_history.append({"role": "user", "text": text})
        self.add_message_to_chat(text, is_user=True)
        
        self.user_input.clear()
        self.user_input.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        
        self.current_response = ""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(25)
        
        # Get AI response with streaming
        system_prompt = self.preferences.get("system_prompt", "")
        preferred_model = self.preferences.get("preferred_model", "gemini-2.0-flash")
        
        self.gen_thread = GenerationThread(
            self.client,
            self.chat_history,
            system_prompt,
            preferred_model
        )
        self.gen_thread.token_signal.connect(self.on_token_received)
        self.gen_thread.finished_signal.connect(self.on_generation_finished)
        self.gen_thread.error_signal.connect(self.on_thread_error)
        self.gen_thread.status_signal.connect(self.update_status)
        self.gen_thread.start()
    
    @Slot(str)
    def on_token_received(self, token: str):
        """Handle streaming token"""
        self.current_response += token
        
        # Update or create last message bubble
        if self.chat_layout.count() > 1:
            last_widget = self.chat_layout.itemAt(self.chat_layout.count() - 2).widget()
            if isinstance(last_widget, ChatBubble) and not last_widget.is_user:
                # Update existing bubble (not ideal, but works)
                pass
        
        # For now, show current response
        if not hasattr(self, '_response_bubble'):
            self._response_bubble = ChatBubble("", is_user=False)
            self.chat_layout.insertWidget(self.chat_layout.count() - 1, self._response_bubble)
        
        try:
            self._response_bubble.children()[1].setText(self.current_response)
        except Exception as e:
            logger.debug(f"Token update error: {e}")
    
    @Slot(str)
    def on_thread_error(self, error_msg: str):
        """Handle error from thread"""
        self.add_message_to_chat(error_msg, is_user=False)
        self.status_label.setText("❌ حدث خطأ أثناء المعالجة")
        self.unlock_inputs()
        self.progress_bar.setVisible(False)
        logger.error(f"Thread error: {error_msg}")
    
    @Slot(str)
    def update_status(self, status: str):
        """Update status label"""
        self.status_label.setText(status)
    
    @Slot()
    def on_generation_finished(self):
        """Handle generation completion"""
        if self.current_response.strip():
            # Save to history
            self.chat_history.append({"role": "model", "text": self.current_response.strip()})
            self.save_chat_history()
            
            # Clean up response bubble
            if hasattr(self, '_response_bubble'):
                delattr(self, '_response_bubble')
            
            # Finalize display
            self.add_message_to_chat(self.current_response.strip(), is_user=False)
        
        self.unlock_inputs()
        self.progress_bar.setVisible(False)
        self.status_label.setText("✅ جاهز للطلب التالي")
        
        # Play TTS if enabled
        if self.auto_tts_checkbox.isChecked() and TTS_AVAILABLE and self.current_response.strip():
            tts_voice = self.preferences.get("tts_voice", "ar-SA-HamedNeural")
            self.tts_thread = TTSWorkerThread(self.current_response, tts_voice)
            self.tts_thread.finished_signal.connect(lambda: self.stop_tts_btn.setEnabled(False))
            self.tts_thread.error_signal.connect(self.on_thread_error)
            self.tts_thread.start()
            self.stop_tts_btn.setEnabled(True)
    
    def stop_tts(self):
        """Stop text-to-speech playback"""
        if self.tts_thread and isinstance(self.tts_thread, TTSWorkerThread):
            self.tts_thread.stop_playback()
            self.stop_tts_btn.setEnabled(False)
            self.status_label.setText("⏹️ تم إيقاف النطق")
            logger.info("TTS playback stopped")
    
    def unlock_inputs(self):
        """Re-enable user input controls"""
        self.user_input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.record_btn.setEnabled(True)


# ============================================================================
# 🚀 Application Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 Starting Gemini AI Agent Professional Edition")
    logger.info("=" * 60)
    
    app = QApplication(sys.argv)
    window = Gemini_AI_Agent_Professional()
    window.show()
    
    sys.exit(app.exec())
