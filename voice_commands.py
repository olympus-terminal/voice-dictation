"""
Voice command recognition for hands-free dictation.
Detects special phrases and executes actions instead of typing them.
"""
import re
from typing import Optional, Callable, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class CommandAction(Enum):
    """Actions that can be triggered by voice commands."""
    DELETE_ALL = "delete_all"
    DELETE_WORD = "delete_word"
    DELETE_LINE = "delete_line"
    SEND = "send"
    ENTER = "enter"
    NEW_LINE = "new_line"
    NEW_PARAGRAPH = "new_paragraph"
    TAB = "tab"
    UNDO = "undo"
    REDO = "redo"
    SELECT_ALL = "select_all"
    COPY = "copy"
    PASTE = "paste"
    CUT = "cut"
    SAVE = "save"
    ESCAPE = "escape"
    # Punctuation
    PERIOD = "period"
    COMMA = "comma"
    QUESTION = "question"
    EXCLAMATION = "exclamation"
    COLON = "colon"
    SEMICOLON = "semicolon"
    # Formatting
    CAPS_ON = "caps_on"
    CAPS_OFF = "caps_off"
    # Mode control
    STOP_LISTENING = "stop_listening"
    PAUSE = "pause"


@dataclass
class VoiceCommand:
    """A recognized voice command."""
    action: CommandAction
    original_text: str
    remaining_text: str  # Text after command is removed


# Wake word required before commands (set to None to disable)
# This prevents accidental command triggers
WAKE_WORD = "kimmy"

# Command patterns - order matters (longer/more specific first)
# These will only trigger if preceded by the wake word (if set)
COMMAND_PATTERNS: Dict[str, CommandAction] = {
    # Deletion
    r"delete all": CommandAction.DELETE_ALL,
    r"delete everything": CommandAction.DELETE_ALL,
    r"clear all": CommandAction.DELETE_ALL,
    r"erase all": CommandAction.DELETE_ALL,
    r"delete that": CommandAction.DELETE_LINE,
    r"delete line": CommandAction.DELETE_LINE,
    r"delete word": CommandAction.DELETE_WORD,
    r"scratch that": CommandAction.DELETE_LINE,
    r"undo that": CommandAction.UNDO,
    r"undo": CommandAction.UNDO,

    # Navigation/Actions
    r"send message": CommandAction.SEND,
    r"send it": CommandAction.SEND,
    r"press enter": CommandAction.ENTER,
    r"new line": CommandAction.NEW_LINE,
    r"next line": CommandAction.NEW_LINE,
    r"new paragraph": CommandAction.NEW_PARAGRAPH,
    r"press tab": CommandAction.TAB,

    # Editing
    r"redo": CommandAction.REDO,
    r"select all": CommandAction.SELECT_ALL,
    r"copy that": CommandAction.COPY,
    r"paste": CommandAction.PASTE,
    r"cut that": CommandAction.CUT,
    r"save file": CommandAction.SAVE,
    r"escape": CommandAction.ESCAPE,

    # Mode control
    r"stop listening": CommandAction.STOP_LISTENING,
    r"pause listening": CommandAction.PAUSE,
}

# Punctuation commands - these DON'T require wake word (commonly used)
# Set to empty dict {} to disable punctuation commands entirely
PUNCTUATION_PATTERNS: Dict[str, CommandAction] = {
    # Disabled by default - too easy to trigger accidentally
    # Uncomment to enable:
    # r"period$": CommandAction.PERIOD,
    # r"comma$": CommandAction.COMMA,
    # r"question mark$": CommandAction.QUESTION,
}


class VoiceCommandProcessor:
    """
    Processes transcribed text to detect and handle voice commands.

    Commands require a wake word (default: "computer") to trigger.
    Example: "computer delete all" will delete, but "delete all" will be typed.
    """

    def __init__(self, wake_word: Optional[str] = WAKE_WORD):
        self.wake_word = wake_word.lower() if wake_word else None

        # Compile command patterns
        self._patterns: list[Tuple[re.Pattern, CommandAction]] = []
        for pattern, action in COMMAND_PATTERNS.items():
            if self.wake_word:
                # Require wake word before command
                full_pattern = self.wake_word + r'\s+' + pattern
            else:
                full_pattern = pattern
            regex = re.compile(r'\b' + full_pattern + r'\b', re.IGNORECASE)
            self._patterns.append((regex, action))

        # Punctuation patterns (no wake word needed)
        self._punct_patterns: list[Tuple[re.Pattern, CommandAction]] = []
        for pattern, action in PUNCTUATION_PATTERNS.items():
            regex = re.compile(r'\b' + pattern + r'\b', re.IGNORECASE)
            self._punct_patterns.append((regex, action))

        # State
        self._caps_mode = False
        self._last_typed_text = ""
        self._history: list[str] = []  # For undo support
        self._max_history = 20

    def process(self, text: str) -> Tuple[Optional[VoiceCommand], str]:
        """
        Process transcribed text for commands.

        Commands only trigger if preceded by wake word (e.g., "computer delete all").
        Regular text like "delete all the files" will be typed normally.

        Returns:
            Tuple of (command if found, remaining text to type)
        """
        if not text:
            return None, ""

        text_lower = text.lower().strip()

        # Check for wake word + command patterns
        for pattern, action in self._patterns:
            match = pattern.search(text_lower)
            if match:
                # Found a command with wake word
                start, end = match.span()
                remaining = text[:start] + text[end:]
                remaining = remaining.strip()

                command = VoiceCommand(
                    action=action,
                    original_text=text,
                    remaining_text=remaining,
                )
                return command, remaining

        # Check punctuation patterns (no wake word needed)
        for pattern, action in self._punct_patterns:
            match = pattern.search(text_lower)
            if match:
                start, end = match.span()
                remaining = text[:start] + text[end:]
                remaining = remaining.strip()

                command = VoiceCommand(
                    action=action,
                    original_text=text,
                    remaining_text=remaining,
                )
                return command, remaining

        # No command found, return text as-is
        return None, text

    def add_to_history(self, text: str):
        """Add typed text to history for undo support."""
        if text:
            self._history.append(text)
            if len(self._history) > self._max_history:
                self._history.pop(0)

    def get_last_typed_length(self) -> int:
        """Get length of last typed text (for undo)."""
        if self._history:
            return len(self._history[-1])
        return 0

    def pop_history(self) -> Optional[str]:
        """Remove and return last typed text from history."""
        if self._history:
            return self._history.pop()
        return None

    def apply_formatting(self, text: str) -> str:
        """Apply current formatting state to text."""
        if self._caps_mode:
            return text.upper()
        return text

    def set_caps_mode(self, enabled: bool):
        """Set caps lock mode."""
        self._caps_mode = enabled

    def set_last_typed(self, text: str):
        """Remember last typed text for 'delete that' etc."""
        self._last_typed_text = text

    def get_last_typed(self) -> str:
        """Get last typed text."""
        return self._last_typed_text


class CommandExecutor:
    """
    Executes voice commands using system tools.
    """

    def __init__(self, text_output):
        """
        Args:
            text_output: TextOutput instance for typing/clipboard
        """
        self.text_output = text_output
        self._last_typed_length = 0

    def execute(self, command: VoiceCommand) -> bool:
        """
        Execute a voice command.

        Returns:
            True if command was handled
        """
        action = command.action

        # Deletion commands
        if action == CommandAction.DELETE_ALL:
            # Select all and delete
            self.text_output.press_key("ctrl+a")
            self.text_output.press_key("Delete")
            print(" [DELETE ALL]")
            return True

        elif action == CommandAction.DELETE_LINE:
            # Delete current line
            self.text_output.press_key("Home")
            self.text_output.press_key("shift+End")
            self.text_output.press_key("Delete")
            print(" [DELETE LINE]")
            return True

        elif action == CommandAction.DELETE_WORD:
            # Delete last word (ctrl+w is "delete word back" in terminal/readline)
            self.text_output.press_key("ctrl+w")
            print(" [DELETE WORD]")
            return True

        # Send/Enter
        elif action == CommandAction.SEND:
            self.text_output.press_key("Return")
            print(" [SEND]")
            return True

        elif action == CommandAction.ENTER:
            self.text_output.press_key("Return")
            print(" [ENTER]")
            return True

        elif action == CommandAction.NEW_LINE:
            self.text_output.type_text("\n")
            print(" [NEW LINE]")
            return True

        elif action == CommandAction.NEW_PARAGRAPH:
            self.text_output.type_text("\n\n")
            print(" [NEW PARAGRAPH]")
            return True

        elif action == CommandAction.TAB:
            self.text_output.press_key("Tab")
            print(" [TAB]")
            return True

        # Editing (ctrl+shift for terminal compatibility)
        elif action == CommandAction.UNDO:
            self.text_output.press_key("ctrl+shift+z")
            print(" [UNDO]")
            return True

        elif action == CommandAction.REDO:
            self.text_output.press_key("ctrl+shift+y")
            print(" [REDO]")
            return True

        elif action == CommandAction.SELECT_ALL:
            self.text_output.press_key("ctrl+shift+a")
            print(" [SELECT ALL]")
            return True

        elif action == CommandAction.COPY:
            self.text_output.press_key("ctrl+shift+c")
            print(" [COPY]")
            return True

        elif action == CommandAction.PASTE:
            self.text_output.press_key("ctrl+shift+v")
            print(" [PASTE]")
            return True

        elif action == CommandAction.CUT:
            self.text_output.press_key("ctrl+shift+x")
            print(" [CUT]")
            return True

        elif action == CommandAction.SAVE:
            self.text_output.press_key("ctrl+s")
            print(" [SAVE]")
            return True

        elif action == CommandAction.ESCAPE:
            self.text_output.press_key("Escape")
            print(" [ESCAPE]")
            return True

        # Punctuation
        elif action == CommandAction.PERIOD:
            self.text_output.type_text(".")
            return True

        elif action == CommandAction.COMMA:
            self.text_output.type_text(",")
            return True

        elif action == CommandAction.QUESTION:
            self.text_output.type_text("?")
            return True

        elif action == CommandAction.EXCLAMATION:
            self.text_output.type_text("!")
            return True

        elif action == CommandAction.COLON:
            self.text_output.type_text(":")
            return True

        elif action == CommandAction.SEMICOLON:
            self.text_output.type_text(";")
            return True

        return False


def test_commands():
    """Test voice command recognition."""
    processor = VoiceCommandProcessor()

    print("=== Voice Command Test ===")
    print(f"Wake word: '{processor.wake_word}'\n")

    test_phrases = [
        # These should NOT trigger (no wake word)
        "hello world",
        "delete all",
        "please delete all the text",
        "send message",
        "I want to undo that decision",
        # These SHOULD trigger (with wake word)
        "computer delete all",
        "computer undo",
        "computer send message",
        "computer new line",
        "hey computer stop listening",
        "computer scratch that I made a mistake",
    ]

    print("Testing phrases:\n")
    for phrase in test_phrases:
        cmd, remaining = processor.process(phrase)
        if cmd:
            print(f"'{phrase}'")
            print(f"  -> COMMAND: {cmd.action.value}")
            if remaining:
                print(f"  -> remaining text: '{remaining}'")
        else:
            print(f"'{phrase}' -> (type as-is)")
        print()


if __name__ == "__main__":
    test_commands()
