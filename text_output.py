"""
Text output module for Voxtral dictation.
Handles typing text into active window (xdotool) and clipboard operations.
"""
import subprocess
import shutil
from typing import Optional
from enum import Enum


class OutputMode(Enum):
    TYPE = "type"           # Type directly into active window
    CLIPBOARD = "clipboard" # Copy to clipboard only
    BOTH = "both"           # Type and copy to clipboard


class TextOutput:
    """
    Outputs transcribed text to the system.
    Supports typing via xdotool and clipboard via xclip.
    """

    def __init__(
        self,
        mode: OutputMode = OutputMode.TYPE,
        typing_delay_ms: int = 0,
    ):
        self.mode = mode
        self.typing_delay_ms = typing_delay_ms

        # Check for required tools
        self._xdotool = shutil.which("xdotool")
        self._xclip = shutil.which("xclip")

        if mode in (OutputMode.TYPE, OutputMode.BOTH) and not self._xdotool:
            print("Warning: xdotool not found. Install with: sudo apt install xdotool")

        if mode in (OutputMode.CLIPBOARD, OutputMode.BOTH) and not self._xclip:
            print("Warning: xclip not found. Install with: sudo apt install xclip")

    def type_text(self, text: str) -> bool:
        """
        Type text into the currently focused window using xdotool.
        Returns True on success.
        """
        if not self._xdotool:
            print("xdotool not available")
            return False

        if not text:
            return True

        try:
            cmd = ["xdotool", "type", "--clearmodifiers"]

            if self.typing_delay_ms > 0:
                cmd.extend(["--delay", str(self.typing_delay_ms)])

            cmd.append(text)

            subprocess.run(cmd, check=True, timeout=10)
            return True
        except subprocess.TimeoutExpired:
            print("xdotool timed out")
            return False
        except subprocess.CalledProcessError as e:
            print(f"xdotool error: {e}")
            return False

    def copy_to_clipboard(self, text: str) -> bool:
        """
        Copy text to system clipboard using xclip.
        Returns True on success.
        """
        if not self._xclip:
            print("xclip not available")
            return False

        if not text:
            return True

        try:
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode("utf-8"),
                check=True,
                timeout=5,
            )
            return True
        except subprocess.TimeoutExpired:
            print("xclip timed out")
            return False
        except subprocess.CalledProcessError as e:
            print(f"xclip error: {e}")
            return False

    def output(self, text: str) -> bool:
        """
        Output text according to configured mode.
        Returns True if all operations succeeded.
        """
        success = True

        if self.mode in (OutputMode.TYPE, OutputMode.BOTH):
            success = self.type_text(text) and success

        if self.mode in (OutputMode.CLIPBOARD, OutputMode.BOTH):
            success = self.copy_to_clipboard(text) and success

        return success

    def output_incremental(self, text: str, previous_text: str = "") -> bool:
        """
        Output only the new portion of text (for streaming transcription).
        Useful when Voxtral sends updated transcription as it processes.
        """
        if text.startswith(previous_text):
            new_text = text[len(previous_text):]
            if new_text:
                return self.output(new_text)
            return True
        else:
            # Text was revised, need to handle differently
            # For now, just output the full new text
            # A more sophisticated approach would backspace and retype
            return self.output(text)

    def backspace(self, count: int = 1) -> bool:
        """Send backspace keys to delete characters."""
        if not self._xdotool:
            return False

        try:
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers"] + ["BackSpace"] * count,
                check=True,
                timeout=5,
            )
            return True
        except Exception as e:
            print(f"Backspace error: {e}")
            return False

    def press_key(self, key: str) -> bool:
        """Press a specific key (e.g., 'Return', 'Tab')."""
        if not self._xdotool:
            return False

        try:
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", key],
                check=True,
                timeout=5,
            )
            return True
        except Exception as e:
            print(f"Key press error: {e}")
            return False


class TextProcessor:
    """
    Processes transcribed text before output.
    Handles capitalization, punctuation, and formatting.
    """

    def __init__(
        self,
        auto_capitalize: bool = True,
        auto_punctuation: bool = True,
    ):
        self.auto_capitalize = auto_capitalize
        self.auto_punctuation = auto_punctuation
        self._sentence_start = True

    def process(self, text: str) -> str:
        """Process text with configured transformations."""
        if not text:
            return text

        result = text

        # Capitalize first letter of sentences
        if self.auto_capitalize and self._sentence_start:
            result = result[0].upper() + result[1:] if len(result) > 0 else result

        # Update sentence tracking
        if result:
            # Check if we end with sentence-ending punctuation
            self._sentence_start = result.rstrip().endswith(('.', '!', '?'))

        return result

    def reset(self) -> None:
        """Reset processor state (e.g., for new dictation session)."""
        self._sentence_start = True


def test_output():
    """Test text output functionality."""
    print("=== Text Output Test ===\n")

    output = TextOutput(mode=OutputMode.CLIPBOARD)

    print("Testing clipboard...")
    if output.copy_to_clipboard("Hello from Voxtral dictation!"):
        print("✓ Text copied to clipboard")
    else:
        print("✗ Clipboard copy failed")

    print("\nTo test typing, focus a text field and run:")
    print("  python -c \"from text_output import TextOutput, OutputMode; TextOutput(OutputMode.TYPE).type_text('Test typing')\"")


if __name__ == "__main__":
    test_output()
