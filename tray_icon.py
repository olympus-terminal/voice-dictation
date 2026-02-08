"""
System tray icon for Voxtral dictation.
Provides visual status indicator and quick controls.
"""
import threading
from typing import Optional, Callable
from enum import Enum

try:
    import pystray
    from PIL import Image, ImageDraw
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False


class TrayStatus(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"


class TrayIcon:
    """
    System tray icon with status indicator.
    """

    def __init__(
        self,
        on_quit: Optional[Callable[[], None]] = None,
        on_toggle: Optional[Callable[[], None]] = None,
    ):
        if not TRAY_AVAILABLE:
            raise RuntimeError("pystray not available. Install with: pip install pystray Pillow")

        self.on_quit = on_quit
        self.on_toggle = on_toggle
        self._status = TrayStatus.IDLE
        self._icon: Optional[pystray.Icon] = None
        self._thread: Optional[threading.Thread] = None
        self._audio_level: float = 0.0

        # Colors
        self._colors = {
            TrayStatus.IDLE: "#666666",       # Gray
            TrayStatus.RECORDING: "#ff4444",  # Red
            TrayStatus.PROCESSING: "#44ff44", # Green
            TrayStatus.ERROR: "#ffaa00",      # Orange
        }

    def _create_icon_image(self, status: TrayStatus, level: float = 0.0) -> Image.Image:
        """Create icon image based on status."""
        size = 64
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Main circle color based on status
        color = self._colors[status]

        # Draw outer circle
        margin = 4
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=color,
            outline="#ffffff",
            width=2,
        )

        # Draw audio level indicator (inner arc) when recording
        if status == TrayStatus.RECORDING and level > 0:
            inner_margin = 16
            # Draw level as filled portion
            level_height = int((size - 2 * inner_margin) * level)
            if level_height > 0:
                draw.rectangle(
                    [
                        inner_margin,
                        size - inner_margin - level_height,
                        size - inner_margin,
                        size - inner_margin,
                    ],
                    fill="#ffffff",
                )

        # Draw microphone icon in center
        mic_color = "#ffffff" if status != TrayStatus.IDLE else "#cccccc"
        center = size // 2

        # Microphone body
        mic_width = 8
        mic_height = 16
        draw.rounded_rectangle(
            [
                center - mic_width // 2,
                center - mic_height // 2 - 4,
                center + mic_width // 2,
                center + mic_height // 2 - 4,
            ],
            radius=4,
            fill=mic_color,
        )

        # Microphone stand
        draw.arc(
            [
                center - 10,
                center - 6,
                center + 10,
                center + 14,
            ],
            start=0,
            end=180,
            fill=mic_color,
            width=2,
        )
        draw.line(
            [center, center + 14, center, center + 20],
            fill=mic_color,
            width=2,
        )

        return image

    def _create_menu(self) -> pystray.Menu:
        """Create context menu."""
        return pystray.Menu(
            pystray.MenuItem(
                "Toggle Recording",
                self._menu_toggle,
                default=True,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Status",
                pystray.Menu(
                    pystray.MenuItem(
                        lambda item: f"● {self._status.value.title()}",
                        None,
                        enabled=False,
                    ),
                ),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._menu_quit),
        )

    def _menu_toggle(self, icon, item):
        """Handle toggle menu item."""
        if self.on_toggle:
            self.on_toggle()

    def _menu_quit(self, icon, item):
        """Handle quit menu item."""
        self.stop()
        if self.on_quit:
            self.on_quit()

    def start(self):
        """Start tray icon in background thread."""
        if not TRAY_AVAILABLE:
            return

        self._icon = pystray.Icon(
            name="voxtral-dictation",
            icon=self._create_icon_image(TrayStatus.IDLE),
            title="Voxtral Dictation",
            menu=self._create_menu(),
        )

        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop tray icon."""
        if self._icon:
            self._icon.stop()
            self._icon = None

    def set_status(self, status: TrayStatus):
        """Update status and refresh icon."""
        self._status = status
        if self._icon:
            self._icon.icon = self._create_icon_image(status, self._audio_level)
            self._icon.title = f"Voxtral Dictation - {status.value.title()}"

    def set_audio_level(self, level: float):
        """Update audio level indicator (0.0 to 1.0)."""
        self._audio_level = min(1.0, max(0.0, level))
        if self._icon and self._status == TrayStatus.RECORDING:
            self._icon.icon = self._create_icon_image(self._status, self._audio_level)


class ConsoleIndicator:
    """
    Fallback console-based status indicator.
    Used when system tray is not available.
    """

    def __init__(self):
        self._status = TrayStatus.IDLE
        self._audio_level = 0.0

    def start(self):
        pass

    def stop(self):
        pass

    def set_status(self, status: TrayStatus):
        self._status = status
        symbols = {
            TrayStatus.IDLE: "⚪",
            TrayStatus.RECORDING: "🔴",
            TrayStatus.PROCESSING: "🟢",
            TrayStatus.ERROR: "🟠",
        }
        print(f"\r{symbols[status]} {status.value.title()}", end="", flush=True)

    def set_audio_level(self, level: float):
        self._audio_level = level
        if self._status == TrayStatus.RECORDING:
            bars = int(level * 20)
            print(f"\r🔴 Recording [{'█' * bars}{' ' * (20 - bars)}]", end="", flush=True)


def create_indicator(use_tray: bool = True):
    """
    Create appropriate status indicator.

    Args:
        use_tray: If True, try to use system tray. Falls back to console.

    Returns:
        TrayIcon or ConsoleIndicator
    """
    if use_tray and TRAY_AVAILABLE:
        try:
            return TrayIcon()
        except Exception as e:
            print(f"Could not create tray icon: {e}")

    return ConsoleIndicator()


def test_tray():
    """Test tray icon functionality."""
    import time

    print("Testing tray icon...")

    indicator = create_indicator(use_tray=True)
    indicator.start()

    print("Tray icon should be visible. Testing status changes...")

    time.sleep(2)
    indicator.set_status(TrayStatus.RECORDING)
    print("Status: RECORDING")

    for i in range(20):
        indicator.set_audio_level(i / 20)
        time.sleep(0.1)

    time.sleep(1)
    indicator.set_status(TrayStatus.PROCESSING)
    print("Status: PROCESSING")

    time.sleep(2)
    indicator.set_status(TrayStatus.IDLE)
    print("Status: IDLE")

    time.sleep(2)
    indicator.stop()
    print("Test complete!")


if __name__ == "__main__":
    test_tray()
