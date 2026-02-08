#!/usr/bin/env python3
"""
Simple CLI Voice Recorder for TTS sample collection.
Supports the RØDE NT-USB+ and other audio devices.
"""

import argparse
import subprocess
import sys
import os
import signal
from datetime import datetime
from pathlib import Path

# Default settings optimized for TTS
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 1  # Mono for TTS
DEFAULT_FORMAT = "wav"
DEFAULT_DEVICE = "hw:2,0"  # RØDE NT-USB+


def list_devices():
    """List available recording devices."""
    print("Available recording devices:\n")
    result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
    print(result.stdout)

    print("\nDevice format: hw:CARD,DEVICE")
    print("  - hw:0,0 = HDA Intel PCH (built-in)")
    print("  - hw:2,0 = RØDE NT-USB+")


def generate_filename(output_dir: Path, prefix: str, format: str) -> Path:
    """Generate a timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.{format}"
    return output_dir / filename


def record_audio(
    output_file: Path,
    device: str = DEFAULT_DEVICE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    duration: int = None,
):
    """Record audio using arecord."""

    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",  # 16-bit signed little-endian
        "-r", str(sample_rate),
        "-c", str(channels),
        "-t", "wav",
    ]

    if duration:
        cmd.extend(["-d", str(duration)])

    cmd.append(str(output_file))

    print(f"\nRecording to: {output_file}")
    print(f"Device: {device}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    if duration:
        print(f"Duration: {duration} seconds")
    else:
        print("Duration: Press Ctrl+C to stop")
    print("\n" + "=" * 50)
    print("RECORDING... (Ctrl+C to stop)")
    print("=" * 50 + "\n")

    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        process.wait()
        print("\n\nRecording stopped.")

    if output_file.exists():
        size = output_file.stat().st_size
        print(f"\nSaved: {output_file}")
        print(f"Size: {size / 1024:.1f} KB")
        return True
    return False


def record_with_sox(
    output_file: Path,
    device: str = DEFAULT_DEVICE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    duration: int = None,
):
    """Record audio using sox (rec command) with level meter."""

    # Set AUDIODEV environment variable for sox
    env = os.environ.copy()
    env["AUDIODEV"] = device

    cmd = [
        "rec",
        "-r", str(sample_rate),
        "-c", str(channels),
        "-b", "16",  # 16-bit
        str(output_file),
    ]

    if duration:
        cmd.extend(["trim", "0", str(duration)])

    print(f"\nRecording to: {output_file}")
    print(f"Device: {device}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    if duration:
        print(f"Duration: {duration} seconds")
    else:
        print("Duration: Press Ctrl+C to stop")
    print("\n" + "=" * 50)
    print("RECORDING... (Ctrl+C to stop)")
    print("=" * 50 + "\n")

    try:
        process = subprocess.Popen(cmd, env=env)
        process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        process.wait()
        print("\n\nRecording stopped.")

    if output_file.exists():
        size = output_file.stat().st_size
        print(f"\nSaved: {output_file}")
        print(f"Size: {size / 1024:.1f} KB")
        return True
    return False


def interactive_mode(output_dir: Path, device: str, sample_rate: int, channels: int):
    """Interactive recording session for multiple takes."""

    print("\n" + "=" * 50)
    print("INTERACTIVE RECORDING MODE")
    print("=" * 50)
    print(f"\nOutput directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Sample rate: {sample_rate} Hz")
    print("\nCommands:")
    print("  [Enter]  - Start new recording")
    print("  l        - List recordings")
    print("  p <num>  - Play recording number")
    print("  d <num>  - Delete recording number")
    print("  q        - Quit")

    recordings = []
    take_num = 1

    while True:
        try:
            cmd = input(f"\n[Take {take_num}] > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if cmd == "q":
            break
        elif cmd == "l":
            if not recordings:
                print("No recordings yet.")
            else:
                print("\nRecordings:")
                for i, rec in enumerate(recordings, 1):
                    size = rec.stat().st_size / 1024
                    print(f"  {i}. {rec.name} ({size:.1f} KB)")
        elif cmd.startswith("p "):
            try:
                num = int(cmd.split()[1]) - 1
                if 0 <= num < len(recordings):
                    print(f"Playing: {recordings[num].name}")
                    subprocess.run(["aplay", str(recordings[num])])
                else:
                    print("Invalid recording number.")
            except (ValueError, IndexError):
                print("Usage: p <number>")
        elif cmd.startswith("d "):
            try:
                num = int(cmd.split()[1]) - 1
                if 0 <= num < len(recordings):
                    rec = recordings.pop(num)
                    rec.unlink()
                    print(f"Deleted: {rec.name}")
                else:
                    print("Invalid recording number.")
            except (ValueError, IndexError):
                print("Usage: d <number>")
        elif cmd == "" or cmd == "r":
            # Record
            filename = generate_filename(output_dir, f"take_{take_num:03d}", "wav")
            success = record_audio(
                filename,
                device=device,
                sample_rate=sample_rate,
                channels=channels,
            )
            if success:
                recordings.append(filename)
                take_num += 1
        else:
            print("Unknown command. Press Enter to record, 'q' to quit.")

    print(f"\nSession complete. {len(recordings)} recordings saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Simple CLI Voice Recorder for TTS samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Quick record with defaults
  %(prog)s -o my_sample.wav         # Record to specific file
  %(prog)s -d 30                    # Record for 30 seconds
  %(prog)s -i                       # Interactive mode for multiple takes
  %(prog)s --list-devices           # Show available devices
  %(prog)s --device hw:0,0          # Use built-in mic
        """
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output filename (default: auto-generated timestamp)"
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        help="Recording duration in seconds (default: until Ctrl+C)"
    )
    parser.add_argument(
        "-r", "--rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Sample rate in Hz (default: {DEFAULT_SAMPLE_RATE})"
    )
    parser.add_argument(
        "-c", "--channels",
        type=int,
        default=DEFAULT_CHANNELS,
        choices=[1, 2],
        help=f"Number of channels (default: {DEFAULT_CHANNELS})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Audio device (default: {DEFAULT_DEVICE} = RØDE NT-USB+)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available recording devices and exit"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode for recording multiple takes"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./recordings",
        help="Output directory for recordings (default: ./recordings)"
    )

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # Create output directory
    output_dir = Path(args.dir)
    output_dir.mkdir(exist_ok=True)

    if args.interactive:
        interactive_mode(output_dir, args.device, args.rate, args.channels)
    else:
        # Single recording mode
        if args.output:
            output_file = Path(args.output)
            if not output_file.suffix:
                output_file = output_file.with_suffix(".wav")
        else:
            output_file = generate_filename(output_dir, "recording", "wav")

        record_audio(
            output_file,
            device=args.device,
            sample_rate=args.rate,
            channels=args.channels,
            duration=args.duration,
        )


if __name__ == "__main__":
    main()
