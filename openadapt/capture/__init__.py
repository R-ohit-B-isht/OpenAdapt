"""Capture the screen, audio, and camera as a video on macOS and Windows.

Module: capture.py
"""
import sys

if sys.platform == "darwin":
    from . import _macos as impl
elif sys.platform == "win32":
    from . import _windows as impl
elif sys.platform == "linux":
    class Capture:
        def start(self, audio: bool = False, camera: bool = False) -> None:
            print("Capture functionality is not supported on Linux.")

        def stop(self) -> None:
            print("Capture functionality is not supported on Linux.")
else:
    raise Exception(f"Unsupported platform: {sys.platform}")

device = impl.Capture() if sys.platform in ["darwin", "win32"] else Capture()

def get_capture() -> impl.Capture:
    """Get the capture object.

    Returns:
        Capture: The capture object.
    """
    return device

def start(audio: bool = False, camera: bool = False) -> None:
    """Start the capture."""
    device.start(audio=audio, camera=camera)

def stop() -> None:
    """Stop the capture."""
    device.stop()

def test() -> None:
    """Test the capture."""
    device.start()
    input("Press enter to stop")
    device.stop()

if __name__ in ("__main__", "capture"):
    test()
