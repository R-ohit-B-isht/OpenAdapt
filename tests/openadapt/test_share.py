import subprocess
import tempfile
import unittest
import os
from unittest.mock import patch
from openadapt import share


class ShareTestCase(unittest.TestCase):
    """
    A test case for the share module.

    This test case verifies the functionality of the share module,
    including exporting a recording to a folder, sending a file,
    sending a recording, and receiving a recording.
    """

    def test_export_recording_to_folder(self):
        # Create a temporary recording database file
        recording_id = 1
        recording_db_path = "temp.db"
        with open(recording_db_path, "w") as f:
            f.write("Recording data")

        # Mock the crud.export_recording() function to return the temporary file path
        with patch(
            "openadapt.share.db.export_recording", return_value=recording_db_path
        ):
            zip_file_path = share.export_recording_to_folder(recording_id)

            self.assertIsNotNone(zip_file_path)
            self.assertTrue(os.path.exists(zip_file_path))

        # Assert that the file is removed after calling export_recording_to_folder
        self.assertFalse(
            os.path.exists(recording_db_path), "Temporary file was not removed."
        )

    def test_send_file(self):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_path = temp_file.name
            temp_file.write(b"File data")

        # Mock the subprocess.run() function to avoid executing the command
        with patch("openadapt.share.subprocess.run") as mock_run:
            share.send_file(file_path)

            # Verify that the command is called with the correct arguments
            mock_run.assert_called_once_with(
                ["wormhole", "send", file_path], check=True
            )

        # Clean up the temporary file
        os.remove(file_path)

    def test_send_recording(self):
        # Mock the export_recording_to_folder() function to return a zip file path
        with patch(
            "openadapt.share.export_recording_to_folder", return_value="temp.zip"
        ):
            # Mock the send_file() function to avoid sending the file
            with patch("openadapt.share.send_file"):
                share.send_recording(1)

                # Verify that export_recording_to_folder() and send_file() are called
                self.assertTrue(share.export_recording_to_folder.called)
                self.assertTrue(share.send_file.called)

    def test_receive_recording(self):
        # Mock the subprocess.run() function to avoid executing the command
        with patch("openadapt.share.subprocess.run"):
            share.receive_recording("wormhole_code")

            # Verify that the command is called with the correct arguments
            subprocess.run.assert_called_once_with(
                ["wormhole", "receive", "wormhole_code"], check=True
            )
