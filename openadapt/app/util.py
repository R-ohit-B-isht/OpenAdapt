import bz2
import os
import sys
from shutil import copyfileobj
from nicegui import ui
from openadapt.scripts.reset_db import reset_db


def clear_db(log=None):
    if log:
        log.log.clear()
        o = sys.stdout
        sys.stdout = sys.stderr

    reset_db()
    ui.notify("Cleared database.")
    sys.stdout = o


def on_import(selected_file, delete=False, src="openadapt.db"):
    with open(src, "wb") as f:
        with bz2.BZ2File(selected_file, "rb") as f2:
            copyfileobj(f2, f)

    if delete:
        os.remove(selected_file)

    ui.notify("Imported data.")


def on_export(dest):
    # TODO: add ui card for configuration
    ui.notify("Exporting data...")

    # compress db with bz2
    with open("openadapt.db", "rb") as f:
        with bz2.BZ2File("openadapt.db.bz2", "wb", compresslevel=9) as f2:
            copyfileobj(f, f2)

    # TODO: magic wormhole
    # # upload to server with requests, and keep file name
    # files = {
    #     "files": open("openadapt.db.bz2", "rb"),
    # }
    # #requests.post(dest, files=files)

    # delete compressed db
    os.remove("openadapt.db.bz2")

    ui.notify("Exported data.")


def sync_switch(switch, prop):
    switch.value = prop.value


def set_dark(dark_mode, value):
    dark_mode.value = value
