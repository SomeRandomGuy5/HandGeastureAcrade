# -*- mode: python ; coding: utf-8 -*-

import sys
from PyInstaller.utils.hooks import collect_data_files

# Use collect_data_files to gather all necessary data/model files from the MediaPipe installation
# This will fix the FileNotFoundError and improve build stability.
mediapipe_datas = collect_data_files('mediapipe')

a = Analysis(
    ['main_app.py'],
    pathex=[],
    binaries=[],
    datas=mediapipe_datas,  # Inject the collected mediapipe files here
    hiddenimports=['mediapipe.python.solutions.hands'], # Ensure the hands solution is fully imported
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HandGestureArcade',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HandGestureArcade',
)