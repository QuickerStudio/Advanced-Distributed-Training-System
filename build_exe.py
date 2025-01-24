import PyInstaller.__main__

PyInstaller.__main__.run([
    '--name=%s' % "AdvancedDistributedTrainingSystem",
    '--onefile',
    '--windowed',
    'main.py',
])
