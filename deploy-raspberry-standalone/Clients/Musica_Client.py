import os
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUSICA_DIR = os.path.join(HERE, "musica")
REPRODUCCION_MAX_SEG = 20


def reproducir(nombre_archivo):
    ruta = os.path.join(MUSICA_DIR, nombre_archivo)
    if not os.path.isfile(ruta):
        print(f"    [música] no existe {ruta}, no se reproduce")
        return False
    try:
        subprocess.Popen(
            ["mpv", "--no-video", f"--length={REPRODUCCION_MAX_SEG}",
             "--really-quiet", ruta],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except FileNotFoundError:
        print("    [música] mpv no está instalado, no se puede reproducir")
        return False
