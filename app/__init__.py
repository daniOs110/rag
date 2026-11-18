import os

# Chroma intenta enviar telemetría al arrancar. Se apaga antes de importar el cliente.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

__version__ = "0.1.0"
