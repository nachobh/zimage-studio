#!/usr/bin/env python3
"""
Z-IMAGE SERVER V4 - CON SOPORTE LoRA (basado en v3)
"""
import torch
import sys
import os
import gc
import json
import time
import struct
import socket
import threading
import traceback
from datetime import datetime

# === FIX CRÍTICO PARA AMD ROCm ===
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['TORCH_HIP_USE_LARGE_WAVE_ELDER'] = '1'
os.environ['HIP_FORCE_DEV_KERN'] = '1'
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MIOPEN_FIND_MODE'] = '2'
os.environ['MIOPEN_DEBUG_CONV_IMPLICIT_GEMM'] = '1'
os.environ['FLASH_ATTENTION_TRITON_AMD_ENABLE'] = 'TRUE'
os.environ['FLASH_ATTENTION_TRITON_AMD_AUTOTUNE'] = 'FALSE'

# === CONFIGURACION ===
MODEL_DIR = os.environ.get("ZIMAGE_BASE_PIPELINE", "~/models/zimage-official")
LORA_DIR = os.environ.get("ZIMAGE_LORA_DIR", "~/lora_models_peft")
OUTPUT_DIR = os.environ.get("ZIMAGE_OUTPUT_DIR", os.path.expanduser("~/zimage_outputs"))
# Loopback by default like the rest of the stack; override via env vars.
# (Legacy V4 hardcoded 0.0.0.0:9876; the stack standard backend port is 9877.)
SERVER_HOST = os.environ.get("ZIMAGE_SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("ZIMAGE_SERVER_PORT", "9877"))
PID_FILE = "/tmp/zimage_server_lora.pid"
LOG_FILE = "/tmp/zimage_server_lora.log"

os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOG ===
log_lock = threading.Lock()

def log(msg=""):
    with log_lock:
        print(msg, flush=True)
        try:
            with open(LOG_FILE, 'a') as f:
                f.write(msg + '\n')
        except:
            pass

# === MODELO GLOBAL ===
pipeline = None
lora_adapters = {}
lock = threading.Lock()

# === CARGA DE LoRAS ===
def load_lora(name, path):
    """Carga un LoRA en el pipeline."""
    global pipeline, lora_adapters
    try:
        log(f"  Cargando LoRA '{name}'...")
        pipeline.transformer.load_lora_adapter(LORA_DIR, weight_name=os.path.basename(path), adapter_name=name, prefix=None)
        from verify_zimage_lora import verify_loaded_lora
        checked = verify_loaded_lora(pipeline.transformer, path, name)
        log(f"  Verificados {checked['tensors_verified']} tensores contra el archivo")
        lora_adapters[name] = path
        log(f"  OK LoRA '{name}' cargado")
        return True
    except Exception as e:
        if name in getattr(pipeline.transformer, 'peft_config', {}):
            pipeline.transformer.delete_adapters(name)
        log(f"  ERROR LoRA '{name}': {e}")
        return False

def load_all_loras():
    """Carga todos los LoRAs del directorio."""
    log("Escaneando LoRAs en " + LORA_DIR)
    if not os.path.exists(LORA_DIR):
        log(f"  Directorio no existe: {LORA_DIR}")
        return
    for fname in sorted(os.listdir(LORA_DIR)):
        if fname.endswith('.safetensors'):
            path = os.path.join(LORA_DIR, fname)
            name = fname.replace('.safetensors', '')
            load_lora(name, path)
    log(f"  Total LoRAs cargados: {len(lora_adapters)}")

def list_loras():
    """Lista todos los LoRAs disponibles."""
    result = []
    for name, path in lora_adapters.items():
        size = os.path.getsize(path) / 1024 / 1024
        result.append({"name": name, "path": path, "size_mb": round(size, 1)})
    return result

# === CARGA DEL MODELO (basado en v3) ===
def load_model():
    """Carga el modelo con optimizaciones del v3."""
    global pipeline

    log("=" * 60)
    log("  Z-IMAGE SERVER V4 - CARGANDO MODELO CON LoRA")
    log("=" * 60)
    log()

    log(f"  GPU: {torch.cuda.get_device_name(0)}")
    log(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
    log(f"  Modelo: {MODEL_DIR}")
    log(f"  LoRA dir: {LORA_DIR}")
    log()

    from diffusers import ZImagePipeline

    log("  [1/7] Cargando pipeline...")
    try:
        log("  Cargando con bf16...")
        pipeline = ZImagePipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
        )
        log("  OK Cargado con bf16")
    except Exception as e:
        log(f"  bf16 fallo: {e}")
        log("  Cargando con fp16...")
        pipeline = ZImagePipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float16,
        )
        log("  OK Cargado con fp16")

    log("  [2/7] Deshabilitando force_upcast del VAE...")
    if hasattr(pipeline.vae, 'config') and hasattr(pipeline.vae.config, 'force_upcast'):
        old_val = pipeline.vae.config.force_upcast
        pipeline.vae.config.force_upcast = False
        log(f"  OK force_upcast del VAE: {old_val} -> False")

    log("  [3/7] Moviendo a GPU...")
    pipeline = pipeline.to("cuda")

    log("  [4/7] Activando gradient checkpointing...")
    if hasattr(pipeline.transformer, 'gradient_checkpointing'):
        pipeline.transformer.gradient_checkpointing = True
        log("  OK Gradient checkpointing ACTIVADO")

    log("  [5/7] Optimizando attention...")
    try:
        if hasattr(pipeline.transformer, 'set_attn_processor'):
            from diffusers.models.attention_processor import AttnProcessor2_0
            pipeline.transformer.set_attn_processor(AttnProcessor2_0())
            log("  OK Attention processor OPTIMIZADO (2.0)")
    except Exception as e:
        log(f"  Attention processor: {e}")

    log("  [6/7] Verificando dispositivos...")
    log(f"  Transformer: {next(pipeline.transformer.parameters()).device}")
    log(f"  Text encoder: {pipeline.text_encoder.device}")
    log(f"  VAE: {pipeline.vae.device}")

    vram_after = torch.cuda.memory_allocated(0) / 1024**3
    log(f"  VRAM usada: {vram_after:.1f} GB")

    log()

    # Cargar LoRAs
    load_all_loras()

    log()
    log("  OK MODELO LISTO - ESCUCHANDO PETICIONES")
    log()

def get_vram():
    """Obtiene VRAM usada."""
    try:
        return round(torch.cuda.memory_allocated(0) / 1024**3, 1)
    except:
        return 0

# === GENERACION DE IMAGEN (basado en v3 + LoRA) ===
def generate_image(params):
    """Genera una imagen con soporte LoRA."""
    global pipeline

    if pipeline is None:
        return {"error": "Modelo no cargado"}

    prompt = params.get("prompt", "A beautiful sunset")
    negative_prompt = params.get("negative_prompt", "")
    width = params.get("width", 1024)
    height = params.get("height", 1024)
    steps = params.get("steps", 8)
    cfg = params.get("cfg", 7.5)
    seed = params.get("seed", 0)
    lora = params.get("lora", "")
    filename = params.get("filename", None)
    output_dir = params.get("output_dir", OUTPUT_DIR)

    os.makedirs(output_dir, exist_ok=True)

    log(f"\n[SERVER] Peticion:")
    log(f"  Prompt: {prompt[:80]}...")
    log(f"  Size: {width}x{height} | Steps: {steps} | CFG: {cfg}")
    log(f"  Seed: {seed}")
    if lora:
        log(f"  LoRA: {lora}")
    log(f"  VRAM actual: {get_vram()} GB")

    log("  [1/6] Creando generator...")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    start_time = time.time()

    # Configurar LoRAs
    active_loras = []
    if lora:
        lora_list = []
        for part in lora.split(';'):
            part = part.strip()
            if not part:
                continue
            if ':' in part:
                name, weight = part.split(':', 1)
                name = name.strip()
                weight = float(weight.strip())
                if name in lora_adapters:
                    lora_list.append((name, weight))
            else:
                if part.strip() in lora_adapters:
                    lora_list.append((part.strip(), 1.0))

        if lora_list:
            adapter_names = [n for n, w in lora_list]
            # Usar el primer LoRA como principal con su peso
            # Para múltiples LoRAs, fusionar con el primero
            primary_name, primary_weight = lora_list[0]
            pipeline.transformer.fuse_lora(lora_scale=primary_weight, adapter_names=[primary_name])
            active_loras = lora_list
            log(f"  LoRA activo: {primary_name}@{primary_weight} (fusionado)")
        else:
            # Desactivar LoRA fusionado
            pipeline.transformer.unfuse_lora()
            active_loras = []
    else:
        # No LoRA especificado, desactivar LoRA fusionado
        pipeline.transformer.unfuse_lora()
        active_loras = []

    log("  [2/6] Ejecutando pipeline...")
    try:
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        )
        log("  OK Pipeline completado")
    except Exception as e:
        log(f"\n[SERVER] ERROR durante pipeline: {e}")
        traceback.print_exc()
        return {"error": str(e)}

    elapsed = time.time() - start_time
    log(f"  [3/6] Accediendo a result.images[0]...")
    image = result.images[0]
    log(f"  OK Imagen obtenida")

    log("  [4/6] Guardando imagen...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename:
        if not filename.endswith('.png'):
            filename += '.png'
    else:
        lora_tag = ''
        if active_loras:
            parts = [f"{n}@{w}" for n, w in active_loras]
            lora_tag = '_' + '_'.join(parts)
        filename = f"zimage_{width}x{height}_seed{seed}{lora_tag}_{timestamp}.png"

    output_path = os.path.join(output_dir, filename)
    image.save(output_path)
    log(f"  OK Imagen guardada: {output_path}")

    file_size = os.path.getsize(output_path)

    log(f"  OK Generado en {elapsed:.1f}s")
    log(f"  OK Guardado: {output_path} ({file_size / 1024 / 1024:.1f} MB)")
    log(f"  VRAM despues: {get_vram()} GB")

    # Liberar solo el resultado
    del result, image, generator
    gc.collect()

    log("  [5/6] Limpiando memoria...")

    return {
        "status": "ok",
        "path": output_path,
        "filename": filename,
        "size": f"{width}x{height}",
        "steps": steps,
        "duration": f"{elapsed:.1f}s",
        "file_size_mb": round(file_size / 1024 / 1024, 1),
        "vram_gb": get_vram(),
        "lora": active_loras,
    }

# === MANEJO DE CLIENTES (protocolo binario del v3) ===
def handle_client(conn, addr):
    """Maneja una peticion de un cliente."""
    try:
        log(f"  [CLIENT] Conectado: {addr}")

        # Recibir header (4 bytes)
        header = conn.recv(4)
        if not header:
            return

        data_length = struct.unpack("!I", header)[0]
        data = b""
        while len(data) < data_length:
            chunk = conn.recv(data_length - len(data))
            if not chunk:
                break
            data += chunk

        # Parsear JSON
        params = json.loads(data.decode("utf-8"))
        cmd = params.get("cmd", "generate")
        log(f"  [CLIENT] Comando: {cmd} | Datos: {len(data)} bytes")

        if cmd == "generate":
            with lock:
                result = generate_image(params)
        elif cmd == "status":
            result = {
                "status": "ok",
                "model_loaded": pipeline is not None,
                "lora_count": len(lora_adapters),
                "lora_list": list_loras(),
                "vram_gb": get_vram(),
            }
        elif cmd == "list_loras":
            result = {"status": "ok", "lora_list": list_loras()}
        else:
            result = {"error": f"Comando desconocido: {cmd}"}

        # Enviar respuesta (header + JSON)
        response = json.dumps(result).encode("utf-8")
        conn.sendall(struct.pack("!I", len(response)))
        conn.sendall(response)
        log(f"  [CLIENT] Respuesta enviada: {len(response)} bytes")

    except Exception as e:
        log(f"\n[SERVER] Error en cliente {addr}: {e}")
        traceback.print_exc()

        error_response = json.dumps({"error": str(e)})
        try:
            conn.sendall(struct.pack("!I", len(error_response)))
            conn.sendall(error_response.encode("utf-8"))
        except:
            pass
    finally:
        conn.close()
        log(f"  [CLIENT] Desconectado: {addr}")

# === SERVIDOR ===
def start_server():
    """Inicia el servidor TCP."""
    global pipeline

    # Cargar modelo
    load_model()

    # Escribir PID
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    # Crear socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((SERVER_HOST, SERVER_PORT))
    server.listen(5)

    log(f"  Escuchando en {SERVER_HOST}:{SERVER_PORT}")
    log()
    log("  MODO: Modelo persistente en VRAM con LoRA")
    log("  - Gradient checkpointing ACTIVADO")
    log("  - Attention processor OPTIMIZADO")
    log("  - force_upcast del VAE: DESACTIVADO")
    log("  - LoRAs cargados: " + str(len(lora_adapters)))
    log()
    log("  USO:")
    log("    Comando 'generate': generar imagen")
    log("    Comando 'status': estado del servidor")
    log("    Comando 'list_loras': listar LoRAs")
    log()
    log("  Para detener: ./zimage_server_lora.py --shutdown")
    log("=" * 60)
    log()

    # Servir peticiones
    while True:
        try:
            conn, addr = server.accept()
            handle_client(conn, addr)
        except KeyboardInterrupt:
            print("\n[SERVER] Deteniendo...")
            break
        except Exception as e:
            log(f"\n[SERVER] Error: {e}")
            continue

    server.close()

# === SHUTDOWN ===
def shutdown_server():
    """Apaga el servidor limpiamente."""
    log("[SERVER] Deteniendo servidor...")

    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())

        if os.getpid() != pid:
            try:
                os.kill(pid, 9)
                log(f"  OK Proceso {pid} matado")
            except:
                log(f"  Proceso {pid} no encontrado")
    else:
        import subprocess
        subprocess.run(["pkill", "-9", "-f", "zimage_server_lora.py"], capture_output=True)

    global pipeline
    if pipeline is not None:
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

    log("  OK Servidor detenido")

# === MAIN ===
if __name__ == "__main__":
    # Limpiar log anterior
    try:
        os.remove(LOG_FILE)
    except:
        pass

    if "--shutdown" in sys.argv or "--stop" in sys.argv or "stop" in sys.argv:
        shutdown_server()
    else:
        start_server()
