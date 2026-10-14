#!/usr/bin/env python3
"""
Z-IMAGE SERVER GGUF - SOPORTE LoRA CON MODELO GGUF
Duplicado de zimage_server_lora.py adaptado para:
  - Modelo de imagen (transformer): jayn7/Z-Image-Turbo-GGUF  (GGUF, DiT S3-DiT)
  - Text encoder: BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1  (GGUF, Qwen3-4B)

El transformer se carga desde un archivo GGUF con from_single_file + GGUFQuantizationConfig.
El text encoder (Qwen3Model) se carga desde su GGUF correspondiente.
La base del pipeline (VAE + tokenizer + scheduler) procede del repo local Tongyi-MAI/Z-Image-Turbo
(caché en ~/models/zimage-official), que ya contiene esas piezas.

Variables de entorno configurables:
  ZIMAGE_TRANSFORMER_QUANT = Q3_K_M (por defecto)  -> z_image_turbo-<QUANT>.gguf
  ZIMAGE_TEXTENCODER_QUANT = Q3_K_M (por defecto)  -> Z-Image-AbliteratedV1.<QUANT>.gguf
  ZIMAGE_TRANSFORMER_GGUF  = ruta al GGUF del transformer (override completo)
  ZIMAGE_TEXT_ENCODER_DIR  = carpeta del text encoder (override completo)
  ZIMAGE_BASE_PIPELINE     = carpeta del pipeline base (VAE/tokenizer/scheduler)
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
from zimage_lora_admin import reload_lora_adapters

# Importacións de diffusers/transformers para el pipeline (cargadas globalmente)
from diffusers import (  # noqa: F401
    ZImagePipeline,
    ZImageTransformer2DModel,
    GGUFQuantizationConfig,
)
from transformers import Qwen3Model, AutoTokenizer  # noqa: F401

# === FIX CRÍTICO PARA AMD ROCm ===
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
os.environ['TORCH_HIP_USE_LARGE_WAVE_ELDER'] = '1'
os.environ['HIP_FORCE_DEV_KERN'] = '1'
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MIOPEN_FIND_MODE'] = '2'
os.environ['MIOPEN_DEBUG_CONV_IMPLICIT_GEMM'] = '1'
os.environ['FLASH_ATTENTION_TRITON_AMD_ENABLE'] = 'TRUE'
os.environ['FLASH_ATTENTION_TRITON_AMD_AUTOTUNE'] = 'FALSE'

# Suprimir barras de progreso y logging verboso (LoRAs / GGUF)
os.environ['TQDM_DISABLE'] = '1'

# === CONFIGURACION ===
MODELS_DIR = os.environ.get("ZIMAGE_MODELS_DIR", "~/models")
BASE_PIPELINE = os.environ.get("ZIMAGE_BASE_PIPELINE", os.path.join(MODELS_DIR, "zimage-official"))
TRANSFORMER_DIR = os.path.join(MODELS_DIR, "Z-Image-Turbo-GGUF")
TEXT_ENCODER_DIR = os.environ.get("ZIMAGE_TEXT_ENCODER_DIR",
                                   os.path.join(MODELS_DIR, "Qwen3-4b-Z-Image-Turbo-AbliteratedV1"))
TRANSFORMER_QUANT = os.environ.get("ZIMAGE_TRANSFORMER_QUANT", "Q3_K_M")
TEXT_ENCODER_QUANT = os.environ.get("ZIMAGE_TEXT_ENCODER_QUANT", "Q3_K_M")

TRANSFORMER_GGUF = os.environ.get("ZIMAGE_TRANSFORMER_GGUF",
                                   os.path.join(TRANSFORMER_DIR, f"z_image_turbo-{TRANSFORMER_QUANT}.gguf"))
TEXT_ENCODER_GGUF = os.path.join(TEXT_ENCODER_DIR, f"Z-Image-AbliteratedV1.{TEXT_ENCODER_QUANT}.gguf")

LORA_DIR = os.environ.get("ZIMAGE_LORA_DIR", "~/lora_models_peft")
# LORA_DIR = "~/lora_models/old"
OUTPUT_DIR = os.environ.get("ZIMAGE_OUTPUT_DIR", os.path.expanduser("~/zimage_outputs_gguf"))
SERVER_HOST = os.environ.get("ZIMAGE_SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("ZIMAGE_SERVER_PORT", "9877"))
PID_FILE = os.path.join(os.environ.get("ZIMAGE_LOG_DIR", os.path.dirname(__file__)), "zimage_server_gguf.pid")
LOG_FILE = os.path.join(os.environ.get("ZIMAGE_LOG_DIR", os.path.dirname(__file__)), "zimage_server_gguf.log")

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

def get_vram():
    """Obtiene VRAM usada (GB)."""
    try:
        return round(torch.cuda.memory_allocated(0) / 1024**3, 1)
    except:
        return 0

# === MODELO GLOBAL ===
pipeline = None
lora_adapters = {}
lock = threading.Lock()

# === DESCARGA DE ARCHIVOS DEL TEXT ENCODER (si faltan) ===
def ensure_text_encoder_files():
    """Asegura config.json, tokenizer.json, chat_template.jinja, tokenizer_config.json en TEXT_ENCODER_DIR."""
    import urllib.request
    needed = {
        "config.json": None,
        "tokenizer.json": None,
        "chat_template.jinja": None,
        "tokenizer_config.json": None,
    }
    missing = [f for f in needed if not os.path.exists(os.path.join(TEXT_ENCODER_DIR, f))]
    if not missing:
        return True
    try:
        from huggingface_hub import hf_hub_download
        for f in missing:
            src = hf_hub_download("BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1", f, repo_type="model")
            import shutil
            shutil.copy(src, os.path.join(TEXT_ENCODER_DIR, f))
            log(f"  Descargado: {f} ({os.path.getsize(os.path.join(TEXT_ENCODER_DIR, f))} bytes)")
        return True
    except Exception as e:
        log(f"  ERROR descargando archivos del text encoder: {e}")
        return False

# === CARGA DEL TRANSFORMER (GGUF) ===
def load_transformer_gguf():
    """Carga el DiT desde un archivo GGUF (jayn7/Z-Image-Turbo-GGUF)."""
    from diffusers import ZImageTransformer2DModel, GGUFQuantizationConfig
    log(f"  [1/7] Transformer GGUF: {TRANSFORMER_GGUF}")
    if not os.path.exists(TRANSFORMER_GGUF):
        raise FileNotFoundError(f"Transformer GGUF no encontrado: {TRANSFORMER_GGUF}")
    transformer = ZImageTransformer2DModel.from_single_file(
        TRANSFORMER_GGUF,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16,
    )
    log("  OK Transformer cargado desde GGUF")
    return transformer

# === CARGA DEL TEXT ENCODER (GGUF) ===
def load_text_encoder_gguf():
    """Carga el Qwen3-4B desde su GGUF (BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1)."""
    from transformers import Qwen3Model, AutoTokenizer
    log(f"  [2/7] Text encoder GGUF: {TEXT_ENCODER_GGUF}")
    if not os.path.exists(TEXT_ENCODER_GGUF):
        raise FileNotFoundError(f"Text encoder GGUF no encontrado: {TEXT_ENCODER_GGUF}")
    if not ensure_text_encoder_files():
        raise FileNotFoundError("Faltan archivos de configuración del text encoder")
    te = Qwen3Model.from_pretrained(
        TEXT_ENCODER_DIR,
        gguf_file=TEXT_ENCODER_GGUF,
        torch_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(TEXT_ENCODER_DIR)
    log("  OK Text encoder (Qwen3Model) cargado desde GGUF")
    return te, tok

# === CARGA DEL MODELO ===
def load_model():
    """Carga el pipeline con transformer GGUF + text encoder GGUF."""
    global pipeline
    log("=" * 60)
    log("  Z-IMAGE SERVER GGUF - CARGANDO MODELO")
    log("=" * 60)
    log()
    log(f"  GPU: {torch.cuda.get_device_name(0)}")
    log(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
    log()
    log(f"  Transformer:  {TRANSFORMER_GGUF}")
    log(f"  Text encoder: {TEXT_ENCODER_GGUF}")
    log(f"  Base pipeline (VAE/tokenizer/scheduler): {BASE_PIPELINE}")
    log()

    try:
        transformer = load_transformer_gguf()
        text_encoder, tokenizer = load_text_encoder_gguf()

        log("  [3/7] Ensamblando pipeline...")
        pipeline = ZImagePipeline.from_pretrained(
            BASE_PIPELINE,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch.bfloat16,
        )
        pipeline.tokenizer = tokenizer
        log("  OK Pipeline ensamblado")
    except Exception as e:
        log(f"\n[SERVER] ERROR cargando el modelo: {e}")
        traceback.print_exc()
        pipeline = None
        return

    log("  [4/7] Deshabilitando force_upcast del VAE...")
    if hasattr(pipeline.vae, 'config') and hasattr(pipeline.vae.config, 'force_upcast'):
        old_val = pipeline.vae.config.force_upcast
        pipeline.vae.config.force_upcast = False
        log(f"  OK force_upcast del VAE: {old_val} -> False")

    log("  [5/7] Moviendo a GPU...")
    pipeline = pipeline.to("cuda")

    log("  [6/7] Activando gradient checkpointing...")
    if hasattr(pipeline.transformer, 'gradient_checkpointing'):
        pipeline.transformer.gradient_checkpointing = True
        log("  OK Gradient checkpointing ACTIVADO")

    log("  [7/7] Verificando dispositivos...")
    log(f"  Transformer: {next(pipeline.transformer.parameters()).device}")
    log(f"  Text encoder: {pipeline.text_encoder.device}")
    log(f"  VAE: {pipeline.vae.device}")
    log(f"  VRAM usada: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
    log()

    # Cargar LoRAs (mismo mecanismo que el original; se ignora si no aplica al transformer GGUF)
    load_all_loras()

    log()
    log("  OK MODELO LISTO - ESCUCHANDO PETICIONES")
    log("  - Transformer GGUF (quantizada) + gradient checkpointing")
    log("  - Text encoder GGUF abliterated (Qwen3-4B)")
    log()

# === CARGA DE LoRAS (basado en el original) ===
def load_lora(name, path):
    """Carga un LoRA en el transformer del pipeline."""
    global pipeline, lora_adapters
    try:
        log(f"  Cargando LoRA '{name}'...")
        # Suprimir el logging verboso de transformers con LoRAs
        import logging
        logging.getLogger("transformers.models.qwen2_moe.modeling_qwen2_moe").setLevel(logging.WARNING)
        from transformers.utils.logging import get_logger as _gl
        try:
            _gl("transformers").setLevel(logging.WARNING)
        except Exception:
            pass

        # Hay dos formatos de LoRA de Z-Image en circulación:
        #
        #   * original:  transformer.blocks.0....lora_down
        #   * convertido a PEFT: blocks.0....lora_down
        #
        # ``load_lora_adapter`` necesita que el prefijo del primer formato
        # se elimine antes de convertir las claves a PEFT. Pasar prefix=None
        # para el formato original deja ``transformer`` dentro del nombre y
        # el conversor de Diffusers puede acabar en ``list index out of
        # range``.
        lora_prefix = None
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as sf:
                first_key = next(iter(sf.keys()), "")
            if first_key.startswith("transformer."):
                lora_prefix = "transformer"
        except Exception as inspect_error:
            log(f"  AVISO: no se pudo inspeccionar el formato de '{name}': {inspect_error}")

        log(f"  Formato LoRA '{name}': prefix={lora_prefix!r}")
        pipeline.transformer.load_lora_adapter(
            LORA_DIR,
            weight_name=os.path.basename(path),
            adapter_name=name,
            prefix=lora_prefix,
        )
        from verify_zimage_lora import verify_loaded_lora
        checked = verify_loaded_lora(pipeline.transformer, path, name)
        log(f"  Verificados {checked['tensors_verified']} tensores contra el archivo")
        lora_adapters[name] = path
        log(f"  OK LoRA '{name}' cargado")
        return True
    except Exception as e:
        if name in getattr(pipeline.transformer, 'peft_config', {}):
            pipeline.transformer.delete_adapters(name)
        log(f"  INFO LoRA '{name}' no aplicable (transformer GGUF): {e}")
        return False

def load_all_loras():
    """Escanea el directorio de LoRAs."""
    log("Escaneando LoRAs en " + LORA_DIR)
    if not os.path.exists(LORA_DIR):
        log(f"  Directorio no existe: {LORA_DIR}")
        return
    # Los LoRAs pueden ser enormes; redirigir stderr (logging de transformers) al log
    import sys
    import contextlib
    with open(LOG_FILE, 'a') as logf:
        with contextlib.redirect_stderr(logf):
            for fname in sorted(os.listdir(LORA_DIR)):
                if fname.endswith('.safetensors'):
                    path = os.path.join(LORA_DIR, fname)
                    name = fname.replace('.safetensors', '')
                    load_lora(name, path)

def reload_loras():
    return reload_lora_adapters(pipeline, lora_adapters, LORA_DIR, load_lora)

def list_loras():
    """Lista todos los LoRAs disponibles."""
    result = []
    for name, path in lora_adapters.items():
        size = os.path.getsize(path) / 1024 / 1024
        result.append({"name": name, "path": path, "size_mb": round(size, 1)})
    return result

# === GENERACION DE IMAGEN ===
def generate_image(params):
    """Genera una imagen con el transformer GGUF y text encoder GGUF."""
    global pipeline

    if pipeline is None:
        return {"error": "Modelo no cargado"}

    prompt = params.get("prompt", "A beautiful sunset")
    negative_prompt = params.get("negative_prompt", "")
    width = params.get("width", 1024)
    height = params.get("height", 1024)
    steps = params.get("steps", 8)
    from zimage_progress import progress
    progress_id = params.get('request_id', '')
    progress.update(progress_id, 'preparing', total=steps)
    cfg = params.get("cfg", 0.0)  # Turbo -> guidance 0
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

    # Configurar LoRAs. En un transformer GGUF no se debe fusionar el LoRA
    # en los pesos cuantizados: PEFT puede aplicarlo en el forward, pero
    # ``fuse_lora`` intenta hacer una suma de matrices sobre la representación
    # GGUF y produce errores como 2160 vs 3840.
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
                    return {'error': f'LoRA no cargado: {name}'}
            else:
                if part.strip() in lora_adapters:
                    lora_list.append((part.strip(), 1.0))
                else:
                    return {'error': f'LoRA no cargado: {part.strip()}'}

        if lora_list:
            try:
                names = [name for name, _weight in lora_list]
                weights = [weight for _name, weight in lora_list]
                # Puede estar desactivado por una petición anterior sin LoRA.
                pipeline.transformer.enable_lora()
                pipeline.transformer.set_adapters(names, weights)
                active_loras = lora_list
                log(f"  LoRA activo: {', '.join(f'{n}@{w}' for n, w in lora_list)} (PEFT en forward)")
            except Exception as e:
                log(f"  INFO LoRA no aplicado: {e}")
                return {'error': f'No se pudo aplicar la mezcla LoRA: {e}'}
        else:
            try:
                pipeline.transformer.disable_lora()
            except Exception:
                pass
            active_loras = []
    else:
        try:
            pipeline.transformer.disable_lora()
        except Exception:
            pass
        active_loras = []

    log("  [2/6] Ejecutando pipeline...")
    try:
        # Los modelos Turbo usan guidance_scale=0.0; se respeta el cfg del cliente salvo que sea <=0.
        guidance = float(cfg)
        if guidance <= 0.0:
            guidance = 0.0
        def on_step(pipe, index, timestep, callback_kwargs):
            total = len(pipe.scheduler.timesteps)
            progress.update(progress_id, 'decoding' if index + 1 == total else 'sampling', index + 1, total)
            return callback_kwargs

        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            callback_on_step_end=on_step,
            callback_on_step_end_tensor_inputs=[],
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
    progress.update(progress_id, 'saving', steps, steps)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename:
        if not filename.endswith('.png'):
            filename += '.png'
    else:
        lora_tag = ''
        if active_loras:
            parts = [f"{n}@{w}" for n, w in active_loras]
            lora_tag = '_' + '_'.join(parts)
        # Long combinations can exceed the filesystem's 255-byte filename limit.
        import hashlib
        if len(lora_tag.encode('utf-8')) > 100:
            lora_tag = '_mix_' + hashlib.sha256(lora_tag.encode()).hexdigest()[:12]
        filename = f"zimage_gguf_{width}x{height}_seed{seed}{lora_tag}_{timestamp}_{time.time_ns() % 1000000000}.png"

    output_path = os.path.join(output_dir, filename)
    image.save(output_path)
    log(f"  OK Imagen guardada: {output_path}")

    file_size = os.path.getsize(output_path)

    log(f"  OK Generado en {elapsed:.1f}s")
    log(f"  OK Guardado: {output_path} ({file_size / 1024 / 1024:.1f} MB)")
    log(f"  VRAM despues: {get_vram()} GB")

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

# === MANEJO DE CLIENTES (protocolo binario) ===
def handle_client(conn, addr):
    """Maneja una peticion de un cliente."""
    try:
        # Una conexión que no envíe nuestro encabezado binario no debe dejar
        # bloqueado el servidor completo (por ejemplo, un cliente HTTP que
        # se conecte por error al puerto TCP interno).
        conn.settimeout(15)
        log(f"  [CLIENT] Conectado: {addr}")

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

        params = json.loads(data.decode("utf-8"))
        cmd = params.get("cmd", "generate")
        log(f"  [CLIENT] Comando: {cmd} | Datos: {len(data)} bytes")

        if cmd == "generate":
            from zimage_progress import progress
            progress.update(params.get('request_id', ''), 'queued')
            with lock:
                try:
                    result = generate_image(params)
                    progress.update(params.get('request_id', ''), 'error' if result.get('error') else 'done')
                except Exception:
                    progress.update(params.get('request_id', ''), 'error')
                    raise
        elif cmd == 'progress':
            from zimage_progress import progress
            result = progress.get(params.get('request_id', ''))
        elif cmd == "status":
            with lock:
                result = {
                    "status": "ok",
                    "model_loaded": pipeline is not None,
                    "lora_count": len(lora_adapters),
                    "lora_list": list_loras(),
                    "vram_gb": get_vram(),
                    "transformer_gguf": TRANSFORMER_GGUF,
                    "text_encoder_gguf": TEXT_ENCODER_GGUF,
                }
        elif cmd == "list_loras":
            with lock:
                result = {"status": "ok", "lora_list": list_loras()}
        elif cmd == 'reload_loras':
            with lock:
                result = reload_loras()
        else:
            result = {"error": f"Comando desconocido: {cmd}"}

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

    load_model()
    if pipeline is None:
        raise RuntimeError("El modelo no se cargó; no se abrirá el puerto TCP")

    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((SERVER_HOST, SERVER_PORT))
    except OSError as exc:
        if exc.errno == 98:
            log(f"[SERVER] ERROR: el puerto {SERVER_PORT} ya está en uso.")
            log(f"[SERVER] Sugerencia: prueba con ZIMAGE_SERVER_PORT={SERVER_PORT + 1} o libera ese puerto.")
        raise
    server.listen(5)

    log(f"  Escuchando en {SERVER_HOST}:{SERVER_PORT}")
    log()
    log("  MODO: Transformer GGUF + text encoder GGUF persistente")
    log()
    log("  USO:")
    log("    Comando 'generate': generar imagen")
    log("    Comando 'status': estado del servidor")
    log("    Comando 'list_loras': listar LoRAs")
    log()
    log("  Para detener: ./zimage_server_gguf.py --shutdown")
    log("=" * 60)
    log()

    while True:
        try:
            conn, addr = server.accept()
            # Igual que el adaptador MCP HTTP, no bloquear la aceptación de
            # nuevas peticiones mientras se procesa una conexión anterior.
            threading.Thread(
                target=handle_client,
                args=(conn, addr),
                daemon=True,
            ).start()
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
        subprocess.run(["pkill", "-9", "-f", "zimage_server_gguf.py"], capture_output=True)

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
    try:
        os.remove(LOG_FILE)
    except:
        pass

    if "--shutdown" in sys.argv or "--stop" in sys.argv or "stop" in sys.argv:
        shutdown_server()
    else:
        start_server()
