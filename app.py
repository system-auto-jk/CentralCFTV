import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_from_directory


APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "security_data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
RECORDING_DIR = DATA_DIR / "recordings"
EVENTS_FILE = DATA_DIR / "events.json"
SETTINGS_FILE = DATA_DIR / "settings.json"

for directory in (DATA_DIR, SNAPSHOT_DIR, RECORDING_DIR):
    directory.mkdir(exist_ok=True)


DEFAULT_SETTINGS = {
    "camera_base_url": "http://192.168.1.71:8080",
    "camera_url": "http://192.168.1.71:8080/video",
    "width": 640,
    "height": 480,
    "armed": True,
    "motion_enabled": True,
    "motion_threshold": 1800,
    "detection_zones": [],
    "schedule_enabled": False,
    "schedule_start": "00:00",
    "schedule_end": "23:59",
    "min_event_interval": 4,
    "record_on_motion": True,
    "pre_record_seconds": 5,
    "post_record_seconds": 5,
    "snapshot_on_motion": True,
    "auto_flash": True,
    "flash_on_dark_only": True,
    "flash_hold_seconds": 8,
    "flash_settle_seconds": 2,
    "darkness_threshold": 82,
    "show_motion_boxes": True,
    "jpeg_quality": 75,
    "record_fps": 15,
    "record_every_n_frames": 1,
    "notifications_enabled": True,
    "notification_webhook_url": "",
}

EVENTS_LOCK = threading.RLock()


def default_camera_config():
    settings = DEFAULT_SETTINGS.copy()
    settings["camera_base_url"] = get_base_url(
        settings["camera_url"], settings["camera_base_url"]
    )
    return {
        "id": "camera-1",
        "name": "Camera 1",
        "settings": settings,
    }


def normalize_camera_settings(saved_settings):
    settings = DEFAULT_SETTINGS.copy()
    if isinstance(saved_settings, dict):
        settings.update({key: saved_settings[key] for key in DEFAULT_SETTINGS if key in saved_settings})
        if "record_seconds" in saved_settings and "post_record_seconds" not in saved_settings:
            settings["post_record_seconds"] = saved_settings["record_seconds"]
    settings["camera_base_url"] = get_base_url(
        settings["camera_url"], settings["camera_base_url"]
    )
    return settings


def load_config():
    config = {"active_camera_id": "camera-1", "cameras": [default_camera_config()]}
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if isinstance(saved, dict) and isinstance(saved.get("cameras"), list):
                cameras = []
                for index, item in enumerate(saved["cameras"], start=1):
                    if not isinstance(item, dict):
                        continue
                    camera_id = str(item.get("id") or f"camera-{index}")
                    cameras.append(
                        {
                            "id": camera_id,
                            "name": str(item.get("name") or f"Camera {index}"),
                            "settings": normalize_camera_settings(item.get("settings", item)),
                        }
                    )
                if cameras:
                    config = {
                        "active_camera_id": str(saved.get("active_camera_id") or cameras[0]["id"]),
                        "cameras": cameras,
                    }
            elif isinstance(saved, dict):
                config = {
                    "active_camera_id": "camera-1",
                    "cameras": [
                        {
                            "id": "camera-1",
                            "name": str(saved.get("name") or "Camera 1"),
                            "settings": normalize_camera_settings(saved),
                        }
                    ],
                }
        except (json.JSONDecodeError, OSError):
            pass

    if not any(camera["id"] == config["active_camera_id"] for camera in config["cameras"]):
        config["active_camera_id"] = config["cameras"][0]["id"]
    return config


def save_config(config):
    SETTINGS_FILE.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_base_url(url, fallback=""):
    if is_local_camera_source(url):
        return ""
    parts = url.split("/")
    if len(parts) >= 3 and parts[0].startswith("http"):
        return f"{parts[0]}//{parts[2]}"
    return fallback


def is_local_camera_source(source):
    source = str(source).strip().lower()
    if source.isdigit():
        return True
    return source.startswith("camera:") and source.split(":", 1)[1].strip().isdigit()


def video_capture_source(source):
    source = str(source).strip()
    if source.isdigit():
        return int(source)
    if source.lower().startswith("camera:"):
        index = source.split(":", 1)[1].strip()
        if index.isdigit():
            return int(index)
    return source


def media_name(prefix, extension):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.{extension}"


INDEX_HTML = """
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Central CFTV</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #191d20;
      --panel-2: #20262a;
      --line: #303840;
      --text: #f4f7f8;
      --muted: #9aa8af;
      --accent: #3fb37f;
      --warn: #d8a13f;
      --danger: #e26d5c;
      --blue: #5ba8f0;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      letter-spacing: 0;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 22px;
      border-bottom: 1px solid var(--line);
      background: #14181b;
    }

    .header-status {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border-radius: 6px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      white-space: nowrap;
    }

    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
    }

    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: calc(100vh - 66px);
    }

    .viewer {
      padding: 18px;
      min-width: 0;
    }

    .camera-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      align-items: start;
    }

    .feed-wrap {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #050607;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      cursor: pointer;
    }

    .feed-wrap.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(63, 179, 127, 0.22);
    }

    .feed-wrap img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    .live-badge {
      position: absolute;
      top: 12px;
      left: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border-radius: 6px;
      background: rgba(0, 0, 0, 0.62);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .camera-name {
      position: absolute;
      right: 12px;
      top: 12px;
      max-width: calc(100% - 150px);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      padding: 7px 10px;
      border-radius: 6px;
      background: rgba(0, 0, 0, 0.62);
      font-size: 12px;
      font-weight: 700;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--danger);
    }

    .dot.ok { background: var(--accent); }
    .dot.warn { background: var(--warn); }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }

    .notice {
      min-height: 22px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }

    .notice.ok { color: #83d6aa; }
    .notice.error { color: #ff9a8c; }

    button, input, select {
      font: inherit;
    }

    button {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      background: var(--panel-2);
      padding: 8px 12px;
      cursor: pointer;
    }

    button:hover { border-color: var(--blue); }
    button.primary { background: #22704f; border-color: #2f9269; }
    button.danger { background: #733128; border-color: #9b4539; }

    aside {
      border-left: 1px solid var(--line);
      background: var(--panel);
      overflow: auto;
    }

    .section {
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }

    .section h2 {
      margin: 0 0 12px;
      font-size: 14px;
      text-transform: uppercase;
      color: var(--muted);
    }

    .config-block {
      padding: 12px;
      margin-bottom: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #15191c;
    }

    .config-block h3 {
      margin: 0 0 4px;
      font-size: 14px;
    }

    .hint {
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .zone-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr) auto;
      gap: 6px;
      align-items: end;
      margin-bottom: 8px;
    }

    .zone-row label {
      margin-bottom: 0;
    }

    .status-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .metric {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-height: 66px;
    }

    .metric strong {
      display: block;
      font-size: 18px;
      margin-top: 6px;
    }

    .metric span, label span {
      color: var(--muted);
      font-size: 12px;
    }

    label {
      display: grid;
      gap: 7px;
      margin-bottom: 12px;
    }

    input[type="text"], input[type="number"], textarea {
      width: 100%;
      min-height: 38px;
      padding: 8px 10px;
      border-radius: 6px;
      border: 1px solid var(--line);
      color: var(--text);
      background: #0f1214;
    }

    textarea {
      min-height: 86px;
      resize: vertical;
      font-family: Consolas, Monaco, monospace;
      font-size: 12px;
    }

    select {
      width: 100%;
      min-height: 38px;
      padding: 8px 10px;
      border-radius: 6px;
      border: 1px solid var(--line);
      color: var(--text);
      background: #0f1214;
    }

    .settings-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .switch-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }

    .switch-row input {
      width: 20px;
      height: 20px;
      accent-color: var(--accent);
    }

    .events {
      display: grid;
      gap: 10px;
      max-height: 380px;
      overflow: auto;
    }

    .event {
      display: grid;
      gap: 6px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
    }

    .event a {
      color: #9ed0ff;
      text-decoration: none;
      font-size: 13px;
    }

    .event small {
      color: var(--muted);
    }

    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { border-left: 0; border-top: 1px solid var(--line); }
      .camera-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Central CFTV</h1>
    <div class="header-status"><span id="top-dot" class="dot"></span><span id="top-status">Inicializando</span></div>
  </header>

  <main>
    <section class="viewer">
      <div id="camera_grid" class="camera-grid"></div>

      <div class="toolbar">
        <button class="primary" onclick="cameraAction('start')">Iniciar camera</button>
        <button onclick="cameraAction('restart')">Reiniciar</button>
        <button class="danger" onclick="cameraAction('stop')">Parar</button>
        <button onclick="snapshot()">Capturar foto</button>
        <button onclick="flash('on')">Flash ligado</button>
        <button onclick="flash('off')">Flash desligado</button>
        <button onclick="setMotionDetection(true)">Ativar Detectar Movimento</button>
        <button onclick="setMotionDetection(false)">Desativar Detectar Movimento</button>
      </div>
      <div id="notice" class="notice"></div>
    </section>

    <aside>
      <section class="section">
        <h2>Status</h2>
        <label>
          <span>Camera ativa</span>
          <select id="camera_select" onchange="selectCamera(this.value)"></select>
        </label>
        <div class="toolbar">
          <button onclick="addCamera()">Adicionar link</button>
          <button onclick="addLocalCamera()">Adicionar local</button>
          <button class="danger" onclick="removeCamera()">Remover atual</button>
        </div>
        <label>
          <span>Nome da nova camera</span>
          <input id="new_camera_name" type="text" placeholder="Entrada, garagem, webcam">
        </label>
        <label>
          <span>Link ou numero local</span>
          <input id="new_camera_source" type="text" placeholder="http://192.168.1.72:8080/video ou 0">
        </label>
        <div class="status-grid">
          <div class="metric"><span>Camera</span><strong id="connected">-</strong></div>
          <div class="metric"><span>FPS</span><strong id="fps">0</strong></div>
          <div class="metric"><span>Movimento</span><strong id="motion-score">0</strong></div>
          <div class="metric"><span>Flash</span><strong id="flash-state">-</strong></div>
        </div>
      </section>

      <section class="section">
        <h2>Configuracoes</h2>
        <div class="config-block">
          <h3>Camera</h3>
          <p class="hint">Fonte de video e tamanho usado para processar esta camera.</p>
          <label><span>URL ou dispositivo local</span><input id="camera_url" type="text"></label>
          <div class="settings-grid">
            <label><span>Largura</span><input id="width" type="number" min="160" max="1920" step="10"></label>
            <label><span>Altura</span><input id="height" type="number" min="120" max="1080" step="10"></label>
            <label><span>Qualidade da imagem</span><input id="jpeg_quality" type="number" min="30" max="95"></label>
          </div>
        </div>

        <div class="config-block">
          <h3>Deteccao</h3>
          <p class="hint">Controla quando o sistema considera que houve movimento.</p>
          <div class="switch-row"><span>Sistema armado</span><input id="armed" type="checkbox"></div>
          <div class="switch-row"><span>Detector de movimento</span><input id="motion_enabled" type="checkbox"></div>
          <div class="switch-row"><span>Caixas de movimento</span><input id="show_motion_boxes" type="checkbox"></div>
          <label><span>Sensibilidade do movimento</span><input id="motion_threshold" type="number" min="100" step="100"></label>
          <div class="toolbar"><button onclick="calibrateSensitivity()">Calibrar sensibilidade</button></div>
        </div>

        <div class="config-block">
          <h3>Zonas</h3>
          <p class="hint">Use porcentagens da tela. Exemplo: x 10, y 20, largura 50, altura 40.</p>
          <div id="zone_editor"></div>
          <div class="toolbar">
            <button onclick="addZone()">Adicionar zona</button>
            <button onclick="clearZones()">Limpar zonas</button>
          </div>
          <textarea id="detection_zones" hidden></textarea>
        </div>

        <div class="config-block">
          <h3>Gravacao</h3>
          <p class="hint">Ao detectar movimento, grava antes do evento, o movimento e alguns segundos depois.</p>
          <div class="switch-row"><span>Gravar ao detectar</span><input id="record_on_motion" type="checkbox"></div>
          <div class="switch-row"><span>Foto ao detectar</span><input id="snapshot_on_motion" type="checkbox"></div>
          <div class="settings-grid">
            <label><span>Antes do movimento</span><input id="pre_record_seconds" type="number" min="1" max="30"></label>
            <label><span>Depois do movimento</span><input id="post_record_seconds" type="number" min="1" max="120"></label>
            <label><span>FPS da gravacao</span><input id="record_fps" type="number" min="5" max="30"></label>
            <label><span>Gravar a cada N frames</span><input id="record_every_n_frames" type="number" min="1" max="5"></label>
          </div>
        </div>

        <div class="config-block">
          <h3>Flash</h3>
          <p class="hint">Flash automatico funciona em cameras IP que suportam lanterna pelo app.</p>
          <div class="switch-row"><span>Flash automatico</span><input id="auto_flash" type="checkbox"></div>
          <div class="switch-row"><span>Flash so no escuro</span><input id="flash_on_dark_only" type="checkbox"></div>
          <div class="settings-grid">
            <label><span>Limiar de escuro</span><input id="darkness_threshold" type="number" min="1" max="255"></label>
            <label><span>Ligado por segundos</span><input id="flash_hold_seconds" type="number" min="1" max="60"></label>
            <label><span>Ignorar apos flash</span><input id="flash_settle_seconds" type="number" min="0" max="10"></label>
          </div>
        </div>

        <div class="config-block">
          <h3>Agenda</h3>
          <p class="hint">Quando ativa, deteccao e eventos so funcionam nesse horario. Pode cruzar meia-noite.</p>
          <div class="switch-row"><span>Agenda ativa</span><input id="schedule_enabled" type="checkbox"></div>
          <div class="settings-grid">
            <label><span>Inicio</span><input id="schedule_start" type="text" placeholder="22:00"></label>
            <label><span>Fim</span><input id="schedule_end" type="text" placeholder="06:00"></label>
          </div>
        </div>

        <div class="config-block">
          <h3>Notificacoes</h3>
          <p class="hint">Mostra alerta no navegador e, se informado, envia JSON para um webhook.</p>
          <div class="switch-row"><span>Notificacoes</span><input id="notifications_enabled" type="checkbox"></div>
          <label><span>Webhook</span><input id="notification_webhook_url" type="text" placeholder="https://..."></label>
          <div class="toolbar"><button onclick="enableBrowserNotifications()">Permitir notificacoes</button></div>
        </div>

        <button class="primary" onclick="saveSettings()">Salvar configuracoes</button>
      </section>

      <section class="section">
        <h2>Eventos</h2>
        <div id="events" class="events"></div>
      </section>
    </aside>
  </main>

  <script>
    let currentCameraId = null;
    let lastEventKey = null;

    const fields = [
      "camera_url", "width", "height", "motion_threshold", "pre_record_seconds",
      "post_record_seconds",
      "darkness_threshold", "flash_hold_seconds", "flash_settle_seconds", "jpeg_quality",
      "record_fps", "record_every_n_frames",
      "schedule_start", "schedule_end", "detection_zones", "notification_webhook_url",
      "armed", "motion_enabled", "schedule_enabled", "record_on_motion", "snapshot_on_motion",
      "notifications_enabled", "auto_flash", "flash_on_dark_only", "show_motion_boxes"
    ];

    function showNotice(message, type = "") {
      const notice = document.getElementById("notice");
      notice.textContent = message;
      notice.className = "notice " + type;
      if (message) setTimeout(() => showNotice(""), 3500);
    }

    function beep() {
      try {
        const context = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = context.createOscillator();
        const gain = context.createGain();
        oscillator.frequency.value = 880;
        gain.gain.value = 0.05;
        oscillator.connect(gain);
        gain.connect(context.destination);
        oscillator.start();
        setTimeout(() => {
          oscillator.stop();
          context.close();
        }, 180);
      } catch {}
    }

    async function enableBrowserNotifications() {
      if (!("Notification" in window)) {
        showNotice("Notificacoes nao suportadas neste navegador.", "error");
        return;
      }
      const permission = await Notification.requestPermission();
      showNotice(permission === "granted" ? "Notificacoes permitidas." : "Notificacoes bloqueadas.", permission === "granted" ? "ok" : "error");
    }

    function notifyEvent(event, settings) {
      if (!settings.notifications_enabled || !event) return;
      const key = `${event.camera_id || ""}-${event.created_at}-${event.recording || event.snapshot || event.motion_score}`;
      if (lastEventKey === null) {
        lastEventKey = key;
        return;
      }
      if (key === lastEventKey) return;
      lastEventKey = key;
      beep();
      if ("Notification" in window && Notification.permission === "granted") {
        new Notification(`Movimento: ${event.camera_name || "Camera"}`, {
          body: `${event.created_at} | movimento ${event.motion_score}`,
        });
      }
    }

    function getZonesFromEditor() {
      const rows = document.querySelectorAll("#zone_editor .zone-row");
      return Array.from(rows).map(row => ({
        x: Number(row.querySelector('[data-zone-field="x"]').value || 0),
        y: Number(row.querySelector('[data-zone-field="y"]').value || 0),
        w: Number(row.querySelector('[data-zone-field="w"]').value || 100),
        h: Number(row.querySelector('[data-zone-field="h"]').value || 100),
      }));
    }

    function syncZonesTextarea() {
      document.getElementById("detection_zones").value = JSON.stringify(getZonesFromEditor(), null, 2);
    }

    function renderZoneEditor(zones) {
      const editor = document.getElementById("zone_editor");
      if (document.activeElement && editor.contains(document.activeElement)) return;
      editor.innerHTML = "";
      for (const zone of zones || []) {
        appendZoneRow(zone);
      }
      syncZonesTextarea();
    }

    function appendZoneRow(zone = { x: 0, y: 0, w: 100, h: 100 }) {
      const editor = document.getElementById("zone_editor");
      const row = document.createElement("div");
      row.className = "zone-row";
      row.innerHTML = `
        <label><span>X</span><input data-zone-field="x" type="number" min="0" max="100" value="${zone.x ?? 0}"></label>
        <label><span>Y</span><input data-zone-field="y" type="number" min="0" max="100" value="${zone.y ?? 0}"></label>
        <label><span>Larg.</span><input data-zone-field="w" type="number" min="1" max="100" value="${zone.w ?? 100}"></label>
        <label><span>Alt.</span><input data-zone-field="h" type="number" min="1" max="100" value="${zone.h ?? 100}"></label>
        <button type="button">Remover</button>
      `;
      row.querySelector("button").onclick = () => {
        row.remove();
        syncZonesTextarea();
      };
      row.querySelectorAll("input").forEach(input => input.oninput = syncZonesTextarea);
      editor.appendChild(row);
    }

    function addZone() {
      appendZoneRow({ x: 0, y: 0, w: 100, h: 100 });
      syncZonesTextarea();
    }

    function clearZones() {
      document.getElementById("zone_editor").innerHTML = "";
      syncZonesTextarea();
    }

    function cssEscape(value) {
      if (window.CSS && typeof window.CSS.escape === "function") {
        return window.CSS.escape(value);
      }
      return String(value).replace(/["\\\\]/g, "\\\\$&");
    }

    async function api(url, options = {}) {
      try {
        const response = await fetch(url, {
          headers: { "Content-Type": "application/json" },
          ...options
        });
        const data = await response.json();
        if (!response.ok || data.ok === false) {
          throw new Error(data.error || "Falha na requisicao");
        }
        return data;
      } catch (error) {
        showNotice(error.message, "error");
        throw error;
      }
    }

    async function refresh() {
      let data;
      try {
        const suffix = currentCameraId ? `?camera_id=${encodeURIComponent(currentCameraId)}` : "";
        data = await api(`/api/status${suffix}`);
      } catch {
        document.getElementById("connected").textContent = "Erro";
        return;
      }
      currentCameraId = data.active_camera_id;
      renderCameraSelect(data.cameras, data.active_camera_id);
      document.getElementById("connected").textContent = data.connected ? "Online" : "Offline";
      document.getElementById("fps").textContent = data.fps.toFixed(1);
      document.getElementById("motion-score").textContent = data.motion_score;
      document.getElementById("flash-state").textContent = data.flash_on ? "Ligado" : "Desligado";
      document.getElementById("top-status").textContent = data.connected ? "Ao vivo" : "Offline";
      document.getElementById("top-dot").className = "dot " + (data.connected ? "ok" : "");

      for (const name of fields) {
        const input = document.getElementById(name);
        if (!input || document.activeElement === input) continue;
        if (input.type === "checkbox") input.checked = Boolean(data.settings[name]);
        else if (name === "detection_zones") {
          input.value = JSON.stringify(data.settings[name] || [], null, 2);
          renderZoneEditor(data.settings[name] || []);
        }
        else input.value = data.settings[name] ?? "";
      }

      const events = document.getElementById("events");
      events.innerHTML = "";
      notifyEvent(data.events[0], data.settings);
      for (const event of data.events) {
        const item = document.createElement("div");
        item.className = "event";
        item.innerHTML = `
          <strong>${event.kind}</strong>
          <small>${event.created_at} | movimento ${event.motion_score} | brilho ${event.brightness}</small>
          ${event.snapshot ? `<a href="/media/snapshots/${event.snapshot}?t=${Date.now()}" target="_blank">Ver foto</a>` : ""}
          ${event.recording && event.recording_ready ? `<a href="/media/recordings/${event.recording}?t=${Date.now()}" target="_blank">Ver gravacao</a>` : ""}
          ${event.recording && !event.recording_ready ? `<small>Gravando...</small>` : ""}
        `;
        events.appendChild(item);
      }
    }

    function renderCameraSelect(cameras, activeCameraId) {
      const select = document.getElementById("camera_select");
      const focused = document.activeElement === select;
      select.innerHTML = "";
      for (const camera of cameras) {
        const option = document.createElement("option");
        option.value = camera.id;
        option.textContent = `${camera.name} ${camera.connected ? "(online)" : "(offline)"}`;
        option.selected = camera.id === activeCameraId;
        select.appendChild(option);
      }
      if (!focused) select.value = activeCameraId || "";
      renderCameraGrid(cameras, activeCameraId);
    }

    function renderCameraGrid(cameras, activeCameraId) {
      const grid = document.getElementById("camera_grid");
      const knownIds = new Set(cameras.map(camera => camera.id));

      for (const child of Array.from(grid.children)) {
        if (!knownIds.has(child.dataset.cameraId)) child.remove();
      }

      for (const camera of cameras) {
        let tile = grid.querySelector(`[data-camera-id="${cssEscape(camera.id)}"]`);
        if (!tile) {
          tile = document.createElement("div");
          tile.className = "feed-wrap";
          tile.dataset.cameraId = camera.id;
          tile.onclick = () => selectCamera(camera.id);
          tile.innerHTML = `
            <img src="/video_feed/${encodeURIComponent(camera.id)}?t=${Date.now()}" alt="Camera ao vivo">
            <div class="live-badge"><span class="dot"></span><span class="motion-text">Sem movimento</span></div>
            <div class="camera-name"></div>
          `;
          grid.appendChild(tile);
        }

        tile.classList.toggle("active", camera.id === activeCameraId);
        tile.querySelector(".camera-name").textContent = camera.name;
        tile.querySelector(".motion-text").textContent = camera.motion_detected ? "Movimento" : (camera.connected ? "Ao vivo" : "Offline");
        tile.querySelector(".dot").className = "dot " + (camera.motion_detected ? "warn" : (camera.connected ? "ok" : ""));
      }
    }

    async function selectCamera(cameraId) {
      currentCameraId = cameraId;
      await api(`/api/cameras/${encodeURIComponent(cameraId)}/active`, { method: "POST" });
      await refresh();
    }

    async function addCamera() {
      const nameInput = document.getElementById("new_camera_name");
      const sourceInput = document.getElementById("new_camera_source");
      const name = nameInput.value.trim() || "Camera";
      const cameraUrl = sourceInput.value.trim();
      if (!cameraUrl || /^\\d+$/.test(cameraUrl)) {
        showNotice("Informe um link de video valido para adicionar por link.", "error");
        return;
      }
      const data = await api("/api/cameras", {
        method: "POST",
        body: JSON.stringify({ name, camera_url: cameraUrl })
      });
      currentCameraId = data.camera.id;
      nameInput.value = "";
      sourceInput.value = "";
      showNotice("Camera adicionada.", "ok");
      await refresh();
    }

    async function addLocalCamera() {
      const nameInput = document.getElementById("new_camera_name");
      const sourceInput = document.getElementById("new_camera_source");
      const name = nameInput.value.trim() || "Webcam";
      const index = sourceInput.value.trim() || "0";
      if (!/^\\d+$/.test(index)) {
        showNotice("Use apenas o numero da camera, como 0, 1 ou 2.", "error");
        return;
      }
      const data = await api("/api/cameras", {
        method: "POST",
        body: JSON.stringify({ name, camera_url: index })
      });
      currentCameraId = data.camera.id;
      nameInput.value = "";
      sourceInput.value = "";
      showNotice("Camera local adicionada.", "ok");
      await refresh();
    }

    async function removeCamera() {
      if (!currentCameraId) return;
      if (!confirm("Remover a camera atual?")) return;
      await fetch(`/api/cameras/${encodeURIComponent(currentCameraId)}`, { method: "DELETE" })
        .then(async response => {
          const data = await response.json();
          if (!response.ok || data.ok === false) throw new Error(data.error || "Falha ao remover");
          return data;
        })
        .catch(error => {
          showNotice(error.message, "error");
          throw error;
      });
      currentCameraId = null;
      showNotice("Camera removida.", "ok");
      await refresh();
    }

    async function saveSettings() {
      const payload = {};
      for (const name of fields) {
        const input = document.getElementById(name);
        if (input.type === "checkbox") {
          payload[name] = input.checked;
        } else if (name === "detection_zones") {
          payload[name] = getZonesFromEditor();
        } else {
          payload[name] = input.value;
        }
      }
      payload.camera_id = currentCameraId;
      await api("/api/settings", { method: "POST", body: JSON.stringify(payload) });
      showNotice("Configuracoes salvas em disco.", "ok");
      await cameraAction("restart");
    }

    async function setMotionDetection(enabled) {
      document.getElementById("motion_enabled").checked = enabled;
      const payload = { camera_id: currentCameraId, motion_enabled: enabled };
      await api("/api/settings", { method: "POST", body: JSON.stringify(payload) });
      showNotice(enabled ? "Detector de movimento ativado." : "Detector de movimento desativado.", "ok");
      await refresh();
    }

    async function calibrateSensitivity() {
      const suffix = currentCameraId ? `?camera_id=${encodeURIComponent(currentCameraId)}` : "";
      const data = await api(`/api/calibrate${suffix}`, { method: "POST" });
      document.getElementById("motion_threshold").value = data.motion_threshold;
      showNotice("Sensibilidade calibrada.", "ok");
      refresh();
    }

    async function cameraAction(action) {
      const suffix = currentCameraId ? `?camera_id=${encodeURIComponent(currentCameraId)}` : "";
      await api(`/api/camera/${action}${suffix}`, { method: "POST" });
      showNotice("Comando enviado.", "ok");
      setTimeout(() => {
        const tile = currentCameraId ? document.querySelector(`[data-camera-id="${cssEscape(currentCameraId)}"] img`) : null;
        if (tile) tile.src = `/video_feed/${encodeURIComponent(currentCameraId)}?t=${Date.now()}`;
        refresh();
      }, 400);
    }

    async function flash(mode) {
      const suffix = currentCameraId ? `?camera_id=${encodeURIComponent(currentCameraId)}` : "";
      const data = await api(`/api/flash/${mode}${suffix}`, { method: "POST" });
      showNotice(data.message || "Comando de flash enviado.", data.supported === false ? "" : "ok");
      refresh();
    }

    async function snapshot() {
      const suffix = currentCameraId ? `?camera_id=${encodeURIComponent(currentCameraId)}` : "";
      const data = await api(`/api/snapshot${suffix}`, { method: "POST" });
      showNotice(data.snapshot ? "Foto salva." : "Nenhum frame disponivel para salvar.", data.snapshot ? "ok" : "error");
      refresh();
    }

    refresh();
    setInterval(refresh, 1500);
  </script>
</body>
</html>
"""


class SecurityCamera:
    def __init__(self, camera_id, name, settings, persist_callback):
        self.camera_id = camera_id
        self.name = name
        self.persist_callback = persist_callback
        self.settings = settings.copy()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.capture_thread = None
        self.cap = None
        self.writer = None
        self.record_until = 0
        self.recording_name = None
        self.recording_path = None
        self.recording_started_at = 0
        self.recording_frame_count = 0
        self.recording_input_frame_count = 0
        self.recording_event_id = None
        self.pre_record_buffer = deque()
        self.frame = None
        self.raw_frame = None
        self.encoded = None
        self.connected = False
        self.motion_detected = False
        self.motion_score = 0
        self.brightness = 0
        self.flash_on = False
        self.flash_off_at = 0
        self.ignore_motion_until = 0
        self.last_event_at = 0
        self.last_frame_at = 0
        self.frame_index = 0
        self.fps = 0.0
        self.events = self.load_events()
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=150, varThreshold=36, detectShadows=True
        )

    def load_events(self):
        if not EVENTS_FILE.exists():
            return []
        try:
            events = json.loads(EVENTS_FILE.read_text(encoding="utf-8"))
            if not isinstance(events, list):
                return []
            camera_events = []
            for event in events:
                if event.get("camera_id", "camera-1") != self.camera_id:
                    continue
                recording = event.get("recording")
                if recording and "recording_ready" not in event:
                    path = RECORDING_DIR / recording
                    event["recording_ready"] = path.exists() and path.stat().st_size > 0
                    event["recording_size"] = path.stat().st_size if path.exists() else 0
                event.setdefault("recording_ready", False)
                event.setdefault("recording_size", 0)
                camera_events.append(event)
            return camera_events
        except (json.JSONDecodeError, OSError):
            return []

    def save_events(self):
        with EVENTS_LOCK:
            all_events = []
            if EVENTS_FILE.exists():
                try:
                    loaded = json.loads(EVENTS_FILE.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        all_events = loaded
                except (json.JSONDecodeError, OSError):
                    all_events = []
            others = [
                event
                for event in all_events
                if event.get("camera_id", "camera-1") != self.camera_id
            ]
            EVENTS_FILE.write_text(
                json.dumps((self.events[:80] + others)[:500], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def start(self):
        if self.capture_thread and self.capture_thread.is_alive():
            return
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        self.release_capture()
        self.finish_recording()
        with self.lock:
            self.connected = False
            self.motion_detected = False

    def restart(self):
        self.stop()
        time.sleep(0.2)
        self.start()

    def update_settings(self, incoming):
        typed = {
            "camera_url": str,
            "schedule_start": str,
            "schedule_end": str,
            "notification_webhook_url": str,
            "motion_threshold": int,
            "pre_record_seconds": int,
            "post_record_seconds": int,
            "darkness_threshold": int,
            "flash_hold_seconds": int,
            "flash_settle_seconds": int,
            "record_fps": int,
            "record_every_n_frames": int,
            "armed": bool,
            "motion_enabled": bool,
            "schedule_enabled": bool,
            "record_on_motion": bool,
            "snapshot_on_motion": bool,
            "notifications_enabled": bool,
            "auto_flash": bool,
            "flash_on_dark_only": bool,
            "show_motion_boxes": bool,
        }
        with self.lock:
            for key, caster in typed.items():
                if key not in incoming:
                    continue
                value = incoming[key]
                if caster is bool:
                    self.settings[key] = bool(value)
                elif caster is int:
                    self.settings[key] = max(0, int(value))
                else:
                    self.settings[key] = str(value).strip()
            self.settings["width"] = max(160, int(incoming.get("width", self.settings["width"])))
            self.settings["height"] = max(120, int(incoming.get("height", self.settings["height"])))
            self.settings["jpeg_quality"] = min(
                95, max(30, int(incoming.get("jpeg_quality", self.settings["jpeg_quality"])))
            )
            self.settings["record_fps"] = min(
                30, max(5, int(incoming.get("record_fps", self.settings["record_fps"])))
            )
            self.settings["record_every_n_frames"] = min(
                5,
                max(1, int(incoming.get("record_every_n_frames", self.settings["record_every_n_frames"]))),
            )
            self.settings["pre_record_seconds"] = min(
                30, max(1, self.settings["pre_record_seconds"])
            )
            self.settings["post_record_seconds"] = min(
                120, max(1, self.settings["post_record_seconds"])
            )
            self.settings["flash_hold_seconds"] = min(
                60, max(1, self.settings["flash_hold_seconds"])
            )
            self.settings["flash_settle_seconds"] = min(
                10, max(0, self.settings["flash_settle_seconds"])
            )
            self.settings["darkness_threshold"] = min(
                255, max(1, self.settings["darkness_threshold"])
            )
            if "detection_zones" in incoming:
                self.settings["detection_zones"] = self.normalize_zones(incoming["detection_zones"])
            self.settings["camera_base_url"] = get_base_url(
                self.settings["camera_url"], self.settings["camera_base_url"]
            )
            self.persist_callback()

    def get_base_url(self, url):
        return get_base_url(url, self.settings.get("camera_base_url", ""))

    def normalize_zones(self, zones):
        if not isinstance(zones, list):
            return []
        normalized = []
        for zone in zones:
            if not isinstance(zone, dict):
                continue
            try:
                x = max(0, min(100, float(zone.get("x", 0))))
                y = max(0, min(100, float(zone.get("y", 0))))
                w = max(1, min(100 - x, float(zone.get("w", 100))))
                h = max(1, min(100 - y, float(zone.get("h", 100))))
            except (TypeError, ValueError):
                continue
            normalized.append({"x": x, "y": y, "w": w, "h": h})
        return normalized

    def is_schedule_active(self):
        if not self.settings["schedule_enabled"]:
            return True
        try:
            start_hour, start_minute = [int(part) for part in self.settings["schedule_start"].split(":", 1)]
            end_hour, end_minute = [int(part) for part in self.settings["schedule_end"].split(":", 1)]
        except (ValueError, AttributeError):
            return True

        now = datetime.now()
        current = now.hour * 60 + now.minute
        start = max(0, min(1439, start_hour * 60 + start_minute))
        end = max(0, min(1439, end_hour * 60 + end_minute))

        if start <= end:
            return start <= current <= end
        return current >= start or current <= end

    def is_detection_active(self):
        return (
            self.settings["armed"]
            and self.settings["motion_enabled"]
            and self.is_schedule_active()
        )

    def zone_rects(self, width, height):
        zones = self.settings.get("detection_zones") or []
        rects = []
        for zone in zones:
            x = int(width * float(zone["x"]) / 100)
            y = int(height * float(zone["y"]) / 100)
            w = int(width * float(zone["w"]) / 100)
            h = int(height * float(zone["h"]) / 100)
            rects.append((x, y, max(1, w), max(1, h)))
        return rects

    def open_capture(self):
        cap = cv2.VideoCapture(video_capture_source(self.settings["camera_url"]))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.settings["width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.settings["height"]))
        return cap

    def release_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def capture_loop(self):
        while not self.stop_event.is_set():
            self.release_capture()
            self.cap = self.open_capture()

            if not self.cap.isOpened():
                with self.lock:
                    self.connected = False
                time.sleep(2)
                continue

            with self.lock:
                self.connected = True

            while not self.stop_event.is_set():
                success, frame = self.cap.read()
                if not success:
                    with self.lock:
                        self.connected = False
                    break

                processed, raw = self.process_frame(frame)
                self.store_frame(processed, raw)
                self.update_pre_record_buffer(raw)
                self.handle_recording(raw)

            time.sleep(0.5)

    def process_frame(self, frame):
        frame = cv2.resize(
            frame,
            (int(self.settings["width"]), int(self.settings["height"])),
            interpolation=cv2.INTER_AREA,
        )
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        brightness = int(np.mean(gray))
        motion_score = 0
        motion_found = False
        boxes = []
        height, width = gray.shape[:2]
        zone_rects = self.zone_rects(width, height)

        now = time.time()
        ignore_motion = now < self.ignore_motion_until

        if self.is_detection_active() and not ignore_motion:
            mask = self.subtractor.apply(gray)
            mask = cv2.threshold(mask, 245, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.dilate(mask, None, iterations=2)
            if zone_rects:
                zone_mask = np.zeros_like(mask)
                for x, y, w, h in zone_rects:
                    zone_mask[y : y + h, x : x + w] = 255
                mask = cv2.bitwise_and(mask, zone_mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            self.subtractor.apply(gray)
            contours = []

        for contour in contours:
            area = int(cv2.contourArea(contour))
            if area < 350:
                continue
            motion_score += area
            boxes.append(cv2.boundingRect(contour))

        motion_found = motion_score >= int(self.settings["motion_threshold"])

        if motion_found and self.settings["show_motion_boxes"]:
            for x, y, w, h in boxes:
                cv2.rectangle(display, (x, y), (x + w, y + h), (63, 179, 127), 2)
        if zone_rects:
            for x, y, w, h in zone_rects:
                cv2.rectangle(display, (x, y), (x + w, y + h), (91, 168, 240), 1)

        self.draw_overlay(display, motion_found, motion_score, brightness)

        with self.lock:
            self.motion_detected = motion_found
            self.motion_score = int(motion_score)
            self.brightness = brightness

        if motion_found:
            self.on_motion(frame.copy(), motion_score, brightness)
        elif (
            self.settings["auto_flash"]
            and self.flash_on
            and self.flash_off_at
            and self.writer is None
            and now >= self.flash_off_at
        ):
            self.set_flash(False)

        return display, frame

    def draw_overlay(self, frame, motion_found, motion_score, brightness):
        status = "MOVIMENTO" if motion_found else "MONITORANDO"
        color = (63, 179, 127) if motion_found else (91, 168, 240)
        cv2.rectangle(frame, (10, 10), (310, 74), (0, 0, 0), -1)
        cv2.putText(frame, status, (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(
            frame,
            f"mov {motion_score} | luz {brightness}",
            (22, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 230, 235),
            1,
        )

    def on_motion(self, raw_frame, motion_score, brightness):
        now = time.time()
        dark = brightness <= int(self.settings["darkness_threshold"])
        should_flash = self.settings["auto_flash"] and (
            dark or not self.settings["flash_on_dark_only"]
        )

        if should_flash:
            self.set_flash(True)
            self.flash_off_at = 0

        if self.settings["record_on_motion"] and self.writer is not None:
            self.extend_recording()
            return

        if now - self.last_event_at < int(self.settings["min_event_interval"]):
            return

        self.last_event_at = now
        event = {
            "camera_id": self.camera_id,
            "camera_name": self.name,
            "kind": "Movimento",
            "created_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "motion_score": int(motion_score),
            "brightness": int(brightness),
            "snapshot": None,
            "recording": None,
            "recording_ready": False,
            "recording_size": 0,
        }

        if self.settings["snapshot_on_motion"]:
            event["snapshot"] = self.write_snapshot(raw_frame)

        if self.settings["record_on_motion"]:
            event["recording"] = self.start_recording(raw_frame, event["created_at"])
            event["recording_ready"] = False

        with self.lock:
            self.events.insert(0, event)
            self.events = self.events[:80]
            self.save_events()
        self.send_notification(event)

    def write_snapshot(self, frame):
        name = media_name(f"{self.camera_id}_snapshot", "jpg")
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return None
        (SNAPSHOT_DIR / name).write_bytes(encoded.tobytes())
        return name

    def send_notification(self, event):
        if not self.settings["notifications_enabled"]:
            return
        webhook_url = str(self.settings.get("notification_webhook_url") or "").strip()
        if not webhook_url:
            return

        def worker():
            payload = json.dumps(event, ensure_ascii=False).encode("utf-8")
            request = Request(
                webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urlopen(request, timeout=3).read()
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def calibrate_sensitivity(self):
        base = max(350, int(self.motion_score * 1.8))
        if self.motion_score < 100:
            base = 1200
        self.settings["motion_threshold"] = min(50000, base)
        self.persist_callback()
        return self.settings["motion_threshold"]

    def manual_snapshot(self):
        with self.lock:
            frame = None if self.raw_frame is None else self.raw_frame.copy()
        if frame is None:
            return None
        name = self.write_snapshot(frame)
        event = {
            "camera_id": self.camera_id,
            "camera_name": self.name,
            "kind": "Snapshot manual",
            "created_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "motion_score": int(self.motion_score),
            "brightness": int(self.brightness),
            "snapshot": name,
            "recording": None,
            "recording_ready": False,
            "recording_size": 0,
        }
        with self.lock:
            self.events.insert(0, event)
            self.save_events()
        return name

    def update_pre_record_buffer(self, frame):
        now = time.time()
        if self.should_keep_record_frame():
            self.pre_record_buffer.append((now, frame.copy()))
        keep_after = now - int(self.settings["pre_record_seconds"])
        while self.pre_record_buffer and self.pre_record_buffer[0][0] < keep_after:
            self.pre_record_buffer.popleft()

    def should_keep_record_frame(self):
        every = int(self.settings["record_every_n_frames"])
        return every <= 1 or self.frame_index % every == 0

    def start_recording(self, frame, event_id):
        if self.writer is not None:
            self.extend_recording()
            return self.recording_name

        name = media_name(f"{self.camera_id}_recording", "mp4")
        path = RECORDING_DIR / name
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_fps = self.recording_output_fps()
        self.writer = cv2.VideoWriter(str(path), fourcc, output_fps, (w, h))
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            return None
        self.recording_name = name
        self.recording_path = path
        self.recording_started_at = time.time()
        self.recording_frame_count = 0
        self.recording_input_frame_count = 0
        self.recording_event_id = event_id

        for _, buffered_frame in list(self.pre_record_buffer):
            self.write_recording_frame(buffered_frame)
        self.extend_recording()
        return name

    def recording_output_fps(self):
        every = max(1, int(self.settings["record_every_n_frames"]))
        configured_fps = float(self.settings["record_fps"]) / every
        measured_fps = 0.0

        if len(self.pre_record_buffer) >= 2:
            first_time = self.pre_record_buffer[0][0]
            last_time = self.pre_record_buffer[-1][0]
            duration = max(last_time - first_time, 0.001)
            measured_fps = (len(self.pre_record_buffer) - 1) / duration
        elif self.fps > 0:
            measured_fps = self.fps / every

        if measured_fps <= 0:
            return max(1.0, configured_fps)

        return min(30.0, max(1.0, measured_fps))

    def extend_recording(self):
        self.record_until = max(
            self.record_until,
            time.time() + int(self.settings["post_record_seconds"]),
        )

    def handle_recording(self, frame):
        if self.writer is None:
            return
        self.recording_input_frame_count += 1
        if self.should_keep_record_frame():
            self.write_recording_frame(frame)
        if time.time() >= self.record_until:
            self.finish_recording()

    def write_recording_frame(self, frame):
        self.writer.write(frame)
        self.recording_frame_count += 1

    def finish_recording(self):
        name = self.recording_name
        path = self.recording_path
        if self.writer is not None:
            writer = self.writer
            self.writer = None
            writer.release()
        if name and path:
            size = path.stat().st_size if path.exists() else 0
            with self.lock:
                for event in self.events:
                    if event.get("recording") == name:
                        event["recording_ready"] = size > 0 and self.recording_frame_count > 1
                        event["recording_size"] = size
                        event["recording_frames"] = self.recording_frame_count
                self.save_events()
        if self.settings["auto_flash"] and self.flash_on:
            self.set_flash(False)
        self.recording_name = None
        self.recording_path = None
        self.record_until = 0
        self.recording_started_at = 0
        self.recording_frame_count = 0
        self.recording_input_frame_count = 0
        self.recording_event_id = None

    def set_flash(self, enabled):
        if self.flash_on == enabled:
            return True

        base_url = self.settings.get("camera_base_url") or self.get_base_url(
            self.settings["camera_url"]
        )
        if not base_url:
            return False

        endpoint = "enabletorch" if enabled else "disabletorch"
        changed = False
        try:
            urlopen(f"{base_url}/{endpoint}", timeout=1.2).read()
            with self.lock:
                self.flash_on = enabled
                changed = True
            ok = True
        except URLError:
            with self.lock:
                if enabled is False:
                    self.flash_on = False
                    changed = True
            ok = False

        if changed:
            self.ignore_motion_until = time.time() + int(self.settings["flash_settle_seconds"])
            if not enabled:
                self.flash_off_at = 0

        return ok

    def store_frame(self, display_frame, raw_frame):
        now = time.time()
        fps = 0 if self.last_frame_at == 0 else 1 / max(now - self.last_frame_at, 0.001)
        self.last_frame_at = now
        self.frame_index += 1
        ok, encoded = cv2.imencode(
            ".jpg",
            display_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.settings["jpeg_quality"])],
        )
        if not ok:
            return
        with self.lock:
            self.frame = display_frame.copy()
            self.raw_frame = raw_frame.copy()
            self.encoded = encoded.tobytes()
            self.fps = (self.fps * 0.85) + (fps * 0.15)

    def get_jpeg(self):
        with self.lock:
            return self.encoded

    def status(self):
        with self.lock:
            return {
                "connected": self.connected,
                "motion_detected": self.motion_detected,
                "motion_score": self.motion_score,
                "brightness": self.brightness,
                "flash_on": self.flash_on,
                "fps": self.fps,
                "camera_id": self.camera_id,
                "camera_name": self.name,
                "armed_now": self.is_detection_active(),
                "settings": self.settings,
                "events": self.events[:30],
            }


class CameraManager:
    def __init__(self, config):
        self.lock = threading.RLock()
        self.config = config
        self.cameras = {}
        for item in self.config["cameras"]:
            self.cameras[item["id"]] = SecurityCamera(
                item["id"], item["name"], item["settings"], self.save
            )

    def start_all(self):
        for camera in self.cameras.values():
            camera.start()

    def stop_all(self):
        for camera in self.cameras.values():
            camera.stop()

    def save(self):
        with self.lock:
            self.config["cameras"] = [
                {
                    "id": camera.camera_id,
                    "name": camera.name,
                    "settings": camera.settings,
                }
                for camera in self.cameras.values()
            ]
            if self.config["active_camera_id"] not in self.cameras and self.cameras:
                self.config["active_camera_id"] = next(iter(self.cameras))
            save_config(self.config)

    def list_cameras(self):
        return [
            {
                "id": camera.camera_id,
                "name": camera.name,
                "connected": camera.connected,
                "motion_detected": camera.motion_detected,
                "motion_score": camera.motion_score,
                "fps": camera.fps,
            }
            for camera in self.cameras.values()
        ]

    def get(self, camera_id=None):
        with self.lock:
            selected_id = camera_id or self.config.get("active_camera_id")
            camera = self.cameras.get(selected_id)
            if camera is None and self.cameras:
                camera = next(iter(self.cameras.values()))
                self.config["active_camera_id"] = camera.camera_id
            return camera

    def set_active(self, camera_id):
        with self.lock:
            if camera_id not in self.cameras:
                return False
            self.config["active_camera_id"] = camera_id
            self.save()
            return True

    def next_camera_id(self):
        index = 1
        while f"camera-{index}" in self.cameras:
            index += 1
        return f"camera-{index}"

    def add_camera(self, name, camera_url):
        with self.lock:
            camera_id = self.next_camera_id()
            settings = DEFAULT_SETTINGS.copy()
            if camera_url:
                settings["camera_url"] = str(camera_url).strip()
            settings["camera_base_url"] = get_base_url(
                settings["camera_url"], settings["camera_base_url"]
            )
            camera_name = str(name or f"Camera {len(self.cameras) + 1}").strip()
            camera = SecurityCamera(camera_id, camera_name, settings, self.save)
            self.cameras[camera_id] = camera
            self.config["active_camera_id"] = camera_id
            self.save()
            camera.start()
            return camera

    def remove_camera(self, camera_id):
        with self.lock:
            if camera_id not in self.cameras:
                return False, "camera nao encontrada"
            if len(self.cameras) <= 1:
                return False, "mantenha pelo menos uma camera"
            camera = self.cameras.pop(camera_id)
            camera.stop()
            if self.config["active_camera_id"] == camera_id:
                self.config["active_camera_id"] = next(iter(self.cameras))
            self.save()
            return True, None

    def status(self, camera_id=None):
        camera = self.get(camera_id)
        if camera is None:
            return {
                "connected": False,
                "motion_detected": False,
                "motion_score": 0,
                "brightness": 0,
                "flash_on": False,
                "fps": 0,
                "settings": DEFAULT_SETTINGS.copy(),
                "events": [],
                "cameras": [],
                "active_camera_id": None,
            }
        data = camera.status()
        data["cameras"] = self.list_cameras()
        data["active_camera_id"] = camera.camera_id
        return data


app = Flask(__name__)
manager = CameraManager(load_config())
manager.save()
manager.start_all()


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.get("/video_feed")
def video_feed():
    return stream_camera(manager.get(request.args.get("camera_id")))


@app.get("/video_feed/<camera_id>")
def video_feed_camera(camera_id):
    return stream_camera(manager.get(camera_id))


def stream_camera(selected_camera):
    def generate():
        while True:
            if selected_camera is None:
                time.sleep(0.2)
                continue
            frame = selected_camera.get_jpeg()
            if frame is None:
                time.sleep(0.2)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.04)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def api_status():
    return jsonify(manager.status(request.args.get("camera_id")))


@app.post("/api/settings")
def api_settings():
    payload = request.get_json(force=True, silent=True) or {}
    selected_camera = manager.get(payload.get("camera_id") or request.args.get("camera_id"))
    if selected_camera is None:
        return jsonify({"ok": False, "error": "camera nao encontrada"}), 404
    selected_camera.update_settings(payload)
    return jsonify({"ok": True, "settings": selected_camera.status()["settings"]})


@app.post("/api/camera/<action>")
def api_camera(action):
    selected_camera = manager.get(request.args.get("camera_id"))
    if selected_camera is None:
        return jsonify({"ok": False, "error": "camera nao encontrada"}), 404
    if action == "start":
        selected_camera.start()
    elif action == "stop":
        selected_camera.stop()
    elif action == "restart":
        selected_camera.restart()
    else:
        return jsonify({"ok": False, "error": "acao invalida"}), 400
    return jsonify({"ok": True})


@app.post("/api/flash/<mode>")
def api_flash(mode):
    selected_camera = manager.get(request.args.get("camera_id"))
    if selected_camera is None:
        return jsonify({"ok": False, "error": "camera nao encontrada"}), 404
    if mode not in {"on", "off"}:
        return jsonify({"ok": False, "error": "modo invalido"}), 400
    if is_local_camera_source(selected_camera.settings["camera_url"]):
        return jsonify(
            {
                "ok": True,
                "flash_on": False,
                "supported": False,
                "message": "flash nao disponivel para camera local",
            }
        )
    ok = selected_camera.set_flash(mode == "on")
    return jsonify({"ok": ok, "flash_on": selected_camera.flash_on})


@app.post("/api/snapshot")
def api_snapshot():
    selected_camera = manager.get(request.args.get("camera_id"))
    if selected_camera is None:
        return jsonify({"ok": False, "error": "camera nao encontrada"}), 404
    name = selected_camera.manual_snapshot()
    return jsonify({"ok": name is not None, "snapshot": name})


@app.post("/api/calibrate")
def api_calibrate():
    selected_camera = manager.get(request.args.get("camera_id"))
    if selected_camera is None:
        return jsonify({"ok": False, "error": "camera nao encontrada"}), 404
    threshold = selected_camera.calibrate_sensitivity()
    return jsonify({"ok": True, "motion_threshold": threshold})


@app.post("/api/cameras")
def api_add_camera():
    payload = request.get_json(force=True, silent=True) or {}
    camera_url = str(payload.get("camera_url") or "").strip()
    if not camera_url:
        return jsonify({"ok": False, "error": "informe a URL ou numero da camera"}), 400
    created = manager.add_camera(payload.get("name"), camera_url)
    return jsonify({"ok": True, "camera": {"id": created.camera_id, "name": created.name}})


@app.delete("/api/cameras/<camera_id>")
def api_remove_camera(camera_id):
    ok, error = manager.remove_camera(camera_id)
    if not ok:
        return jsonify({"ok": False, "error": error}), 400
    return jsonify({"ok": True})


@app.post("/api/cameras/<camera_id>/active")
def api_active_camera(camera_id):
    if not manager.set_active(camera_id):
        return jsonify({"ok": False, "error": "camera nao encontrada"}), 404
    return jsonify({"ok": True})


@app.get("/media/snapshots/<path:name>")
def media_snapshot(name):
    return send_from_directory(SNAPSHOT_DIR, name)


@app.get("/media/recordings/<path:name>")
def media_recording(name):
    return send_from_directory(RECORDING_DIR, name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5051, threaded=True)
