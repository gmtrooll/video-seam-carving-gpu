#!/usr/bin/env python3
# Interfata grafica pentru GPU Seam Carving

import asyncio
import os
import re
import sys
import base64
import subprocess as sp
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

from nicegui import ui, app

# Try to load OpenCV for video info
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

PROJECT_DIR = Path(__file__).parent.resolve()
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}

def find_binary() -> Optional[Path]:
    # cauta binarul compilat
    exts = ['.exe', ''] if sys.platform == 'win32' else ['', '.exe']
    for ext in exts:
        p = PROJECT_DIR / f"v2_jnd{ext}"
        if p.exists():
            return p
    return None

def scan_videos() -> List[Path]:
    # scanare folder input pt videouri
    skip_tags = ['-cuda-carved', '-v2-jnd', '_viz', '_carved']
    result = []
    for f in sorted(INPUT_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            if not any(tag in f.stem.lower() for tag in skip_tags):
                result.append(f)
    return result

def get_video_info(path: str) -> Optional[dict]:
    # rezolutie si fps
    if not HAS_CV2:
        return None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        info = {
            'width':  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps':    round(cap.get(cv2.CAP_PROP_FPS), 2),
        }
        info['duration'] = round(info['frames'] / info['fps'], 1) if info['fps'] > 0 else 0
        cap.release()
        return info
    except Exception:
        return None

def extract_thumbnail(path: str, max_w: int = 360) -> Optional[str]:
    # extrage primul cadru pt thumbnail
    if not HAS_CV2:
        return None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        h, w = frame.shape[:2]
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, (max_w, int(h * scale)))
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    except Exception:
        return None

@dataclass
class Job:
    id: int
    video_path: str
    video_name: str
    algorithm: str          
    algo_label: str
    vertical_seams: int
    horizontal_seams: int
    visualize: bool
    video_width: int = 0
    video_height: int = 0
    status: str = 'queued'  
    progress: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    fps_str: str = ''
    output_path: str = ''
    viz_path: str = ''
    is_expand: bool = False
    lambda_val: float = 0.4
    skin_bias: float = 1000.0
    motion_weight: float = 5.0
    batch_size: int = 1
    log: list = field(default_factory=list)

THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base: #08090d;
    --bg-card: rgba(16, 18, 27, 0.85);
    --bg-card-hover: rgba(22, 25, 38, 0.92);
    --accent-orange: #ff6b35;
    --accent-red: #ff2e63;
    --text-main: #e2e4f0;
    --text-dim: #6b7294;
    --border-line: rgba(255,255,255,0.06);
    --glow-a: rgba(255,107,53,0.12);
    --glow-b: rgba(255,46,99,0.08);
}

body {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background: var(--bg-base) !important;
    background-image:
        radial-gradient(ellipse at 15% 45%, var(--glow-a), transparent 55%),
        radial-gradient(ellipse at 85% 15%, var(--glow-b), transparent 55%) !important;
    background-attachment: fixed !important;
}
.q-page { background: transparent !important; }

.glass {
    background: var(--bg-card) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 1px solid var(--border-line) !important;
    border-radius: 16px !important;
    transition: border-color 0.3s ease !important;
}
.glass:hover {
    border-color: rgba(255,107,53,0.12) !important;
}

.grad-text {
    background: linear-gradient(135deg, var(--accent-orange), var(--accent-red));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hdr {
    background: rgba(8,9,13,0.92) !important;
    backdrop-filter: blur(30px) !important;
    border-bottom: 1px solid var(--border-line) !important;
}

.btn-accent {
    background: linear-gradient(135deg, var(--accent-orange), var(--accent-red)) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    text-transform: none !important; letter-spacing: 0 !important;
    transition: all 0.3s ease !important;
}
.btn-accent:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(255,107,53,0.35) !important;
}
.btn-sec {
    background: rgba(255,255,255,0.04) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border-line) !important;
    border-radius: 12px !important; font-weight: 500 !important;
    text-transform: none !important; letter-spacing: 0 !important;
    transition: all 0.3s ease !important;
}
.btn-sec:hover {
    background: rgba(255,255,255,0.09) !important;
    border-color: rgba(255,107,53,0.25) !important;
}

.job-card {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid var(--border-line) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    transition: background 0.2s ease !important;
}
.job-card:hover { background: rgba(255,255,255,0.04) !important; }

.chip {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border-line);
    border-radius: 6px; padding: 2px 10px;
    font-size: 11px; color: var(--text-dim);
    font-family: 'JetBrains Mono', monospace;
}

.log-box {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11.5px !important;
    background: rgba(0,0,0,0.35) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-line) !important;
    color: var(--text-dim) !important;
}

.sec-title {
    color: var(--text-main) !important; font-weight: 600 !important;
    font-size: 13px !important; text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}

.st-queued  { color: var(--text-dim); }
.st-running { color: var(--accent-orange); }
.st-done    { color: #22c55e; }
.st-error   { color: var(--accent-red); }

.thumb-img {
    border-radius: 10px;
    border: 1px solid var(--border-line);
    max-width: 100%; height: auto;
}

.dim-label { color: var(--text-dim) !important; font-size: 13px; }
.result-dims {
    color: var(--accent-orange) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px;
}
.fade-in {
    animation: fadeSlide 0.35s ease-out;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
"""

class SeamCarvingStudio:

    def __init__(self):
        self.jobs: List[Job] = []
        self.job_id = 0
        self.is_processing = False
        self.running_proc: Optional[asyncio.subprocess.Process] = None

        self.selected_video = ''
        self.video_info: Optional[dict] = None

        self._job_container = None
        self._log_scroll = None
        self._log_container = None
        self._progress = None
        self._progress_text = None
        self._dims_label = None
        self._info_label = None
        self._thumb_img = None
        self._process_btn = None
        self._cancel_btn = None
        
        self._v_slider = None
        self._h_slider = None
        self._batch_slider = None
        self._lambda_slider = None
        self._skin_slider = None
        self._motion_slider = None
        self._algo_toggle = None
        self._mode_toggle = None
        self._viz_switch = None
        self._video_select = None

    def _output_name(self, job: Job) -> str:
        # genereaza numele fisierului rezultat
        base = Path(job.video_path).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        s = f"{job.vertical_seams}v"
        if job.horizontal_seams > 0:
            s += f"_{job.horizontal_seams}h"
        return f"{base}_{ts}_{s}.mp4"

    def _viz_name(self, job: Job) -> str:
        # nume fisier vizualizare
        base = Path(job.video_path).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        s = f"{job.vertical_seams}v"
        if job.horizontal_seams > 0:
            s += f"_{job.horizontal_seams}h"
        return f"{base}_{ts}_{s}_viz.mp4"

    def _on_video_change(self, path: str):
        # actualizeaza info la selectie video
        # print(f"video schimbat: {path}")
        self.selected_video = path
        if path and Path(path).exists():
            self.video_info = get_video_info(path)
            thumb = extract_thumbnail(path)
        else:
            self.video_info = None
            thumb = None

        if self._info_label:
            if self.video_info:
                vi = self.video_info
                self._info_label.set_text(
                    f"{vi['width']}×{vi['height']}  •  {vi['frames']} frames  •  "
                    f"{vi['fps']} FPS  •  {vi['duration']}s"
                )
            else:
                self._info_label.set_text("Select a video" if not path else "Error reading video")

        if self._thumb_img:
            if thumb:
                self._thumb_img.set_source(thumb)
                self._thumb_img.set_visibility(True)
            else:
                self._thumb_img.set_visibility(False)

        self._update_dims()

    def _handle_upload(self, e):
        # upload video nou
        name = e.name
        content = e.content
        target = INPUT_DIR / name
        with open(target, 'wb') as f:
            f.write(content.read())
        
        videos = scan_videos()
        self._video_select.options = {str(f): f.name for f in videos}
        self._video_select.value = str(target)
        self._on_video_change(str(target))
        ui.notify(f"Uploaded: {name}", type='positive')

    def _update_dims(self, v_val: int = None, h_val: int = None):
        # recalculeaza dimensiunile rezultat
        if not self._dims_label:
            return
        if self.video_info:
            vi = self.video_info
            v = v_val if v_val is not None else int(self._v_slider.value)
            h = h_val if h_val is not None else int(self._h_slider.value)
            
            is_expand = (self._mode_toggle.value == 'expand') if hasattr(self, '_mode_toggle') else False

            if is_expand:
                rw = vi['width'] + v
                rh = vi['height'] + h 
            else:
                rw = vi['width'] - v
                rh = vi['height'] - h
            
            rw = max(rw, 1)
            rh = max(rh, 1)
            self._dims_label.set_text(f"{vi['width']}×{vi['height']}  →  {rw}×{rh} ({'WIDEN' if is_expand else 'SHRINK'})")
        else:
            self._dims_label.set_text("")

    def _add_job(self):
        # adauga job in coada
        path = self.selected_video
        if not path or not Path(path).exists():
            ui.notify("Select a video", type='warning')
            return
        try:
            v = int(self._v_slider.value)
            h = int(self._h_slider.value)
            if v <= 0 and h <= 0:
                ui.notify("Set at least 1 seam", type='warning')
                return

            vw = self.video_info['width'] if self.video_info else 0
            vh = self.video_info['height'] if self.video_info else 0

            if v >= vw and vw > 0:
                ui.notify("Too many seams", type='warning')
                return

            self.job_id += 1
            job = Job(
                id=self.job_id,
                video_path=path,
                video_name=Path(path).name,
                algorithm='v2',
                algo_label='JND Perceptual',
                vertical_seams=v,
                horizontal_seams=h,
                visualize=self._viz_switch.value,
                video_width=vw,
                video_height=vh,
                is_expand=(self._mode_toggle.value == 'expand'),
                batch_size=int(self._batch_slider.value),
                lambda_val=float(self._lambda_slider.value),
                skin_bias=float(self._skin_slider.value),
                motion_weight=float(self._motion_slider.value),
            )
            self.jobs.append(job)
            self._render_jobs()
            ui.notify(f"Added: {job.video_name}", type='positive')
        except Exception as e:
            ui.notify(f"Error adding job: {e}", type='negative')
            print(f"Error adding job: {e}")

    def _remove_job(self, jid: int):
        # sterge din coada
        self.jobs = [j for j in self.jobs if j.id != jid]
        self._render_jobs()

    def _clear_finished(self):
        # curata joburile terminate
        self.jobs = [j for j in self.jobs if j.status not in ('done', 'error')]
        self._render_jobs()

    def _render_jobs(self):
        # redeseneaza lista de joburi
        if not self._job_container:
            return
        self._job_container.clear()
        with self._job_container:
            if not self.jobs:
                ui.label("Queue is empty").classes('dim-label')
                return
            for job in self.jobs:
                self._render_one_job(job)

    def _render_one_job(self, job: Job):
        icons = {'queued': 'schedule', 'running': 'play_circle',
                 'done': 'check_circle', 'error': 'error'}
        css   = {'queued': 'st-queued', 'running': 'st-running',
                 'done': 'st-done', 'error': 'st-error'}

        with ui.element('div').classes('job-card fade-in'):
            with ui.row().classes('w-full items-center no-wrap gap-3'):
                ui.icon(icons.get(job.status, 'help')).classes(f"{css.get(job.status, '')} text-lg")
                with ui.column().classes('flex-grow gap-0'):
                    ui.label(job.video_name).classes('text-sm font-medium').style('color: var(--text-main)')
                    with ui.row().classes('gap-1 mt-1'):
                        ui.html(f'<span class="chip">JND</span>')
                        ui.html(f'<span class="chip">{job.vertical_seams}v</span>')
                        if job.horizontal_seams > 0:
                            ui.html(f'<span class="chip">{job.horizontal_seams}h</span>')
                        ui.html(f'<span class="chip">{"EXP" if job.is_expand else "SHR"}</span>')
                        if job.visualize:
                            ui.html('<span class="chip">VIZ</span>')

                if job.status == 'running':
                    ui.label(f"{job.progress:.0%}").classes('st-running text-sm')
                elif job.status == 'queued':
                    ui.button(icon='close', on_click=lambda _, jid=job.id: self._remove_job(jid)).props('flat round dense size=sm')
                elif job.status == 'done':
                    with ui.row().classes('gap-1'):
                        if job.output_path and Path(job.output_path).exists():
                            ui.button('Result', icon='play_arrow', on_click=lambda _, p=job.output_path: _open_file(p)).classes('btn-sec').props('dense no-caps size=sm')
                        if job.viz_path and Path(job.viz_path).exists():
                            ui.button('Viz', icon='visibility', on_click=lambda _, p=job.viz_path: _open_file(p)).classes('btn-sec').props('dense no-caps size=sm')
                elif job.status == 'error':
                    ui.icon('warning').classes('st-error')

    async def _run_job(self, job: Job):
        # lanseaza binarul cuda
        binary = find_binary()
        if not binary:
            job.status = 'error'
            job.log.append("Binar negasit.")
            self._render_jobs()
            self._push_log(job)
            return

        OUTPUT_DIR.mkdir(exist_ok=True)
        out_name = self._output_name(job)
        job.output_path = str(OUTPUT_DIR / out_name)

        bin_str = str(binary)
        vid_str = str(job.video_path)
        out_str = job.output_path
        viz_str = ""

        if job.visualize:
            viz_name = self._viz_name(job)
            job.viz_path = str(OUTPUT_DIR / viz_name)
            viz_str = job.viz_path

        cmd_prefix = []
        if sys.platform == 'win32' and binary.suffix != '.exe':
            def to_wsl(p_str):
                if not p_str: return ""
                p = Path(p_str).resolve()
                if p.is_relative_to(PROJECT_DIR):
                    return p.relative_to(PROJECT_DIR).as_posix()
                p_posix = p.as_posix()
                if p_posix[1:3] == ':/':
                    return f"/mnt/{p_posix[0].lower()}{p_posix[2:]}"
                return p_posix
            cmd_prefix = ['wsl']
            bin_str = f"./{binary.name}"
            vid_str = to_wsl(vid_str)
            out_str = to_wsl(out_str)
            if job.visualize:
                viz_str = to_wsl(viz_str)

        cmd = cmd_prefix + [bin_str, vid_str, str(job.vertical_seams)]
        if job.horizontal_seams > 0:
            cmd.append(str(job.horizontal_seams))
        cmd.extend(['--output', out_str])
        if job.is_expand:
            cmd.append('--expand')
        if job.batch_size > 1:
            cmd.extend(['--batch', str(job.batch_size)])
        if job.visualize:
            cmd.extend(['--visualize', viz_str])
        
        cmd.extend(['--lambda', str(job.lambda_val)])
        cmd.extend(['--skin-bias', str(job.skin_bias)])
        cmd.extend(['--motion-weight', str(job.motion_weight)])

        job.status = 'running'
        self._render_jobs()
        self._push_log(job)

        try:
            self.running_proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_DIR),
            )

            buf = ''
            while True:
                chunk = await self.running_proc.stdout.read(512)
                if not chunk:
                    break
                buf += chunk.decode('utf-8', errors='replace')
                parts = re.split(r'[\r\n]+', buf)
                buf = parts[-1]

                for part in parts[:-1]:
                    line = part.strip()
                    if not line: continue
                    if len(job.log) > 300: job.log = job.log[-150:]
                    job.log.append(line)

                    m = re.search(r'Cadru\s+(\d+)\s*/?\s*(\d+)', line)
                    if m:
                        job.current_frame = int(m.group(1))
                        job.total_frames = int(m.group(2))
                        job.progress = job.current_frame / job.total_frames if job.total_frames else 0

                    fm = re.search(r'([\d.]+)\s*FPS', line)
                    if fm: job.fps_str = f"{float(fm.group(1)):.1f} FPS"

                    self._update_progress(job)
                    self._push_log(job)

            await self.running_proc.wait()
            if self.running_proc.returncode == 0:
                job.status = 'done'
                job.progress = 1.0
            else:
                job.status = 'error'
        except asyncio.CancelledError:
            job.status = 'error'
        except Exception as e:
            job.status = 'error'
        finally:
            self.running_proc = None

        self._render_jobs()
        self._update_progress(job)
        self._push_log(job)

    async def _process_all(self):
        # porneste procesarea cozii
        # print("procesare coada start")
        if self.is_processing:
            return
        queued = [j for j in self.jobs if j.status == 'queued']
        if not queued:
            return

        self.is_processing = True
        self._process_btn.disable()
        self._cancel_btn.set_visibility(True)

        for job in queued:
            if not self.is_processing: break
            await self._run_job(job)

        self.is_processing = False
        self._process_btn.enable()
        self._cancel_btn.set_visibility(False)

    async def _cancel(self):
        # opreste procesul curent
        self.is_processing = False
        if self.running_proc:
            try:
                self.running_proc.terminate()
            except Exception:
                pass

    async def _compile(self):
        # ruleaza make
        ui.notify("Compilare...", type='info')
        cmd = ['make', 'all']
        if sys.platform == 'win32':
             cmd = ['wsl'] + cmd
        try:
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=str(PROJECT_DIR))
            out, _ = await proc.communicate()
            if proc.returncode == 0:
                ui.notify("OK", type='positive')
            else:
                ui.notify("Eroare compilare", type='negative')
        except Exception as e:
            ui.notify(f"Err: {e}", type='negative')

    def _update_progress(self, job: Job):
        # update bara de progres
        if self._progress:
            self._progress.set_value(job.progress)
        if self._progress_text:
            if job.status == 'running':
                self._progress_text.set_text(f"Frame {job.current_frame}/{job.total_frames}  •  {job.fps_str}")
            elif job.status == 'done':
                self._progress_text.set_text("Done")
            elif job.status == 'error':
                self._progress_text.set_text("Error")

    def _push_log(self, job: Job):
        # update log
        if not self._log_container:
            return
        self._log_container.clear()
        with self._log_container:
            for line in job.log[-60:]:
                ui.label(line).style('font-size: 11px; white-space: pre-wrap')
        if self._log_scroll:
            self._log_scroll.scroll_to(percent=1.0)

    def build(self):
        # construieste interfata
        ui.add_head_html(f'<style>{THEME_CSS}</style>')
        ui.dark_mode().enable()

        with ui.header().classes('hdr items-center px-6 py-3'):
            with ui.row().classes('items-center gap-3'):
                ui.icon('local_fire_department').classes('text-2xl').style('color: var(--accent-orange)')
                ui.html('<span class="grad-text" style="font-size:20px; font-weight:700;">GPU Seam Carving Studio</span>')
            ui.space()
            with ui.row().classes('gap-3 items-center'):
                v2ok = 'OK' if find_binary() else 'NO'
                ui.html(f'<span class="chip">{v2ok} Engine</span>')
                ui.button('Compile', icon='build', on_click=self._compile).classes('btn-sec').props('dense no-caps size=sm')

        with ui.column().classes('w-full mx-auto px-6 py-8 gap-6').style('max-width: 960px'):
            with ui.card().classes('glass w-full').style('padding: 28px'):
                ui.label('CONFIGURATION').classes('sec-title mb-4')
                videos = scan_videos()
                vid_options = {str(v): v.name for v in videos}
                with ui.row().classes('w-full gap-6 items-start'):
                    with ui.column().classes('flex-grow gap-4'):
                        with ui.row().classes('w-full items-center gap-2'):
                            self._video_select = ui.select(options=vid_options, label='Select Video', on_change=lambda e: self._on_video_change(e.value)).classes('flex-grow').props('dark outlined color="orange"')
                            ui.upload(on_upload=self._handle_upload, auto_upload=True, label="Upload").props('flat color="orange" icon="cloud_upload" dense').classes('w-48')
                        self._info_label = ui.label('Select a video').classes('dim-label -mt-2')
                    with ui.column().classes('items-center').style('min-width: 180px'):
                        self._thumb_img = ui.image('').classes('thumb-img').style('width: 180px')
                        self._thumb_img.set_visibility(False)

                ui.separator().style('background: var(--border-line); margin: 12px 0')

                with ui.row().classes('items-center gap-4 mb-2'):
                    ui.label('Mode:').classes('dim-label text-sm')
                    self._mode_toggle = ui.toggle({'shrink': 'Shrink', 'expand': 'Expand'}, value='shrink').props('no-caps rounded color="orange" text-color="white"').on('update:model-value', lambda _: self._update_dims())

                with ui.row().classes('w-full gap-8'):
                    with ui.column().classes('flex-grow gap-1'):
                        self._v_label = ui.label('Vertical Seams: 100').classes('dim-label text-sm')
                        self._v_slider = ui.slider(min=0, max=500, value=100, step=1).props('color="orange" label-always').on('update:model-value', lambda e: (self._v_label.set_text(f'Vertical Seams: {int(e.args)}'), self._update_dims(v_val=int(e.args))))
                    with ui.column().classes('flex-grow gap-1'):
                        self._h_label = ui.label('Horizontal Seams: 0').classes('dim-label text-sm')
                        self._h_slider = ui.slider(min=0, max=500, value=0, step=1).props('color="deep-orange" label-always').on('update:model-value', lambda e: (self._h_label.set_text(f'Horizontal Seams: {int(e.args)}'), self._update_dims(h_val=int(e.args))))

                with ui.row().classes('w-full items-center gap-4 mt-2'):
                    ui.label('Batch Size:').classes('dim-label text-sm')
                    self._batch_slider = ui.slider(min=1, max=100, value=1, step=1).classes('flex-grow').props('color="amber" label-always')

                with ui.expansion('ADVANCED QUALITY SETTINGS', icon='settings').classes('w-full glass mt-2'):
                    with ui.column().classes('w-full p-4 gap-4'):
                        with ui.column().classes('w-full gap-1'):
                            self._lambda_label = ui.label('JND Lambda (Flattening): 0.4').classes('dim-label text-sm')
                            self._lambda_slider = ui.slider(min=0.0, max=2.0, value=0.4, step=0.05).props('color="orange" label-always').on('update:model-value', lambda e: self._lambda_label.set_text(f'JND Lambda (Flattening): {e.args:.2f}'))
                            ui.label('Lower = preserves more detail. Higher = hides seams better in flat areas.').classes('text-xs text-dim')
                        
                        with ui.column().classes('w-full gap-1'):
                            self._skin_label = ui.label('Skin Bias: 1000').classes('dim-label text-sm')
                            self._skin_slider = ui.slider(min=0, max=5000, value=1000, step=100).props('color="deep-orange" label-always').on('update:model-value', lambda e: self._skin_label.set_text(f'Skin Bias: {int(e.args)}'))
                            ui.label('Protects human subjects.').classes('text-xs text-dim')

                        with ui.column().classes('w-full gap-1'):
                            self._motion_label = ui.label('Motion Weight: 5.0').classes('dim-label text-sm')
                            self._motion_slider = ui.slider(min=0.0, max=20.0, value=5.0, step=0.5).props('color="amber" label-always').on('update:model-value', lambda e: self._motion_label.set_text(f'Motion Weight: {e.args:.1f}'))
                            ui.label('Protects moving objects. If the subject is being carved, increase this.').classes('text-xs text-dim')

                self._dims_label = ui.label('').classes('result-dims mt-1')
                ui.separator().style('background: var(--border-line); margin: 12px 0')

                with ui.row().classes('w-full items-center justify-between'):
                    self._viz_switch = ui.switch('Visualize').props('color="orange"').style('color: var(--text-main)')
                    with ui.row().classes('gap-3'):
                        ui.button('Add to Queue', icon='add', on_click=self._add_job).classes('btn-sec').props('no-caps')
                        self._process_btn = ui.button('Process All', icon='rocket_launch', on_click=self._process_all).classes('btn-accent').props('no-caps')
                        self._cancel_btn = ui.button('Cancel', icon='stop', on_click=self._cancel).classes('btn-sec').props('no-caps')
                        self._cancel_btn.set_visibility(False)

            with ui.card().classes('glass w-full').style('padding: 28px'):
                with ui.row().classes('w-full items-center justify-between mb-4'):
                    ui.label('JOB QUEUE').classes('sec-title mb-0')
                    ui.button('Clear', icon='delete_sweep', on_click=self._clear_finished).classes('btn-sec').props('dense no-caps size=sm')
                self._job_container = ui.column().classes('w-full gap-2')
                with self._job_container:
                    ui.label('Queue is empty').classes('dim-label')

            with ui.card().classes('glass w-full').style('padding: 28px'):
                ui.label('PROCESSING').classes('sec-title mb-4')
                self._progress = ui.linear_progress(value=0, show_value=False).props('color="orange" rounded size="10px"').classes('w-full')
                self._progress_text = ui.label('Ready').classes('dim-label mt-2').style("font-size: 12px")
                self._log_scroll = ui.scroll_area().classes('log-box w-full mt-4').style('height: 220px')
                with self._log_scroll:
                    self._log_container = ui.column().classes('w-full gap-0 p-3')

            with ui.row().classes('w-full justify-center mt-2'):
                ui.label('GPU Seam Carving Studio').style('color: var(--text-dim); font-size: 11px')

def _open_file(path: str):
    # deschide fisierul rezultat
    try:
        if sys.platform == 'win32': os.startfile(path)
        elif sys.platform == 'darwin': sp.Popen(['open', path])
        else: sp.Popen(['xdg-open', path])
    except Exception: pass

studio = SeamCarvingStudio()
studio.build()

ui.run(title='GPU Seam Carving Studio', port=8080, reload=False)
