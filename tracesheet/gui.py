from __future__ import annotations

import queue
import threading
from dataclasses import replace
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .engine import TraceResult, TraceSettings, prepare_raster, trace_image
from .exporters import export_dxf, export_raster
from .version import APP_VERSION


class TraceSheetApp(tk.Tk):
    FACE = "#D4D0C8"
    LIGHT = "#FFFFFF"
    SHADOW = "#808080"
    BLUE = "#000080"

    def __init__(self):
        super().__init__()
        self.title(f"Trace Sheet Studio {APP_VERSION}")
        self.geometry("1280x840")
        self.minsize(1080, 720)
        self.configure(background=self.FACE)
        self.source_path: Path | None = None
        self.source_image: Image.Image | None = None
        self.result: TraceResult | None = None
        self.prepared_raster: Image.Image | None = None
        self.prepared_settings: TraceSettings | None = None
        self.preview_refs: dict[str, ImageTk.PhotoImage] = {}
        self.events: queue.Queue = queue.Queue()
        self.live_after_id: str | None = None
        self.live_token = 0
        self._style()
        self._ui()
        self.after(100, self._poll)

    def _style(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Tahoma", 9), background=self.FACE, foreground="#000")
        style.configure("TFrame", background=self.FACE)
        style.configure("TLabel", background=self.FACE)
        style.configure("TButton", background=self.FACE, padding=(7, 4), borderwidth=2, relief="raised")
        style.map("TButton", background=[("pressed", "#B8B4AC"), ("active", "#E4E0D8")])
        style.configure("Primary.TButton", font=("Tahoma", 9, "bold"), padding=(7, 5))
        style.configure("TLabelframe", background=self.FACE, borderwidth=2, relief="groove")
        style.configure("TLabelframe.Label", background=self.FACE, font=("Tahoma", 9, "bold"))
        style.configure("TNotebook", background=self.FACE, borderwidth=2)
        style.configure("TNotebook.Tab", background=self.FACE, padding=(11, 4))
        style.map("TNotebook.Tab", background=[("selected", self.LIGHT)])
        style.configure("Preview.TLabel", background=self.LIGHT, relief="sunken", borderwidth=2)

    def _menu(self):
        menu = tk.Menu(self, tearoff=False)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Apri immagine...\tCtrl+O", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Salva raster PNG...", command=self.save_raster)
        file_menu.add_command(label="Esporta DXF...", command=self.save_dxf)
        file_menu.add_separator()
        file_menu.add_command(label="Esci", command=self.destroy)
        menu.add_cascade(label="File", menu=file_menu)
        trace_menu = tk.Menu(menu, tearoff=False)
        trace_menu.add_command(label="Prepara raster\tF5", command=self.start_prepare)
        trace_menu.add_command(label="Genera ricalco", command=self.start_trace)
        menu.add_cascade(label="Ricalco", menu=trace_menu)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="Informazioni", command=self.about)
        menu.add_cascade(label="?", menu=help_menu)
        self.config(menu=menu)
        self.bind_all("<Control-o>", lambda _e: self.open_image())
        self.bind_all("<F5>", lambda _e: self.start_prepare())

    def _ui(self):
        self._menu()
        root = ttk.Frame(self, padding=6)
        root.pack(fill="both", expand=True)
        title = tk.Frame(root, background=self.BLUE, relief="sunken", borderwidth=2)
        title.pack(fill="x", pady=(0, 5))
        tk.Label(title, text=" Trace Sheet Studio", font=("Tahoma", 10, "bold"),
                 background=self.BLUE, foreground=self.LIGHT, padx=4, pady=3).pack(side="left")
        tk.Label(title, text=f"Versione {APP_VERSION} ", font=("Tahoma", 8),
                 background=self.BLUE, foreground=self.LIGHT).pack(side="right")

        toolbar = tk.Frame(root, background=self.FACE, relief="raised", borderwidth=1)
        toolbar.pack(fill="x", pady=(0, 5))
        ttk.Button(toolbar, text="Apri...", command=self.open_image).pack(side="left", padx=2, pady=2)
        ttk.Button(toolbar, text="Prepara raster", command=self.start_prepare).pack(side="left", padx=2, pady=2)
        self.trace_toolbar = ttk.Button(toolbar, text="Genera ricalco", command=self.start_trace, state="disabled")
        self.trace_toolbar.pack(side="left", padx=2, pady=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=4, pady=3)
        self.raster_toolbar = ttk.Button(toolbar, text="Salva PNG", command=self.save_raster, state="disabled")
        self.raster_toolbar.pack(side="left", padx=2, pady=2)
        self.dxf_toolbar = ttk.Button(toolbar, text="Esporta DXF", command=self.save_dxf, state="disabled")
        self.dxf_toolbar.pack(side="left", padx=2, pady=2)
        ttk.Label(toolbar, text="Raster → scheletro → geometria DXF", anchor="e").pack(side="right", padx=8)

        panes = ttk.Panedwindow(root, orient="horizontal")
        panes.pack(fill="both", expand=True)
        controls = ttk.Frame(panes, padding=(0, 0, 6, 0), width=330)
        preview = ttk.Frame(panes)
        panes.add(controls, weight=0)
        panes.add(preview, weight=1)

        box = ttk.LabelFrame(controls, text="1. Immagine", padding=7)
        box.pack(fill="x", pady=(0, 4))
        ttk.Button(box, text="Apri immagine...", command=self.open_image).pack(fill="x")
        self.file_label = ttk.Label(box, text="Nessuna immagine selezionata", wraplength=290)
        self.file_label.pack(anchor="w", pady=(7, 0))

        box = ttk.LabelFrame(controls, text="2. Tipo di ricalco", padding=7)
        box.pack(fill="x", pady=4)
        self.mode_var = tk.StringVar(value="Linee centrali")
        combo = ttk.Combobox(box, state="readonly", textvariable=self.mode_var,
                             values=["Linee centrali", "Contorni delle sagome"], width=28)
        combo.pack(fill="x")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._mode_changed())
        self.mode_help = ttk.Label(box, text="Per planimetrie, aste e disegni tecnici.", wraplength=290)
        self.mode_help.pack(anchor="w", pady=(6, 0))

        box = ttk.LabelFrame(controls, text="3. Preparazione raster", padding=7)
        box.pack(fill="x", pady=4)
        ttk.Label(box, text="Metodo soglia").grid(row=0, column=0, sticky="w")
        self.threshold_mode_var = tk.StringVar(value="Otsu automatica")
        threshold_combo = ttk.Combobox(box, state="readonly", textvariable=self.threshold_mode_var,
                                       values=["Manuale", "Otsu automatica", "Sauvola locale"], width=17)
        threshold_combo.grid(row=0, column=1, sticky="e")
        threshold_combo.bind("<<ComboboxSelected>>", lambda _e: self._threshold_mode_changed())
        self.threshold_var = tk.IntVar(value=185)
        self.threshold_row = self._row_scale(box, 1, "Soglia", self.threshold_var, 0, 255)
        self.sauvola_window_var = tk.IntVar(value=31)
        self.sauvola_row = self._row_scale(box, 2, "Finestra Sauvola", self.sauvola_window_var, 9, 81)
        self.contrast_var = tk.DoubleVar(value=1.25)
        self._row_scale(box, 3, "Contrasto", self.contrast_var, 0.2, 3.0)
        self.blur_var = tk.DoubleVar(value=0.6)
        self._row_scale(box, 4, "Riduzione rumore", self.blur_var, 0.0, 3.0)
        self.close_var = tk.IntVar(value=1)
        self._row_scale(box, 5, "Chiudi interruzioni", self.close_var, 0, 3)
        self.invert_var = tk.BooleanVar(value=False)
        invert_check = ttk.Checkbutton(box, text="Inverti bianco/nero", variable=self.invert_var,
                                       command=self._schedule_live_preview)
        invert_check.grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.colors_var = tk.IntVar(value=8)
        self.colors_row = self._row_scale(box, 7, "Aree cromatiche", self.colors_var, 2, 24)
        self.region_color_var = tk.IntVar(value=28)
        self.region_color_row = self._row_scale(box, 8, "Differenza colore", self.region_color_var, 5, 80)
        self.min_region_var = tk.IntVar(value=100)
        self.min_region_row = self._row_scale(box, 9, "Area minima", self.min_region_var, 10, 2000)
        self.live_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Anteprima in tempo reale", variable=self.live_var,
                        command=self._schedule_live_preview).grid(row=10, column=0, columnspan=2, sticky="w", pady=(5, 0))
        box.columnconfigure(1, weight=1)

        box = ttk.LabelFrame(controls, text="4. Geometria", padding=7)
        box.pack(fill="x", pady=4)
        ttk.Label(box, text="Motore").grid(row=0, column=0, sticky="w")
        self.recognition_var = tk.StringVar(value="Ibrido")
        recognition_combo = ttk.Combobox(box, state="readonly", textvariable=self.recognition_var,
                                         values=["Monolinea", "Rette", "Curve morbide", "Ibrido"], width=14)
        recognition_combo.grid(row=0, column=1, sticky="e")
        recognition_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_live_preview())
        self.minimum_var = tk.IntVar(value=8)
        self._row_scale(box, 1, "Tratto minimo (px)", self.minimum_var, 2, 80)
        self.simplify_var = tk.DoubleVar(value=1.5)
        self._row_scale(box, 2, "Semplificazione", self.simplify_var, 0.0, 8.0)
        self.line_tolerance_var = tk.DoubleVar(value=1.5)
        self._row_scale(box, 3, "Tolleranza rette", self.line_tolerance_var, 0.2, 6.0)
        self.mm_var = tk.DoubleVar(value=1.0)
        ttk.Label(box, text="Scala (mm per pixel)").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(box, from_=0.001, to=1000, increment=0.01, textvariable=self.mm_var, width=10).grid(row=4, column=1, sticky="e", pady=(6, 0))
        box.columnconfigure(1, weight=1)

        box = ttk.LabelFrame(controls, text="5. Elaborazione", padding=7)
        box.pack(fill="x", pady=4)
        self.prepare_button = ttk.Button(box, text="1. Prepara e mostra raster", style="Primary.TButton", command=self.start_prepare)
        self.prepare_button.pack(fill="x")
        self.trace_button = ttk.Button(box, text="2. Genera ricalco dal raster", command=self.start_trace, state="disabled")
        self.trace_button.pack(fill="x", pady=(6, 0))
        self.progress = ttk.Progressbar(box, mode="indeterminate")
        self.progress.pack(fill="x", pady=(7, 0))
        self.summary = ttk.Label(box, text="Nessun ricalco generato", wraplength=290)
        self.summary.pack(anchor="w", pady=(6, 0))

        self.notebook = ttk.Notebook(preview)
        self.notebook.pack(fill="both", expand=True)
        for key, title_text, placeholder in [
            ("original", "Originale", "Apri un'immagine"),
            ("raster", "Raster preparato", "Genera il raster"),
            ("skeleton", "Scheletro", "Genera il ricalco"),
            ("overlay", "Sovrapposizione", "Controlla il risultato"),
        ]:
            tab = ttk.Frame(self.notebook, padding=5)
            self.notebook.add(tab, text=title_text)
            label = ttk.Label(tab, text=placeholder, anchor="center", style="Preview.TLabel")
            label.pack(fill="both", expand=True)
            setattr(self, f"{key}_label", label)
            label.bind("<Configure>", lambda _e, k=key: self._refresh_preview(k))

        status = tk.Frame(root, background=self.FACE, relief="sunken", borderwidth=2)
        status.pack(fill="x", pady=(5, 0))
        self.status_label = tk.Label(status, text="Pronto", font=("Tahoma", 8), background=self.FACE, anchor="w", padx=4, pady=2)
        self.status_label.pack(fill="x")
        self._mode_changed()
        self._threshold_mode_changed(schedule=False)

    def _row_scale(self, parent, row, text, variable, start, end):
        label = ttk.Label(parent, text=text)
        label.grid(row=row, column=0, sticky="w", pady=(4, 0))
        scale = ttk.Scale(parent, from_=start, to=end, variable=variable,
                          command=lambda _value: self._schedule_live_preview())
        scale.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))
        return label, scale

    def _threshold_mode_changed(self, schedule=True):
        manual = self.threshold_mode_var.get() == "Manuale"
        sauvola = self.threshold_mode_var.get() == "Sauvola locale"
        for widget in self.threshold_row:
            widget.configure(state="normal" if manual else "disabled")
        for widget in self.sauvola_row:
            widget.configure(state="normal" if sauvola else "disabled")
        if schedule:
            self._schedule_live_preview()

    def _mode_changed(self):
        contours = self.mode_var.get() == "Contorni delle sagome"
        self.mode_help.configure(text=("Per intarsi, campiture e confini fra aree cromatiche."
                                       if contours else "Per planimetrie, aste e disegni tecnici."))
        state = "normal" if contours else "disabled"
        for row in (self.colors_row, self.region_color_row, self.min_region_row):
            for widget in row:
                widget.configure(state=state)
        if contours and self.blur_var.get() < 2.0:
            self.blur_var.set(2.0)
        elif not contours and self.blur_var.get() > 1.5:
            self.blur_var.set(0.6)
        self._schedule_live_preview()

    def about(self):
        messagebox.showinfo("Trace Sheet Studio", f"Trace Sheet Studio {APP_VERSION}\n\nAnteprima raster, ricalco monolinea e DXF.\nInterfaccia Windows 2000 modernizzata.\n\nLicenza MIT")

    def open_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Immagini", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Tutti i file", "*.*")])
        if not filename:
            return
        try:
            with Image.open(filename) as opened:
                self.source_image = opened.convert("RGB")
        except Exception as exc:
            messagebox.showerror("Apertura immagine", str(exc))
            return
        self.source_path = Path(filename)
        self.result = None
        self.prepared_raster = None
        self.prepared_settings = None
        self.trace_button.configure(state="disabled")
        self.trace_toolbar.configure(state="disabled")
        self.raster_toolbar.configure(state="disabled")
        self.dxf_toolbar.configure(state="disabled")
        self.file_label.configure(text=f"{self.source_path.name}\n{self.source_image.width} × {self.source_image.height} px")
        self._set_image("original", self.source_image)
        self.status_label.configure(text="Immagine caricata. Regola i parametri e genera il ricalco.")
        self._schedule_live_preview(immediate=True)

    def _settings(self):
        threshold_modes = {"Manuale": "manual", "Otsu automatica": "otsu", "Sauvola locale": "sauvola"}
        recognition_modes = {"Monolinea": "centerline", "Rette": "lines",
                             "Curve morbide": "curves", "Ibrido": "hybrid"}
        threshold_mode = threshold_modes[self.threshold_mode_var.get()]
        return TraceSettings(
            mode="contours" if self.mode_var.get() == "Contorni delle sagome" else "centerline",
            threshold_mode=threshold_mode,
            threshold=round(self.threshold_var.get()), automatic_threshold=threshold_mode == "otsu",
            sauvola_window=round(self.sauvola_window_var.get()),
            invert=self.invert_var.get(), contrast=float(self.contrast_var.get()),
            blur_radius=float(self.blur_var.get()), close_gaps=round(self.close_var.get()),
            min_path_pixels=round(self.minimum_var.get()), simplify_pixels=float(self.simplify_var.get()),
            recognition_mode=recognition_modes[self.recognition_var.get()],
            line_tolerance=float(self.line_tolerance_var.get()),
            colors=round(self.colors_var.get()),
            region_color_radius=round(self.region_color_var.get()),
            min_region_area=round(self.min_region_var.get()),
        )

    def _schedule_live_preview(self, immediate=False):
        if self.source_image is None:
            return
        self.prepared_raster = None
        self.prepared_settings = None
        self.result = None
        if hasattr(self, "trace_button"):
            self.trace_button.configure(state="disabled")
            self.trace_toolbar.configure(state="disabled")
            self.raster_toolbar.configure(state="disabled")
            self.dxf_toolbar.configure(state="disabled")
        if not hasattr(self, "live_var") or not self.live_var.get():
            self.status_label.configure(text="Parametri modificati. Prepara nuovamente il raster.")
            return
        if self.live_after_id is not None:
            self.after_cancel(self.live_after_id)
        delay = 1 if immediate else 180
        self.live_after_id = self.after(delay, self._start_live_preview)

    def _start_live_preview(self):
        self.live_after_id = None
        if self.source_image is None:
            return
        self.live_token += 1
        token = self.live_token
        maximum = 650 if self.mode_var.get() == "Contorni delle sagome" else 900
        settings = replace(self._settings(), max_dimension=maximum)
        image = self.source_image.copy()
        threading.Thread(target=self._live_worker, args=(token, image, settings), daemon=True).start()

    def _live_worker(self, token, image, settings):
        try:
            raster, _mask, scale = prepare_raster(image, settings)
            self.events.put(("live", (token, raster, scale, settings)))
        except Exception as exc:
            self.events.put(("live_error", (token, exc)))

    def start_prepare(self):
        if self.source_image is None:
            messagebox.showwarning("Preparazione raster", "Apri prima un'immagine.")
            return
        self.prepare_button.configure(state="disabled")
        self.trace_button.configure(state="disabled")
        self.progress.start(12)
        self.status_label.configure(text="Preparazione del raster...")
        image, settings = self.source_image.copy(), self._settings()
        threading.Thread(target=self._prepare_worker, args=(image, settings), daemon=True).start()

    def _prepare_worker(self, image, settings):
        try:
            raster, _mask, scale = prepare_raster(image, settings)
            self.events.put(("prepared", (raster, scale, settings)))
        except Exception as exc:
            self.events.put(("error", exc))

    def start_trace(self):
        if self.source_image is None or self.prepared_raster is None:
            messagebox.showwarning("Ricalco", "Prepara e controlla prima il raster.")
            return
        self.prepare_button.configure(state="disabled")
        self.trace_button.configure(state="disabled")
        self.progress.start(12)
        self.status_label.configure(text="Generazione dello scheletro e dei tracciati DXF...")
        image = self.source_image.copy()
        settings = self.prepared_settings or self._settings()
        threading.Thread(target=self._worker, args=(image, settings), daemon=True).start()

    def _worker(self, image, settings):
        try:
            self.events.put(("done", trace_image(image, settings)))
        except Exception as exc:
            self.events.put(("error", exc))

    def _poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self.prepare_button.configure(state="normal")
                self.progress.stop()
                if kind == "error":
                    messagebox.showerror("Ricalco", str(payload))
                    self.status_label.configure(text="Errore durante il ricalco")
                elif kind == "live":
                    token, raster, scale, settings = payload
                    if token != self.live_token:
                        continue
                    self._set_image("raster", raster)
                    if settings.mode == "contours":
                        description = f"Regioni {settings.colors} colori | area ≥ {settings.min_region_area} px"
                    else:
                        black = 100.0 * raster.histogram()[0] / (raster.width * raster.height)
                        description = settings.threshold_mode.capitalize()
                        if settings.threshold_mode == "manual":
                            description += f" {settings.threshold}"
                        description += f" | nero {black:.1f}%"
                    self.summary.configure(text=f"{description} | anteprima {scale:.3f}")
                    self.status_label.configure(text="Anteprima raster aggiornata in tempo reale.")
                    self.notebook.select(1)
                elif kind == "live_error":
                    token, _exc = payload
                    if token == self.live_token:
                        self.status_label.configure(text="Impossibile aggiornare l'anteprima rapida")
                elif kind == "prepared":
                    self.prepared_raster, scale, self.prepared_settings = payload
                    self.result = None
                    self._set_image("raster", self.prepared_raster)
                    self.summary.configure(text=f"Raster pronto | scala elaborazione {scale:.3f}")
                    self.status_label.configure(text="Raster pronto. Controllalo prima di generare il ricalco.")
                    self.raster_toolbar.configure(state="normal")
                    self.dxf_toolbar.configure(state="disabled")
                    self.trace_button.configure(state="normal")
                    self.trace_toolbar.configure(state="normal")
                    self.notebook.select(1)
                else:
                    self.result = payload
                    self._set_image("raster", payload.raster)
                    self._set_image("skeleton", payload.skeleton)
                    self._set_image("overlay", payload.overlay)
                    self.summary.configure(text=f"{len(payload.paths):,} tracciati vettoriali | scala elaborazione {payload.processing_scale:.3f}")
                    self.status_label.configure(text="Ricalco completato. Controlla le anteprime prima di esportare.")
                    self.raster_toolbar.configure(state="normal")
                    self.dxf_toolbar.configure(state="normal")
                    self.trace_button.configure(state="normal")
                    self.trace_toolbar.configure(state="normal")
                    self.notebook.select(3)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _set_image(self, key, image):
        setattr(self, f"_{key}_image", image.copy())
        self._refresh_preview(key)

    def _refresh_preview(self, key):
        image = getattr(self, f"_{key}_image", None)
        label = getattr(self, f"{key}_label", None)
        if image is None or label is None:
            return
        width, height = max(100, label.winfo_width() - 12), max(100, label.winfo_height() - 12)
        preview = image.copy()
        preview.thumbnail((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(preview)
        self.preview_refs[key] = photo
        label.configure(image=photo, text="")

    def save_raster(self):
        raster = self.result.raster if self.result is not None else self.prepared_raster
        if raster is None:
            messagebox.showwarning("Esportazione", "Prepara prima il raster.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if filename:
            export_raster(raster, filename)
            self.status_label.configure(text=f"Raster salvato: {filename}")

    def save_dxf(self):
        if self.result is None:
            messagebox.showwarning("Esportazione", "Genera prima il ricalco.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".dxf", filetypes=[("DXF", "*.dxf")])
        if filename:
            export_dxf(self.result, filename, float(self.mm_var.get()))
            self.status_label.configure(text=f"DXF salvato: {filename}")


def run():
    TraceSheetApp().mainloop()
