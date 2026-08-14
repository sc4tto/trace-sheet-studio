from __future__ import annotations

import queue
import threading
from dataclasses import replace
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from .engine import TraceResult, TraceSettings, prepare_raster, segment_from_samples, trace_image
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
        self.preview_geometry: dict[str, tuple[int, int, int, int, float]] = {}
        self.positive_samples: list[tuple[int, int]] = []
        self.negative_samples: list[tuple[int, int]] = []
        self.sample_after_id = None
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
        self.mode_var = tk.StringVar(value="Ricalco strutturale")
        combo = ttk.Combobox(box, state="readonly", textvariable=self.mode_var,
                             values=["Linee centrali", "Contorni delle sagome",
                                     "Analisi combinata", "Ricalco strutturale"], width=28)
        combo.pack(fill="x")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._mode_changed())
        self.mode_help = ttk.Label(box, text="Per planimetrie, aste e disegni tecnici.", wraplength=290)
        self.mode_help.pack(anchor="w", pady=(6, 0))
        ttk.Button(box, text="Ripristina preset intarsio",
                   command=self._apply_inlay_preset).pack(fill="x", pady=(6, 0))

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
        self.colors_var = tk.IntVar(value=20)
        self.colors_row = self._row_scale(box, 7, "Aree cromatiche", self.colors_var, 2, 24)
        self.region_color_var = tk.IntVar(value=28)
        self.region_color_row = self._row_scale(box, 8, "Differenza colore", self.region_color_var, 5, 80)
        self.min_region_var = tk.IntVar(value=80)
        self.min_region_row = self._row_scale(box, 9, "Area minima", self.min_region_var, 10, 2000)
        self.texture_suppression_var = tk.IntVar(value=17)
        self.texture_row = self._row_scale(box, 10, "Soppressione texture", self.texture_suppression_var, 1, 31)
        self.region_merge_var = tk.DoubleVar(value=7.0)
        self.region_merge_row = self._row_scale(box, 11, "Fusione aree", self.region_merge_var, 0.0, 35.0)
        self.structural_strength_var = tk.DoubleVar(value=0.50)
        self.structural_row = self._row_scale(
            box, 12, "Persistenza strutturale", self.structural_strength_var, 0.05, 0.95)
        ttk.Label(box, text="Vista strutturale").grid(row=13, column=0, sticky="w", pady=(4, 0))
        self.structural_view_var = tk.StringVar(value="Solo primarie")
        self.structural_view_combo = ttk.Combobox(
            box, state="readonly", textvariable=self.structural_view_var,
            values=["Solo primarie", "Primarie + secondarie", "Probabilità"], width=19)
        self.structural_view_combo.grid(row=13, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(4, 0))
        self.structural_view_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._schedule_live_preview())
        self.live_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Anteprima in tempo reale", variable=self.live_var,
                        command=self._schedule_live_preview).grid(row=14, column=0, columnspan=3, sticky="w", pady=(5, 0))
        box.columnconfigure(1, weight=1)

        box = ttk.LabelFrame(controls, text="4. Geometria", padding=7)
        box.pack(fill="x", pady=4)
        ttk.Label(box, text="Motore").grid(row=0, column=0, sticky="w")
        self.recognition_var = tk.StringVar(value="Ibrido")
        recognition_combo = ttk.Combobox(box, state="readonly", textvariable=self.recognition_var,
                                         values=["Monolinea", "Rette", "Curve morbide", "Ibrido", "Flussi direzionali"], width=18)
        recognition_combo.grid(row=0, column=1, sticky="e")
        self.recognition_combo = recognition_combo
        recognition_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_live_preview())
        self.minimum_var = tk.IntVar(value=10)
        self._row_scale(box, 1, "Tratto minimo (px)", self.minimum_var, 2, 80)
        self.simplify_var = tk.DoubleVar(value=1.5)
        self._row_scale(box, 2, "Semplificazione", self.simplify_var, 0.0, 8.0)
        self.line_tolerance_var = tk.DoubleVar(value=1.5)
        self.line_tolerance_row = self._row_scale(box, 3, "Tolleranza rette", self.line_tolerance_var, 0.2, 6.0)
        self.flow_coherence_var = tk.DoubleVar(value=0.42)
        self.flow_coherence_row = self._row_scale(box, 4, "Coerenza flusso", self.flow_coherence_var, 0.1, 0.9)
        self.flow_gap_var = tk.DoubleVar(value=26.0)
        self.flow_gap_row = self._row_scale(box, 5, "Unione frammenti", self.flow_gap_var, 2.0, 60.0)
        self.flow_angle_var = tk.DoubleVar(value=22.0)
        self.flow_angle_row = self._row_scale(box, 6, "Angolo massimo", self.flow_angle_var, 3.0, 45.0)
        self.preservation_var = tk.DoubleVar(value=0.82)
        self.preservation_row = self._row_scale(
            box, 7, "Conservazione scheletro", self.preservation_var, 0.0, 1.0)
        self.mm_var = tk.DoubleVar(value=1.0)
        ttk.Label(box, text="Scala (mm per pixel)").grid(row=8, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(box, from_=0.001, to=1000, increment=0.01, textvariable=self.mm_var, width=10).grid(row=8, column=1, sticky="e", pady=(6, 0))
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

        sample_box = ttk.LabelFrame(controls, text="6. Segmentazione per campioni", padding=7)
        sample_box.pack(fill="x", pady=4)
        ttk.Label(sample_box, text="Sull'Originale: sinistro = interno, destro = esterno",
                  wraplength=290).pack(anchor="w")
        self.sample_model_var = tk.StringVar(value="Gradiente lineare OKLab")
        model_combo = ttk.Combobox(sample_box, state="readonly", textvariable=self.sample_model_var,
                                   values=["Colore medio OKLab", "Gradiente lineare OKLab"], width=27)
        model_combo.pack(fill="x", pady=(5, 0))
        model_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_sample_preview())
        self.sample_tolerance_var = tk.DoubleVar(value=0.055)
        tolerance_frame = ttk.Frame(sample_box)
        tolerance_frame.pack(fill="x", pady=(5, 0))
        ttk.Label(tolerance_frame, text="Tolleranza").pack(side="left")
        ttk.Scale(tolerance_frame, from_=0.01, to=0.20, variable=self.sample_tolerance_var,
                  command=lambda _v: self._schedule_sample_preview()).pack(
                      side="right", fill="x", expand=True, padx=(8, 0))
        self.sample_radius_var = tk.IntVar(value=8)
        radius_frame = ttk.Frame(sample_box)
        radius_frame.pack(fill="x", pady=(5, 0))
        ttk.Label(radius_frame, text="Raggio campione").pack(side="left")
        ttk.Scale(radius_frame, from_=1, to=30, variable=self.sample_radius_var,
                  command=lambda _v: self._schedule_sample_preview()).pack(
                      side="right", fill="x", expand=True, padx=(8, 0))
        self.lightness_weight_var = tk.DoubleVar(value=0.35)
        light_frame = ttk.Frame(sample_box)
        light_frame.pack(fill="x", pady=(5, 0))
        ttk.Label(light_frame, text="Peso luminosità").pack(side="left")
        ttk.Scale(light_frame, from_=0.0, to=1.5, variable=self.lightness_weight_var,
                  command=lambda _v: self._schedule_sample_preview()).pack(
                      side="right", fill="x", expand=True, padx=(8, 0))
        self.largest_only_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sample_box, text="Conserva solo componente principale",
                        variable=self.largest_only_var,
                        command=self._schedule_sample_preview).pack(anchor="w", pady=(5, 0))
        self.fill_holes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sample_box, text="Ignora fori e venature interne",
                        variable=self.fill_holes_var,
                        command=self._schedule_sample_preview).pack(anchor="w")
        self.sample_count_label = ttk.Label(sample_box, text="Positivi: 0 | Negativi: 0")
        self.sample_count_label.pack(anchor="w", pady=(5, 0))
        ttk.Button(sample_box, text="Calcola regione dai campioni",
                   command=self.start_sample_segmentation).pack(fill="x", pady=(5, 0))
        ttk.Button(sample_box, text="Cancella campioni",
                   command=self.clear_samples).pack(fill="x", pady=(5, 0))

        self.notebook = ttk.Notebook(preview)
        self.notebook.pack(fill="both", expand=True)
        compare_tab = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(compare_tab, text="Confronto raster + vettore")
        compare_panes = ttk.Panedwindow(compare_tab, orient="horizontal")
        compare_panes.pack(fill="both", expand=True)
        for key, title_text, placeholder in [
            ("compare_raster", "Raster / regioni", "Genera il raster"),
            ("compare_overlay", "Vettore sovrapposto", "Genera il ricalco"),
        ]:
            side = ttk.LabelFrame(compare_panes, text=title_text, padding=4)
            compare_panes.add(side, weight=1)
            label = ttk.Label(side, text=placeholder, anchor="center", style="Preview.TLabel")
            label.pack(fill="both", expand=True)
            setattr(self, f"{key}_label", label)
            label.bind("<Configure>", lambda _e, k=key: self._refresh_preview(k))
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
            if key == "original":
                label.bind("<Button-1>", lambda event: self._add_sample(event, True))
                label.bind("<Button-3>", lambda event: self._add_sample(event, False))

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
        integer = isinstance(variable, tk.IntVar)
        value = ttk.Spinbox(
            parent, from_=start, to=end, textvariable=variable, width=6,
            increment=1 if integer else 0.1,
            format="%.0f" if integer else "%.2f",
            command=self._schedule_live_preview,
        )
        value.grid(row=row, column=2, sticky="e", padx=(5, 0), pady=(4, 0))
        value.bind("<Return>", lambda _e: self._schedule_live_preview(immediate=True))
        value.bind("<FocusOut>", lambda _e: self._schedule_live_preview())
        return label, scale, value

    def _apply_inlay_preset(self):
        """Recommended editable starting point for the reference wood-inlay photo."""
        self.mode_var.set("Ricalco strutturale")
        self.threshold_mode_var.set("Otsu automatica")
        self.threshold_var.set(185)
        self.sauvola_window_var.set(31)
        self.contrast_var.set(1.25)
        self.blur_var.set(2.0)
        self.close_var.set(1)
        self.invert_var.set(False)
        self.colors_var.set(20)
        self.region_color_var.set(28)
        self.min_region_var.set(80)
        self.texture_suppression_var.set(17)
        self.region_merge_var.set(7.0)
        self.structural_strength_var.set(0.50)
        self.structural_view_var.set("Solo primarie")
        self.recognition_var.set("Ibrido")
        self.minimum_var.set(10)
        self.simplify_var.set(1.5)
        self.line_tolerance_var.set(1.5)
        self.flow_coherence_var.set(0.42)
        self.flow_gap_var.set(26.0)
        self.flow_angle_var.set(22.0)
        self.preservation_var.set(0.82)
        self._mode_changed()

    def _threshold_mode_changed(self, schedule=True):
        threshold_active = self.mode_var.get() in {"Linee centrali", "Analisi combinata"}
        manual = self.threshold_mode_var.get() == "Manuale"
        sauvola = self.threshold_mode_var.get() == "Sauvola locale"
        for widget in self.threshold_row:
            widget.configure(state="normal" if threshold_active and manual else "disabled")
        for widget in self.sauvola_row:
            widget.configure(state="normal" if threshold_active and sauvola else "disabled")
        if schedule:
            self._schedule_live_preview()

    def _mode_changed(self):
        contours = self.mode_var.get() in {"Contorni delle sagome", "Analisi combinata"}
        combined = self.mode_var.get() == "Analisi combinata"
        structural = self.mode_var.get() == "Ricalco strutturale"
        if structural:
            help_text = "Conserva i bordi persistenti a più scale ed esclude la venatura fine."
        elif combined:
            help_text = "Combina aree cromatiche con soglia, fessure e linee persistenti."
        elif contours:
            help_text = "Per intarsi, campiture e confini fra aree cromatiche."
        else:
            help_text = "Per planimetrie, aste e disegni tecnici."
        self.mode_help.configure(text=help_text)
        state = "normal" if contours else "disabled"
        for row in (self.colors_row, self.region_color_row, self.min_region_row,
                    self.region_merge_row):
            for widget in row:
                widget.configure(state=state)
        for widget in self.texture_row:
            widget.configure(state="normal" if contours or structural else "disabled")
        for widget in self.structural_row:
            widget.configure(state="normal" if structural else "disabled")
        self.structural_view_combo.configure(state="readonly" if structural else "disabled")
        if contours and self.blur_var.get() < 2.0:
            self.blur_var.set(2.0)
        elif not contours and self.blur_var.get() > 1.5:
            self.blur_var.set(0.6)
        self.recognition_combo.configure(state="readonly")
        for widget in self.line_tolerance_row:
            widget.configure(state="normal")
        for row in (self.flow_coherence_row, self.flow_gap_row):
            for widget in row:
                widget.configure(state="normal" if combined or not contours else "disabled")
        for widget in self.flow_angle_row:
            widget.configure(state="normal")
        self._threshold_mode_changed(schedule=False)
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
        self.positive_samples.clear()
        self.negative_samples.clear()
        self._update_sample_label()
        self.file_label.configure(text=f"{self.source_path.name}\n{self.source_image.width} × {self.source_image.height} px")
        self._set_image("original", self.source_image)
        self.status_label.configure(text="Immagine caricata. Regola i parametri e genera il ricalco.")
        self._schedule_live_preview(immediate=True)

    def _settings(self):
        threshold_modes = {"Manuale": "manual", "Otsu automatica": "otsu", "Sauvola locale": "sauvola"}
        recognition_modes = {"Monolinea": "centerline", "Rette": "lines",
                             "Curve morbide": "curves", "Ibrido": "hybrid",
                             "Flussi direzionali": "flows"}
        threshold_mode = threshold_modes[self.threshold_mode_var.get()]
        mode_names = {"Linee centrali": "centerline",
                      "Contorni delle sagome": "contours",
                      "Analisi combinata": "combined",
                      "Ricalco strutturale": "structural"}
        structural_views = {"Solo primarie": "primary",
                            "Primarie + secondarie": "details",
                            "Probabilità": "probability"}
        return TraceSettings(
            mode=mode_names[self.mode_var.get()],
            threshold_mode=threshold_mode,
            threshold=round(self.threshold_var.get()), automatic_threshold=threshold_mode == "otsu",
            sauvola_window=round(self.sauvola_window_var.get()),
            invert=self.invert_var.get(), contrast=float(self.contrast_var.get()),
            blur_radius=float(self.blur_var.get()), close_gaps=round(self.close_var.get()),
            min_path_pixels=round(self.minimum_var.get()), simplify_pixels=float(self.simplify_var.get()),
            recognition_mode=recognition_modes[self.recognition_var.get()],
            line_tolerance=float(self.line_tolerance_var.get()),
            flow_coherence=float(self.flow_coherence_var.get()),
            flow_gap=float(self.flow_gap_var.get()),
            flow_angle=float(self.flow_angle_var.get()),
            colors=round(self.colors_var.get()),
            region_color_radius=round(self.region_color_var.get()),
            min_region_area=round(self.min_region_var.get()),
            texture_suppression=round(self.texture_suppression_var.get()),
            region_merge_delta=float(self.region_merge_var.get()),
            generator_angle=float(self.flow_angle_var.get()),
            structural_strength=float(self.structural_strength_var.get()),
            structural_view=structural_views[self.structural_view_var.get()],
            skeleton_preservation=float(self.preservation_var.get()),
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
        if self.recognition_var.get() == "Flussi direzionali":
            maximum = 450
        else:
            maximum = 600 if self.mode_var.get() in {
                "Contorni delle sagome", "Analisi combinata",
                "Ricalco strutturale"} else 750
        settings = replace(self._settings(), max_dimension=maximum)
        image = self.source_image.copy()
        threading.Thread(target=self._live_worker, args=(token, image, settings), daemon=True).start()

    def _live_worker(self, token, image, settings):
        try:
            result = trace_image(image, settings)
            self.events.put(("live", (token, result, settings)))
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

    def _add_sample(self, event, positive):
        geometry = self.preview_geometry.get("original")
        if self.source_image is None or geometry is None:
            return
        offset_x, offset_y, shown_w, shown_h, scale = geometry
        if not (offset_x <= event.x < offset_x + shown_w and offset_y <= event.y < offset_y + shown_h):
            return
        x = min(self.source_image.width - 1, max(0, round((event.x - offset_x) / scale)))
        y = min(self.source_image.height - 1, max(0, round((event.y - offset_y) / scale)))
        (self.positive_samples if positive else self.negative_samples).append((x, y))
        self._update_sample_label()
        self._set_image("original", self._marked_original())
        self._schedule_sample_preview()

    def _marked_original(self):
        marked = self.source_image.copy()
        draw = ImageDraw.Draw(marked)
        radius = max(3, round(max(marked.size) / 300))
        for x, y in self.positive_samples:
            draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                         fill="#00D020", outline="white", width=2)
        for x, y in self.negative_samples:
            draw.line((x-radius, y-radius, x+radius, y+radius), fill="#FF2020", width=3)
            draw.line((x-radius, y+radius, x+radius, y-radius), fill="#FF2020", width=3)
        return marked

    def _update_sample_label(self):
        if hasattr(self, "sample_count_label"):
            self.sample_count_label.configure(
                text=f"Positivi: {len(self.positive_samples)} | Negativi: {len(self.negative_samples)}")

    def clear_samples(self):
        self.positive_samples.clear()
        self.negative_samples.clear()
        self._update_sample_label()
        if self.source_image is not None:
            self._set_image("original", self.source_image)

    def _schedule_sample_preview(self):
        required = 3 if self.sample_model_var.get().startswith("Gradiente") else 1
        if len(self.positive_samples) < required:
            return
        if self.sample_after_id is not None:
            self.after_cancel(self.sample_after_id)
        self.sample_after_id = self.after(220, self.start_sample_segmentation)

    def start_sample_segmentation(self):
        self.sample_after_id = None
        required = 3 if self.sample_model_var.get().startswith("Gradiente") else 1
        if self.source_image is None or len(self.positive_samples) < required:
            messagebox.showwarning("Segmentazione per campioni",
                                   f"Aggiungi almeno {required} campioni positivi nella scheda Originale.")
            return
        self.progress.start(12)
        self.status_label.configure(text="Calcolo della regione OKLab dai campioni...")
        args = (self.source_image.copy(), self.positive_samples.copy(), self.negative_samples.copy(),
                float(self.sample_tolerance_var.get()), required == 3, float(self.simplify_var.get()),
                round(self.sample_radius_var.get()), float(self.lightness_weight_var.get()),
                self.largest_only_var.get(), self.fill_holes_var.get())
        threading.Thread(target=self._sample_worker, args=args, daemon=True).start()

    def _sample_worker(self, image, positive, negative, tolerance, linear, simplify,
                       radius, lightness_weight, keep_largest, fill_holes):
        try:
            sample = segment_from_samples(
                image, positive, negative, tolerance=tolerance,
                linear_gradient=linear, simplify_pixels=simplify,
                sample_radius=radius, lightness_weight=lightness_weight,
                keep_largest=keep_largest, fill_holes=fill_holes)
            self.events.put(("sample_done", sample))
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
                    token, result, settings = payload
                    if token != self.live_token:
                        continue
                    self._set_image("raster", result.raster)
                    self._set_image("skeleton", result.skeleton)
                    self._set_image("overlay", result.overlay)
                    if settings.mode in {"contours", "combined"}:
                        layer_count = len(result.vector_layers.get("01_TESSERE_CHIUSE", [])) if result.vector_layers else 0
                        description = (f"{layer_count} tessere | fusione {settings.region_merge_delta:.1f}"
                                       f" | texture {settings.texture_suppression}")
                    elif settings.mode == "structural":
                        secondary = len(result.vector_layers.get("05_DETTAGLI_SECONDARI", [])) if result.vector_layers else 0
                        description = (f"Strutturale {settings.structural_strength:.2f}"
                                       f" | secondarie {secondary}")
                    else:
                        black = 100.0 * result.raster.histogram()[0] / (result.raster.width * result.raster.height)
                        description = settings.threshold_mode.capitalize()
                        if settings.threshold_mode == "manual":
                            description += f" {settings.threshold}"
                        description += f" | nero {black:.1f}%"
                    self.summary.configure(text=f"{description} | {len(result.paths):,} vettori | anteprima {result.processing_scale:.3f}")
                    self.status_label.configure(text="Anteprima raster e vettoriale aggiornata in tempo reale.")
                    self.notebook.select(0)
                elif kind == "live_error":
                    token, _exc = payload
                    if token == self.live_token:
                        self.status_label.configure(text="Impossibile aggiornare l'anteprima rapida")
                elif kind == "sample_done":
                    sample = payload
                    self.result = TraceResult(
                        original=self.source_image.copy(), raster=sample.mask,
                        skeleton=sample.contour, overlay=sample.overlay,
                        paths=sample.paths, processing_scale=1.0,
                        source_size=self.source_image.size,
                    )
                    self.prepared_raster = sample.mask
                    self._set_image("raster", sample.mask)
                    self._set_image("skeleton", sample.contour)
                    self._set_image("overlay", sample.overlay)
                    self.summary.configure(
                        text=f"Regione: {sample.accepted_pixels:,} px | {len(sample.paths)} contorni chiusi")
                    self.status_label.configure(
                        text="Regione OKLab pronta. Aggiungi campioni o regola la tolleranza.")
                    self.raster_toolbar.configure(state="normal")
                    self.dxf_toolbar.configure(state="normal")
                    self.notebook.select(4)
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
                    self.notebook.select(2)
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
                    self.notebook.select(0)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _set_image(self, key, image):
        setattr(self, f"_{key}_image", image.copy())
        self._refresh_preview(key)
        if key == "raster":
            self._compare_raster_image = image.copy()
            self._refresh_preview("compare_raster")
        elif key == "overlay":
            self._compare_overlay_image = image.copy()
            self._refresh_preview("compare_overlay")

    def _refresh_preview(self, key):
        image = getattr(self, f"_{key}_image", None)
        label = getattr(self, f"{key}_label", None)
        if image is None or label is None:
            return
        width, height = max(100, label.winfo_width() - 12), max(100, label.winfo_height() - 12)
        preview = image.copy()
        preview.thumbnail((width, height), Image.Resampling.LANCZOS)
        scale = preview.width / image.width
        offset_x = max(0, (label.winfo_width() - preview.width) // 2)
        offset_y = max(0, (label.winfo_height() - preview.height) // 2)
        self.preview_geometry[key] = (offset_x, offset_y, preview.width, preview.height, scale)
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
            try:
                export_dxf(self.result, filename, float(self.mm_var.get()))
            except Exception as exc:
                messagebox.showerror("Esportazione DXF", f"Impossibile esportare il DXF:\n\n{exc}")
                self.status_label.configure(text="Esportazione DXF non riuscita")
                return
            self.status_label.configure(text=f"DXF salvato: {filename}")
            messagebox.showinfo("Esportazione DXF", f"File esportato correttamente:\n{filename}")


def run():
    TraceSheetApp().mainloop()
