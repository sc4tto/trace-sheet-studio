# Trace Sheet Studio

Applicazione desktop portatile per Windows dedicata alla preparazione raster e al ricalco
vettoriale di scansioni tecniche, planimetrie e fotografie. L'interfaccia riprende lo stile
Windows 2000 modernizzato di Pixel Sheet Converter, mentre motore, release e sviluppo sono
indipendenti.

Versione corrente: **0.1.0-test**.

## Funzioni del primo prototipo

- apertura di PNG, JPEG, BMP e TIFF;
- flusso separato **Originale → Raster preparato → Scheletro → Sovrapposizione**;
- anteprima raster prima della generazione vettoriale;
- salvataggio del raster preparato in PNG;
- modalità **Linee centrali** per planimetrie, aste e disegni tecnici;
- modalità sperimentale **Contorni delle sagome** per intarsi e aree cromatiche;
- soglia automatica o manuale, contrasto, sfocatura e chiusura delle interruzioni;
- assottigliamento monolinea e semplificazione dei tracciati;
- esportazione DXF ASCII con `LWPOLYLINE`, unità in millimetri e livello `RICALCO`.

## Avvio dal sorgente

Richiede Python 3.12 a 64 bit.

```bash
python -m pip install -r requirements.txt
python main.py
```

Su Windows è disponibile anche `avvia_trace_sheet.bat`.

## Build portatile Windows

Eseguire:

```text
build_trace_sheet.bat
```

L'applicazione verrà generata in `dist\TraceSheetStudio\TraceSheetStudio.exe`.

Una build automatica viene generata anche da GitHub Actions. Aprire la scheda **Actions**,
selezionare l'ultima esecuzione riuscita di **Windows portable build** e scaricare l'artifact
`TraceSheetStudio-Windows-x64` dalla sezione Artifacts. Estrarre l'intero archivio prima di
avviare `TraceSheetStudio.exe`.

## Stato e limiti noti

La modalità monolinea è il percorso principale del primo checkpoint. La modalità a sagome è
ancora sperimentale: fotografie con venature o texture marcate possono generare troppi
segmenti. Ritaglio, correzione prospettica, calibrazione grafica su una misura nota e livelli
DXF separati per colore sono previsti nei prossimi checkpoint.

## Licenza

[MIT](LICENSE)
