# Trace Sheet Studio

Applicazione desktop portatile per Windows dedicata alla preparazione raster e al ricalco
vettoriale di scansioni tecniche, planimetrie e fotografie. L'interfaccia riprende lo stile
Windows 2000 modernizzato di Pixel Sheet Converter, mentre motore, release e sviluppo sono
indipendenti.

Versione corrente: **0.8.0-test**.

La modalità **Ricalco strutturale** confronta i bordi a tre scale: le risposte
persistenti diventano curve generatrici primarie, mentre i dettagli presenti
soltanto alla scala fine vengono separati nel layer `05_DETTAGLI_SECONDARI`.

Ogni cursore dispone ora di un campo numerico sincronizzato e modificabile. Il
programma si apre con il preset **Intarsio – direttrici principali**, calibrato
sull'immagine di riferimento; il preset può essere ripristinato dal pannello
Tipo di ricalco senza impedire successive regolazioni manuali.

La scheda **Confronto raster + vettore** mostra simultaneamente regioni elaborate
e sovrapposizione geometrica. La nuova **Analisi combinata** integra segmentazione
cromatica e soglia nello stesso grafo di generatrici; il motore geometrico resta
selezionabile anche durante il riconoscimento delle sagome.

La ricostruzione delle sagome produce ora un grafo di bordi condivisi: ogni
separazione fisica è trasformata in un asse unico e i frammenti tangenti vengono
concatenati in curve generatrici massimali. Il DXF separa tessere chiuse, curve
generatrici, perimetro e venature su layer distinti.

La modalità **Flussi direzionali** usa un tensore di struttura locale per stimare
le direzioni dominanti e unisce i frammenti compatibili in direttrici più lunghe.
L'anteprima in tempo reale mostra ora anche le geometrie vettoriali sovrapposte
all'immagine prima dell'esportazione DXF.

## Funzioni del primo prototipo

- apertura di PNG, JPEG, BMP e TIFF;
- flusso separato **Originale → Raster preparato → Scheletro → Sovrapposizione**;
- anteprima raster prima della generazione vettoriale;
- aggiornamento raster in tempo reale durante la regolazione dei parametri;
- soglia manuale, Otsu automatica e Sauvola locale;
- salvataggio del raster preparato in PNG;
- modalità **Linee centrali** per planimetrie, aste e disegni tecnici;
- modalità sperimentale **Contorni delle sagome** per intarsi e aree cromatiche;
- soglia automatica o manuale, contrasto, sfocatura e chiusura delle interruzioni;
- assottigliamento monolinea e semplificazione dei tracciati;
- motori Monolinea, Rette, Curve morbide e Ibrido;
- segmentazione degli intarsi in spazio colore Lab tramite Mean Shift e clustering;
- fusione delle piccole regioni e controllo dell'area minima;
- contorni chiusi delle tessere, pronti per l'esportazione DXF;
- segmentazione assistita mediante campioni positivi e negativi;
- modello OKLab a colore medio o gradiente lineare 2D;
- crescita connessa della regione e aggiornamento della tolleranza;
- campioni circolari robusti con scarto delle venature anomale;
- peso separato della luminosità e della cromaticità;
- opzioni per la sola componente principale e il riempimento dei fori interni;
- conferma esplicita dell'esportazione DXF e messaggi diagnostici in caso di errore;
- DXF R2010 conforme, generato con ezdxf e verificato mediante riapertura automatica;
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

La modalità Curve morbide usa Chaikin come approssimazione stabile di una B-spline quadratica
e viene ancora esportata come polilinea campionata; l'entità DXF `SPLINE` nativa è prevista
nel checkpoint successivo. La modalità a sagome usa regioni cromatiche chiuse; la qualità
dipende ancora dalla differenza di colore fra legni adiacenti e dall'illuminazione. Ritaglio,
correzione prospettica, calibrazione grafica su una misura nota e livelli
DXF separati per colore sono previsti nei prossimi checkpoint.

## Licenza

[MIT](LICENSE)
