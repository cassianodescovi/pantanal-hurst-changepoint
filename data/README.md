# Data

## ANA records are not redistributed

This repository does **not** contain files downloaded from the Brazilian National Water and Sanitation Agency (ANA). Out of caution regarding ANA's terms of use, the raw records, and series derived directly from them, are not published here. Everything else (code, results, figures) is. The scripts rebuild all processed series from the files you download yourself.

## Source

- **Provider:** Agência Nacional de Águas e Saneamento Básico (ANA), Hidroweb system
- **Portal:** <https://www.snirh.gov.br/hidroweb/serieshistoricas>
- **Station:** Ladário, Paraguay River (ANA code **66825000**)
- **Variable:** daily river stage ("Cotas")
- **Period used:** 1 January 1900 to 31 December 2025
- **Access date:** [to be filled in]

Alternative source, to be confirmed before publication: ANA also publishes series from conventional stations at <https://github.com/anagovbr/hidro-dados-estacoes-convencionais>. Check whether station 66825000 is available there and, if so, prefer that link.

Please cite ANA as the data source.

## How to obtain and place the files

1. Download the daily stage series for station 66825000 from Hidroweb.
2. Save the file as `66825000_Cotas.csv` in `data/raw/`.

## Expected file format

The Hidroweb export has the following structure, which `01_preprocessing` assumes:

- Latin-1 encoding, semicolon-separated, with a block of metadata lines before the table (the table header is on line 16)
- One row per month, with the daily values stored side by side in columns `Cota01` to `Cota31`, and matching `Cota01Status` to `Cota31Status` columns
- Several rows can exist for the same month: instantaneous readings (07:00, 17:00), the daily-summary row, and, for older data, both raw and consistency-reviewed versions. Only the daily-summary rows are used.

## What the preprocessing does

`scripts/01_preprocessing` reads the file above and produces, in `data/processed/` (not tracked by git):

- the daily series, one value per calendar day, 1900–2025 (46,021 days);
- the annual-mean series used in the change-point and Hurst analyses.

Steps: keep daily-summary rows only; where raw and consistency-reviewed versions of the same month exist, keep the one with more non-missing values (ties go to the consistency-reviewed version); convert the monthly rows to a daily sequence; fill gaps of up to three days by linear interpolation (27 days, 0.06%); compute the daily climatology and anomalies for the deseasonalized series.

## Quality summary

About 99.5% of days carry the ANA "real" quality flag. The unflagged days (206) are concentrated in 2024–2025, consistent with the lag between telemetric collection and formal consistency review. No days carry a "doubtful" or "dry-gauge" flag.