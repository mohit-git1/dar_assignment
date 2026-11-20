Manual Public Page Collection Format for DarGlobal
===================================================

Since DarGlobal's website (https://www.darglobal.co.uk) blocks automated bot
traffic with Imperva Incapsula bot protection, public pages are collected manually
and placed in this folder.

File requirements:
1. Each file must be a plain text file ending with '.txt' (e.g. 'aida.txt', 'about.txt').
2. Line 1 must begin with 'URL: ' followed by the page URL.
   Example: URL: https://www.darglobal.co.uk/projects/aida
3. Line 2 must begin with 'TITLE: ' followed by the page title.
   Example: TITLE: AIDA by DarGlobal | Luxury Coastal Community in Oman
4. Line 3 onwards contains the readable public text from the page (descriptions, amenities, unit types, locations, etc.).
5. Files under 200 characters in total length will be skipped.

During ingestion (`python -m app.ingest`), all .txt files in this directory
are automatically parsed into:
  {"source": "darglobal", "url": "...", "title": "...", "text": "..."}
and synced to data/raw/darglobal.json before chunking and embedding.
