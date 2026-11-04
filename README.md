# DocumentParsing

## Roadmap

- [ ] Have close to perfect parsing on:
  - [x] readable pdfs -- found DocLayout-YOLO with amazing performance and GNU license
  - [ ] scanned pdfs -- to find good OCR with permisive license (I will check PaddleOCR first)

- [ ] Pipeline to match image (table) footnotes, image (table) captions and images (tables)

- [ ] Detector for text regions that have not been succesfully parsed. Needed for routing
      challenging text regions to expensive VLM (treat similarly to images/tables).
      Fallback for when the OCR model fails.

- [ ] Module that sends (crop, footnote, caption) tuple, optionally with some context from
      page to a VLM via API call, that can output the, say, LaTeX code to generate an
      approximation of that table, graph, math formula, or detect that it is a natural
      image and return a caption as such

- [ ] Coarse clustering of JSON output bundle based on keywords to reduce token count.
  - [ ] Pure wording
  - [ ] + localization (x pages before, y pages after) -- text regions may be split across
        pages, or longer context needed

- [ ] Module that sends API call to more economically viable LLM for text processing
  - [ ] text regions
  - [ ] text regions + results from VLM if in local context window

- [ ] Pipeline for selecting ground truth for validation.

- [ ] Create a dataset and train (if finetuning necessary / development), validation and test splits

- [ ] Metrics and benchmarking: TP, FP, TN, FN, precision, recall, F1, etc.

- [ ] Token/Cloud compute usage dashboard/tracker.

- [ ] Prompt optimization

- [ ] Cost and token usage reduction

- [ ] Speed, latency and resource consumption (VRAM, RAM, Disk Space) optimizations


## Requirements

- Access to GPU (dirt cheap GPU that can run CUDA 12.6 is enough)
- Python Environment with PyTorch and pdf libraries installed (TODO: document libraries properly), ideally conda
- API key to dirt cheap text model that can understand legal text
- API key to VLM to be used for images, tables (and as fallback for hard regions)
- Some database for storing files, simplest would be a key-value store

## Runtime 

- PDF parser for readable documents runs in real time (based on YOLO). On my laptop, with no flash_attn, 62 pages document in 0.9 seconds.
- TBD other components


## Costs

## Compliance

- PDF parsing via DocLayout-YOLO is local. License allows free commercial use.
