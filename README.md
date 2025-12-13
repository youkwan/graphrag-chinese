# graphrag chinese

## Installation

```bash
git clone https://github.com/youkwan/graphrag-chinese.git
cd graphrag-chinese
```

```bash
uv sync
```

## Prepare Data & Settings

### Download and extract dataset

```bash
curl -L "https://drive.google.com/uc?export=download&id=1wKoQY56kmfPSzOdrVT8k4J4mf04iihqb" -o raw.zip
```

Linux / macOS:

```bash
unzip raw.zip -d raw
```

Windows (PowerShell):

```powershell
Expand-Archive -Path "./raw.zip" -DestinationPath "./raw"
```

### Set up input directory

```bash
mkdir -p ./data/input
```

Convert `.doc` documents to plain text:

```bash
uv run graphrag to-txt -s ./raw/ -o ./data/input/ --overwrite
```

### Pre-chunk for Chinese

The default Graphrag chunker may not handle Chinese content optimally. Use this helper script to pre-chunk your data before indexing. You can adjust the `chunk-size` and `chunk-overlap` parameters as needed to suit your dataset.

```bash
uv run graphrag pre-chunk -s ./data/input/ -o ./data/input_chunked --chunk-size 800 --chunk-overlap 400
```

### Copy prompts

Linux / macOS:

```bash
cp -r ./templates/prompts ./templates/prompts_chinese ./data/
```

Windows (PowerShell):

```powershell
Copy-Item -Path ./templates/prompts, ./templates/prompts_chinese -Destination ./data/ -Recurse
```

### Set up API keys

Linux / macOS:

```bash
cp ./templates/.env.example ./data/.env
```

Windows (PowerShell):

```powershell
Copy-Item -Path "./templates/.env.example" -Destination "./data/.env"
```

Fill in `GRAPHRAG_API_KEY` in `.env`.

## Choose settings

Settings are pre-tuned; copy the desired variant into `data/settings.yaml`.

### Baseline

- Linux / macOS:

  ```bash
  cp ./templates/default.yaml ./data/settings.yaml
  ```

- Windows (PowerShell):

  ```powershell
  Copy-Item -Path "./templates/default.yaml" -Destination "./data/settings.yaml"
  ```

### Chinese prompts

- Linux / macOS:

  ```bash
  cp ./templates/chinese_prompts.yaml ./data/settings.yaml
  ```

- Windows (PowerShell):

  ```powershell
  Copy-Item -Path "./templates/chinese_prompts.yaml" -Destination "./data/settings.yaml"
  ```

### Custom chunking

- Linux / macOS:

  ```bash
  cp ./templates/chunked.yaml ./data/settings.yaml
  ```

- Windows (PowerShell):

  ```powershell
  Copy-Item -Path "./templates/chunked.yaml" -Destination "./data/settings.yaml"
  ```

### Chinese prompts + Custom chunking

- Linux / macOS:

  ```bash
  cp ./templates/chinese_prompts_chunked.yaml ./data/settings.yaml
  ```

- Windows (PowerShell):

  ```powershell
  Copy-Item -Path "./templates/chinese_prompts_chunked.yaml" -Destination "./data/settings.yaml"
  ```

> Additional configurations will be added later.

## Start indexing

```bash
uv run graphrag index --root ./data/
```

## Question generation

```bash
uv run question-gen
```

## Batch Query

TODO

## Evaluation

TODO
