# graphrag chinese

## data

.doc to .txt in batch

```text
uv run graphrag to-txt -s .\taipei_council_docs\ -o out --overwrite  
```

set up setting.example

```text
cp 00_setting.example.yaml  setting.yaml
```

00 = base defulat setting didn't change any thing

```text
uv run graphrag pre-chunk -s .\data\input\ -o .\data\input_chunk_800_400 --chunk-size 800 --chunk-overlap 400
```

## evaluation
