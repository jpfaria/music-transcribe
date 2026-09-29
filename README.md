# music-transcribe

Letra, cifra com diagramas e tab com articulações de uma música, a partir do áudio, tudo local. Nenhum áudio sai da máquina.

## Requisitos
- macOS Apple Silicon (MPS) ou CPU (`--device cpu`); `brew install ffmpeg whisper-cpp uv`
- Python 3.11 (exigido pelo basic-pitch; o `uv` cuida disso)
- Na primeira vez, a etapa `lyrics` baixa o modelo Whisper large-v3 (~3 GB) para `~/.cache/music-transcribe/` (`--yes` aceita sem perguntar)

## Uso
```bash
uv sync
uv run music-transcribe run ~/Music/song.flac --yes
open ~/Music/song.transcribe/render/cifra.html
```
A saída padrão é `<pasta do áudio>/<nome sem extensão>.transcribe/` (`--out` muda). Outras opções de `run`: `--instruments guitar,bass,piano`, `--model large-v3|medium.en|small.en`, `--device mps|cpu`, `--force`.

Etapas: `tags` `stems` `lyrics` `harmony` `notes` `render`, cada uma também é um subcomando (`uv run music-transcribe harmony ~/Music/song.flac --force`). Os resultados ficam em `<saída>/<etapa>/` e são reaproveitados; `--force` refaz a etapa. Quem depende dela (ex.: `notes` depois de `harmony`) precisa ser refeito também.

Em `render/`:
- `cifra.html`: página completa, abre no navegador
- `cifra.fragment.html`: o mesmo conteúdo sem `<html>/<head>`, para publicar como artifact do Claude
- `cifra.txt`: cifra e tab em texto
- `<instrumento>.mid`: notas transcritas

## Plugin Claude Code
`/plugin marketplace add jpfaria/music-transcribe` e depois `/plugin install music-transcribe@music-transcribe`, que traz a skill `transcribing-a-song`: pede a cifra de um arquivo e ela roda o pipeline, revisa a confiança e publica a página.

## Confiança
Toda nota, acorde e linha de letra traz confiança (nota: high/medium/low + motivo `bleed`/`weak`/`unstable`; acorde e letra: 0..1). Linha de letra abaixo de 0,6 e acorde abaixo de 0,3 aparecem marcados com `?`. A página nunca esconde isso.

## Limites conhecidos
- A letra é transcrita com Whisper em inglês (`-l en`); música em outro idioma sai com confiança baixa.

## Docs
Spec em `docs/superpowers/specs/`, plano em `docs/superpowers/plans/`.
