# music-transcribe

Letra, cifra com diagramas e tab com articulações de uma música, a partir do áudio, tudo local. Nenhum áudio sai da máquina.

## Requisitos
- macOS Apple Silicon (MPS) ou CPU (`--device cpu`); `brew install ffmpeg whisper-cpp uv`
- Python 3.11 (exigido pelo basic-pitch; o `uv` cuida disso)
- Na primeira vez, a etapa `lyrics` baixa o modelo Whisper large-v3 (~3 GB) para `~/.cache/music-transcribe/`, a menos que ele já exista em `~/.cache/whisper/` (convenção do whisper.cpp, reaproveitado sem baixar de novo). A etapa `stems` também pode baixar os pesos do demucs `htdemucs_6s` (~55 MB) para `~/.cache/music-transcribe/torch/` (reaproveita o cache do torch.hub se já existir). Os dois downloads pedem confirmação; `--yes` aceita sem perguntar.

## Uso
```bash
uv sync
uv run music-transcribe run ~/Music/song.flac --yes
open ~/Music/song.transcribe/render/cifra.html
```
A saída padrão é `<pasta do áudio>/<nome sem extensão>.transcribe/` (`--out` muda). Outras opções de `run`: `--instruments guitar,bass,piano`, `--model large-v3|medium|medium.en|small.en`, `--language auto|en|pt|…` (padrão `auto` detecta; música em português: `--language pt`), `--device mps|cpu` (se o demucs falhar em `mps`, tenta `cpu` uma vez), `--force`. Se a letra falhar (modelo recusado, `whisper-cli` ausente), `run` avisa `letra indisponível` e segue com cifra e tab.

Etapas: `tags` `stems` `lyrics` `harmony` `notes` `render`, cada uma também é um subcomando (`uv run music-transcribe harmony ~/Music/song.flac --force`). Os resultados ficam em `<saída>/<etapa>/` e são reaproveitados; `--force` refaz a etapa e invalida as que dependem dela (`stems` → `lyrics`/`harmony`/`notes`, `harmony` → `notes`). Se o áudio mudar (hash em `<saída>/input.sha1`), todo o cache é invalidado.

Em `render/`:
- `cifra.html`: página completa, abre no navegador
- `cifra.fragment.html`: o mesmo conteúdo sem `<html>/<head>`, para publicar como artifact do Claude
- `cifra.txt`: cifra e tab em texto
- `<instrumento>.mid`: notas transcritas

## Plugin Claude Code
`/plugin marketplace add jpfaria/music-transcribe` e depois `/plugin install music-transcribe@music-transcribe`, que traz a skill `transcribing-a-song`: pede a cifra de um arquivo e ela roda o pipeline, revisa a confiança e publica a página.

## Confiança
Toda nota, acorde e linha de letra traz confiança (nota: high/medium/low + motivo `bleed`/`weak`/`unstable`/`chord`; acorde e letra: 0..1). Na tab, nota `low` aparece como `(6)` e `medium` como `6?`. Linha de letra abaixo de 0,6 e acorde abaixo de 0,3 aparecem marcados com `?`. A página nunca esconde isso.

## Limites conhecidos
- Guitarra e baixo são transcritos como linha monofônica: em notas simultâneas fica a mais forte, marcada `medium`/`chord`. Piano mantém as notas simultâneas.
- Com `--language auto`, trechos curtos de voz podem ser detectados no idioma errado; passe `--language` quando souber.
- Ver `docs/pending.md` para o estado atual e o que ficou para depois.

## Docs
Spec em `docs/superpowers/specs/`, plano em `docs/superpowers/plans/`.
