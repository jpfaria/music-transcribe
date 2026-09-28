# music-transcribe — design

Data: 2026-09-28. Estado: aprovado em conversa, aguardando revisão do texto.

## Objetivo

Dado um arquivo de áudio (FLAC/WAV/MP3), produzir a **cifra completa** da música: letra
sincronizada, tom, andamento, progressão de acordes por compasso com diagramas, e tab
nota a nota dos instrumentos (guitarra com bends/vibrato/slides, baixo, piano). Tudo
local, sem enviar áudio para serviço externo. O resultado é revisável: cada nota, acorde
e linha de letra carrega uma confiança visível.

Usuário: João, guitarrista, quer tocar junto com a gravação. Sucesso = abrir `cifra.html`
e conseguir tocar a música inteira, sabendo onde o transcritor está inseguro.

## Forma

Repo `jpfaria/music-transcribe` = plugin Claude Code (`.claude-plugin/plugin.json` +
`marketplace.json`, mesmo padrão de `jpfaria/mackie-control`) + pacote Python
`music_transcribe` com CLI `music-transcribe`. Gerenciado com `uv`, Python 3.11.

Dependências pesadas (torch, demucs, basic-pitch, whisper.cpp via `whisper-cli` do Homebrew,
ffmpeg) são pré-requisitos declarados; modelos baixam sob demanda em
`~/.cache/music-transcribe/` com aviso de tamanho antes do download.

## Pipeline

`music-transcribe run <audio> [--out DIR]` executa as etapas abaixo em ordem; cada etapa
grava seu resultado em `<out>/<etapa>/` e é pulada se já existir (cache por hash do input).
Cada etapa é também um subcomando (`music-transcribe lyrics <audio>`).

| etapa | entrada | saída | ferramenta |
|---|---|---|---|
| `tags` | áudio | `tags.json` (título, artista, letra embutida se houver) | ffprobe |
| `stems` | áudio | `stems/{vocals,guitar,bass,drums,piano,other}.flac` | demucs `htdemucs_6s` |
| `lyrics` | stem `vocals` | `lyrics.json` (segmentos com tempo, texto, confiança) | whisper-cli large-v3, `-sns`, dois passes (medium.en como fallback) |
| `harmony` | stems `bass`+`piano`+`guitar`+`other` | `harmony.json` (bpm, métrica, tom, grade de compassos, acorde por compasso, loop detectado) | librosa: beat, chroma, pyin no baixo |
| `notes` | um stem por instrumento | `notes/<inst>.json` (notas com início, fim, pitch, amplitude, contorno f0, articulação, confiança) | basic-pitch + pyin por nota |
| `render` | tudo acima | `cifra.html`, `cifra.txt`, `<inst>.mid` | jinja-free: string templates + SVG gerado |

### harmony

- Tempo: tempograma com prior em 40–100 BPM; se o pico estiver em dobro/triplo, escolhe
  o candidato cujo compasso bate com a taxa de troca de raiz do baixo.
- Métrica: 4/4 ou 12/8 (shuffle) por autocorrelação de onsets em subdivisão 2 vs 3.
- Grade de compassos: ajusta origem e duração do compasso às trocas de raiz do baixo.
- Acorde por compasso: chroma dos stems harmônicos ponderado pela raiz do baixo, casado
  contra templates maj, min, 7, maj7, m7, dim, m7b5, sus2, sus4, add9, 9. Inversão
  quando a raiz do baixo difere da fundamental (`Db/F`). Confiança = margem entre o
  melhor e o segundo template.
- Loop: detecta repetição da sequência de acordes (ex.: 4 compassos) e reporta.

### notes e articulações

Para cada nota do basic-pitch, recorta o f0 (pyin) dentro da nota:
- **bend**: subida contínua ≥ 40 cents que se sustenta → `7b8` (½ tom) / `7b9` (1 tom);
  bend-release se volta.
- **vibrato**: oscilação periódica 4–8 Hz com amplitude ≥ 20 cents → `7~`.
- **slide**: glide entre duas notas sem re-ataque → `7/9` ou `9\7`.
- **hammer/pull**: nota seguinte com ataque fraco (amplitude de onset baixa) → `7h9` / `9p7`.
- Confiança por nota: alta (f0 estável, amplitude alta), média, baixa (bleed provável,
  amplitude baixa, f0 com falhas).

Posição no braço: escolhe corda/casa minimizando salto em relação à nota anterior,
preferindo a região onde a escala detectada cai (caixa da pentatônica). Baixo: 4 cordas
E A D G. Piano: sem posição, notas por compasso.

### render

`cifra.html`: página única, temas claro/escuro, seções: cabeçalho (título, artista, tom,
andamento, métrica), progressão com diagramas SVG (banco de shapes + fallback que calcula
um voicing a partir das notas do acorde), letra por seção com acordes acima, escala do
solo, tab por compasso por instrumento com legenda de articulações e cor por confiança.
`cifra.txt`: mesmo conteúdo em texto. `<inst>.mid`: notas para ouvir em DAW.

## Skill

`skills/transcribing-a-song/SKILL.md`. Gatilho: usuário pede letra, cifra, acordes ou
tab de um arquivo de áudio. Fluxo: (1) roda `music-transcribe run`; (2) lê os JSON e a
página, revisa: linhas de letra com confiança baixa ganham `?`, acordes fora da escala
detectada são questionados, compassos de tab com muita nota repetida ganham aviso de
possível bleed; (3) publica `cifra.html` como artifact; (4) copia a pasta de saída para
onde o usuário indicar (padrão: iCloud `Musica/Transcricoes/<Artista - Título>/`).
Nunca esconde confiança baixa; nunca apresenta nota incerta como certa.

## Testes

- Unit (pytest, rápidos, sem modelo): harmonia (tempo/métrica/acordes) sobre áudio
  sintético gerado com senoides; articulações sobre contornos f0 sintéticos; posição no
  braço; diagramas SVG (shape correto para acordes conhecidos); render (HTML válido,
  contém todas as seções).
- Integração (marcado `slow`, manual): pipeline inteiro sobre um trecho de 30 s
  incluído em `tests/fixtures/` gerado sinteticamente (sem áudio comercial no repo).

## Fora de escopo

Voz de fundo (harmonias vocais), letras em mais de um idioma no mesmo arquivo, edição
interativa na página, partitura tradicional.
