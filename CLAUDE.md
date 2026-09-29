# music-transcribe

CLI Python + skill Claude Code. Pipeline em etapas cacheadas em `<out>/`: tags → stems → lyrics → harmony → notes → render.

- Python 3.11 obrigatório (basic-pitch). `uv run pytest` roda os testes rápidos; `uv run pytest -m slow` roda o pipeline real.
- Ferramentas pesadas ficam atrás de adaptadores com `runner` injetável. Teste de unidade nunca chama demucs, whisper-cli ou basic-pitch.
- Áudio comercial nunca entra no repo. Fixtures são sintetizadas em `tests/conftest.py`.
- Confiança (high/medium/low) acompanha cada nota, acorde e linha de letra até o HTML. Nunca esconder.
- Spec: `docs/superpowers/specs/2026-09-28-music-transcribe-design.md`. Plano: `docs/superpowers/plans/2026-09-28-music-transcribe.md`.
- Aprendizado de projeto vai em `docs/` ou na skill, não em memória de usuário.
