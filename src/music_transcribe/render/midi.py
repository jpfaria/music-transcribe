from __future__ import annotations
from pathlib import Path
import pretty_midi
from music_transcribe.schema import Note


def write_midi(notes: list[Note], path: Path, program: int = 27) -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    for n in notes:
        inst.notes.append(pretty_midi.Note(velocity=int(40 + 80 * min(1.0, n.amplitude)), pitch=int(n.pitch),
                                           start=float(n.start), end=float(max(n.end, n.start + 0.05))))
    pm.instruments.append(inst)
    pm.write(str(path))
