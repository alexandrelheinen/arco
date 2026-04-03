#!/usr/bin/env python
"""Generate a CC0 8-bit racing chiptune loop as a WAV file.

Outputs a loopable square-wave / triangle-wave track in the style of
classic car-racing video games (F-Zero / Top Gear era chiptune).

The generated audio is entirely synthetic (no samples), built from
first principles — it is original work released under CC0.

Usage::

    python tools/simulator/generate_chiptune.py
    python tools/simulator/generate_chiptune.py --output path/to/out.wav
"""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE: int = 44100
BPM: int = 175
_BEAT: float = 60.0 / BPM  # seconds per quarter note
_EIGHTH: float = _BEAT / 2
_SIXTEENTH: float = _BEAT / 4


# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------


def _semitone(st_from_a4: float) -> float:
    """Return frequency in Hz for *st_from_a4* semitones above/below A4."""
    return 440.0 * (2.0 ** (st_from_a4 / 12.0))


# Chromatic semitone offsets from C4 (middle C)
_CHROMA = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_A4_OFFSET = -9  # C4 is 9 semitones below A4


def note(name: str, octave: int) -> float:
    """Return the frequency in Hz for *name* (e.g. ``'A'``, ``'F#'``).

    Args:
        name: Note name, optionally followed by ``'#'`` for a sharp.
        octave: Octave number (C4 = middle C).

    Returns:
        Frequency in Hz.
    """
    base = _CHROMA[name[0]]
    if len(name) > 1 and name[1] == "#":
        base += 1
    st = _A4_OFFSET + (octave - 4) * 12 + base
    return _semitone(st)


# ---------------------------------------------------------------------------
# Waveform generators  (all return list[float] in [-1, 1])
# ---------------------------------------------------------------------------


def _square(freq: float, duration: float, duty: float = 0.5) -> list[float]:
    n = int(SAMPLE_RATE * duration)
    if freq <= 0:
        return [0.0] * n
    period = SAMPLE_RATE / freq
    return [1.0 if (i % period) < period * duty else -1.0 for i in range(n)]


def _triangle(freq: float, duration: float) -> list[float]:
    n = int(SAMPLE_RATE * duration)
    if freq <= 0:
        return [0.0] * n
    period = SAMPLE_RATE / freq
    out = []
    for i in range(n):
        ph = (i % period) / period  # 0 → 1
        out.append(4.0 * abs(ph - 0.5) - 1.0)
    return out


def _silence(duration: float) -> list[float]:
    return [0.0] * int(SAMPLE_RATE * duration)


# ---------------------------------------------------------------------------
# Sequencer helpers
# ---------------------------------------------------------------------------


def _seq(
    freqs: list[float],
    step: float,
    wave_fn=_square,
    vol: float = 0.3,
    gate: float = 0.85,
) -> list[float]:
    """Build a track from a list of frequencies, each lasting *step* seconds.

    Args:
        freqs: List of frequencies (use 0 for a rest).
        step: Duration of each note step in seconds.
        wave_fn: Waveform generator function.
        vol: Amplitude scaling factor.
        gate: Fraction of *step* to keep the note sounding (rest of step
            is silence).

    Returns:
        Flat list of audio samples.
    """
    out: list[float] = []
    for f in freqs:
        if f <= 0:
            out += _silence(step)
        else:
            on = wave_fn(f, step * gate)
            off = _silence(step * (1.0 - gate))
            out += [s * vol for s in on] + off
    return out


def _mix(tracks: list[list[float]]) -> list[float]:
    n = max(len(t) for t in tracks)
    result: list[float] = []
    for i in range(n):
        s = sum(t[i] if i < len(t) else 0.0 for t in tracks)
        result.append(max(-1.0, min(1.0, s)))
    return result


# ---------------------------------------------------------------------------
# Song data  (4-bar loop, 8th-note resolution, Am-based racing theme)
# ---------------------------------------------------------------------------
#
# Chord progression: Am | G | F | E7
# Each bar has 8 eighth-note steps.

_E = _EIGHTH  # shorthand

# Lead melody (square wave, 8th notes × 32 steps = 4 bars)
_MELODY_FREQS = [
    # Bar 1 – Am (fast ascending arpeggio, then fill)
    note("A", 4),
    note("E", 5),
    note("A", 5),
    note("C", 6),
    note("B", 5),
    note("A", 5),
    note("E", 5),
    note("C", 5),
    # Bar 2 – G major
    note("G", 4),
    note("D", 5),
    note("G", 5),
    note("B", 5),
    note("A", 5),
    note("G", 5),
    note("D", 5),
    note("B", 4),
    # Bar 3 – F major
    note("F", 4),
    note("C", 5),
    note("F", 5),
    note("A", 5),
    note("G", 5),
    note("F", 5),
    note("C", 5),
    note("A", 4),
    # Bar 4 – E7 → Am (tension / resolution)
    note("E", 4),
    note("G#", 4),
    note("B", 4),
    note("E", 5),
    note("D", 5),
    note("B", 4),
    note("G#", 4),
    note("A", 4),
]

# Counter-melody (triangle wave, same rhythm, lower octave, offset harmony)
_COUNTER_FREQS = [
    # Bar 1
    note("C", 4),
    note("E", 4),
    note("G", 4),
    note("E", 5),
    note("G", 4),
    note("E", 4),
    note("C", 4),
    note("E", 4),
    # Bar 2
    note("B", 3),
    note("D", 4),
    note("G", 4),
    note("D", 5),
    note("B", 4),
    note("G", 4),
    note("D", 4),
    note("G", 3),
    # Bar 3
    note("A", 3),
    note("C", 4),
    note("F", 4),
    note("C", 5),
    note("A", 4),
    note("F", 4),
    note("C", 4),
    note("F", 3),
    # Bar 4
    note("G#", 3),
    note("B", 3),
    note("E", 4),
    note("G#", 4),
    note("B", 4),
    note("E", 4),
    note("B", 3),
    note("E", 3),
]

# Bass line (triangle wave, quarter notes = 2 eighth steps × 16 quarter notes)
_BASS_FREQS: list[float] = []
for _f in [
    note("A", 2),
    note("A", 2),
    note("E", 3),
    note("E", 3),  # Am
    note("G", 2),
    note("G", 2),
    note("D", 3),
    note("D", 3),  # G
    note("F", 2),
    note("F", 2),
    note("C", 3),
    note("C", 3),  # F
    note("E", 2),
    note("E", 2),
    note("E", 3),
    note("A", 2),  # E7→Am
]:
    _BASS_FREQS += [_f, _f]  # each repeated twice to fill an eighth-note slot

# Percussion: kick pattern (noise burst every beat = every 2 eighth-notes)
_KICK_NOISE_AMP = 0.12
_KICK_DURATION = _E * 0.25  # short attack noise


def _kick_track(n_bars: int) -> list[float]:
    """Build a kick-drum track using decaying white noise."""
    import random

    rng = random.Random(42)
    out: list[float] = []
    steps_per_bar = 8
    total_steps = n_bars * steps_per_bar
    for step in range(total_steps):
        is_beat = step % 2 == 0  # kick on every beat (every 2 eighth-notes)
        is_snare = step % 4 == 2  # snare on beats 2 and 4
        if is_beat or is_snare:
            amp = _KICK_NOISE_AMP if is_beat else _KICK_NOISE_AMP * 0.7
            on = int(SAMPLE_RATE * _KICK_DURATION)
            off = int(SAMPLE_RATE * (_E - _KICK_DURATION))
            for i in range(on):
                decay = math.exp(-6.0 * i / on)
                out.append(rng.uniform(-amp, amp) * decay)
            out += [0.0] * off
        else:
            out += _silence(_E)
    return out


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate(output: str = "racing_theme.wav") -> None:
    """Generate the chiptune loop and save to *output*.

    Args:
        output: Destination WAV file path.
    """
    n_bars = 4

    lead = _seq(_MELODY_FREQS, _E, wave_fn=_square, vol=0.28, gate=0.88)
    counter = _seq(_COUNTER_FREQS, _E, wave_fn=_triangle, vol=0.18, gate=0.82)
    bass = _seq(_BASS_FREQS, _E, wave_fn=_triangle, vol=0.22, gate=0.78)
    kick = _kick_track(n_bars)

    mixed = _mix([lead, counter, bass, kick])

    # Tiny fade-in / fade-out to avoid clicks at loop point
    fade_samples = int(SAMPLE_RATE * 0.005)
    for i in range(fade_samples):
        t = i / fade_samples
        mixed[i] *= t
        mixed[-(i + 1)] *= t

    # Write WAV (16-bit PCM, mono)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(output, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        pcm = struct.pack(f"<{len(mixed)}h", *[int(s * 32767) for s in mixed])
        wf.writeframes(pcm)

    duration = len(mixed) / SAMPLE_RATE
    print(f"Generated {output!r}  ({duration:.2f} s, {BPM} BPM, CC0)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a CC0 8-bit racing chiptune WAV loop."
    )
    parser.add_argument(
        "--output",
        default="racing_theme.wav",
        help="Output WAV file path (default: racing_theme.wav)",
    )
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    _main()
