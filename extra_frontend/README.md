# extra_frontend — overhaul & experiment lab

Gitignored scratch space for richer UI you can hand-integrate into `frontend/src`.
Nothing here ships in the pristine milestone commits; it's a staging ground for
visual overhauls, animations, and drop-in enhanced variants of the tracked
components. Copy what you like into `frontend/src/components` and wire it in.

Design-system compatible: everything reuses the CSS variables and utility classes
already defined in `frontend/src/app/globals.css` (`--accent`, `.glass`, `.badge`,
`.chip`, etc.), so pasted components inherit the light/dark theme automatically.

## Contents

- **`OrionVoiceOrb.tsx`** (M5) — an animated push-to-talk "orb" overhaul of the
  plain `MicButton`. Same Web Speech STT logic, but a pulsing, listening-state
  visual meant for a hero placement on the Orion page. Drop-in replacement:
  `import { OrionVoiceOrb } from "@/components/OrionVoiceOrb"` after copying it over.

## Roadmap (added per milestone)

- M6: authenticated shell (sign-in gate, session avatar) experiments.
- M7/M8: signal-source cards (Reddit/Kaggle), landing-page hero, polish pass.
