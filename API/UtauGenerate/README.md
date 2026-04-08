# UtauGenerate Memory Palace

C# wrapper around OpenUtau engine, loaded into Python via PythonNET (`pythonnet.load("coreclr")`). Provides the `Player` class used by `DataSynthesizer/stages/synthesizePrior.py` to render singing voice audio from note data.

## File Details

### `Player.cs`
Main API class implementing `ICmdSubscriber`. Manages an OpenUtau project with a single voice part and track.

**Constructor**: `Player(string phonemizer)` — Initializes DocManager, searches singers (uses first found), loads phonemizer (default: ArpasingPlusPhonemizer), sets WORLDLINER renderer, creates DummyAudioOutput.

**Core note management:**
- `resetParts()`: Clears all parts and creates a fresh `UVoicePart` with 100-tick duration. Destroys cached phonemizer state — use for first iteration of a new chunk.
- `clearNotes()`: Clears notes and phonemes from the current part without replacing the part object. Preserves the part identity so `validateLight()` can re-run the phonemizer efficiently. Use for iterations 2+ when only note durations change.
- `addNote(int position, int duration, int tone, string lyric)`: Adds a note to the part. Extends part duration if needed. Lyric can include phoneme hints in `word[ah n d]` format.

**Validation (private):**
- `validate()`: Full validation — `project.ValidateFull()`, manual phonemizer `SetUp`/`Process` (synchronous, `Testing=true`), `part.Validate(SkipPhonemizer=false)`, waits for `PhonemesUpToDate`. Verbose console logging.
- `validateLight()`: Lightweight validation — `project.ValidateFull()` + `part.Validate(SkipPhonemizer=false)`, waits for `PhonemesUpToDate`. Skips the redundant manual phonemizer setup/process cycle and verbose logging. Faster than `validate()`.

**Rendering (private helpers):**
- `renderWavInternal(string outputPath)`: Renders to WAV via `RenderMixdown().Wait()`, polls for file creation and lock release. No validation — callers must validate first.
- `saveUstxInternal(string outputPath)`: Saves OpenUtau project file via `project.BeforeSave()` + `Ustx.Save()`. No validation.

**Public export API (composed from helpers):**
- `export(string wavPath, string ustxPath)`: Full validate + WAV render + USTX save. Single validate call for both outputs. Used on iteration 1 of alignment.
- `exportFast(string wavPath, string ustxPath)`: Light validate + WAV render + USTX save. Skips manual phonemizer overhead. Used on iterations 2+ of alignment via `rerender_prior_with_adjusted_durations()`.
- `exportWav(string wavPath)`: Full validate + WAV render only.
- `exportWavOnly(string wavPath)`: Light validate + WAV render only.
- `exportUstx(string ustxPath)`: Full validate + USTX save only.

**Other:**
- `addPitchBend`, `setPitchBend`, `addDynamic`, `setDynamics`: Curve expression manipulation.
- `play()`, `testAudio()`: Audio playback (requires MiniAudioOutput, not used in synthesis pipeline).
- `getDevices()`, `setDevice()`: Audio device management.
- `diagnose()`: Debug dump of phonemes.

### `Program.cs`
Console entry point (not used by the data pipeline — pipeline uses Player via PythonNET).
