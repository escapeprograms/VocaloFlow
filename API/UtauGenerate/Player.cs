using System;
using OpenUtau.Core.Ustx;
using OpenUtau.Core.Format;
using OpenUtau.Core;
using OpenUtau.Audio;
using OpenUtau.Classic;
using OpenUtau.Core.Util;
using System.Threading.Tasks;
using System.Threading;

namespace UtauGenerate
{
    public class Player : ICmdSubscriber
    {
        public UProject project = DocManager.Inst.Project;
        public UVoicePart part = new UVoicePart();

        public Boolean audioInitialized = false;
        public Tuple<Guid, int> selectedDevice = null;

        public void OnNext(UCommand cmd, bool isUndo)
        {
            if (cmd is ErrorMessageNotification err)
            {
                Console.WriteLine("ERROR NOTIFICATION: " + err.message);
                if (err.e != null) Console.WriteLine("Exception: " + err.e);
            }
        }

        public Player(string phonemizer = "OpenUtau.Plugin.Builtin.ArpasingPlusPhonemizer")
        {
            AppDomain.CurrentDomain.FirstChanceException += (sender, eventArgs) => {
                Console.WriteLine($"[EXCEPTION DUMP] {eventArgs.Exception.ToString()}");
            };
            
            //list of phonemizers:
            //"OpenUtau.Plugin.Builtin.JapaneseCVVCPhonemizer"
            //"OpenUtau.Plugin.Builtin.ArpasingPlusPhonemizer"
            //"OpenUtau.Core.DefaultPhonemizer"
            //"OpenUtau.Plugin.Builtin.EnXSampaPhonemizer"

            DocManager.Inst.AddSubscriber(this);
            // Console.WriteLine("happy birthday");
            System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
            //initialize and load stuff
            Thread mainThread = Thread.CurrentThread;
            TaskScheduler mainScheduler = TaskScheduler.Default;
            DocManager.Inst.Initialize(mainThread, mainScheduler); // Initialize DocManager to load phonemizers and singers

            SingerManager.Inst.SearchAllSingers();
            var singers = ClassicSingerLoader.FindAllSingers();
            Console.WriteLine($"Found {singers.Count()} singers");

            var teto = singers.First();  //singers.FirstOrDefault(s => s.Name.Contains("Teto"));
            if (teto != null)
            {
                Console.WriteLine($"Found Teto: {teto.Id} - {teto.Name}");
            }
            else
            {
                throw new Exception("TETO not found.");
            }
            
            Console.WriteLine($"[TRACE] PathManager.Inst.DataPath: {PathManager.Inst.DataPath ?? "NULL"}");
            Console.WriteLine($"[TRACE] PathManager.Inst.PluginsPath: {PathManager.Inst.PluginsPath ?? "NULL"}");

            //load wavtool and resampler
            ToolsManager.Inst.SearchResamplers(); //TODO: make it actually load stuff
            ToolsManager.Inst.SearchWavtools();

            // Create a new UProject
            project = DocManager.Inst.Project;
            project.name = "Project";
            project.ustxVersion = new Version(0, 6);
            Ustx.AddDefaultExpressions(project);

            //assign singer and phonemizer
            project.tracks[0].singer = teto.Id;
            project.tracks[0].phonemizer = phonemizer;
            project.tracks[0].RendererSettings.renderer = OpenUtau.Core.Render.Renderers.WORLDLINER;
            project.tracks[0].AfterLoad(project); // load the singer + phonemes

            project.tracks[0].Phonemizer.Testing = true;
            project.tracks[0].Phonemizer.SetSinger(project.tracks[0].Singer);

            Console.WriteLine($"Track phonemizer: {project.tracks[0].Phonemizer}");
            // await Task.Sleep(2000); //WAIT TO LOAD ALL THE SHI
            PlaybackManager.Inst.AudioOutput = new DummyAudioOutput();
            resetParts();
        }

        public void resetParts()
        {
            //reset parts
            project.parts.Clear();
            part = new UVoicePart();
            part.trackNo = 0;
            part.position = 0;      // Start at the beginning
            part.Duration = 100;    // Duration in ticks (adjust as needed)
            part.name = "Main Speech";

            project.parts.Add(part);
        }

        /// <summary>Clear all notes from the current part without replacing the part object.
        /// Preserves cached phonemizer state so exportFast/exportWavOnly can skip
        /// phonemizer re-setup on subsequent renders.</summary>
        public void clearNotes()
        {
            part.notes.Clear();
            part.phonemes.Clear();
            part.Duration = 100;
        }

        public void addNote(int position, int duration, int tone, string lyric)
        {
            //Extend part duration
            if (position + duration > part.Duration)
            {
                part.Duration = position + duration;
            }

            UNote note = project.CreateNote();
            note.position = position;      // Start at the specified position
            note.duration = duration;    // Duration in ticks
            note.tone = tone;         // MIDI number for the note
            note.lyric = lyric;       // Assign lyric to the note

            // Add the note to the voice part
            part.notes.Add(note);
        }

        public void addPitchBend(int position, int pitch)
        {
            //pitch ranges [-1200, 1200], aka 1 octave down to 1 octave up
            var command = new SetCurveCommand(project, part, Ustx.PITD, position, pitch, position, 0);
            command.Execute();
        }

        public void setPitchBend(int[] pitches, int time_step = 1)
        {
            //set all pitch bends at a given time per array item (in ticks, 960 ticks = 1 beat)
            for (int i = 0; i < pitches.Length; i++)
            {
                for (int j = 0; j < time_step; j++)
                {
                    int position = i * time_step + j;
                    if (position >= part.Duration) break;
                    addPitchBend(position, pitches[i]);
                }
            }
        }
        public void addDynamic(int position, int dynamic)
        {
            //dynamic ranges [-240, 120]
            var command = new SetCurveCommand(project, part, Ustx.DYN, position, dynamic, position, 0);
            command.Execute();
        }
        public void setDynamics(int[] dynamics, int resolution = 1)
        {
            //set all dynamics at a given resolution
            for (int i = 0; i < dynamics.Length; i++)
            {
                for (int j = 0; j < resolution; j++)
                {
                    int position = i * resolution + j;
                    if (position >= part.Duration) break;
                    addDynamic(position, dynamics[i]);
                }
            }
        }

        public void validate()
        {
            project.ValidateFull();
            var track = project.tracks[0];
            var phonemizer = track.Phonemizer;
            
            Console.WriteLine("[SYNC PHONEMIZER] SetUp...");
            phonemizer.Testing = true; // force load dictionary synchronously
            if (track.Singer != null)
            {
                Console.WriteLine($"[SYNC PHONEMIZER] Singer Location: {track.Singer.Location ?? "NULL"}");
            }
            phonemizer.SetSinger(track.Singer);
            phonemizer.SetTiming(project.timeAxis.Clone());
            
            var pNotes = part.notes.Select(note => {
                string raw = note.lyric ?? "";
                string parsedLyric;
                string parsedHint;
                int open = raw.IndexOf('[');
                int close = raw.IndexOf(']');
                if (open >= 0 && close > open) {
                    parsedLyric = raw.Substring(0, open).Trim();
                    parsedHint  = raw.Substring(open + 1, close - open - 1).Trim();
                    if (parsedHint.Length == 0) parsedHint = null;
                } else {
                    parsedLyric = raw.Trim();
                    parsedHint  = null;
                }
                return new OpenUtau.Api.Phonemizer.Note() {
                    lyric             = parsedLyric,
                    phoneticHint      = parsedHint,
                    tone              = note.tone,
                    position          = part.position + note.position,
                    duration          = note.duration,
                    phonemeAttributes = new OpenUtau.Api.Phonemizer.PhonemeAttributes[0]
                };
            }).ToArray();
            phonemizer.SetUp(new[] { pNotes }, project, track);
            
            Console.WriteLine("[SYNC PHONEMIZER] Process...");
            OpenUtau.Api.Phonemizer.Result result = phonemizer.Process(
                 pNotes,
                 null,
                 null,
                 null,
                 null,
                 new OpenUtau.Api.Phonemizer.Note[0]);
                 
            Console.WriteLine($"[SYNC PHONEMIZER] Process returned {result.phonemes.Length} phonemes.");
            foreach (var p in result.phonemes) {
                Console.WriteLine($"  -> Phoneme: '{p.phoneme}', Pos: {p.position}");
            }
            
            part.Validate(new OpenUtau.Core.ValidateOptions { SkipPhonemizer = false }, project, project.tracks[0]);
            
            int waitTime = 5000;
            while (!part.PhonemesUpToDate && waitTime > 0)
            {
                Thread.Sleep(50);
                waitTime -= 50;
            }
            
            Console.WriteLine($"Number of Phrases: {part.renderPhrases.Count}");
            foreach (var phrase in part.renderPhrases) {
                Console.WriteLine($"Phrase with {phrase.phones.Length} phones");
            }
        }
        public List<Tuple<Guid, int, string>> getDevices()
        {
            //get all audio devices
            var devices = PlaybackManager.Inst.AudioOutput.GetOutputDevices();
            List<Tuple<Guid, int, string>> deviceList = new List<Tuple<Guid, int, string>>();
            foreach (var device in devices)
            {
                deviceList.Add(Tuple.Create(device.guid, device.deviceNumber, device.name));
            }
            return deviceList;
        }
        public void setDevice(Guid guid, int deviceNumber)
        {
            //set audio output device
            PlaybackManager.Inst.AudioOutput.SelectDevice(guid, deviceNumber);
            selectedDevice = Tuple.Create(guid, deviceNumber);
        }

        public void setDevice(string name)
        {
            //set audio output device that contains name
            var devices = PlaybackManager.Inst.AudioOutput.GetOutputDevices();
            var device = devices.FirstOrDefault(d => d.name.Contains(name, StringComparison.OrdinalIgnoreCase));
            if (device != null)
            {
                PlaybackManager.Inst.AudioOutput.SelectDevice(device.guid, device.deviceNumber);
                selectedDevice = Tuple.Create(device.guid, device.deviceNumber);
            }
            else
            {
                Console.WriteLine($"Device with name '{name}' not found.");
            }
        }
        public void testAudio()
        {
            if (!audioInitialized)
            {
                PlaybackManager.Inst.AudioOutput = new MiniAudioOutput();//NAudioOutput()
                if (selectedDevice != null)
                {
                    PlaybackManager.Inst.AudioOutput.SelectDevice(selectedDevice.Item1, selectedDevice.Item2);
                }
                audioInitialized = true;
            }
            PlaybackManager.Inst.PlayTestSound();
        }
        public void play()
        {
            validate();
            //initialize audio output
            if (!audioInitialized)
            {
                PlaybackManager.Inst.AudioOutput = new MiniAudioOutput();//NAudioOutput()
                if (selectedDevice != null)
                {
                    PlaybackManager.Inst.AudioOutput.SelectDevice(selectedDevice.Item1, selectedDevice.Item2);
                }
                audioInitialized = true;
            }
            Thread.Sleep(300);
            PlaybackManager.Inst.Play(project, 0);
        }

        // ---------------------------------------------------------------
        // Internal helpers (no validation — callers must validate first)
        // ---------------------------------------------------------------

        private void renderWavInternal(string outputPath)
        {
            PlaybackManager.Inst.AudioOutput = new DummyAudioOutput();
            try
            {
                if (System.IO.File.Exists(outputPath)) System.IO.File.Delete(outputPath);
                PlaybackManager.Inst.RenderMixdown(project, outputPath).Wait();
                Console.WriteLine("EXPORTED TO: " + outputPath);

                // wait for file to be created
                int waitTime = 10000;
                while (!System.IO.File.Exists(outputPath) && waitTime > 0)
                {
                    System.Threading.Thread.Sleep(100);
                    waitTime -= 100;
                }

                // wait for file to be unlocked
                waitTime = 10000;
                while (waitTime > 0 && System.IO.File.Exists(outputPath))
                {
                    try
                    {
                        using (var fs = System.IO.File.Open(outputPath, System.IO.FileMode.Open, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None))
                        {
                            break; // Successfully grabbed lock, meaning wave writer finished!
                        }
                    }
                    catch (System.IO.IOException)
                    {
                        System.Threading.Thread.Sleep(100);
                        waitTime -= 100;
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
            }
        }

        private void saveUstxInternal(string outputPath)
        {
            project.BeforeSave();
            Ustx.Save(outputPath, project);
        }

        /// <summary>Lightweight validation: runs phonemizer via part.Validate but
        /// skips the redundant manual SetUp/Process cycle and verbose logging.
        /// Use after notes have been replaced with clearNotes() + addNote().</summary>
        private void validateLight()
        {
            project.ValidateFull();
            part.Validate(new OpenUtau.Core.ValidateOptions { SkipPhonemizer = false }, project, project.tracks[0]);

            int waitTime = 5000;
            while (!part.PhonemesUpToDate && waitTime > 0)
            {
                Thread.Sleep(50);
                waitTime -= 50;
            }
        }

        // ---------------------------------------------------------------
        // Public export API
        // ---------------------------------------------------------------

        /// <summary>Full validate + render WAV.</summary>
        public void exportWav(string outputPath)
        {
            validate();
            renderWavInternal(outputPath);
        }

        /// <summary>Light validate (skip manual phonemizer setup) + render WAV.
        /// Faster than exportWav but still runs the phonemizer via part.Validate.</summary>
        public void exportWavOnly(string outputPath)
        {
            validateLight();
            renderWavInternal(outputPath);
        }

        /// <summary>Full validate + save USTX project file.</summary>
        public void exportUstx(string outputPath)
        {
            validate();
            saveUstxInternal(outputPath);
        }

        /// <summary>Full validate + render WAV + save USTX under a single validate call.</summary>
        public void export(string wavPath, string ustxPath)
        {
            validate();
            renderWavInternal(wavPath);
            saveUstxInternal(ustxPath);
        }

        /// <summary>Light validate (skip manual phonemizer setup) + render WAV + save USTX.
        /// Faster than export() but still runs the phonemizer via part.Validate.</summary>
        public void exportFast(string wavPath, string ustxPath)
        {
            validateLight();
            renderWavInternal(wavPath);
            saveUstxInternal(ustxPath);
        }

        public void diagnose()
        {
            validate();
            Console.WriteLine("phonemes:");
            foreach (var p in part.phonemes)
            {
                Console.WriteLine($"Phoneme: {p.phoneme}, Duration: {p.Duration}");
            }
        }
    }
}