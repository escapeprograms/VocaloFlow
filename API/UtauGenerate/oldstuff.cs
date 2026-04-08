// using System;
// using OpenUtau.Core.Ustx;
// using OpenUtau.Core.Format;
// using OpenUtau.Core;
// using OpenUtau.Audio;
// using OpenUtau.Classic;



// System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance); //language support

// //initialize and load stuff
// Thread mainThread = Thread.CurrentThread;
// TaskScheduler mainScheduler = TaskScheduler.Default;
// DocManager.Inst.Initialize(mainThread, mainScheduler); // Initialize DocManager to load phonemizers and singers

// SingerManager.Inst.SearchAllSingers();
// // var singer = SingerManager.Inst.GetSinger("重音テト（かさねてと）音声ライブラリー"); //load teto
// // Console.WriteLine($"Found Teto {singer.Id}");

// var singers = ClassicSingerLoader.FindAllSingers();
// Console.WriteLine($"Found {singers.Count()} singers");

// var teto = singers.First();  //singers.FirstOrDefault(s => s.Name.Contains("Teto"));
// if (teto != null)
// {
//     Console.WriteLine($"Found Teto: {teto.Id} - {teto.Name}");
// }
// else
// {
//     throw new Exception("TETO not found.");
// }

// //load wavtool and resampler
// ToolsManager.Inst.SearchResamplers(); //TODO: make it actually load stuff
// ToolsManager.Inst.SearchWavtools();

// // Create a new UProject
// UProject project = DocManager.Inst.Project;

// project.name = "My New Project";
// project.ustxVersion = new Version(0, 6);
// Ustx.AddDefaultExpressions(project);

// // Add a track (required for USTX structure)
// // UTrack track = new UTrack(project);
// // project.tracks.Add(track);

// //assign singer and phonemizer
// project.tracks[0].singer = teto.Id;
// project.tracks[0].phonemizer = "OpenUtau.Plugin.Builtin.EnXSampaPhonemizer";
// //"OpenUtau.Plugin.Builtin.JapaneseCVVCPhonemizer"
// //"OpenUtau.Plugin.Builtin.ArpasingPlusPhonemizer";
// //"OpenUtau.Core.DefaultPhonemizer"
// //"OpenUtau.Plugin.Builtin.EnXSampaPhonemizer"
// project.tracks[0].AfterLoad(project); // load the singer + phonemes

// project.tracks[0].Phonemizer.SetSinger(project.tracks[0].Singer);//set the singer for the phonemizer (english phonemizers) NOTE: THIS TAKES A LONG TIME TO LOAD
// Thread.Sleep(2000); //WAIT TO LOAD ALL THE SHI

// Console.WriteLine($"Track singer: {project.tracks[0].Singer.Name}");
// // project.tracks[0].Phonemizer.SetSinger(project.tracks[0].Singer);
// Console.WriteLine($"Track phonemizer: {project.tracks[0].Phonemizer}");
// Console.WriteLine($"renderer: {project.tracks[0].RendererSettings.renderer}");
// Console.WriteLine($"resampler: {project.tracks[0].RendererSettings.resampler}");
// Console.WriteLine($"wavtool: {project.tracks[0].RendererSettings.wavtool}");


// // Create a voice part and add it to the project
// UVoicePart part = new UVoicePart();
// part.trackNo = 0;
// part.position = 0;      // Start at the beginning
// part.Duration = 6000;    // Duration in ticks (adjust as needed)
// part.name = "Main Melody Skibidi";

// // Create a note
// // var lyrics = new List<string> { "fA", "king", "dum", "ass"}; 
// var lyrics = new List<string> { "red", "miku", "fa king", "works" }; // Example lyrics
// for (int i = 0; i < 4; i++)
// {
//     UNote note = project.CreateNote();
//     note.position = i * 960;      // Start at the beginning
//     note.duration = 480;    // Duration in ticks (quarter note if resolution is 480)
//     note.tone = 60;         // MIDI number for Middle C (C4)
//     note.lyric = lyrics[i]; // Assign lyric from the list
//     // note.phonemes = new List<string> { "a" }; // Assign phoneme

//     // Add the note to the voice part
//     part.notes.Add(note);
// }

// // Add the part to the project
// project.parts.Add(part);

// //final validation to add phrases
// project.ValidateFull();

// foreach (var phoneme in part.phonemes) {
//     Console.WriteLine($"Phoneme: {phoneme.phoneme}, Duration: {phoneme.Duration}");
// }
// // Export the project as a USTx file
// // string savePath = @"..\..\..\outputs\output.ustx";
// // project.BeforeSave();
// // Ustx.Save(savePath, project);

// // Console.WriteLine($"USTx exported to: {savePath}");

// ////////////////////////////////////////////////////////////////////
// // string ustxPath = @"..\..\..\testing 2.ustx";
// // try
// // {
// //     // Load the project from the .ustx file using UstxLoader
// //     project = Ustx.Load(ustxPath);

// //     Console.WriteLine("Loaded USTX file successfully.");
// //     // You can now work with the project object.
// // }
// // catch (Exception ex)
// // {
// //     Console.WriteLine("Error loading USTX file: " + ex);
// // }
// // Console.WriteLine($"Track singer: {project.tracks[0].Singer.Name}");
// // // project.tracks[0].Phonemizer.SetSinger(project.tracks[0].Singer);
// // Console.WriteLine($"Track phonemizer: {project.tracks[0].Phonemizer}");


// //add voicebanks to search
// // PathManager.Inst.SingersPaths.Clear();
// // PathManager.Inst.SingersPaths.Add(@"C:\Users\archi\Documents\OpenUtau\Singers"); //Note, this doesn't work lol


// //play sound
// PlaybackManager.Inst.AudioOutput = new NAudioOutput(); //MiniAudioOutput()
// // PlaybackManager.Inst.PlayTestSound();
// PlaybackManager.Inst.Play(project, 0);
// Thread.Sleep(10000);

// //export
// string outputWav = @"..\..\..\outputs\output_audio.wav";
// try
// {
//     await PlaybackManager.Inst.RenderToFiles(project, outputWav);
// }
// catch (Exception e)
// {

// }
