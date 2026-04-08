//build with 
using OpenUtau.Core;
using OpenUtau.Classic;
using System.Linq;

var player = new UtauGenerate.Player();

var devices = player.getDevices();
foreach (var device in devices)
{
    Console.WriteLine($"Device: {device.Item3}, GUID: {device.Item1}, Number: {device.Item2}");
}
player.setDevice("speaker"); //this is the virtual device

// player.setDevice(devices[2].Item1, devices[2].Item2); // this is the normal device

Thread.Sleep(3000);
Console.WriteLine("Player initialized.");


for (int i = 0; i < 4; i++)
{
    player.addNote(100, 500, 60 + i, "stop");
    player.addNote(700, 500, 60, "world");

    int[] pitches = { 0, 100, 200, 300 };
    player.setPitchBend(pitches, 50);
    // for (int j = 0; j < 50; j++)
    // {
    //     player.addNote(j * 960, 960, 60, "la");
    // }

    player.play();

    Thread.Sleep(3000);

    player.resetParts();
    Console.WriteLine("reset");
}

// string savePath = @"..\..\..\outputs\output.ustx";
// player.exportUstx(savePath);

SingerManager.Inst.SearchAllSingers();
var singers = ClassicSingerLoader.FindAllSingers();
var teto = singers.FirstOrDefault(s => s.Id.Contains("Teto") || s.Name.Contains("テト"));
teto.EnsureLoaded();
int count = 0;
foreach (var oto in teto.Otos) {
    if (oto.Alias.Contains("hh iy") || oto.Alias.Contains("iy r") || oto.Alias.Contains("w iy") || oto.Alias.Contains("aa r")) {
        Console.WriteLine(oto.Alias);
        count++;
    }
}
Console.WriteLine($"Total English otos found: {count}");