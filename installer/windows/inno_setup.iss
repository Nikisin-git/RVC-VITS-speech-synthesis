; Inno Setup script for VoiceGen.
; Build prerequisite: run `conda pack -n voicegen -o build/voicegen-env.tar.gz`
; before compiling this script (the env bundle is required for offline install).

#define MyAppName "VoiceGen"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "VoiceGen"
#define MyAppExeName "voicegen.exe"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={pf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=VoiceGen-Setup-{#MyAppVersion}
SetupIconFile=..\..\app\assets\icons\app.ico
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "..\..\build\voicegen-env\*"; DestDir: "{app}\env"; Flags: recursesubdirs createallsubdirs
Source: "..\..\app\*"; DestDir: "{app}\app"; Flags: recursesubdirs createallsubdirs
Source: "..\..\scripts\*"; DestDir: "{app}\scripts"; Flags: recursesubdirs createallsubdirs
Source: "..\..\third_party\*"; DestDir: "{app}\third_party"; Flags: recursesubdirs createallsubdirs
Source: "..\..\environment.yml"; DestDir: "{app}"
Source: "..\..\LICENSE"; DestDir: "{app}"
Source: "..\..\README.md"; DestDir: "{app}"
Source: "..\..\DEPLOYMENT.md"; DestDir: "{app}"
; Bundle a portable ffmpeg.exe + espeak-ng next to the env so PATH stays clean.
Source: "vendor\ffmpeg.exe"; DestDir: "{app}\env\Library\bin"
Source: "vendor\espeak-ng\*"; DestDir: "{app}\espeak-ng"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\env\python.exe"; Parameters: "-m app.main"; WorkingDir: "{app}"; IconFilename: "{app}\app\assets\icons\app.ico"
Name: "{group}\{#MyAppName}"; Filename: "{app}\env\python.exe"; Parameters: "-m app.main"; WorkingDir: "{app}"

[Run]
Filename: "{app}\env\python.exe"; Parameters: "-m app.main"; Description: "Запустить VoiceGen"; Flags: nowait postinstall skipifsilent
