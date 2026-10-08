// Host-only CLI adapter. No serving or benchmark parameters are changed.
// Compile with the Windows .NET Framework csc compiler into an isolated PATH.
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

public static class PolicyDockerRetry {
    static string Quote(string arg) {
        // Windows CommandLineToArgvW escaping; never invoke a shell.
        var output = new StringBuilder("\"");
        int slashes = 0;
        foreach (char ch in arg) {
            if (ch == '\\') { slashes++; continue; }
            if (ch == '"') {
                output.Append('\\', slashes * 2 + 1); output.Append('"');
            } else {
                output.Append('\\', slashes); output.Append(ch);
            }
            slashes = 0;
        }
        output.Append('\\', slashes * 2); output.Append('"');
        return output.ToString();
    }

    static bool ReadOnly(string[] args) {
        if (args.Length == 0) return false;
        switch (args[0]) {
            case "inspect": case "version": case "logs": case "ps": case "stats": return true;
            case "image": return args.Length > 1 && args[1] == "inspect";
            case "exec":
                // Only the study's cat / JSON-stat reads. Do not replay resets,
                // Python mutations, stdin commands, or arbitrary user commands.
                if (args.Length < 3 || args[1] != "vllm-serving-lab-cache-policy") return false;
                if (args.Length == 4 && args[2] == "cat" && args[3] == "/var/lib/vllm-kv/policy-events.jsonl") return true;
                return args.Length == 5 && args[2] == "python3" && args[3] == "-c" &&
                    args[4] == "import pathlib,json; r=pathlib.Path('/var/lib/vllm-kv'); print(json.dumps({p.stem:json.loads(p.read_text()) for p in r.glob('*.json') if p.name.endswith('-stats.json') or p.name=='policy-state.json'}))";
            default: return false;
        }
    }

    static void Log(string message) {
        string path = Environment.GetEnvironmentVariable("POLICY_DOCKER_RETRY_LOG");
        if (String.IsNullOrEmpty(path)) return;
        try { File.AppendAllText(path, DateTime.UtcNow.ToString("o")+" "+message+Environment.NewLine); }
        catch (IOException) { } // Diagnostics cannot change the command's result.
    }

    public static int Main(string[] args) {
        string native = Environment.GetEnvironmentVariable("POLICY_DOCKER_NATIVE_EXE");
        if (String.IsNullOrEmpty(native) || !File.Exists(native)) {
            Console.Error.WriteLine("POLICY_DOCKER_NATIVE_EXE must name the original Docker executable");
            return 125;
        }
        Console.OutputEncoding = new UTF8Encoding(false);
        bool liveRunningProbe = args.Length == 4 && args[0] == "inspect" &&
            args[1] == "--format" && args[2] == "{{.State.Running}}" &&
            args[3] == "vllm-serving-lab-cache-policy";
        if (liveRunningProbe) {
            // Docker Desktop may serve inspect from its API cache. A filtered
            // list is forwarded to the engine; change only our startup probe,
            // preserving metadata inspect and every mutation verbatim.
            args = new string[] { "ps", "--all", "--filter",
                "name=^/vllm-serving-lab-cache-policy$", "--format", "{{.State}}" };
        }
        var quoted = new string[args.Length];
        for (int i=0;i<args.Length;i++) quoted[i]=Quote(args[i]);
        bool query=ReadOnly(args);
        for (int attempt=1;attempt<=5;attempt++) {
            var start = new ProcessStartInfo(native,String.Join(" ",quoted));
            start.UseShellExecute=false;start.CreateNoWindow=true;
            start.RedirectStandardOutput=true;start.RedirectStandardError=true;
            start.StandardOutputEncoding=new UTF8Encoding(false);
            start.StandardErrorEncoding=new UTF8Encoding(false);
            using (var process=Process.Start(start)) {
                Task<string> stdout=process.StandardOutput.ReadToEndAsync();
                Task<string> stderr=process.StandardError.ReadToEndAsync();
                bool timedOut=query && !process.WaitForExit(15000);
                if (timedOut) {
                    process.Kill();
                    Log("command="+args[0]+" attempt="+attempt+" readonly=true timeout=15s");
                }
                process.WaitForExit();Task.WaitAll(stdout,stderr);
                bool allocationFailure=stderr.Result.Contains("cannot allocate memory") ||
                                       stderr.Result.Contains("fatal error: out of memory");
                if (process.ExitCode!=0) {
                    Log("command="+(args.Length==0?"<empty>":args[0])+" attempt="+attempt+
                        " exit="+process.ExitCode+" readonly="+query+" stderr="+stderr.Result);
                }
                if (query && (allocationFailure || timedOut) && process.ExitCode!=0 && attempt<5) {
                    Thread.Sleep(250*attempt);
                    continue;
                }
                if (timedOut) {
                    Console.Error.WriteLine("Docker read-only query exceeded 15s on all retry attempts.");
                    return 124;
                }
                if (liveRunningProbe && process.ExitCode == 0) {
                    string state = stdout.Result.Trim();
                    if (state == "running" || state == "paused") Console.Out.WriteLine("true");
                    else if (state == "" || state == "created" || state == "restarting" ||
                             state == "exited" || state == "removing" || state == "dead") Console.Out.WriteLine("false");
                    else {
                        Log("unexpected engine state="+state);
                        Console.Error.WriteLine("Unexpected Docker engine state for the study container");
                        return 125;
                    }
                    Console.Error.Write(stderr.Result);
                    return 0;
                }
                Console.Out.Write(stdout.Result);Console.Error.Write(stderr.Result);
                return process.ExitCode;
            }
        }
        return 125;
    }
}
