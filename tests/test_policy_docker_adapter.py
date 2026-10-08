"""Host-controller checks; fixture commands never generate experiment results."""
import base64
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CSC = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
pytestmark = pytest.mark.skipif(os.name != "nt" or not CSC.exists(), reason="Windows .NET Framework CLI adapter")

FIXTURE = r'''
using System;
using System.IO;
using System.Text;
using System.Threading;
public class NativeFixture {
    public static int Main(string[] args) {
        string state=Environment.GetEnvironmentVariable("POLICY_TEST_STATE");
        string mode=Environment.GetEnvironmentVariable("POLICY_TEST_MODE");
        int previous=File.Exists(state)?Int32.Parse(File.ReadAllText(state)):0;
        File.WriteAllText(state,(previous+1).ToString());
        var encoded=new string[args.Length];
        for(int i=0;i<args.Length;i++) encoded[i]=Convert.ToBase64String(Encoding.UTF8.GetBytes(args[i]));
        File.AppendAllText(state+".calls",String.Join("\t",encoded)+"\n");
        if(mode=="missing") {
            Console.Error.WriteLine("Error: No such object: vllm-serving-lab-cache-policy"); return 1;
        }
        if(mode=="daemon-error") {
            Console.Error.WriteLine("Cannot connect to the Docker daemon"); return 1;
        }
        if(mode=="allocation" && previous==0) {
            Console.Error.WriteLine("fatal error: runtime: cannot allocate memory"); return 2;
        }
        if(mode=="hang-once" && previous==0) Thread.Sleep(20000);
        foreach(string arg in encoded) Console.WriteLine(arg);
        return 0;
    }
}
'''


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    directory = tmp_path_factory.mktemp("policy-docker-adapter")
    native = directory / "native.exe"
    adapter = directory / "docker.exe"
    fixture_source = directory / "NativeFixture.cs"
    fixture_source.write_text(FIXTURE, encoding="utf-8")
    for source, target in ((fixture_source, native), (ROOT / "scripts/PolicyDockerRetry.cs", adapter)):
        subprocess.run([str(CSC), "/nologo", "/target:exe", f"/out:{target}", str(source)],
                       check=True, capture_output=True, timeout=30)
    return adapter, native


def execute(binaries, tmp_path, mode, args):
    adapter, native = binaries
    state = tmp_path / "state"
    env = dict(os.environ, POLICY_DOCKER_NATIVE_EXE=str(native),
               POLICY_DOCKER_RETRY_LOG=str(tmp_path / "retry.log"),
               POLICY_TEST_STATE=str(state), POLICY_TEST_MODE=mode)
    result = subprocess.run([str(adapter), *args], env=env, capture_output=True,
                            text=True, encoding="utf-8", timeout=25)
    calls = [[base64.b64decode(x).decode("utf-8") for x in line.split("\t")]
             for line in state.with_suffix(".calls").read_text().splitlines()]
    return result, calls


def test_query_retries_allocation_failure_without_corrupting_arguments(binaries, tmp_path):
    args = ["inspect", "中文参数", 'quote"value', "trailing\\", "line1\nline2", '{"key":"value"}']
    result, calls = execute(binaries, tmp_path, "allocation", args)
    assert result.returncode == 0
    assert calls == [args, args]
    assert [base64.b64decode(x).decode("utf-8") for x in result.stdout.splitlines()] == args


def test_mutations_are_not_replayed(binaries, tmp_path):
    args = ["run", "--name", "vllm-serving-lab-cache-policy"]
    result, calls = execute(binaries, tmp_path, "allocation", args)
    assert result.returncode == 2
    assert calls == [args]


@pytest.mark.parametrize("mode,code", [("missing", 0), ("daemon-error", 1)])
def test_exact_owned_lookup_preserves_missing_and_error_semantics(binaries, tmp_path, mode, code):
    result, calls = execute(binaries, tmp_path, mode,
        ["ps", "--all", "--quiet", "--filter", "name=^/vllm-serving-lab-cache-policy$"])
    assert result.returncode == code
    assert calls == [["inspect", "--format", "{{.Id}}", "vllm-serving-lab-cache-policy"]]
    if mode == "missing":
        assert result.stdout == result.stderr == ""
    else:
        assert "Cannot connect" in result.stderr


def test_stalled_read_only_query_is_killed_and_retried(binaries, tmp_path):
    args = ["version"]
    result, calls = execute(binaries, tmp_path, "hang-once", args)
    assert result.returncode == 0
    assert calls == [args, args]
    assert "timeout=15s" in (tmp_path / "retry.log").read_text()
