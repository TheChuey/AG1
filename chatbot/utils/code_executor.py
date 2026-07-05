import subprocess
import sys
import io


class CodeExecutor:
    @staticmethod
    def execute(code: str, timeout: int = 10) -> dict:
        try:
            process = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=timeout,
            )
            return {
                "stdout": process.stdout,
                "stderr": process.stderr,
                "exit_code": process.returncode,
                "success": process.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Execution timed out", "exit_code": -1, "success": False}
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "exit_code": 1, "success": False}

    @staticmethod
    def execute_with_test(code: str, timeout: int = 8) -> dict:
        harness = f"""
import sys, io
print("--- TEST START ---")
try:
    buf = io.StringIO()
    sys.stdout = buf
    {code}
    sys.stdout = sys.__stdout__
    print("CHECK 1: SYNTAX... OK")
    print("CHECK 2: EXECUTION... OK")
    print("\\n[DATA]:\\n" + buf.getvalue())
except Exception as e:
    sys.stdout = sys.__stdout__
    print(f"FAIL: {{type(e).__name__}}")
    print(f"INFO: {{str(e)}}")
    sys.exit(1)
"""
        try:
            p = subprocess.run([sys.executable, "-c", harness], capture_output=True, text=True, timeout=timeout)
            return {"success": p.returncode == 0, "report": p.stdout, "error": p.stderr}
        except Exception as e:
            return {"success": False, "report": "", "error": str(e)}
