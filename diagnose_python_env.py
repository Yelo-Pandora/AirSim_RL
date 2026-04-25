import os
import sys


def try_import(name):
    try:
        mod = __import__(name)
        return True, getattr(mod, "__file__", None), None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def main():
    sys.path.insert(0, os.getcwd())
    print("exe:", sys.executable)
    print("cwd:", os.getcwd())

    for name in ("numpy", "airsim", "msgpackrpc", "tornado"):
        ok, path, err = try_import(name)
        if ok:
            print(f"{name}: OK ({path})")
        else:
            print(f"{name}: FAIL ({err})")

    ok, path, err = try_import("airsim")
    if ok:
        import airsim  # noqa: F401
        print("airsim_module_path:", airsim.__file__)


if __name__ == "__main__":
    main()

