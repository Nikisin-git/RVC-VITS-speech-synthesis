"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.config import ensure_dirs


def _enable_cross_integrity_drag_drop() -> None:
    """Let Explorer (medium IL) drop files onto our window even if we are
    running elevated. Windows blocks WM_DROPFILES across integrity levels
    by default — UIPI. ChangeWindowMessageFilterEx whitelists the messages.
    Silently no-op on non-Windows.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        MSGFLT_ALLOW = 1
        WM_DROPFILES = 0x0233
        WM_COPYDATA = 0x004A
        WM_COPYGLOBALDATA = 0x0049
        # We can't get the HWND yet (no top-level window), so we use the
        # global form: pass NULL hwnd so the rule applies process-wide.
        # That's the legacy ChangeWindowMessageFilter; in Win7+ the per-
        # window form (ChangeWindowMessageFilterEx) is preferred but the
        # global call still works and is enough for our use case.
        cwmf = getattr(user32, "ChangeWindowMessageFilter", None)
        if cwmf is None:
            return
        cwmf.argtypes = [wintypes.UINT, wintypes.DWORD]
        cwmf.restype = wintypes.BOOL
        for msg in (WM_DROPFILES, WM_COPYDATA, WM_COPYGLOBALDATA):
            cwmf(msg, MSGFLT_ALLOW)
    except Exception:
        # Best-effort: if anything goes wrong, fall back to default OS
        # behaviour rather than crashing the app on startup.
        pass


def main() -> int:
    ensure_dirs()
    _enable_cross_integrity_drag_drop()
    app = QApplication(sys.argv)
    # Defer import so config dirs exist before any subwidget queries them
    from app.ui.main_window import MainWindow
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
