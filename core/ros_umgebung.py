"""ROS-2-Installation erkennen: Humble (Ubuntu 22.04) oder Jazzy (Ubuntu 24.04).

Qt-frei. Die App laeuft ohne gesourctes ROS; nur die Subprozesse von FAST-LIO
und RViz sourcen eine Umgebung, und die rosbags-Leser holen Nicht-Standard-
Nachrichtentypen (mavros_msgs) als .msg-Datei aus dem share-Verzeichnis.
Welche Distribution das ist, steht nur hier.

Reihenfolge: ``ROS_DISTRO`` (wenn vor dem Start gesourct wurde und die
Installation dazu existiert), sonst die erste vorhandene aus ``BEKANNTE``,
sonst irgendeine unter ``/opt/ros`` mit setup.bash.

Die eigenen Workspaces (``ws_livox``, ``fastlio2_ws``, ``EPIC_ros2``) liegen im
Home; ``SUPER360_ROS_WS`` legt sie woanders hin (Docker-Image, in dem HOME je
nach Lauf ein anderer Ordner ist).
"""
from __future__ import annotations

import os

__all__ = ["ROS_WURZEL", "BEKANNTE", "distro", "setup_bash", "share_pfad",
           "msg_pfad", "passende_distro", "erwartet", "workspace"]

ROS_WURZEL = "/opt/ros"
#: Unterstuetzte Distributionen, bevorzugte zuerst. Auf einem Rechner liegt
#: normalerweise nur eine (Humble gibt es nur fuer 22.04, Jazzy nur fuer 24.04).
BEKANNTE = ("jazzy", "humble")
#: Ubuntu-Codename -> passende ROS-2-Distribution (fuer Installationshinweise).
_PASSEND = {"jammy": "humble", "noble": "jazzy"}


def _hat_setup(wurzel: str, name: str) -> bool:
    return bool(name) and os.path.isfile(os.path.join(wurzel, name, "setup.bash"))


def distro(wurzel: str = ROS_WURZEL) -> str | None:
    """Name der zu nutzenden ROS-2-Distribution, None ohne Installation."""
    gesetzt = os.environ.get("ROS_DISTRO", "").strip()
    if _hat_setup(wurzel, gesetzt):
        return gesetzt
    try:
        vorhanden = sorted(n for n in os.listdir(wurzel) if _hat_setup(wurzel, n))
    except OSError:
        return None
    for name in BEKANNTE:
        if name in vorhanden:
            return name
    return vorhanden[0] if vorhanden else None


def setup_bash(wurzel: str = ROS_WURZEL) -> str | None:
    """Pfad der setup.bash der erkannten Distribution, None ohne Installation."""
    name = distro(wurzel)
    return os.path.join(wurzel, name, "setup.bash") if name else None


def share_pfad(*teile: str, wurzel: str = ROS_WURZEL) -> str | None:
    """Pfad unter <wurzel>/<distro>/share (muss nicht existieren), None ohne ROS."""
    name = distro(wurzel)
    return os.path.join(wurzel, name, "share", *teile) if name else None


def msg_pfad(paket: str, typ: str, wurzel: str = ROS_WURZEL) -> str | None:
    """.msg-Datei eines Nachrichtentyps, z. B. ('mavros_msgs', 'GPSRAW')."""
    return share_pfad(paket, "msg", f"{typ}.msg", wurzel=wurzel)


def workspace(name: str) -> str:
    """Ordner eines eigenen ROS-Workspaces: unter $SUPER360_ROS_WS, sonst ~."""
    basis = os.environ.get("SUPER360_ROS_WS") or os.path.expanduser("~")
    return os.path.join(basis, name)


def passende_distro(os_release: str = "/etc/os-release") -> str:
    """Die ROS-2-Distribution, die zur laufenden Ubuntu-Version gehoert."""
    try:
        with open(os_release, encoding="utf-8") as f:
            for zeile in f:
                schluessel, _, wert = zeile.strip().partition("=")
                if schluessel == "VERSION_CODENAME":
                    return _PASSEND.get(wert.strip('"'), "humble")
    except OSError:
        pass
    return "humble"


def erwartet(wurzel: str = ROS_WURZEL) -> str:
    """Kurzer Text fuer Fehlermeldungen: welche setup.bash gesucht wurde."""
    return " oder ".join(os.path.join(wurzel, n, "setup.bash") for n in BEKANNTE)


if __name__ == "__main__":
    import tempfile

    print("== Test 1: Erkennung in einem nachgebauten /opt/ros ==")
    alt = os.environ.pop("ROS_DISTRO", None)
    with tempfile.TemporaryDirectory() as tmp:
        assert distro(tmp) is None and setup_bash(tmp) is None, "leer: keine Distribution"
        assert msg_pfad("mavros_msgs", "GPSRAW", wurzel=tmp) is None
        os.makedirs(os.path.join(tmp, "humble"))
        assert distro(tmp) is None, "ohne setup.bash zaehlt ein Verzeichnis nicht"
        open(os.path.join(tmp, "humble", "setup.bash"), "w").close()
        assert distro(tmp) == "humble"
        assert setup_bash(tmp) == os.path.join(tmp, "humble", "setup.bash")
        assert msg_pfad("mavros_msgs", "GPSRAW", wurzel=tmp) == os.path.join(
            tmp, "humble", "share", "mavros_msgs", "msg", "GPSRAW.msg")
        os.makedirs(os.path.join(tmp, "jazzy"))
        open(os.path.join(tmp, "jazzy", "setup.bash"), "w").close()
        assert distro(tmp) == "jazzy", "beide da: Jazzy zuerst"
        os.environ["ROS_DISTRO"] = "humble"
        assert distro(tmp) == "humble", "gesourctes ROS_DISTRO gewinnt"
        os.environ["ROS_DISTRO"] = "rolling"
        assert distro(tmp) == "jazzy", "ROS_DISTRO ohne Installation zaehlt nicht"
        del os.environ["ROS_DISTRO"]
        for n in ("humble", "jazzy"):
            os.remove(os.path.join(tmp, n, "setup.bash"))
        os.makedirs(os.path.join(tmp, "kilted"))
        open(os.path.join(tmp, "kilted", "setup.bash"), "w").close()
        assert distro(tmp) == "kilted", "unbekannte Distribution als letzte Wahl"
        rel = os.path.join(tmp, "os-release")
        for code, soll in (("jammy", "humble"), ("noble", "jazzy")):
            with open(rel, "w") as f:
                f.write(f'NAME="Ubuntu"\nVERSION_CODENAME={code}\n')
            assert passende_distro(rel) == soll
    if alt is not None:
        os.environ["ROS_DISTRO"] = alt
    ws_alt = os.environ.pop("SUPER360_ROS_WS", None)
    assert workspace("ws_livox") == os.path.expanduser("~/ws_livox")
    os.environ["SUPER360_ROS_WS"] = "/opt/ws"
    assert workspace("fastlio2_ws") == "/opt/ws/fastlio2_ws"
    del os.environ["SUPER360_ROS_WS"]
    if ws_alt is not None:
        os.environ["SUPER360_ROS_WS"] = ws_alt
    print("  leer, Humble, beide, ROS_DISTRO, unbekannte, os-release, Workspaces — OK")

    print("== Test 2: dieser Rechner ==")
    print(f"  distro={distro()} setup={setup_bash()} passend={passende_distro()}")
    print(f"  GPSRAW.msg={msg_pfad('mavros_msgs', 'GPSRAW')}")
    print(f"  ws_livox={workspace('ws_livox')} fastlio2_ws={workspace('fastlio2_ws')}")
