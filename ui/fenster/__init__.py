"""Die Bereiche des Hauptfensters als Mixins.

Jede Datei hält Aufbau und Handler eines Fachbereichs als Mixin-Klasse; das
Hauptfenster (:class:`ui.main_window.MainWindow`) erbt von allen. ``self`` ist
in jedem Mixin das Hauptfenster.

Regeln:

- Den Sitzungszustand legt nur ``MainWindow.__init__`` an. Kein Mixin hat ein
  eigenes ``__init__``.
- Jede Mixin-Datei nennt im Kopf, welche Attribute sie schreibt.
- Neue Helfer tragen ein Bereichspräfix; kein Name darf in zwei Mixins
  vorkommen (sonst überdeckt einer still den anderen).
"""
