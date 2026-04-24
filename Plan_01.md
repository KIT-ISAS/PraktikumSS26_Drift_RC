
## Vorschlag Arbeitsplan

- Überblick verschaffen
    - Ausarbeitung von WS25/26
- Hardware in Betrieb nehmen
    - Holodeck Raumnutzung muss mit anderen Nutzern abgesprochen werden
- Hardware 
    - robuster machen: mechanischer Bumper rundum 
    - Kabel aufräumen
    - Stromversorgung vereinheitlichen
- Bremsen verbessern
    - erforderlich für sofortigen Abbruch des Trainings bei Verlassen der Strecke zur Vermeidung von Kollisionen
    - durch ESP-Programmierung
    - ... bei den meisten RC-Autos ist das Rückwärtsfahr-Signal gesperrt während man vorwärts fährt → nicht als Bremse nutzbar → warum? kann man das umgehen?
    - separate Hardware
- Loss-Funktion verbessern
    - Ende der Kurve wo er rausfliegt im Buffer zum Lernen mehr gewichten, damit er gezielt da dazulernt
- Overfitting-freies Lernen
    - als Input nur ein Stück des weiteren Straßenverlaufs, nicht ganze Strecke 
    - so sollte es allgemeiner lernen und dann für beliebige Strecken einsetzbar sein
- Einparken mit Driften 
    - wenig Platz notwendig
    - Loss-Funktion leicht aufstellbar (benötigter Längsplatz)
    - sicherlich schwierig überhaupt in die Nähe des Optimums zu kommen
- Holodeck PC
    - Holodeck PC: Simulation mit Tracking-Daten setzt manchmal 1 Sekunde aus → parallelisieren, auslesen auf anderem Thread, Zeitabstände berücksichtigen 
- weitere Baustellen: Ausarbeitung von WS25/26, Ausblick 
- Christoph Michelfeit uwzju@student.kit.edu, gerne kontaktieren, er will mitkriegen wie es mit dem Projekt weitergeht

