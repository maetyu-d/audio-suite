The applications below are built in Python and tested on Mac OS Ventura 13.7.8. Once you have installed PyQt6, pyqtgraph, numpy and soundfile (in the Terminal, run "pip install PyQt6 pyqtgraph numpy soundfile"), it should just be a case of cd into the folder of the app you want to run, then run "python main.py" (depending on which version of Python you have installed, this may be "python3 main.py"). 

Microsound is a sound-generation application focused on microsound, offering a wide range of unusual, experimental, and novel synthesis techniques. Most notably, it can generate an initial transient at extremely high sample rates (optionally band-limited), then unfold this transient by up to hundreds of times in the time domain and/or stretch its spectrum by up to ×4.0. [video demo](https://youtu.be/s1LnN1odIZE)

Grid Audio is a deliberately warped take on a DAW: simple in some respects, but highly flexible in others. Each track can operate with its own sense of time, featuring independently modulatable clocks and a programmable timing grid. At any point within this grid, a track can trigger WAV files or execute Python code snippets for sound generation. These code snippets can be aware of their position within a track, the state of other tracks, how many times tracks have been restarted, and can even restart other tracks in response. [video demo](https://youtu.be/vZM6BZ9ggYg)

Pattern Lab explores algorithmically generated musical patterns, realised through a Mega Drive–style chiptunes synthesizer. Extensive pattern generation and parameter modification are actively encouraged.

Forest Fire OSC is a real-time forest fire simulation built in Python using NumPy and Qt/pyqtgraph, modelling fuel, moisture, wind, terrain, rain, embers, and forest regrowth while visualising both the landscape and system metrics. A rule-based OSC layer lets the evolving dynamics drive external systems such as Pure Data via thresholds and events. [video demo](https://youtu.be/KwqHsVlGpEM?si=WCwv7E0Q_PDpaE5V)

All applications are functional, but their UIs are often basic and ugly. Note that you will probably get some really weird sounds too.


