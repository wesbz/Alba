# Alba
A music sheet writer !

Alba is a python project that can write a complete score of a given .WAV file ! I managed to get a mere 60% of accuracy. The score is given using the Lilypond format. You can then export them into .pdf using Lilypond.

# How does it work ?

Alba uses Fourier decomposition to analyze the spectrum of the audio file and associates frequencies to notes, giving a prototype of what the score looks like !
