# Alba
A music sheet writer !

Alba is a python project that can write a complete score of a given .WAV file ! I managed to get a mere 60% of accuracy. The score is given using the Lilypond format. You can then export them into .pdf using Lilypond.

## How does it work ?

For the first version, Alba uses Fourier decomposition to analyze the spectrum of the audio file and associates frequencies to notes, giving a prototype of what the score looks like !

I also managed to use an unsupervised machine learning algorithm, Non-Negative Matrix Factorization, which iteratively decompose a given matrix (here the spectrogram of the recording) into a matrix of base vectors and a matrix of coefficients, or "activations".
