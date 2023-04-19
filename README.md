Transmit data over audio.

This project was created primarily for me to learn and understand the concepts of digital communications.

If you are looking for a mature digital signal processing library take a look at [liquid-dsp](https://github.com/jgaeddert/liquid-dsp) or [quiet](https://github.com/quiet/quiet) instead.

# Goals/Roadmap
- [ ] Physical layer
    - [x] On Off Keyed (OOK) Frequency division multiplexing (FDM).
    - [ ] Forward error correction
        - [x] Reed solomon.
        - [x] Parity checks.
        - [ ] Hamming codes.
            - [x] 74
            - [ ] 84
        - [ ] Convolutional/Viterbi.
    - [ ] Orthogonal FDM (OFDM).
        - [x] Cyclic prefix.
        - [ ] Multiple packets.
            - [x] Multiple frames.
            - [x] Course timing estimation.
                - [x] Autocorrelation.
                - [x] Cross-correlation.
            - [ ] Precise timing estimation.
                - [ ] Pilot Symbol.
                - [ ] Pilot Channel.
    - [ ] Quadrature Phase Shift Keying (QPSK)
        - [x] bpsk (2 point)
        - [ ] qpsk (4 point)
        - [ ] n point
    - [ ] Quadrature Amplitude Modulation (QAM).
        - [ ] 2 point.
        - [ ] 4 point.
        - [ ] n point.
    - [ ] Different sampling frequency of transmitter vs receiver.
    - [ ] Realtime.
    - [x] From file.
- [ ] Interface
    - [x] Command line interface (cli).
    - [x] Native gui.
    - [ ] WASM static website.
    - [ ] Operating system network interface/adapter
