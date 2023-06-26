Transmit data over audio.

This project was created primarily for me to learn and understand the concepts of digital communications.

If you are looking for a mature digital signal processing library take a look at [liquid-dsp](https://github.com/jgaeddert/liquid-dsp) or [quiet](https://github.com/quiet/quiet) instead.

# Usage
`cargo run --release --bin transceiver_cli -- <OPTIONS>`

or

`cargo run --release --bin transceiver_gui -F gui`

The cli has help information if `-h` or `--help` or `help` are used as options.

# Goals/Roadmap
- [ ] Physical layer
    - [x] On Off Keyed (OOK) Frequency division multiplexing (FDM)
    - [ ] Forward error correction
        - [x] [Reed solomon](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction)
        - [x] Parity checks
        - [ ] Hamming codes
            - [x] 74
            - [ ] 84
        - [ ] Convolutional/Viterbi
        - [ ] Cyclic redundancy check [CRC-32](https://en.wikipedia.org/wiki/Cyclic_redundancy_check)
    - [ ] Orthogonal FDM ([OFDM](https://en.wikipedia.org/wiki/Orthogonal_frequency-division_multiplexing))
        - [x] Cyclic prefix
        - [ ] Multiple packets
            - [x] Multiple frames
            - [x] Coarse timing estimation
                - [x] Autocorrelation
                - [x] Cross-correlation
            - [ ] Precise timing estimation
                - [ ] Pilot Symbol
                - [ ] Pilot Channel
    - [ ] Quadrature Phase Shift Keying ([QPSK](https://en.wikipedia.org/wiki/Phase-shift_keying#Quadrature_phase-shift_keying_(QPSK)))
        - [x] bpsk (2 point)
        - [ ] qpsk (4 point)
        - [ ] n point
    - [ ] Quadrature Amplitude Modulation ([QAM](https://en.wikipedia.org/wiki/Quadrature_amplitude_modulation))
        - [ ] 2 point
        - [ ] 4 point
        - [ ] n point
    - [ ] Different sampling frequency of transmitter vs receiver
    - [ ] Realtime
    - [x] From file
- [ ] Interface
    - [x] Command line interface (cli)
    - [x] Native gui
    - [ ] WASM static website
    - [ ] Operating system network interface/adapter
