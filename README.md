Transmit data over audio.

This project was created primarily for me to learn and understand the concepts of digital communications.

If you are looking for a mature digital signal processing library take a look at [liquid-dsp](https://github.com/jgaeddert/liquid-dsp) or [quiet](https://github.com/quiet/quiet) instead. Although, it does appear at least equal to the efficacy of [quiet-js](https://github.com/quiet/quiet-js)

# Usage
[Usage Instructions Overview](UsageInstructions.md)
## Run Instructions
### Cli
`cargo run --release --bin transceiver_cli -- <OPTIONS>`

The cli has help information if `-h` or `--help` or `help` are used as options.
### Native Gui
`cargo run --release --bin transceiver_gui -F gui`

The gui has help information on hover.
### Wasm Gui
Hosted demo at https://easyoakland.github.io/audio-network-interface/

To run locally: Install [`trunk`](trunkrs.dev/) then run
`trunk serve --release`. You may need to install [wasm-opt](https://github.com/WebAssembly/binaryen) first.

The gui has help information on hover.

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
        - [ ] Multiple frames
            - [x] Multiple contiguous frames
            - [ ] Multiple non-contiguous frames
            - [x] Coarse timing estimation
                - [x] Autocorrelation
                - [x] Cross-correlation
        - [ ] Precise timing estimation
            - [x] Pilot Symbol
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
        - Faster than realtime but don't yet have option to decode direct from microphone.
    - [x] From file
- [ ] Interface
    - [x] Command line interface (cli)
    - [x] Native gui
    - [x] WASM static website
        - Note: Not great because poor AudioWorklet and/or Multithreading in Wasm support. The single-threaded [implementation in cpal](https://github.com/RustAudio/cpal/issues/780) is sometimes too choppy/imprecise causing phase to rapidly become incorrect. Also see [here](https://github.com/bevyengine/bevy/issues/4078) for a good summary of the state of things.
        - [x] Local server.
        - [x] Public site.
    - [ ] Operating system network interface/adapter
