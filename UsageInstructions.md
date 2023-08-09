To transmit a binary file:

1. Load the software on the device that has the file and a speaker.
2. Get a device (it can be the same one) which has a microphone.
3. Start recording on the second device. Match sample rate of output device when recording. On web this is likely 44.1kHz.
4. Use the `transmit` option with specified parameters to encode the file to sound and output through the speaker.
5. End the recording after the audio transmission finishes playing.
6. Use the `receive` option on any device with access to the recorded audio file with the same parameters as set when using `transmit`.
7. The file should be reconstructed on the receive device.