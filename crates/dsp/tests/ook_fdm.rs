use dsp::ook_fdm::OokFdmConfig;

#[test]
fn size_one_bit() {
    let bits = [true].into_iter();
    let channel_width = 1.0;
    let parallel_channels_num = 1;
    let sample_rate = 4.0;
    let start_freq = 1.0;
    let symbol_length = 1;
    let config = OokFdmConfig::new(
        channel_width,
        parallel_channels_num,
        sample_rate,
        start_freq,
        symbol_length,
    );
    let iter = config.encode(bits);
    // If there is one bit it should be transmitted for one symbol length + 2 guard symbols;
    let signal_length = symbol_length + 2 * symbol_length;
    assert_eq!(signal_length, iter.clone().count());
    assert_eq!(iter.clone().count(), iter.clone().size_hint().0);
    assert_eq!(iter.clone().count(), iter.clone().size_hint().1.unwrap());

    // Size hint should stay correct
    let mut iter = iter.enumerate();
    while let Some((i, _)) = iter.next() {
        assert_eq!(
            signal_length - 1 - i,
            iter.clone().count(),
            "Length failed on iteration {i}"
        );
        assert_eq!(
            iter.clone().count(),
            iter.clone().size_hint().0,
            "Lower bound failed on iteration {i}"
        );
        assert_eq!(
            iter.clone().count(),
            iter.clone().size_hint().1.unwrap(),
            "Upper bound failed on iteration {i}"
        );
    }
}

#[test]
fn size_multi_bit() {
    let bits = [false, true].into_iter();
    let channel_width = 1.0;
    let parallel_channels_num = 2;
    let sample_rate = 6.0;
    let start_freq = 1.0;
    let symbol_length = 2;
    let config = OokFdmConfig::new(
        channel_width,
        parallel_channels_num,
        sample_rate,
        start_freq,
        symbol_length,
    );
    let iter = config.encode(bits);

    // If there are two bits in parallel should be transmitted for one symbol length + 2 guard symbol;
    let signal_length = symbol_length + 2 * symbol_length;
    assert_eq!(signal_length, iter.clone().count());
    assert_eq!(iter.clone().count(), iter.clone().size_hint().0);
    assert_eq!(iter.clone().count(), iter.clone().size_hint().1.unwrap());

    // Size hint should stay correct
    let mut iter = iter.enumerate();
    while let Some((i, _)) = iter.next() {
        assert_eq!(
            signal_length - 1 - i,
            iter.clone().count(),
            "Length failed on iteration {i}"
        );
        assert_eq!(
            iter.clone().count(),
            iter.clone().size_hint().0,
            "Lower bound failed on iteration {i}"
        );
        assert_eq!(
            iter.clone().count(),
            iter.clone().size_hint().1.unwrap(),
            "Upper bound failed on iteration {i}"
        );
    }
}
