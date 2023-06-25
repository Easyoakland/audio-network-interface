use bitvec::{prelude::Lsb0, vec::BitVec};
use dsp::{
    carrier_modulation::{
        bpsk_decode, bpsk_encode, null_decode, null_encode, ook_decode, ook_encode,
    },
    ofdm::{OfdmDataDecoder, OfdmDataEncoder, SubcarrierDecoder, SubcarrierEncoder},
};
use num_complex::Complex;
use num_traits::One;
use proptest::{prelude::ProptestConfig, proptest};

proptest! {
    #![proptest_config(ProptestConfig {cases: 10, max_shrink_iters: 1000, .. ProptestConfig::default()})]
    #[test]
    fn ofdm_data_encode_decode(input: Vec<bool>, cyclic_prefix_len in 0usize..1000, end_bits in 0usize..1000) {
        let input = input.into_iter();

        // Set subcarriers to modulate on.
        let mut subcarriers = [SubcarrierEncoder::T0(null_encode); 1000];
        subcarriers[1] = SubcarrierEncoder::T1(ook_encode::<f64>);
        subcarriers[2] = SubcarrierEncoder::T1(bpsk_encode);
        subcarriers[3] = SubcarrierEncoder::T1(ook_encode);

        // Create modulation.
        let ofdm = OfdmDataEncoder::new(input.clone(), &subcarriers, cyclic_prefix_len, end_bits);

         // Decode the modulation with the same subcarriers settings. Scale factors are set to multiplicative identity.
        let mut decoder_subcarriers: [_; 1000] = [SubcarrierDecoder::Data(null_decode); 1000];
        decoder_subcarriers[1] = SubcarrierDecoder::Data(ook_decode);
        decoder_subcarriers[2] = SubcarrierDecoder::Data(bpsk_decode);
        decoder_subcarriers[3] = SubcarrierDecoder::Data(ook_decode);
        let ofdm_decoder = OfdmDataDecoder::new(ofdm.flatten(), decoder_subcarriers, cyclic_prefix_len, [Complex::one(); 1000], end_bits);

        let decoded: BitVec = OfdmDataDecoder::decode(ofdm_decoder);
        assert_eq!(BitVec::<usize, Lsb0>::from_iter(input.into_iter().collect::<Vec<_>>()), decoded, "left (input) != right (decoded)")
    }

}
