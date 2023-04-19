use num_complex::Complex;
use num_traits::identities::Zero;
use stft::fft::FourierFloat;

/// Always returns a complex zero.
#[inline]
pub fn null_encode<T: FourierFloat>(_: [bool; 0]) -> Complex<T> {
    Complex::zero()
}

/// Always returns no bits
#[inline]
pub fn null_decode<T: FourierFloat>(_: Complex<T>) -> Vec<bool> {
    vec![]
}

/// On off keying takes 1 bit and converts it to a real value.
/// This function converts a bit into a real valued complex number.
pub fn ook_encode<T: FourierFloat>(bit: [bool; 1]) -> Complex<T> {
    // Send either real value 1 or zero depending on bit.
    match bit[0] {
        true => Complex {
            re: T::one(),
            im: T::zero(),
        },
        false => Complex::zero(),
    }
}

/// On off keying takes 1 bit and converts it to a real value.
/// This function converts that real valued complex number back into a bit.
pub fn ook_decode<T: FourierFloat>(complex: Complex<T>) -> Vec<bool> {
    let dist_from_1 = (complex
        - Complex {
            re: T::one(),
            im: T::zero(),
        })
    .norm();
    let dist_from_0 = complex.norm();

    // Returns whichever value is closer.
    if dist_from_0 < dist_from_1 {
        vec![false]
    } else {
        vec![true]
    }
}

/// Binary phase modulation encoding. Sends constant magnitude and varies phase between 0 and pi.
pub fn bpsk_encode<T: FourierFloat>(bit: [bool; 1]) -> Complex<T> {
    match bit[0] {
        true => Complex {
            re: T::one(),
            im: T::zero(),
        },
        false => Complex {
            re: -T::one(),
            im: T::zero(),
        },
    }
}

/// Binary phase modulation decoding. Decodes each value depending on if it is closer to 1.0 or -1.0
pub fn bpsk_decode<T: FourierFloat>(complex: Complex<T>) -> Vec<bool> {
    let dist_from_0 = (complex
        - Complex {
            re: T::one(),
            im: T::zero(),
        })
    .norm();
    let dist_from_pi = (complex
        - Complex {
            re: -T::one(),
            im: T::zero(),
        })
    .norm();

    // Returns whichever value is closer.
    if dist_from_0 < dist_from_pi {
        vec![true]
    } else {
        vec![false]
    }
}

/* #[derive(Debug, Clone, Copy)]
pub struct Pilot<T: FourierFloat> {
    pub val: Complex<T>,
}

impl<T: FourierFloat> Pilot<T> {
    /// Encodes a constant value.
    #[inline]
    pub fn encode(&self) -> Complex<T> {
        self.val
    }

    /// Decodes scale factor of received pilot.
    pub fn decode(&self, received_val: Complex<T>) -> Complex<T> {
        self.val / received_val
    }
} */

#[cfg(test)]
mod tests {
    use num_complex::Complex;

    use super::{ook_decode, ook_encode};

    #[test]
    fn ook_encode_decode() {
        fn test(bit: bool) {
            let encoded: Complex<f64> = ook_encode([bit]);
            let decoded = ook_decode(encoded)[0];
            assert_eq!(
                bit, decoded,
                "{bit} was encoded as {encoded} and decoded as {decoded}"
            )
        }
        test(true);
        test(false);
    }
}
