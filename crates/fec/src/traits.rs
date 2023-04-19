/// Object has a configuration state. It can take an input, process it, and return output.
pub trait Function {
    type Input;
    type Output;

    /// Map the input to the output.
    fn map(&self, input: Self::Input) -> Self::Output;
}

/// Composes a new function that takes the input of `a` passes it into `b` and returns the output.
pub fn compose<A, B>(a: A, b: B) -> impl Fn(A::Input) -> B::Output
where
    A: Function,
    B: Function<Input = A::Output>,
{
    move |input: A::Input| b.map(a.map(input))
}
