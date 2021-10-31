use std::{
    error::Error,
    fmt::{self, Display, Formatter},
};

/// This will be thrown at you if the somehting within Exmex went wrong. Ok, obviously it is not an
/// exception, so thrown needs to be understood figuratively.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct ExError {
    pub msg: String,
}
impl Display for ExError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
impl Error for ExError {}

/// Exmex' result type with [`ExError`](ExError) as error type.
pub type ExResult<U> = Result<U, ExError>;
