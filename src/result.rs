use std::{
    error::Error,
    fmt::{self, Display, Formatter},
};

/// This will be thrown at you if the parsing went wrong. Ok, obviously it is not an
/// exception, so thrown needs to be understood figuratively.
#[derive(Debug, Clone)]
pub struct ExError {
    pub msg: String,
}
impl Display for ExError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
impl Error for ExError {}

pub type ExResult<U> = Result<U, ExError>;
