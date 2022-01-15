use std::{
    error::Error,
    fmt::{self, Display, Formatter},
};

/// This will be thrown at you if the somehting within Exmex went wrong. Ok, obviously it is not an
/// exception, so thrown needs to be understood figuratively.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct ExError {
    msg: String,
}
impl ExError {
    pub fn new(msg: &str) -> ExError {
        ExError {
            msg: msg.to_string(),
        }
    }
    pub fn msg(&self) -> &str {
        self.msg.as_str()
    }
}
impl Display for ExError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
impl Error for ExError {}

/// Exmex' result type with [`ExError`](ExError) as error type.
pub type ExResult<U> = Result<U, ExError>;

/// Creates an [`ExError`](ExError) with a formatted message.
/// ```rust
/// # use std::error::Error;
/// use exmex::{format_exerr, ExError};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// assert_eq!(format_exerr!("some error {}", 1), ExError::new(format!("some error {}", 1).as_str()));
/// #     Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! format_exerr {
    ($s:literal, $( $exps:expr ),*) => {
        ExError::new(format!($s, $($exps,)*).as_str())
    }
}