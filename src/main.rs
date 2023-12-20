#[cfg(feature = "value")]
use exmex::{statements, Statement, StatementsVal};
#[cfg(feature = "value")]
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "value")]
    {
        let mut buffer = String::new();
        let mut stdout = io::stdout();
        let stdin = io::stdin();
        let mut statements = StatementsVal::<i32, f64>::default();
        loop {
            stdout.write_all("> ".as_bytes())?;
            stdout.flush()?;
            stdin.read_line(&mut buffer)?;
            let Statement { var, rhs } = statements::line_2_statement(buffer.trim())?;
            if let Some(var) = var {
                statements = statements.insert(var, rhs);
            } else {
                match rhs.eval(&statements) {
                    Ok(x) => println!("{x:?}"),
                    Err(e) => eprintln!("Error {e:?}"),
                }
            }

            buffer.clear();
        }
    }
    #[cfg(not(feature = "value"))]
    Ok(())
}
