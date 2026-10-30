//! Interactive tensor calculator; pipe input for reproducible script execution.
use std::io::{self, IsTerminal};
fn main() -> io::Result<()> {
    matrix_calc::repl::run(
        io::stdin().lock(),
        io::stdout().lock(),
        io::stdin().is_terminal(),
    )
}
