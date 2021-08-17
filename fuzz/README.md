# Fuzzing exmex
> Fuzzing or fuzz testing is an automated software testing technique that involves providing invalid, unexpected, or random data as inputs to a computer program. The program is then monitored for exceptions such as crashes, failing built-in code assertions, or potential memory leaks. 

_from [Wikipedia](https://en.m.wikipedia.org/wiki/Fuzzing)_

## Installation
### Nightly on Linux
To run fuzzing, Linux or WSL and a nightly compiler is necenssary due to [unstable compiler flags](https://doc.rust-lang.org/beta/unstable-book/compiler-flags.html). To install nightly, one can run
```
curl https://sh.rustup.rs -sSf | sh
rustup toolchain install nightly
```
and switch to `nightly` via
```
rustup default nightly
```
For more information about the different channels, see [the rustup documentation](https://rust-lang.github.io/rustup/concepts/channels.html).
### Cargo Fuzz
Install cargo fuzz via `cargo install cargo-fuzz`.

## Fuzzing
List targets via 
```
cargo fuzz list
```

Choose one of the targets and begin fuzzing via 
```
cargo fuzz run TARGET
```

## Extras
For multithreading use
```
cargo fuzz run TARGET --jobs=THREADS
```

To print `std::fmt::Debug` for a test case, use
```
cargo fuzz fmt TARGET INPUT
```

For coverage use
```
cargo fuzz coverage TARGET
```

For more information about fuzzing look at the [Rust Fuzz Book](https://rust-fuzz.github.io/book/introduction.html). 
