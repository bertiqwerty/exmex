# Fuzzing exmex

## Prerequisites:
Install cargo fuzz via:
`cargo install cargo-fuzz`

## Fuzzing:
List targets via:
`cargo fuzz list`

Choose one of the targets and begin fuzzing via: 
```
cargo fuzz run TARGET
```

## Extras:
Multithreading:
```
cargo fuzz run TARGET --jobs=THREADS
```


Print std::fmt::Debug for a test case:
```
cargo fuzz fmt TARGET INPUT
```


Coverage:
```
cargo fuzz coverage TARGET
```

For more information look at the [Rust Fuzz Book](https://rust-fuzz.github.io/book/introduction.html) and the 
[Rust Unstable Book](https://doc.rust-lang.org/beta/unstable-book/compiler-flags.html)