[![Crate](https://img.shields.io/crates/v/exmex.svg)](https://crates.io/crates/exmex)
[![API](https://docs.rs/exmex/badge.svg)](https://docs.rs/exmex)
[![example workflow](https://github.com/bertiqwerty/exmex/actions/workflows/rust.yml/badge.svg)](https://github.com/bertiqwerty/exmex)
# Exmex

Exmex is an **ex**tendable **m**athematical **ex**pression evaluator.  
Users can define their own operators and work with different data types such
as float, integer, or bool.

## Installation

Add
```
[dependencies]
# ...
exmex = "0.5.0"
```
to your `Cargo.toml`.

## Basic Usage
To simply evaluate a string there is
```rust
let result = eval_str("sin(73)")?;
```
To create an expression with variables that represents a mathematical function you can
use curly brackets as in
```rust
let expr = parse_with_default_ops::<f64>("2*{x}^3-4/{z}")?;
```
To evaluate the function at, e.g., `x=5.3` and `z=0.5` you can use
```rust
let value = expr.eval(&[5.3, 0.5]);
```
Besides predefined operators, you can pass custom operators to the 
function `parse` to create an expression. 
```rust
let ops = vec![
    Operator {
        repr: "invert",
        bin_op: None,
        unary_op: Some(|a: f32| 1.0 / a),
    },
    Operator {
        repr: "sqrt",
        bin_op: None,
        unary_op: Some(|a: f32| a.sqrt()),
    },
];
let expr = parse::<f32>("sqrt(invert({a}))", ops)?;
let result = expr.eval(&[0.25]);
```

## Benchmarks

Exmex is not particularly fast during parsing. However, Exmex is efficient during evaluation
that might be more performance critical depending on the application. If you replace
`exmex=0.5.0` with
```
exmex = {git = "https://github.com/bertiqwerty/exmex"}
```
in your `Cargo.toml`, 
you can run benchmarks with on
```cargo bench``` 
to compare Exmex with other crates. We have used
* [Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/)
* [Meval](https://docs.rs/meval/0.2.0/meval/)
* [Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/)
and [Criterion](https://docs.rs/criterion/0.3.4/criterion/) as benchmarking tool. 
Unfortunately, [Mexprp](https://docs.rs/mexprp/0.3.0/mexprp/) did not compile on a
Win10 with i5-8350 processor used for benchmarking. Even faster
evaluation will be implemented in a future Exmex-release.

## Documentation
More documentation and examples also with integer and boolean data types can be found under [docs.rs/exmex/](https://docs.rs/exmex/) or generated via
```
cargo doc
```

## License
You as library user can select between MIT and Apache 2.0.
