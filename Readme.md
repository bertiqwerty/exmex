[![Crate](https://img.shields.io/crates/v/exmex.svg)](https://crates.io/crates/exmex)
[![API](https://docs.rs/exmex/badge.svg)](https://docs.rs/exmex)
[![example workflow](https://github.com/bertiqwerty/exmex/actions/workflows/rust.yml/badge.svg)](https://github.com/bertiqwerty/exmex)
# Exmex

Exmex is an **ex**tendable **m**athematical **ex**pression evaluator.  
Users can define their own operators and work with different data types such
as float, integer, bool, or other types that implement `Copy` and `FromStr`.

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
Besides predefined operators for floats, you can pass custom operators to the 
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
exmex = { git = "https://github.com/bertiqwerty/exmex", branch="main" }
```
in your `Cargo.toml`, 
you can run [Criterion](https://docs.rs/criterion/0.3.4/criterion/)-based benchmarks with
```
cargo bench
``` 
to compare Exmex with other crates. The
expressions used for benchmarking are:
```
flat:    "2 * 6 - 4 - 3 / 2.5 + 3.141 * 0.4 * x - 32 * y + 43 * z",
flatsin: "2 * 6 - 4 - 3 / sin(2.5) + 3.141 * 0.4 * sin(x) - 32 * y + 43 * z",
nested:  "x*0.02*(3*(2*(sin(x - 1 / (sin(y * 5)) + (5.0 - 1/z)))))",
```
The following
table shows mean runtimes of 1000-evaluation-runs on a Win10 machine with i5-8350 processor in micro-seconds.

|        |flat           |flatsin   | nested   | comment|
|--------|---------------|----------|----------|--------|
|[Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/)|          373.6|     385.3|     491.7|
|[Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/)|         2531.8|      3333|    3455.2|
|[Meval](https://docs.rs/meval/0.2.0/meval/)   |            190|**242.35**| **268.2**|
|[Rsc](https://docs.rs/rsc/2.0.0/rsc/)     |         6867.7|   7616.1 |    7731.5|
|**Exmex**   |      **149.3**|     292.1|     374.2|
|[Mexprp](https://docs.rs/mexprp/0.3.0/mexprp/) |-|-|-| did not compile on a Win10 with i5-8350 processor|
|[Asciimath](https://docs.rs/asciimath/0.8.8/asciimath/)|-|-|-|did print a lot of error messages during the run|

More details can be found in the [source file](https://github.com/bertiqwerty/exmex/blob/main/benches/benchmark.rs).
Even faster evaluation will be implemented in a future Exmex-release. 
## Documentation
More documentation and examples also with integer and boolean data types can be found under [docs.rs/exmex/](https://docs.rs/exmex/) or generated via
```
cargo doc
```

## License
You as library user can select between MIT and Apache 2.0.
