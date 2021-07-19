[![Crate](https://img.shields.io/crates/v/exmex.svg)](https://crates.io/crates/exmex)
[![API](https://docs.rs/exmex/badge.svg)](https://docs.rs/exmex)
[![example workflow](https://github.com/bertiqwerty/exmex/actions/workflows/rust.yml/badge.svg)](https://github.com/bertiqwerty/exmex)
# Exmex

Exmex is a fast **ex**tendable **m**athematical **ex**pression evaluator.  
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
        repr: "|",
        bin_op: Some(BinOp {
            op: |a: u32, b: u32| a | b,
            prio: 0,
        }),
        unary_op: None,
    },
    Operator {
        repr: "!",
        bin_op: None,
        unary_op: Some(|a: u32| !a),
    },
];
let expr = parse::<u32>("!({a}|{b})", ops)?;
let result = expr.eval(&[0, 1]);
assert_eq!(result, u32::MAX - 1);
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
to compare Exmex with other crates. The expressions used for benchmarking are:
```
xyz:     "x*y*z"
xx+:     "x*x+y*y+z*z"
x^2+:    "x^2+y^2+z^2"
comp:    "x+2*(6-4)-3/2.5+y+3.141*0.4*(2-32*(7+43*(1+5)))*0.1+x*y*z",
flat:    "2*6-4-3/2.5+3.141*0.4*x-32*y+43*z",
flatsin: "2*6-4-3/sin(2.5)+3.141*0.4*sin(x)-32*y+43*z",
nested:  "x*0.02*(3*(2*(sin(x - 1 / (sin(y * 5)) + (5.0 - 1/z)))))",
```
The following
table shows mean runtimes of 1000-evaluation-runs on an Ubuntu machine with Xeon 2.6 GHz processor in micro-seconds, i.e., smaller means better.

|        |xyz|xx+|x^2+|comp|flat|flatsin|nested| comment|
|--------|---------------|----------|----------|---|--------|---|---|---|
|[Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/)|145.92|233.37|195.11|204.72|183.05| 233.88|305.76|supports a faster, unsafe mode|
|[Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/)|499.14|943.57|801.53|2433.5|1507.1|1900.5|2011.7|
|[Meval](https://docs.rs/meval/0.2.0/meval/)   |50.927|86.030| **90.260**|163.22|**109.02**|201.44|**195.87**|only `f64`, no custom operators|
|[Rsc](https://docs.rs/rsc/2.0.0/rsc/)     |376.46|837.47|791.88|2521.2|1500|1825.5|1732.8|
|**Exmex**   |**37.837**|**70.085**|**86.786**|**93.205**|**103.73**|**140.50**|**195.78**|
|[Mexprp](https://docs.rs/mexprp/0.3.0/mexprp/) |-|-|-|-|-|-|-| did not compile on Win10|
|[Asciimath](https://docs.rs/asciimath/0.8.8/asciimath/)|-|-|-|-|-|-|-|lots of error messages during the run|

More details can be found in the [source file](https://github.com/bertiqwerty/exmex/blob/main/benches/benchmark.rs).

## Documentation
More documentation and examples also with integer and boolean data types can be found under [docs.rs/exmex/](https://docs.rs/exmex/) or generated via
```
cargo doc
```

## License
You as library user can select between MIT and Apache 2.0.
