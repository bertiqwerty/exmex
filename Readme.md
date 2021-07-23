[![Crate](https://img.shields.io/crates/v/exmex.svg)](https://crates.io/crates/exmex)
[![API](https://docs.rs/exmex/badge.svg)](https://docs.rs/exmex)
[![example workflow](https://github.com/bertiqwerty/exmex/actions/workflows/rust.yml/badge.svg)](https://github.com/bertiqwerty/exmex)
# Exmex

Exmex is a fast **ex**tendable **m**athematical **ex**pression evaluator.  
Users can define their own operators and work with different data types such
as float, integer, bool, or other types that implement `Copy` and `FromStr`.

## Installation
To install the latest commit add
```
[dependencies]
# ...
exmex = { git = "https://github.com/bertiqwerty/exmex.git", branch = "main" }
```
to your `Cargo.toml`. For the latest release, see https://crates.io/crates/exmex.

## Basic Usage
To simply evaluate a string there is
```rust
let result = eval_str("sin(73)")?;
```
To create an expression with variables that represents a mathematical function you can
use any string that does not define an operator and matches `r"^[a-zA-Z_]+[a-zA-Z_0-9]*"` as in
```rust
let expr = parse_with_default_ops::<f64>("2*x^3-4/z")?;
```
Especially, you do not need to use a context or tell the parser explicitly what variables are.
To evaluate the function at, e.g., `x=5.3` and `z=0.5` you can use
```rust
let value = expr.eval(&[5.3, 0.5]);
```
Besides predefined operators for floats, you can pass custom operators to the 
function `parse` to create an expression. 
```rust
let ops = [
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
let expr = parse::<u32>("!(a|b)", &ops)?;
let result = expr.eval(&[0, 1])?;
assert_eq!(result, u32::MAX - 1);
```

## Benchmarks

Exmex was created with flexibility (e.g., arbitrary names of binary and unary operators and a custom regex 
for number literals), ergonomics (e.g., just finds variables), and evaluation speed in mind. On the other
hand, Exmex is slower than the other crates during parsing. 
However, evaluation might be more performance critical depending on the application. 
We provide in this section [Criterion](https://docs.rs/criterion/0.3.4/criterion/)-based benchmarks. 
The expressions used for benchmarking are:
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
We run 
```
RUSTFLAGS=--emit=asm cargo bench
```
to compute the results.

|        |xyz|xx+|x^2+|comp|flat|flatsin|nested| comment|
|--------|---------------|----------|----------|---|--------|---|---|---|
|[Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/)|133.2|207.4|183.6|191.8|174| 215|285.8|supports a faster, unsafe mode|
|[Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/)|499.1|943.6|801.5|2433.5|1507.1|1900.5|2011.7| supports more than just math. expressions|
|[Meval](https://docs.rs/meval/0.2.0/meval/)   |50.9|86.0| **90.3**|163.2|**109.0**|201.44|**195.9**|only `f64`, no custom operators|
|[Rsc](https://docs.rs/rsc/2.0.0/rsc/)     |376.5|837.5|791.9|2521.2|1500.0|1825.5|1732.8|
|**Exmex**   |**32.1**|**61.2**|**81.8**|**87.1**|**97.5**|**132.8**|**180.3**|

Note that some crates such as [Meval](https://docs.rs/meval/0.2.0/meval/) did not care 
about the optimization flag `--emit=asm`. [Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/) 
and Exmex, on the other hand, were between 5% and 17% faster than on the same machine without `--emit=asm`.  

Benchmarks for parsing on the aforementioned machine are shown in the following.
|        |parse all expressions (μs)|
|--------|---------------|
|[Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/)|**23.4**|
|[Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/)|42.6|
|[Meval](https://docs.rs/meval/0.2.0/meval/)   |34.4|
|[Rsc](https://docs.rs/rsc/2.0.0/rsc/)     |25.3|
|**Exmex**   |82.9|

We also used a Win10 machine with an i5-8350U 1.7 GHz to benchmark evaluations. We excluded the slow crates from above and
omitted the optimization flag, i.e., we run
```
cargo bench
```
to obtain the following results.

|        |xyz|xx+|x^2+|comp|flat|flatsin|nested|
|--------|---------------|----------|----------|---|--------|---|---|
|[Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/)|412.8|521.9|645.5| 425.1|593.9|607.8|584.84|
|[Meval](https://docs.rs/meval/0.2.0/meval/)|117.7|182.6|298.9|341.9|237.5|322.2|400.4|
|**Exmex**|**35.2**|**69.4**|**169.5**|**109.4**|**134.5**|**141.3**|**150.9**|

We also tried to add the crates [Mexprp](https://docs.rs/mexprp/0.3.0/mexprp/) and [Asciimath](https://docs.rs/asciimath/0.8.8/asciimath/) to the benchmarking. Unfortunately, we could not make them run without errors on Win10. More details about the benchmarking can be found in the [source file](https://github.com/bertiqwerty/exmex/blob/main/benches/benchmark.rs). To see benchmarking results for evaluation and also parsing you can run `cargo bench` locally.

## Documentation
More documentation and examples also with integer and boolean data types can be found under [docs.rs/exmex/](https://docs.rs/exmex/) or generated via
```
cargo doc
```

## License
You as library user can select between MIT and Apache 2.0.
