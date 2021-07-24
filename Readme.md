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
exmex = "0.7.1"
```
to your `Cargo.toml`.

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

The expressions used for benchmarking are:
```
sin:     "sin(x)+sin(y)+sin(z)",
power:   "x^2+y*y+z^z",
nested:  "x*0.02*(3*(2*(sin(x - 1 / (sin(y * 5)) + (5.0 - 1/z)))))",
```
The following
table shows mean runtimes of 5-evaluation-runs with increasing `x`-values on a Win10 machine with an i5-8350U 1.7 GHz processor in micro-seconds, i.e., smaller means better.
We run [Criterion](https://docs.rs/criterion/0.3.4/criterion/)-based benchmarks via
```
cargo bench --bench benchmark -- --noplot --warm-up-time 2 --sample-size 10 --nresamples 250
```
to compute the results.

|        |sin|power|nested| comment|
|--------|---|-----|------|--------|
|[Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/)|2.4|2.3| 1.9|supports a faster, unsafe mode|
|[Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/)|9.1|  7|16.2| supports more than just math. expressions|
|[Meval](https://docs.rs/meval/0.2.0/meval/)   |1|1.1| 1.2|only `f64`, no custom operators|
|[Rsc](https://docs.rs/rsc/2.0.0/rsc/)     |7.9|9.5|17|
|**Exmex**   |**0.3**|**0.6**|**0.7**|

Note that we also tried the optimization flag `--emit=asm` which did not change the results qualitatively. Benchmarks for parsing on the aforementioned machine are shown in the following.
|        |parse all expressions (μs)|
|--------|---------------|
|[Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/)|**13.8**|
|[Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/)|43.1|
|[Meval](https://docs.rs/meval/0.2.0/meval/)   |24.5|
|[Rsc](https://docs.rs/rsc/2.0.0/rsc/)     |16.5|
|**Exmex**   |36.2|


We also tried to add the crates [Mexprp](https://docs.rs/mexprp/0.3.0/mexprp/) and [Asciimath](https://docs.rs/asciimath/0.8.8/asciimath/) to the benchmarking. Unfortunately, we could not make them run without errors on Win10. More details about the benchmarking can be found in the [source file](https://github.com/bertiqwerty/exmex/blob/main/benches/benchmark.rs). 

Note the unfortunate fact that Criterion does neither provide the option to simply report the minimum runtime nor to remove outliers before reporting a mean runtime as mentioned in the following [quote](https://bheisler.github.io/criterion.rs/book/analysis.html).
> Note, however, that outlier samples are not dropped from the data, and are used in the following analysis steps along with all other samples.




## Documentation
More documentation and examples also with integer and boolean data types can be found under [docs.rs/exmex/](https://docs.rs/exmex/) or generated via
```
cargo doc
```

## License
You as library user can select between MIT and Apache 2.0.
