[![Crate](https://img.shields.io/crates/v/exmex.svg)](https://crates.io/crates/exmex)
[![API](https://docs.rs/exmex/badge.svg)](https://docs.rs/exmex)
[![example workflow](https://github.com/bertiqwerty/exmex/actions/workflows/rust.yml/badge.svg)](https://github.com/bertiqwerty/exmex)
![license](https://img.shields.io/crates/l/exmex.svg)
# Exmex

Exmex is a fast, simple, and **ex**tendable **m**athematical **ex**pression evaluator.  
Users can define their own operators and work with different data types such
as float, integer, bool, or other types that implement `Copy` and `FromStr`.

## Installation
Add
```
[dependencies]
# ...
exmex = "0.8.1"
```
to your `Cargo.toml`.

## Basic Usage
To simply evaluate a string there is
```rust
use exmex::eval_str;

let result = eval_str("sin(73)")?;
```
To create an expression with variables that represents a mathematical function you can
use any string that does not define an operator and matches `r"^[a-zA-Z_]+[a-zA-Z_0-9]*"` as in
```rust
use exmex::parse_with_default_ops;

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
use exmex::{parse, Operator};

let ops = [
    Operator {
        repr: "|",
        bin_op: Some(BinOp {
            apply: |a: u32, b: u32| a | b,
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

Exmex was created with flexibility (e.g., use your own operators, literals, and types), ergonomics (e.g., just finds variables), and evaluation speed in mind. On the other
hand, Exmex is slower than the other crates during parsing. 
However, evaluation might be more performance critical depending on the application. 

The expressions used to compare Exmex with other creates are:
```
sin:     "sin(x)+sin(y)+sin(z)",
power:   "x^2+y*y+z^z",
nested:  "x*0.02*sin(-(3*(2*sin(x-1/(sin(y*5)+(5.0-1/z))))))",
compile: "x*0.2*5/4+x*2*4*1*1*1*1*1*1*1+7*sin(y)-z/sin(3/2/(1-x*4*1*1*1*1))",
```
The following
table shows mean runtimes of 5-evaluation-runs with increasing `x`-values on a Win10 machine with an i5-8350U 1.7 GHz processor in micro-seconds, i.e., smaller means better.
[Criterion](https://docs.rs/criterion/0.3.4/criterion/)-based benchmarks can be executed via
```
cargo bench --bench benchmark -- --noplot --sample-size 10 --nresamples 20
```
to compute the results. Reported is the best result over multiple invocations. More about
taking the minimum run-time for benchmarking can be found below.

|        |sin|power|nested| compile|comment|
|--------|---|-----|------|--------|-------|
|[Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/)|9.8|7.36|19.73|27.07|more than mathematical expressions|
|**Exmex**   |**0.32**|**0.66**|**0.78**|**0.75**|
|[Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/)|2.4|2.64| 2.56|2.43|only `f64`, supports a faster, unsafe mode|
|[Meval](https://docs.rs/meval/0.2.0/meval/)   |1.03|1.1| 1.3|1.75|only `f64`, no custom operators|
|[Rsc](https://docs.rs/rsc/2.0.0/rsc/)     |9.03|9.93|36.74|55.77|


Note that we also tried the optimization flag `--emit=asm` which did not change the results qualitatively. Benchmarks for parsing all expressions again in μs on the aforementioned machine are shown in the following.
|        |all expressions|
|--------|--------------------------|
|[Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/)|69.94|
|**Exmex**   |48.69|
|[Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/)|48.12|
|[Meval](https://docs.rs/meval/0.2.0/meval/)   |**41.09**|
|[Rsc](https://docs.rs/rsc/2.0.0/rsc/)     |48.99|

Exmex parsing can be made faster by only passing the relevant operators. 

The crates [Mexprp](https://docs.rs/mexprp/0.3.0/mexprp/) and [Asciimath](https://docs.rs/asciimath/0.8.8/asciimath/) did not run without errors on Win10. More details about the benchmarking can be found in the [source file](https://github.com/bertiqwerty/exmex/blob/main/benches/benchmark.rs). 

Note the unfortunate fact that Criterion does neither provide the option to simply report the minimum runtime nor to remove outliers before reporting a mean runtime as mentioned in the following [quote](https://bheisler.github.io/criterion.rs/book/analysis.html).
> Note, however, that outlier samples are not dropped from the data, and are used in the following analysis steps along with all other samples.

[This talk by
Andrei Alexandrescu](https://youtu.be/vrfYLlR8X8k?t=1024) explains why I think
taking the minimum is a good idea in many cases. See also 
https://github.com/bheisler/criterion.rs/issues/485.




## Documentation
More documentation and examples also with integer and boolean data types can be found under [docs.rs/exmex/](https://docs.rs/exmex/) or generated via
```
cargo doc
```

## License
You as library user can select between MIT and Apache 2.0.
