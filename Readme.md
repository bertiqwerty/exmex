[![Crate](https://img.shields.io/crates/v/exmex.svg)](https://crates.io/crates/exmex)
[![API](https://docs.rs/exmex/badge.svg)](https://docs.rs/exmex)
[![example workflow](https://github.com/bertiqwerty/exmex/actions/workflows/rust.yml/badge.svg)](https://github.com/bertiqwerty/exmex)
![license](https://img.shields.io/crates/l/exmex.svg)
# Exmex

Exmex is a fast, simple, and **ex**tendable **m**athematical **ex**pression evaluator. On the one hand, it comes with a list of default operators for floating point values. For differentiable default operators, exmex can compute partial derivatives. On the other hand, users can define their own operators and work with different data types such as float, integer, bool, or other types that implement `Copy` and `FromStr`.

## Installation
Add
```
[dependencies]
# ...
exmex = "0.9.3"
```
to your `Cargo.toml` for the latest relase. If you want to use the newest version, add
```
[dependencies]
# ...
exmex = { git = "https://github.com/bertiqwerty/exmex.git", branch = "main" }
```
to your `Cargo.toml`.
## Basic Usage
To simply evaluate a string there is
```rust
use exmex;
let result = exmex::eval_str("sin(73)")?;
```
To create an expression with variables that represents a mathematical function you can
use any string that does not define an operator and matches `r"^[a-zA-Z_]+[a-zA-Z_0-9]*"` as in
```rust
use exmex;
let expr = exmex::parse_with_default_ops::<f64>("2*x^3-4/z")?;
```
Especially, you do not need to use a context or tell the parser explicitly what variables are.
To evaluate the function at, e.g., `x=5.3` and `z=0.5` you can use
```rust
let value = expr.eval(&[5.3, 0.5]);
```
The order of the variables' values passed for evaluation has to match the alphabetical order of the variable names.
Besides predefined operators for floats, you can pass custom operators to the 
function `parse` to create an expression. 
```rust
use exmex::{self, Operator};

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
let expr = exmex::parse::<u32>("!(a|b)", &ops)?;
let result = expr.eval(&[0, 1])?;
assert_eq!(result, u32::MAX - 1);
```

## Partial Derivatives

To compute partial derivatives you can use the expression's method `partial`. The result is again an 
expression.

```rust
use exmex;
let expr = exmex::parse_with_default_ops::<f64>("y*x^2")?;

// d_x
let dexpr_dx = expr.partial(0)?;
assert_eq!(format!("{}", dexpr_dx), "({x}*2.0)*{y}");

// d_xy
let ddexpr_dxy = dexpr_dx.partial(1)?;
assert_eq!(format!("{}", ddexpr_dxy), "{x}*2.0");
assert_float_eq_f64(ddexpr_dxy.eval(&[2.0, f64::MAX])?, 4.0);
//                                            |
//                               The partial derivative still 
//                               has 2 variables but is 
//                               constant in y.

// d_xyx
let dddexpr_dxyx = ddexpr_dxy.partial(0)?;
assert_eq!(format!("{}", dddexpr_dxyx), "2.0");
assert_float_eq_f64(dddexpr_dxyx.eval(&[f64::MAX, f64::MAX])?, 2.0);
```

## Serialization and Deserialization

To use [`serde`](https://serde.rs/) for default operators you can activate the feature `serde_support`.
 

## Benchmarks `v0.9.0`

Exmex was created with flexibility (e.g., use your own operators, literals, and types), ergonomics (e.g., just finds variables), and evaluation speed in mind. On the other hand, Exmex is slower than the other crates during parsing. 
However, evaluation might be more performance critical depending on the application. 

The expressions used to compare Exmex with other creates are:
```
sin:     "sin(x)+sin(y)+sin(z)",
power:   "x^2+y*y+z^z",
nested:  "x*0.02*sin(-(3*(2*sin(x-1/(sin(y*5)+(5.0-1/z))))))",
compile: "x*0.2*5/4+x*2*4*1*1*1*1*1*1*1+7*sin(y)-z/sin(3.0/2/(1-x*4*1*1*1*1))",
```
The following
table shows mean runtimes of 5-evaluation-runs with increasing `x`-values on a Win10 machine with an i5-8350U 1.7 GHz processor in micro-seconds, i.e., smaller means better.
[Criterion](https://docs.rs/criterion/0.3.4/criterion/)-based benchmarks can be executed via
```
cargo bench --bench benchmark -- --noplot --sample-size 10 --nresamples 20
```
to compute the results. Reported is the best result over multiple invocations. More about
taking the minimum run-time for benchmarking can be found below.

|                                                      | sin     | power    | nested   | compile  | comment                                    |
| ---------------------------------------------------- | ------- | -------- | -------- | -------- | ------------------------------------------ |
| [Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/) | 9.8     | 7.36     | 19.73    | 27.07    | more than mathematical expressions         |
| **[Exmex](https://docs.rs/exmex)**                   | **0.3** | **0.62** | **0.76** | **0.74** |
| [Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/) | 1.88    | 2.24     | 2.36     | 2.39     | only `f64`, supports a faster, unsafe mode |
| [Meval](https://docs.rs/meval/0.2.0/meval/)          | 0.93    | 1.05     | 1.25     | 1.56     | only `f64`, no custom operators            |
| [Rsc](https://docs.rs/rsc/2.0.0/rsc/)                | 8.2     | 9.25     | 36.74    | 50.56    |


Note that we also tried the optimization flag `--emit=asm` which did not change the results qualitatively. Benchmarks for parsing all expressions again in μs on the aforementioned machine are shown in the following.
|                                                      | all expressions |
| ---------------------------------------------------- | --------------- |
| [Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/) | 69.94           |
| **[Exmex](https://docs.rs/exmex)**                   | 60.1            |
| [Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/) | 48.12           |
| [Meval](https://docs.rs/meval/0.2.0/meval/)          | **41.09**       |
| [Rsc](https://docs.rs/rsc/2.0.0/rsc/)                | 48.99           |

Exmex parsing can be made faster by passing only the relevant operators. 

The crates [Mexprp](https://docs.rs/mexprp/0.3.0/mexprp/) and [Asciimath](https://docs.rs/asciimath/0.8.8/asciimath/) did not run without errors on Win10. More details about the benchmarking can be found in the [source file](https://github.com/bertiqwerty/exmex/blob/main/benches/benchmark.rs). 

Note that Criterion does not provide the option to simply report the minimum runtime as mentioned in the following [quote](https://bheisler.github.io/criterion.rs/book/analysis.html).
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
