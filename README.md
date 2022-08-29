[![Crate](https://img.shields.io/crates/v/exmex.svg)](https://crates.io/crates/exmex)
[![API](https://docs.rs/exmex/badge.svg)](https://docs.rs/exmex)
[![example workflow](https://github.com/bertiqwerty/exmex/actions/workflows/rust.yml/badge.svg)](https://github.com/bertiqwerty/exmex)
![license](https://img.shields.io/crates/l/exmex.svg)
# Exmex

Exmex is an extendable mathematical expression parser and evaluator. Ease of use, flexibility, and efficient evaluations are its main design goals. Exmex can parse mathematical expressions possibly containing variables and operators. On the one hand, it comes with a list of default operators for floating point values. For differentiable default operators, Exmex can compute partial derivatives. On the other hand, users can define their own operators and work with different data types such as float, integer, bool, or other types that implement `Clone`, `FromStr`, and `Debug`.

Parts of Exmex' functionality are accessible from Python via [Mexpress](https://github.com/bertiqwerty/mexpress).

## Installation
Add
```
[dependencies]
# ...
exmex = "0.16.1"
```
to your `Cargo.toml` for the [latest relase](https://crates.io/crates/exmex). If you want to use the newest version of Exmex, add
```
[dependencies]
# ...
exmex = { git = "https://github.com/bertiqwerty/exmex.git", branch = "main" }
```
to your `Cargo.toml`.
## Basic Usage
To simply evaluate a string there is
```rust
let result = exmex::eval_str::<f64>("e^(2*π-τ)")?;
assert!((result - 1.0).abs() < 1e-12);
```
where `π`/`PI`, `τ`/`TAU`, and Euler's number `E`/`e` are available as constants.
To create an expression with variables that represents a mathematical function you can use any string that does not define an operator or constant and matches `r"[a-zA-Zα-ωΑ-Ω_]+[a-zA-Zα-ωΑ-Ω_0-9]*"` as in
```rust
use exmex::prelude::*;
let expr = exmex::parse::<f64>("2*x^3-4/y")?;
```
The wildcard-import from `prelude` makes only the expression-trait `Express` and its implementation `FlatEx`, a flattened expression, accessible. To use variables, you do not need to use a context or tell the parser explicitly what variables are. To evaluate the function at, e.g., `x=2.0` and `y=4.0` you can use
```rust
let result = expr.eval(&[2.0, 4.0])?;
assert!((result - 15.0).abs() < 1e-12);
```
The order of the variables' values passed for evaluation has to match the alphabetical order of the variable names. 

Besides predefined operators for floats, you can implement custom operators and use their factory type as generic argument as shown in the following example.
```rust
use exmex::prelude::*;
use exmex::{BinOp, MakeOperators, Operator};
ops_factory!(
    BitwiseOpsFactory,
    u32,
    Operator::make_bin(
        "|",
        BinOp {
            apply: |a, b| a | b,
            prio: 0,
            is_commutative: true,
        }
    ),
    Operator::make_unary("!", |a| !a)
);
let expr = FlatEx::<_, BitwiseOpsFactory>::from_str("!(a|b)")?;
let result = expr.eval(&[0, 1])?;
assert_eq!(result, u32::MAX - 1);
```
More involved examples of data types are
* operators as operands as used for [day 19 of Advent of Code 2020](https://www.ninety.de/log/index.php/en/2021/11/11/parsing-operators-in-rust/) and
* the type [`Val`](https://docs.rs/exmex/0.16.1/exmex/enum.Val.html) that can be activated with the feature `value`, see below.

## Partial Differentiation

To compute partial derivatives of expressions with floating point numbers, you can use the method [`partial`](https://docs.rs/exmex/latest/exmex/trait.Differentiate.html#method.partial) after activating the Exmex-feature `partial` in the `Cargo.toml` via
```
[dependencies]
exmex = { ..., features = ["partial"] }
```

The result of the method [`partial`](https://docs.rs/exmex/latest/exmex/trait.Differentiate.html#method.partial) is again an expression.

```rust
use exmex::prelude::*;
let expr = exmex::parse::<f64>("y*x^2")?;

// d_x
let dexpr_dx = expr.partial(0)?;
assert_eq!(format!("{}", dexpr_dx), "({x}*2.0)*{y}");

// d_xy
let ddexpr_dxy = dexpr_dx.partial(1)?;
assert_eq!(format!("{}", ddexpr_dxy), "{x}*2.0");
let result = ddexpr_dxy.eval(&[2.0, f64::MAX])?;
assert!((result - 4.0).abs() < 1e-12);

// d_xyx
let dddexpr_dxyx = ddexpr_dxy.partial(0)?;
assert_eq!(format!("{}", dddexpr_dxyx), "2.0");
let result = dddexpr_dxyx.eval(&[f64::MAX, f64::MAX])?;
assert!((result - 2.0).abs() < 1e-12);

// all in one
let dddexpr_dxyx_iter = expr.partial_iter([0, 1, 0].iter())?;
assert_eq!(format!("{}", dddexpr_dxyx_iter), "2.0");
let result = dddexpr_dxyx_iter.eval(&[f64::MAX, f64::MAX])?;
assert!((result - 2.0).abs() < 1e-12);
```

## Mixing Data Types in one Expression with the Feature `value`

After activating the Exmex-feature `value` one can use expressions with data of type [`Val`](https://docs.rs/exmex/0.16.1/exmex/enum.Val.html), inspired by the type `Value` from the crate [Evalexpr](https://crates.io/crates/evalexpr). An instance of `Val` can contain a boolean, an int, or a float. This way, it is possible to use booleans, ints, and floats in the same expression. Further, Exmex provides in terms of [`ValOpsFactory`](https://docs.rs/exmex/0.16.1/exmex/struct.ValOpsFactory.html)  a pre-defined set of operators for `Val`. See the following example of a Python-like `if`-`else`-operator.
```rust
use exmex::{Express, Val};
let expr = exmex::parse_val::<i32, f64>("0 if b < c else 1.2")?;
let res = expr.eval(&[Val::Float(34.0), Val::Int(21)])?.to_float()?;
assert!((res - 1.2).abs() < 1e-12);
```

## Serialization and Deserialization

To use [`serde`](https://serde.rs/) activate the feature `serde`.
 

## Documentation
More documentation and examples including integer data types and boolean literals can be found for the latest release under [docs.rs/exmex/](https://docs.rs/exmex/) or generated via
```
cargo doc --all-features
```

## Benchmarks `v0.13.0`

The benchmarks were run for Exmex `v0.13.0`. However, do not expect a significant change for Exmex `v0.16.1` and the dependencies pinned in `v0.16.1`.

Exmex was created with flexibility (e.g., use your own operators, literals, and types), ergonomics (e.g., just finds variables), and evaluation speed in mind. On the other hand, Exmex is slower than the other crates during parsing. However, evaluation might be more performance critical depending on the application. 

The expressions used to compare Exmex with other creates are:
```
sin:     "sin(x)+sin(y)+sin(z)",
power:   "x^2+y*y+z^z",
nested:  "x*0.02*sin(-(3*(2*sin(x-1/(sin(y*5)+(5.0-1/z))))))",
compile: "x*0.2*5/4+x*2*4*1*1*1*1*1*1*1+7*sin(y)-z/sin(3.0/2/(1-x*4*1*1*1*1))",
```
The following table shows mean runtimes of 5-evaluation-runs with increasing `x`-values on a Win10 machine with an i7-10850H 2.7 GHz processor in micro-seconds, i.e., smaller means better. [Criterion](https://docs.rs/criterion/0.3.4/criterion/)-based benchmarks can be executed via
```
cargo bench --bench benchmark -- --noplot --sample-size 10 --nresamples 10
```
to compute the results. Reported is the best result over multiple invocations. More about taking the minimum run-time for benchmarking can be found below.

|                                                      | sin      | power   | nested   | compile  | comment                                        |
| ---------------------------------------------------- | -------- | ------- | -------- | -------- | ---------------------------------------------- |
| [Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/) | 5.88     | 4.51    | 19.36    | 21.11    | more than mathematical expressions             |
| *[Exmex](https://docs.rs/exmex)* `f64`               | **0.27** | **0.5** | **0.57** | **0.53** | can compute partial derivatives                |
| *[Exmex uncompiled](https://docs.rs/exmex)* `f64`    | **0.27** | **0.5** | **0.57** | 1.17     | can compute partial derivatives                |
| *[Exmex](https://docs.rs/exmex)* `Val`               | 0.77     | 1.13    | 1.87     | 1.73     | multiple data types in one expression possible |
| [Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/) | 1.19     | 1.46    | 1.59     | 1.6      | only `f64`, supports a faster, unsafe mode     |
| [Meval](https://docs.rs/meval/0.2.0/meval/)          | 0.65     | 0.66    | 0.82     | 1.01     | only `f64`, no custom operators                |
| [Rsc](https://docs.rs/rsc/2.0.0/rsc/)                | 4.88     | 8.21    | 13.32    | 24.28    |                                                |


Note that we also tried the optimization flag `--emit=asm` which did not change the results qualitatively. Benchmarks for parsing all expressions again in μs on the aforementioned machine are shown in the following.
|                                                                             | all expressions |
| --------------------------------------------------------------------------- | --------------- |
| [Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/)                        | 35.94           |
| *[Exmex](https://docs.rs/exmex)* `f64`                                      | 24.83           |
| *[Exmex uncompiled](https://docs.rs/exmex)* `f64`                           | 21.56           |
| *[Exmex](https://docs.rs/exmex)* `Val`                                      | 37.45           |
| [Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/)                        | 18.42           |
| [Meval](https://docs.rs/meval/0.2.0/meval/)                                 | **17.99**       |
| [Rsc](https://docs.rs/rsc/2.0.0/rsc/)                                       | 20.50           |

Exmex parsing can be made faster by passing only the relevant operators. 

The crates [Mexprp](https://docs.rs/mexprp/0.3.0/mexprp/) and [Asciimath](https://docs.rs/asciimath/0.8.8/asciimath/) did not run without errors on Win10. More details about the benchmarking can be found in the [source file](https://github.com/bertiqwerty/exmex/blob/main/benches/benchmark.rs). 

Note that Criterion does [not provide the option to simply report the minimum runtime](https://bheisler.github.io/criterion.rs/book/analysis.html). A [talk by
Andrei Alexandrescu](https://youtu.be/vrfYLlR8X8k?t=1024) explains why I think taking the minimum is a good idea in many cases. See also https://github.com/bheisler/criterion.rs/issues/485.

## License
You as library user can select between MIT and Apache 2.0.
