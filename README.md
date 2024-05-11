[![Crate](https://img.shields.io/crates/v/exmex.svg)](https://crates.io/crates/exmex)
[![API](https://docs.rs/exmex/badge.svg)](https://docs.rs/exmex)
[![example workflow](https://github.com/bertiqwerty/exmex/actions/workflows/rust.yml/badge.svg)](https://github.com/bertiqwerty/exmex)
![license](https://img.shields.io/crates/l/exmex.svg)
[![dependency status](https://deps.rs/repo/github/bertiqwerty/exmex/status.svg)](https://deps.rs/repo/github/bertiqwerty/exmex)
[![downloads](https://img.shields.io/crates/d/exmex.svg)](https://crates.io/crates/exmex)

# Exmex

Exmex is an extendable mathematical expression parser and evaluator. Ease of use, flexibility, and efficient evaluations are its main design goals. Exmex can parse mathematical expressions possibly containing variables and operators. On the one hand, it comes with a list of default operators for floating point values. For differentiable default operators, Exmex can compute partial derivatives. On the other hand, users can define their own operators and work with different data types such as float, integer, bool, or other types that implement `Clone`, `FromStr`, `Debug`, `Default`.

Parts of Exmex' functionality are accessible from Python via [Mexpress](https://github.com/bertiqwerty/mexpress).

## Installation
Run
```
cargo add exmex
```
in your project's directory for the [latest relase](https://crates.io/crates/exmex). If you want to use the newest version of Exmex, add
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
let expr = FlatEx::<_, BitwiseOpsFactory>::parse("!(a|b)")?;
let result = expr.eval(&[0, 1])?;
assert_eq!(result, u32::MAX - 1);
```
More involved examples of data types are
* operators as operands as used for [day 19 of Advent of Code 2020](https://www.bertiqwerty.com/posts/operatorenparsen-in-rust/) and
* the type [`Val`](https://docs.rs/exmex/latest/exmex/enum.Val.html) that can be activated with the feature `value`, see below.

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

## Mixing Scalar Data Types and Float Vectors in one Expression with the Feature `value`

After activating the Exmex-feature `value` one can use expressions with data of type [`Val`](https://docs.rs/exmex/latest/exmex/enum.Val.html), inspired by the type `Value` from the crate [Evalexpr](https://crates.io/crates/evalexpr). An instance of `Val` can contain a boolean, an int, a float, or a vector of floats. This way, it is possible to use booleans, ints, floats, and vectors in the same expression. Further, Exmex provides in terms of [`ValOpsFactory`](https://docs.rs/exmex/latest/exmex/struct.ValOpsFactory.html) a pre-defined set of operators for `Val`. See the following example of a Python-like `if`-`else`-operator.
```rust
use exmex::{Express, Val};
let expr = exmex::parse_val::<i32, f64>("0 if b < c else 1.2")?;
let res = expr.eval(&[Val::Float(34.0), Val::Int(21)])?.to_float()?;
assert!((res - 1.2).abs() < 1e-12);
```

See the [`Val`-docs](https://docs.rs/exmex/latest/exmex/enum.Val.html) for an example containing vectors.

If both `partial` and `value` are activated, partial derivatives can be computed for expressions of types such 
as `FlatExVal<i32, f64>`. This is currently not supported for vectors.

```rust
use exmex::{Differentiate, Express, Val};
let expr = exmex::parse_val::<i32, f64>("3*x if x > 1 else x^2")?;
let deri = expr.partial(0)?;
let res = deri.eval(&[Val::Float(1.0)])?.to_float()?;
assert!((res - 2.0).abs() < 1e-12);
let res = deri.eval(&[Val::Float(7.0)])?.to_float()?;
assert!((res - 3.0).abs() < 1e-12);
```

## Serialization and Deserialization

To use [`serde`](https://serde.rs/) activate the feature `serde`.
 

## Documentation
More features and examples including integer data types and boolean literals can be found for the documentation release under [docs.rs/exmex/](https://docs.rs/exmex/) or generated via
```
cargo doc --all-features --no-deps
```

## Benchmarks `v0.17.2`

Exmex was created with flexibility (e.g., use your own operators, literals, and types), ergonomics (e.g., just finds variables), and evaluation speed in mind. On the other hand, Exmex is slower than the other crates during parsing. However, evaluation might be more performance critical depending on the application. 

The expressions used to compare Exmex with other creates are:
```
sin:     "sin(x)+sin(y)+sin(z)",
power:   "x^2+y*y+z^z",
nested:  "x*0.02*sin(-(3*(2*sin(x-1/(sin(y*5)+(5.0-1/z))))))",
compile: "x*0.2*5/4+x*2*4*1*1*1*1*1*1*1+7*sin(y)-z/sin(3.0/2/(1-x*4*1*1*1*1))",
```
The following table shows mean runtimes of 5-evaluation-runs with increasing `x`-values on a Macbook Pro M1 Max in micro-seconds, i.e., smaller means better. [Criterion](https://docs.rs/criterion/latest/criterion/index.html)-based benchmarks can be executed via
```
cargo bench --all-features --bench benchmark -- --noplot --sample-size 10 --nresamples 10
```
to compute the results. Reported is the best result over multiple invocations. More about taking the minimum run-time for benchmarking can be found below.

|                                                      | sin      | power    | nested  | compile | comment                                        |
| ---------------------------------------------------- | -------- | -------- | ------- | ------- | ---------------------------------------------- |
| [Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/) | 3.9      | 3.23     | 7.84    | 11.06   | more than mathematical expressions             |
| *[Exmex](https://docs.rs/exmex)* `f64`               | **0.14** | **0.18** | **0.5** | **0.3** | can compute partial derivatives                |
| *[Exmex uncompiled](https://docs.rs/exmex)* `f64`    | **0.14** | **0.18** | **0.5** | 0.66    | can compute partial derivatives                |
| *[Exmex](https://docs.rs/exmex)* `Val`               | 0.57     | 0.45     | 1.14    | 0.94    | multiple data types in one expression possible |
| [Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/) | 0.68     | 0.78     | 1.19    | 1.03    | only `f64`, supports a faster, unsafe mode     |
| [Meval](https://docs.rs/meval/0.2.0/meval/)          | 0.42     | 0.43     | 0.79    | 0.99    | only `f64`, no custom operators                |
| [Rsc](https://docs.rs/rsc/2.0.0/rsc/)                | 4.52     | 5.0      | 10.47   | 19.8    |                                                |


Note that we also tried the optimization flag `--emit=asm` which did not change the results qualitatively. Benchmarks for parsing all expressions again in μs on the aforementioned machine are shown in the following.
|                                                      | all expressions |
| ---------------------------------------------------- | --------------- |
| [Evalexpr](https://docs.rs/evalexpr/6.3.0/evalexpr/) | 28.95           |
| *[Exmex](https://docs.rs/exmex)* `f64`               | 23.45           |
| *[Exmex uncompiled](https://docs.rs/exmex)* `f64`    | 20.00           |
| *[Exmex](https://docs.rs/exmex)* `Val`               | 33.32           |
| [Fasteval](https://docs.rs/fasteval/0.2.4/fasteval/) | **13.04**       |
| [Meval](https://docs.rs/meval/0.2.0/meval/)          | 15.00           |
| [Rsc](https://docs.rs/rsc/2.0.0/rsc/)                | 14.64           |

Exmex parsing can be made faster by passing only the relevant operators. 

The crates [Mexprp](https://docs.rs/mexprp/0.3.0/mexprp/) and [Asciimath](https://docs.rs/asciimath/0.8.8/asciimath/) did not run without errors on Win10. However, I have not tried to use them in a while. More details about the benchmarking can be found in the [source file](https://github.com/bertiqwerty/exmex/blob/main/benches/benchmark.rs). 

Note that Criterion does [not provide the option to simply report the minimum runtime](https://bheisler.github.io/criterion.rs/book/analysis.html). A [talk by
Andrei Alexandrescu](https://youtu.be/vrfYLlR8X8k?t=1024) explains why I think taking the minimum is a good idea in many cases. See also https://github.com/bheisler/criterion.rs/issues/485.

## License
You as library user can select between MIT and Apache 2.0.
