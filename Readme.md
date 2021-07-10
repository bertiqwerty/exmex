# Exexpress

Exexpress parses and evaluates mathematical expressions. Thereby, library users can
define their own unary and binary operators.

## Installation

Add
```
[dependencies]
exexpress = { git = "https://github.com/bertiqwerty/exexpress.git", branch = "main" } 
```
to your `Cargo.toml`.

## Basic Usage
To simply parse a string there is
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
let value = eval_expr::<f64>(&expr, &[5.3, 0.5]);
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
let result = eval_expr::<f32>(&expr, &[0.25]);  // 2.0
```

## Documentation
More documentation and examples also with integer and boolean data types can be 
generated via
```
cargo doc
```
