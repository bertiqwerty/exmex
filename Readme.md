# Exexpress

Exexpress is an extendable expression evaluator for mathematical expressions.

## Extendability
We have a list of predifined operators, namely 
`^`, `*`, `/`, `+`, `-`, `sin`, `cos`, `tan`, `exp`, `log`, and `log2`.

If you do not like the pre-defined operators, you can define your own unary and binary operators as shown in the following.
```
let custom_ops = [
    ("**", OperatorPair { bin_op: Some(BinOp{op: |a: f32, b| a.powf(b), prio: 2}), unary_op: None }),
    ("*", OperatorPair { bin_op: Some(BinOp{op: |a, b| a * b, prio: 1}), unary_op: None }),
    ("invert", OperatorPair { bin_op: None, unary_op: Some(|a: f32| 1.0/a )}),
]
.iter()
.cloned()
.collect::<Vec<_>>();
let expr = parse::<f32>("2**2*invert(3)", custom_ops).unwrap();
let val = eval_expr::<f32>(&expr);  // == 4.0/3.0
```
Custom operators are defined as a vector of two-element-tuples. The first element is the `&str` that represents the operator in the to be parsed string, e.g., `**` in the example above. The second element is an instance of the struct `OperatorPair<T>` that has a binary and a unary operator of type `Option<fn(T, T) -> T>` and `Option<fn(T) -> T>`, respectively, as members. This means, operators can be both, binary and unary such as `-` as defined in the list of default operators.

## Priorities and Parantheses

In Exexpress-land, unary operators always have higher priority than binary operators, e.g., 
`-2^2=4` instead of `-2^2=-4`. Moreover, we are not too strict regarding parantheses. 
For instance `"---1"` will evalute to `-1`. 
If you want to be on the safe side, we suggest using parantheses.