# Exexpress

Exexpress is an extendable expression evaluator for mathematical expressions.

## Extendability
We have a list of predifined operators, namely 
`^`, `*`, `/`, `+`, `-`, `sin`, `cos`, `tan`, `exp`, `log`, and `log2`.

If we need a different set of operators, we can proceed as shown in the following.
```
let custom_ops = vec![
    ("**", OperatorPair { bin_op: Some(BinOp{op: |a: f32, b| a.powf(b), prio: 2}), unary_op: None }),
    ("*", OperatorPair { bin_op: Some(BinOp{op: |a, b| a * b, prio: 1}), unary_op: None }),
    ("invert", OperatorPair { bin_op: None, unary_op: Some(|a: f32| 1.0/a )}),
];
let expr = parse::<f32>("2**2*invert(3)", custom_ops).unwrap();
let val = eval_expr::<f32>(&expr, &vec![]);  // 4.0/3.0
```
Custom operators are defined as a vector of two-element-tuples. The first element is the `&str` that represents the operator in the to-be-parsed string, e.g., `**` in the example above. The second element is an instance of the struct `OperatorPair<T>` that has a binary and a unary operator of type `Option<BinOp<T>>` and `Option<fn(T) -> T>`, respectively, as members. `BinOp` contains in addition to the operator of type `fn(T, T) -> T` a priority. Operators can be both, binary and unary such as `-` as defined in the list of default operators.

We can also extend the predefined default operators, e.g., via
```
let zero_mapper = (
    "zer0",
    OperatorPair {
        bin_op: Some(BinOp {
            op: |_: f32, _| 0.0,
            prio: 2,
        }),
        unary_op: Some(|_| 0.0),
    },
);
let extended_operators = make_default_operators::<f32>()
    .iter()
    .cloned()
    .chain(once(zero_mapper))
    .collect::<Vec<_>>();
let expr = parse::<f32>("2^2*1/({berti}) + zer0(4)", extended_operators).unwrap();
let val = eval_expr::<f32>(&expr, &[4.0]);  // 1.0
```

## Priorities and Parantheses

In Exexpress-land, unary operators always have higher priority than binary operators, e.g., 
`-2^2=4` instead of `-2^2=-4`. Moreover, we are not too strict regarding parantheses. 
For instance `"---1"` will evalute to `-1`. 
If you want to be on the safe side, we suggest using parantheses.

## Variables

To add variables we can use curly brackets as shown in the following expression.
```
let to_be_parsed = "log({x}) + 2* (-{x}^2 + sin(4*{y}))";
let expr = parse::<f32>(to_be_parsed, make_default_operators::<f32>()).unwrap();
```
To evaluate this expression we can pass a slice with the values of the variables, e.g.,
```
let val = eval_expr::<f32>(&expr, &[2.5, 3.7])  // 14.992794866624788
```
The `n`-th number in the slice corresponds to the `n`-th variable. Thereby only the first occurence of the variables is relevant. In this example, we have `x=2.5` and `y=3.7`.

