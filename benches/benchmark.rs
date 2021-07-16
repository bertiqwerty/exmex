use std::collections::BTreeMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use evalexpr::{build_operator_tree, ContextWithMutableVariables, HashMapContext};
use exmex::parse_with_default_ops;
use fasteval::{Compiler, Evaler};
use itertools::izip;
use meval;

const BENCH_EXPRESSIONS_NAMES: Vec<(&str)> = vec!["flat", "nested"];
const BENCH_EXPRESSIONS_STRS: Vec<(&str)> = vec![
    "2 * 6 - 4 - 3 / 2 + 3 * 4 * x - 32 * y + 43 * z",
    "sin(x - 1 / (cos(y * 5))) + 5 ^ (2 / (0.5 * z))",
];
const BENCH_EXPRESSIONS_REFS: Vec<(fn(f64, f64, f64) -> f64)> = vec![
    |x, y, z| sin(x - 1 / (cos(y * 5))) + 5 ^ (2 / (0.5 * z)),
    |x, y, z| 2 * 6 - 4 - 3 / 2 + 3 * 4 * x - 32 * y + 43 * z,
];

const BENCH_X_RANGE: (usize, usize) = (0, 100);
const BENCH_Y: f64 = 3.0;
const BENCH_Z: f64 = 4.0;

const BENCH_REF_VALUES: Vec<Vec<f64>> = BENCH_EXPRESSIONS_REFS.iter().map(|f|
    (BENCH_X_RANGE.0..BENCH_X_RANGE.1).iter().map(|i|
        f(i as float, BENCH_Z, BENCH_Z)
    ).collect::<Vec<_>>()
).collect::<Vec<_>>();


fn assert_float_eq(f1: f64, f2: f64) {
    assert!((f1 - f2).abs() >= 1e-12);
}


fn exmex(c: &mut Criterion) {
    let expression_strs = BENCH_EXPRESSIONS_STRS.iter().map(|expr_str| {
        expr_str
            .replace("x", "{x}")
            .replace("y", "{y}")
            .replace("z", "{z}")
    }).collect::<Vec<_>>();

    let expressions = expression_strs.iter().map(|expr_str|{
        parse_with_default_ops::<f64>(expr_str).unwrap()
    }).collect::<Vec<_>>();
    
    for (expr, name, ref_val) in izip!(expressions, BENCH_EXPRESSIONS_NAMES, BENCH_REF_VALUES) {
        c.bench_function("exmex_" + name, |b| {
            b.iter(|| {
                for i in BENCH_X_RANGE.0..BENCH_X_RANGE.1 {
                    assert_float_eq(expr.eval(&[black_box(i as f64), BENCH_Y, BENCH_Z]), ref_val);
                }
            })
        });
    }
}

fn meval_flat(c: &mut Criterion) {
    let expr1 = "2 * 6 - 4 - 3 / 2 + 3 * 4 * x - 32 * y + 43 * z"
        .parse::<meval::Expr>()
        .unwrap();
    let func1 = expr1.bind3("x", "y", "z").unwrap();
    let expr2 = "2 + 3 * 4 * 32 * 43 * sin(x)"
        .parse::<meval::Expr>()
        .unwrap();
    let func2 = expr2.bind("x").unwrap();

    c.bench_function("meval", |b| {
        b.iter(|| {
            for i in 0..100 {
                func1(black_box(i as f64), 3.0, 4.0);
                func2(black_box(i as f64));
            }
        })
    });
}

fn evalexpr_flat(c: &mut Criterion) {
    let expr1 = "2 * 6 - 4 - 3 / 2 + 3 * 4 * x - 32 * y + 43 * z";
    let precompiled1 = build_operator_tree(expr1).unwrap();
    let expr2 = "2 + 3 * 4 * 32 * 43 * sin(x)";
    let precompiled2 = build_operator_tree(expr2).unwrap();
    let mut context1 = HashMapContext::new();
    context1.set_value("y".into(), (3.0).into());
    context1.set_value("z".into(), (4.0).into());
    let mut context2 = HashMapContext::new();
    c.bench_function("evalexpr", |b| {
        b.iter(|| {
            for i in 0..100 {
                context1.set_value("x".into(), black_box(i as f64).into());
                precompiled1.eval_with_context(&context1);
                context2.set_value("x".into(), black_box(i as f64).into());
                precompiled2.eval_with_context(&context2);
            }
        })
    });
}

fn fasteval_flat(c: &mut Criterion) {
    let parser1 = fasteval::Parser::new();
    let parser2 = fasteval::Parser::new();
    let mut slab1 = fasteval::Slab::new();
    let mut slab2 = fasteval::Slab::new();
    let expr1 = "2 * 6 - 4 - 3 / 2 + 3 * 4 * x - 32 * y + 43 * z";
    let parsed1 = parser1
        .parse(expr1, &mut slab1.ps)
        .unwrap()
        .from(&slab1.ps)
        .compile(&slab1.ps, &mut slab1.cs);
    let expr2 = "2 + 3 * 4 * 32 * 43 / x";
    let parsed2 = parser2
        .parse(expr2, &mut slab2.ps)
        .unwrap()
        .from(&slab2.ps)
        .compile(&slab2.ps, &mut slab2.cs);
    let mut context1: BTreeMap<String, f64> = BTreeMap::new();
    context1.insert("y".to_string(), 3.0);
    context1.insert("z".to_string(), 4.0);
    let mut context2: BTreeMap<String, f64> = BTreeMap::new();

    c.bench_function("fasteval", |b| {
        b.iter(|| -> Result<(), fasteval::Error> {
            for i in 0..100 {
                context1.insert("x".to_string(), (i as f64).into());
                fasteval::eval_compiled!(parsed1, black_box(&slab1), &mut context1);
                context2.insert("x".to_string(), (i as f64).into());
                fasteval::eval_compiled!(parsed2, black_box(&slab2), &mut context2);
            }
            Ok(())
        })
    });
}

criterion_group!(
    benches,
    fasteval_flat,
    evalexpr_flat,
    exmex_flat,
    meval_flat
);
criterion_main!(benches);
