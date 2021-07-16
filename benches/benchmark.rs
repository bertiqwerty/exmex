use std::{collections::BTreeMap, iter::repeat};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use evalexpr::{build_operator_tree, ContextWithMutableVariables, HashMapContext, Value};
use exmex::parse_with_default_ops;
use fasteval::{Compiler, Evaler};
use itertools::izip;
use meval;

const N: usize = 2;

const BENCH_EXPRESSIONS_NAMES: [&str; N] = ["flat", "nested"];
const BENCH_EXPRESSIONS_STRS: [&str; N] = [
    "2 * 6 - 4 - 3 / 2.5 + 3.141 * 0.4 * x - 32 * y + 43 * z",
    "x*0.02*(3*(2*(sin(x - 1 / (sin(y * 5)) + (5.0 - 1/z)))))",
];
const BENCH_EXPRESSIONS_REFS: [fn(f64, f64, f64) -> f64; N] = [
    |x, y, z| 2.0 * 6.0 - 4.0 - 3.0 / 2.5 + 3.141 * 0.4 * x - 32.0 * y + 43.0 * z,
    |x, y, z| x * 0.02 * (3.0 * (2.0 * (x - 1.0 / (y * 5.0).sin() + (5.0 - 1.0 / z)).sin())),
];
const BENCH_X_RANGE: (usize, usize) = (0, 1000);
const BENCH_Y: f64 = 3.0;
const BENCH_Z: f64 = 4.0;

fn bench_ref_values() -> Vec<Vec<f64>> {
    BENCH_EXPRESSIONS_REFS
        .iter()
        .map(|f| {
            (BENCH_X_RANGE.0..BENCH_X_RANGE.1)
                .map(|i| f(i as f64, BENCH_Y, BENCH_Z))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

fn assert_float_eq(f1: f64, f2: f64) {
    assert!((f1 - f2).abs() <= 1e-12);
}

fn run_benchmark<F: FnMut(f64) -> f64>(funcs: Vec<F>, eval_name: &str, c: &mut Criterion) {
    for (mut func, exp_name, ref_vals) in izip!(
        funcs,
        BENCH_EXPRESSIONS_NAMES.iter(),
        bench_ref_values().iter()
    ) {
        c.bench_function(format!("{}_{}", eval_name, exp_name).as_str(), |b| {
            b.iter(|| {
                for (i, ref_val) in izip!(BENCH_X_RANGE.0..BENCH_X_RANGE.1, ref_vals) {
                    assert_float_eq(func(black_box(i as f64)), *ref_val);
                }
            })
        });
    }
}

fn exmex(c: &mut Criterion) {
    let expression_strs = BENCH_EXPRESSIONS_STRS
        .iter()
        .map(|expr_str| {
            expr_str
                .replace("x", "{x}")
                .replace("y", "{y}")
                .replace("z", "{z}")
        })
        .collect::<Vec<_>>();
    let parsed_exprs = expression_strs
        .iter()
        .map(|expr_str| parse_with_default_ops::<f64>(expr_str).unwrap())
        .collect::<Vec<_>>();
    let funcs = parsed_exprs
        .iter()
        .map(|expr| move |x: f64| expr.eval(&[x, BENCH_Y, BENCH_Z]))
        .collect::<Vec<_>>();
    run_benchmark(funcs, "exmex", c);
}

fn bench_meval(c: &mut Criterion) {
    let parsed_exprs = BENCH_EXPRESSIONS_STRS
        .iter()
        .map(|expr_str| {
            let expr = expr_str.parse::<meval::Expr>().unwrap();
            expr.bind3("x", "y", "z").unwrap()
        })
        .collect::<Vec<_>>();
    let funcs = parsed_exprs
        .iter()
        .map(|expr| move |x: f64| expr(x, BENCH_Y, BENCH_Z))
        .collect::<Vec<_>>();
    run_benchmark(funcs, "meval", c);
}

fn evalexpr(c: &mut Criterion) {
    let parsed_exprs = BENCH_EXPRESSIONS_STRS
        .iter()
        .map(|expr_str| build_operator_tree(expr_str.replace("sin", "math::sin").as_str()).unwrap())
        .collect::<Vec<_>>();
    let mut contexts = repeat(HashMapContext::new()).take(N).collect::<Vec<_>>();
    let funcs = izip!(parsed_exprs
        .iter(), contexts.iter_mut())
        .map(|(expr, context)| {
            move |x: f64| {
                context.set_value("x".into(), x.into()).unwrap();
                context.set_value("y".into(), BENCH_Y.into()).unwrap();
                context.set_value("z".into(), BENCH_Z.into()).unwrap();
                match expr.eval_with_context(context).unwrap() {
                    Value::Float(val) => val,
                    _ => panic!("What?"),
                }
            }
        })
        .collect::<Vec<_>>();
    run_benchmark(funcs, "evalexpr", c);
}

fn fasteval(c: &mut Criterion) {
    let parsed_exprs = BENCH_EXPRESSIONS_STRS
        .iter()
        .map(|expr_str| {
            let parser = fasteval::Parser::new();
            let mut slab = fasteval::Slab::new();
            (
                parser
                    .parse(expr_str, &mut slab.ps)
                    .unwrap()
                    .from(&slab.ps)
                    .compile(&slab.ps, &mut slab.cs),
                slab,
            )
        })
        .collect::<Vec<_>>();
    let mut contexts = repeat(BTreeMap::<String, f64>::new())
        .take(N)
        .collect::<Vec<_>>();
    let funcs = izip!(parsed_exprs.iter(), contexts.iter_mut())
        .map(|tuple_of_tuples| {
            let context = tuple_of_tuples.1;
            let (instr, slab) = tuple_of_tuples.0;
            move |x: f64| {
                context.insert("x".to_string(), x.into());
                context.insert("y".to_string(), BENCH_Y.into());
                context.insert("z".to_string(), BENCH_Z.into());
                || -> Result<f64, fasteval::Error> {
                    Ok(fasteval::eval_compiled_ref!(
                        instr,
                        black_box(slab),
                        context
                    ))
                }()
                .unwrap()
            }
        })
        .collect::<Vec<_>>();
    run_benchmark(funcs, "fasteval", c);
}

criterion_group!(benches, fasteval, evalexpr, exmex, bench_meval);
criterion_main!(benches);
