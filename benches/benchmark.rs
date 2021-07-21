use std::{collections::BTreeMap, iter::repeat};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use evalexpr::{ContextWithMutableVariables, HashMapContext, Node, Value, build_operator_tree};
use exmex::{parse_with_default_ops, FlatEx};
use fasteval::{Compiler, Evaler, Instruction, Slab};
use itertools::{Itertools, izip};
use meval;
use rsc::{
    computer::Computer,
    lexer::tokenize,
    parser::{parse, Expr},
};
const N: usize = 7;

const BENCH_EXPRESSIONS_NAMES: [&str; N] = [
    "xyz",
    "xx+yy+zz",
    "x^2+y^2+z^2",
    "canucompile",
    "flat",
    "flatsin",
    "nested",
];
const BENCH_EXPRESSIONS_STRS: [&str; N] = [
    "x*y*z",
    "x*x+y*y+z*z",
    "x^2+y^2+z^2",
    "x+2 * (6 - 4) - 3 / 2.5 + y + 3.141 * 0.4 * (2 - 32 * (7 + 43 * (1+5))) * 0.1 + x*y*z",
    "2 * 6 - 4 - 3 / 2.5 + 3.141 * 0.4 * x - 32 * y + 43 * z",
    "2 * 6 - 4 - 3 / sin(2.5) + 3.141 * 0.4 * sin(x) - 32 * y + 43 * z",
    "x*0.02*(3*(2*(sin(x - 1 / (sin(y * 5)) + (5.0 - 1/z)))))",
];
const BENCH_EXPRESSIONS_REFS: [fn(f64, f64, f64) -> f64; N] = [
    |x, y, z| x * y * z,
    |x, y, z| x * x + y * y + z * z,
    |x, y, z| x.powi(2) + y.powi(2) + z.powi(2),
    |x, y, z| {
        x + 2.0 * (6.0 - 4.0) - 3.0 / 2.5
            + y
            + 3.141 * 0.4 * (2.0 - 32.0 * (7.0 + 43.0 * (1.0 + 5.0))) * 0.1
            + x * y * z
    },
    |x, y, z| 6.8 + 1.2564 * x - 32.0 * y + 43.0 * z,
    |x, y, z| 8.0 - 5.01276463667604 + 1.2564 * x.sin() - 32.0 * y + 43.0 * z,
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

fn run_benchmark_parse<T, F: Fn(&[&str]) -> Vec<T>>(func: F, parse_name: &str, c: &mut Criterion) {
    c.bench_function(format!("{}", parse_name).as_str(), |b| {
        b.iter(|| {
            func(black_box(&BENCH_EXPRESSIONS_STRS));
        })
    });
}

fn exmex_parse(strings: &[&str]) -> Vec<FlatEx<f64>> {
    strings
        .iter()
        .map(|expr_str| {
            parse_with_default_ops::<f64>(expr_str).unwrap()
        })
        .collect::<Vec<_>>()
}

fn exmex_bench_parse(c: &mut Criterion) {
    run_benchmark_parse(exmex_parse, "exmex_parse", c);
}

fn exmex_bench_eval(c: &mut Criterion) {
    let parsed_exprs = exmex_parse(&BENCH_EXPRESSIONS_STRS);
    let funcs = parsed_exprs
        .iter()
        .map(|expr| move |x: f64| expr.eval(&[x, BENCH_Y, BENCH_Z]).unwrap())
        .collect::<Vec<_>>();
    run_benchmark(funcs, "exmex", c);
}

fn meval_parse(strings: &[&str]) -> Vec<impl Fn(f64, f64, f64) -> f64> {
    strings
        .iter()
        .map(|expr_str| {
            let expr = expr_str.parse::<meval::Expr>().unwrap();
            expr.bind3("x", "y", "z").unwrap()
        })
        .collect::<Vec<_>>()
}

fn meval_bench_parse(c: &mut Criterion) {
    run_benchmark_parse(meval_parse, "meval_parse", c);
}

fn meval_bench_eval(c: &mut Criterion) {
    let parsed_exprs = meval_parse(&BENCH_EXPRESSIONS_STRS);
    let funcs = parsed_exprs
        .iter()
        .map(|expr| move |x: f64| expr(x, BENCH_Y, BENCH_Z))
        .collect::<Vec<_>>();
    run_benchmark(funcs, "meval", c);
}

fn evalexpr_parse(strings: &[&str]) -> Vec<(Node, HashMapContext)> {
    let parsed_exprs = strings
        .iter()
        .map(|expr_str| build_operator_tree(expr_str.replace("sin", "math::sin").as_str()).unwrap());
    let contexts = repeat(HashMapContext::new()).take(N);
    izip!(parsed_exprs, contexts).collect_vec()
}

fn evalexpr_bench_parse(c: &mut Criterion) {
    run_benchmark_parse(evalexpr_parse, "evalexpr_parse", c);
}

fn evalexpr_bench_eval(c: &mut Criterion) {
    let mut parsed_exprs = evalexpr_parse(&BENCH_EXPRESSIONS_STRS);
    let funcs = parsed_exprs.iter_mut()
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

fn fasteval_parse(strings: &[&str]) -> Vec<((Instruction, Slab), BTreeMap<String, f64>)> {
    let parsed_exprs = strings
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
        });
    let contexts = repeat(BTreeMap::<String, f64>::new())
        .take(N);
    izip!(parsed_exprs, contexts).collect::<Vec<_>>()
}

fn fasteval_bench_parse(c: &mut Criterion) {
    run_benchmark_parse(fasteval_parse, "fasteval_parse", c);
}
fn fasteval_bench_eval(c: &mut Criterion) {
    let mut parsed_exprs = fasteval_parse(&BENCH_EXPRESSIONS_STRS);
    let funcs = parsed_exprs.iter_mut()
        .map(|tuple_of_tuples| {
            let context = &mut tuple_of_tuples.1;
            let (instr, slab) = &tuple_of_tuples.0;
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

fn rsc_parse<'a>(strings: &[&str]) -> Vec<(Expr::<f64>, Computer<'a, f64>)> {
    let parsed_exprs = strings
        .iter()
        .map(|expr_str| {
            let tokens = tokenize(expr_str, true).unwrap();
            parse(&tokens).unwrap()
        });
    let computers = repeat(Computer::<f64>::default())
        .take(N);
    izip!(parsed_exprs, computers).collect_vec()
}
fn rsc_bench_parse(c: &mut Criterion) {
    run_benchmark_parse(rsc_parse, "rsc_parse", c);
}

fn rsc_bench_eval(c: &mut Criterion) {
    let mut parsed_exprs = rsc_parse(&BENCH_EXPRESSIONS_STRS);
    let funcs = parsed_exprs.iter_mut()
        .map(|(ast, comp)| {
            move |x: f64| {
                let mut ast = ast.clone();
                ast.replace(&Expr::Identifier("x".to_owned()), &Expr::Constant(x), false);
                ast.replace(
                    &Expr::Identifier("y".to_owned()),
                    &Expr::Constant(BENCH_Y),
                    false,
                );
                ast.replace(
                    &Expr::Identifier("z".to_owned()),
                    &Expr::Constant(BENCH_Z),
                    false,
                );
                comp.compute(&ast).unwrap()
            }
        })
        .collect::<Vec<_>>();
    run_benchmark(funcs, "rsc", c);
}
criterion_group!(
    benches,
    fasteval_bench_eval,
    exmex_bench_eval,
    meval_bench_eval,
    evalexpr_bench_eval,
    rsc_bench_eval,
    fasteval_bench_parse,
    exmex_bench_parse,
    meval_bench_parse,
    rsc_bench_parse,
    evalexpr_bench_parse,
);
criterion_main!(benches);
