#[cfg(feature = "partial")]
use exmex::{
    ops_factory, parse, BinOp, Calculate, DiffDataType, Differentiate, ExError, ExResult, Express,
    FlatEx, MakeOperators, MatchLiteral, MissingOpMode, NeutralElts, Operator,
};
#[cfg(feature = "partial")]
mod utils;
#[cfg(feature = "partial")]
#[cfg(feature = "value")]
use exmex::{FlatExVal, Val};
#[cfg(feature = "partial")]
use rand::{thread_rng, Rng};
#[cfg(feature = "partial")]
use smallvec::{smallvec, SmallVec};
#[cfg(feature = "partial")]
use std::{fmt::Debug, fmt::Display, ops::Index, ops::Range, str::FromStr};
#[cfg(feature = "partial")]
#[test]
fn test_readme_partial() -> ExResult<()> {
    let expr = parse::<f64>("y*x^2")?;

    // d_x
    let dexpr_dx = expr.clone().partial(0)?;
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
    let dddexpr_dxyx_iter = expr.partial_iter([0, 1, 0].iter().copied())?;
    assert_eq!(format!("{}", dddexpr_dxyx_iter), "2.0");
    let result = dddexpr_dxyx_iter.eval(&[f64::MAX, f64::MAX])?;
    assert!((result - 2.0).abs() < 1e-12);

    Ok(())
}
#[cfg(feature = "partial")]
#[cfg(test)]
use exmex::DeepEx;
#[cfg(feature = "partial")]
#[test]
fn test_partial() -> ExResult<()> {
    fn test_expr<'a, F, E: Differentiate<'a, F> + Clone>(
        flatex: &E,
        var_idx: usize,
        n_vars: usize,
        random_range: Range<f64>,
        reference: impl Fn(F) -> F,
        to_float: fn(F) -> f64,
        to_f: fn(f64) -> F,
    ) -> ExResult<()>
    where
        F: DiffDataType,
        <F as FromStr>::Err: Debug,
    {
        let mut rng = rand::thread_rng();
        assert!(flatex.clone().partial(flatex.var_names().len()).is_err());

        // test flatex
        let deri = flatex.clone().partial(var_idx)?;
        println!("flatex {}", flatex);
        println!("partial {}", deri);
        for _ in 0..3 {
            let vut = to_f(rng.gen_range(random_range.clone()));
            let mut vars: SmallVec<[F; 10]> = smallvec![to_f(0.0); n_vars];
            vars[var_idx] = vut.clone();
            println!("value under test {}.", to_float(vut.clone()));
            utils::assert_float_eq_f64(
                to_float(deri.eval(&vars).unwrap()),
                to_float(reference(vut)),
            );
        }
        Ok(())
    }
    fn test(
        sut: &str,
        var_idx: usize,
        n_vars: usize,
        random_range: Range<f64>,
        reference: fn(f64) -> f64,
    ) -> ExResult<()> {
        println!("testing {}...", sut);
        let flatex = FlatEx::<f64>::parse(sut)?;
        test_expr(
            &flatex,
            var_idx,
            n_vars,
            random_range.clone(),
            reference,
            |x| x,
            |x| x,
        )?;
        let deepex = DeepEx::<f64>::parse(sut)?;
        test_expr(
            &deepex,
            var_idx,
            n_vars,
            random_range.clone(),
            reference,
            |x| x,
            |x| x,
        )?;
        #[cfg(feature = "value")]
        {
            let flatex = FlatExVal::<i32, f64>::parse(sut).unwrap();
            test_expr(
                &flatex,
                var_idx,
                n_vars,
                random_range,
                |x| Val::Float(reference(x.to_float().unwrap())),
                |x| x.to_float().unwrap(),
                |x| Val::Float(x),
            )?;
        }
        Ok(())
    }

    let sut = "+x";
    let var_idx = 0;
    let n_vars = 1;
    let reference = |_: f64| 1.0;
    test(sut, var_idx, n_vars, -10000.0..10000.0, reference)?;

    let sut = "++x";
    let var_idx = 0;
    let n_vars = 1;
    let reference = |_: f64| 1.0;
    test(sut, var_idx, n_vars, -10000.0..10000.0, reference)?;

    let sut = "+-+x";
    let var_idx = 0;
    let n_vars = 1;
    let reference = |_: f64| -1.0;
    test(sut, var_idx, n_vars, -10000.0..10000.0, reference)?;

    let sut = "-x";
    let var_idx = 0;
    let n_vars = 1;
    let reference = |_: f64| -1.0;
    test(sut, var_idx, n_vars, -10000.0..10000.0, reference)?;

    let sut = "--x";
    let var_idx = 0;
    let n_vars = 1;
    let reference = |_: f64| 1.0;
    test(sut, var_idx, n_vars, -10000.0..10000.0, reference)?;

    let sut = "sin(sin(x))";
    let var_idx = 0;
    let n_vars = 1;
    let reference = |x: f64| x.sin().cos() * x.cos();
    test(sut, var_idx, n_vars, -10000.0..10000.0, reference)?;

    let sut = "{x}+sin(2.0*{y})";
    let var_idx = 1;
    let n_vars = 2;
    let reference = |y: f64| 2.0 * (2.0 * y).cos();
    test(sut, var_idx, n_vars, -10000.0..10000.0, reference)?;

    let sut = "sin(x)-cos(x)+a";
    let var_idx = 1;
    let n_vars = 2;
    let reference = |x: f64| x.cos() + x.sin();
    test(sut, var_idx, n_vars, -10000.0..10000.0, reference)?;
    let flatex_1 = FlatEx::<f64>::parse(sut)?;
    let deri = flatex_1.partial(var_idx)?;
    let reference = |x: f64| -x.sin() + x.cos();
    test_expr(
        &deri,
        var_idx,
        n_vars,
        -10000.0..10000.0,
        reference,
        |x| x,
        |x| x,
    )?;
    let deri = deri.partial(var_idx)?;
    let reference = |x: f64| -x.cos() - x.sin();
    test_expr(
        &deri,
        var_idx,
        n_vars,
        -10000.0..10000.0,
        reference,
        |x| x,
        |x| x,
    )?;
    let deri = deri.partial(var_idx)?;
    let reference = |x: f64| x.sin() - x.cos();
    test_expr(
        &deri,
        var_idx,
        n_vars,
        -10000.0..10000.0,
        reference,
        |x| x,
        |x| x,
    )?;

    let sut = "sin(x)-cos(x)+tan(x)+a";
    let var_idx = 1;
    let n_vars = 2;
    let reference = |x: f64| x.cos() + x.sin() + 1.0 / (x.cos().powf(2.0));
    test(sut, var_idx, n_vars, -10000.0..10000.0, reference)?;

    let sut = "ln(v)*exp(v)+cos(x)+tan(x)+a";
    let var_idx = 1;
    let n_vars = 3;
    let reference = |x: f64| 1.0 / x * x.exp() + x.ln() * x.exp();
    test(sut, var_idx, n_vars, 0.1..20.0, reference)?;

    let sut = "a+z+sinh(v)/cosh(v)+b+tanh({v})";
    let var_idx = 2;
    let n_vars = 4;
    let reference = |x: f64| {
        (x.cosh() * x.cosh() - x.sinh() * x.sinh()) / x.cosh().powf(2.0)
            + 1.0 / (x.cosh().powf(2.0))
    };
    test(sut, var_idx, n_vars, -100.0..100.0, reference)?;

    let sut = "w+z+acos(v)+asin(v)+b+atan({v})";
    let var_idx = 1;
    let n_vars = 4;
    let reference = |x: f64| {
        1.0 / (1.0 - x.powf(2.0)).sqrt() - 1.0 / (1.0 - x.powf(2.0)).sqrt()
            + 1.0 / (1.0 + x.powf(2.0))
    };
    test(sut, var_idx, n_vars, -1.0..1.0, reference)?;

    let sut = "sqrt(var)*var^1.57";
    let var_idx = 0;
    let n_vars = 1;
    let reference = |x: f64| 1.0 / (2.0 * x.sqrt()) * x.powf(1.57) + x.sqrt() * 1.57 * x.powf(0.57);
    test(sut, var_idx, n_vars, 0.0..100.0, reference)?;
    Ok(())
}

#[cfg(feature = "partial")]
#[test]
fn test_partial_finite() -> ExResult<()> {
    fn test_(sut: &str, range: Range<f64>, skip_subs: bool) -> ExResult<()> {
        fn inner_test<'a, E: Differentiate<'a, f64> + Clone>(
            flatex: &E,
            range: Range<f64>,
        ) -> ExResult<()> {
            let n_vars = flatex.var_names().len();
            let step = 1e-5;
            let mut rng = thread_rng();

            let x0s: Vec<f64> = (0..n_vars).map(|_| rng.gen_range(range.clone())).collect();
            for var_idx in 0..flatex.var_names().len() {
                let x1s: Vec<f64> = x0s
                    .iter()
                    .enumerate()
                    .map(|(i, x0)| if i == var_idx { x0 + step } else { *x0 })
                    .collect();

                let f0 = flatex.eval(&x0s)?;
                let f1 = flatex.eval(&x1s)?;
                let finite_diff = (f1 - f0) / step;
                let deri = flatex.clone().partial(var_idx)?;
                let deri_val = deri.eval(&x0s)?;
                utils::assert_float_eq::<f64>(
                    deri_val,
                    finite_diff,
                    1e-5,
                    1e-3,
                    format!("finite diff error at {x0s:?} for {flatex}").as_str(),
                );
            }
            Ok(())
        }
        let flatex = exmex::parse::<f64>(sut)?;
        inner_test(&flatex, range.clone())?;
        let deepex = exmex::DeepEx::<f64>::parse(sut)?;
        inner_test(&deepex, range.clone())?;
        if !skip_subs {
            let vn0 = deepex.var_names()[0].clone();
            let mut sub = |vn: &str| {
                if vn0 == vn {
                    Some(DeepEx::<f64>::parse("x* 0.1 +0.3 * y+z* 0.1").unwrap())
                } else {
                    None
                }
            };
            let deepex = deepex.subs(&mut sub)?;
            inner_test(&deepex, range.clone())?;
            let flatex = FlatEx::from_deepex(deepex)?;
            inner_test(&flatex, range)?;
        }
        Ok(())
    }
    fn test(sut: &str, range: Range<f64>) -> ExResult<()> {
        test_(sut, range, false)
    }
    fn test_skip_subs(sut: &str, range: Range<f64>) -> ExResult<()> {
        test_(sut, range, true)
    }
    test("sqrt(x)", 0.0..10000.0)?;
    test("asin(x)", -0.9..0.9)?;
    test("acos(x)", -0.9..0.9)?;
    test("atan(x)", -0.9..0.9)?;
    test("-y*(x*(-(1-y))) + 1.7", 2.0..10.0)?;
    test("1/x", -10.0..10.0)?;
    test("x^x", 0.01..2.0)?;
    test("x^y", 4.036286084344371..4.036286084344372)?;
    test("z+sin(x)+cos(y)", -1.0..1.0)?;
    test("sin(cos(sin(z)))", -10.0..10.0)?;
    test("sin(x+z)", -10.0..10.0)?;
    test("sin(x-z)", -10.0..10.0)?;
    test("y-sin(x-z)", -10.0..10.0)?;
    test("(sin(x)^2)/x/4", -10.0..10.0)?;
    test("sin(y+x)/((x*2)/y)*(2*x)", -1.0..1.0)?;
    test("z*sin(x)+cos(y)^(1 + x^2)/(sin(z))", 0.1..1.0)?;
    test("ln(x^2)", 0.1..10.0)?;
    test("log2(x^2)", 0.1..10.0)?;
    test("log10(x^2)", 0.1..10.0)?;
    test("tan(x)", -1.0..1.0)?;
    test("tan(exp(x))", -1000.0..0.0)?;
    test("exp(y-x)", -1.0..1.0)?;
    test("sqrt(exp(y-x))", -100.0..0.0)?;
    test("sin(sin(x+z))", -10.0..10.0)?;
    test("asin(sqrt(x+y))", 0.0..0.5)?;
    println!("atanh");
    test("atanh(x)+atanh(y)", -0.9..0.9)?;
    println!("asinh");
    test("asinh(x)+asinh(y)", -0.9..1.5)?;
    println!("acosh");
    test_skip_subs("acosh(x)+acosh(y)", 1.1..1.5)?;
    Ok(())
}

#[cfg(feature = "partial")]
#[test]
fn test_partial_iter() -> ExResult<()> {
    let sut = "a^2+b^2+c^2+x^2+y^2+z^2";
    let expr = exmex::parse::<f64>(sut)?;
    let deri = expr.partial_iter([0, 1, 2, 3, 4, 5].iter().copied())?;
    utils::assert_float_eq::<f64>(
        0.0,
        deri.eval(&[7.0, 7.0, 7.0, 7.0, 7.0, 7.0])?,
        1e-12,
        1e-12,
        sut,
    );

    fn test3(sut: &str) -> ExResult<()> {
        let expr = exmex::parse::<f64>(sut)?;
        let deri = expr.clone().partial_iter([0, 1, 2].iter().copied())?;
        let mut deri_seq = expr;
        for i in 0..3 {
            deri_seq = deri_seq.partial(i)?;
        }
        let vals = [7.3, 4.2, 423.9];
        utils::assert_float_eq_f64(deri_seq.eval(&vals)?, deri.eval(&vals)?);
        Ok(())
    }

    test3("a^2*b^2*c^2")?;
    test3("a^2+b^2*c^2")?;
    test3("a^2-cos(sin(b^2))*c^3")?;
    test3("a^2*b^2/sin(c^2)")?;
    Ok(())
}

#[cfg(feature = "partial")]
#[test]
fn test_log() -> ExResult<()> {
    let test_vals = [0.001, 5.0, 10.0, 1000.0, 12341.2345];
    let deri_ln = exmex::parse::<f64>("ln(x)")?.partial(0)?;
    let deri_log = exmex::parse::<f64>("log(x)")?.partial(0)?;

    let expr = exmex::parse::<f64>("log10(x)")?;
    let deri = expr.partial(0)?;
    for v in test_vals {
        utils::assert_float_eq_f64(deri_ln.eval(&[v])? * 1.0 / 10.0f64.ln(), deri.eval(&[v])?);
        utils::assert_float_eq_f64(deri_log.eval(&[v])? * 1.0 / 10.0f64.ln(), deri.eval(&[v])?);
    }

    let expr = exmex::parse::<f64>("log2(x)")?;
    let deri = expr.partial(0)?;
    for v in test_vals {
        utils::assert_float_eq_f64(deri_ln.eval(&[v])? * 1.0 / 2.0f64.ln(), deri.eval(&[v])?);
    }
    Ok(())
}

#[cfg(feature = "partial")]
#[test]
fn test_operatorsubset() {
    use exmex::{FloatOpsFactory, Operator};

    #[derive(Debug, Clone)]
    struct SubsetFloatOpsFactory;
    impl MakeOperators<f32> for SubsetFloatOpsFactory {
        fn make<'a>() -> Vec<Operator<'a, f32>> {
            let ops = FloatOpsFactory::<f32>::make();
            ops.into_iter()
                .filter(|o| {
                    let r = o.repr();
                    r == "+" || r == "*" || r == "/" || r == "-" || r == "^" || r == "sin"
                })
                .collect::<Vec<_>>()
        }
    }
    let flatex = FlatEx::<f32, SubsetFloatOpsFactory>::parse("sin(x)").unwrap();
    let cosx = flatex.partial(0);
    assert!(cosx.is_err());
    assert!(format!("{cosx:?}").contains("cos"));
    let flatex = FlatEx::<f32, SubsetFloatOpsFactory>::parse("1/x").unwrap();
    println!("{}", flatex.clone().partial(0).unwrap());
    let dflatex = flatex.partial(0).unwrap();
    assert_eq!("-1.0/({x}*{x})", format!("{dflatex}"));
}

#[cfg(feature = "partial")]
#[test]
fn test_custom_data() {
    #[derive(Clone, Default, PartialEq)]
    struct Arr {
        data: [f64; 2],
    }
    impl Arr {
        fn new(data: [f64; 2]) -> Self {
            Arr { data }
        }
    }
    impl Index<usize> for Arr {
        type Output = f64;
        fn index(&self, index: usize) -> &Self::Output {
            &self.data[index]
        }
    }
    ops_factory!(
        ArrOpsFactory,
        Arr,
        Operator::make_unary("set0", |_| Arr::new([0.0, 0.0])),
        Operator::make_bin(
            ">>",
            BinOp {
                apply: |a, b| Arr::new([
                    if a[0] > b[0] { 1.0 } else { 0.0 },
                    if a[1] > b[1] { 1.0 } else { 0.0 }
                ]),
                prio: 0,
                is_commutative: false
            }
        ),
        Operator::make_bin(
            "+",
            BinOp {
                apply: |a, b| Arr::new([a[0] + b[0], a[1] + b[1]]),
                prio: 1,
                is_commutative: true
            }
        ),
        Operator::make_bin(
            "-",
            BinOp {
                apply: |a, b| Arr::new([a[0] - b[0], a[1] - b[1]]),
                prio: 2,
                is_commutative: false
            }
        ),
        Operator::make_bin(
            "*",
            BinOp {
                apply: |a, b| Arr::new([a[0] * b[0], a[1] * b[1]]),
                prio: 3,
                is_commutative: true
            }
        ),
        Operator::make_bin(
            "/",
            BinOp {
                apply: |a, b| Arr::new([a[0] / b[0], a[1] / b[1]]),
                prio: 4,
                is_commutative: false
            }
        )
    );

    impl Debug for Arr {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(format!("[{}, {}]", self[0], self[1]).as_str())
        }
    }
    impl Display for Arr {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            <Self as Debug>::fmt(&self, f)
        }
    }

    impl FromStr for Arr {
        type Err = ExError;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            let s = s.trim();
            let mut it = s[1..s.len() - 1].split(',');
            let a = it.next().unwrap().parse::<f64>().unwrap();
            let b = it.next().unwrap().parse::<f64>().unwrap();
            Ok(Self::new([a, b]))
        }
    }

    impl From<f32> for Arr {
        fn from(value: f32) -> Self {
            Self::new([value as f64, value as f64])
        }
    }

    impl NeutralElts for Arr {
        fn zero() -> Self {
            Self::new([0.0, 0.0])
        }
        fn one() -> Self {
            Self::new([1.0, 1.0])
        }
    }

    #[derive(Debug, Clone)]
    struct ArrMatcher;
    impl MatchLiteral for ArrMatcher {
        fn is_literal(text: &str) -> Option<&str> {
            let text = text.trim();
            if text.starts_with('[') && text.contains(',') {
                let end = text
                    .chars()
                    .enumerate()
                    .find(|(_, c)| c == &']')
                    .map(|(i, _)| i);
                if let Some(end) = end {
                    Some(&text[..end + 1])
                } else {
                    None
                }
            } else {
                None
            }
        }
    }

    let expr = FlatEx::<Arr, ArrOpsFactory, ArrMatcher>::parse("a+b*c+d*d").unwrap();
    let deri = expr.clone().partial(0).unwrap();
    assert_eq!(deri.unparse(), "[1, 1]");
    let deri = expr.clone().partial(1).unwrap();
    assert_eq!(deri.unparse(), "{c}");
    let deri = expr.clone().partial(2).unwrap();
    assert_eq!(deri.unparse(), "{b}");
    let deri = expr.clone().partial(3).unwrap();
    assert_eq!(deri.unparse(), "{d}+{d}");
    let expr = FlatEx::<Arr, ArrOpsFactory, ArrMatcher>::parse("a+b*c >> d*d").unwrap();
    let deri = expr
        .clone()
        .partial_relaxed(1, MissingOpMode::PerOperand)
        .unwrap();
    assert_eq!(deri.unparse(), "{c}>>[0, 0]");
    let deri = expr
        .clone()
        .partial_relaxed(1, MissingOpMode::None)
        .unwrap();
    assert_eq!(deri.unparse(), "({a}+({b}*{c}))>>({d}*{d})");
    let deri = expr.clone().partial_relaxed(1, MissingOpMode::Error);
    assert!(deri.is_err());
    let deri = expr.clone().partial(1);
    assert!(deri.is_err());
    FlatEx::<Arr, ArrOpsFactory, ArrMatcher>::parse("[1,1] + set0(a)").unwrap();
}

#[test]
fn test_minmax() {
    // currently partial does not support min and max
    let expr = exmex::parse::<f64>("min(x, y)").unwrap();
    assert!(expr.partial(0).is_err());
    let expr = exmex::parse::<f64>("max(x, y)").unwrap();
    assert!(expr.partial(1).is_err());
}
