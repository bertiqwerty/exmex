mod utils;
use std::ops::{BitAnd, BitOr};
use std::str::FromStr;
use std::{iter::once, ops::Range};

use rand::Rng;
use smallvec::{smallvec, SmallVec};

use exmex::{
    eval_str, parse, ExResult, OwnedFlatEx, {BinOp, FloatOpsFactory, MakeOperators, Operator},
};
use exmex::{ops_factory, prelude::*, ExError};

use crate::utils::assert_float_eq_f64;

#[test]
fn test_readme() {
    fn readme_partial() -> ExResult<()> {
        let expr = parse::<f64>("y*x^2")?;

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

        Ok(())
    }
    fn readme() -> ExResult<()> {
        let result = eval_str::<f64>("sin(73)")?;
        assert!((result - 73f64.sin()).abs() < 1e-12);
        let expr = parse::<f64>("2*β^3-4/τ")?;
        let result = expr.eval(&[5.3, 0.5])?;
        assert!((result - 289.75399999999996).abs() < 1e-12);
        Ok(())
    }
    fn readme_int() -> ExResult<()> {
        ops_factory!(
            BitwiseOpsFactory,
            u32,
            Operator::make_bin(
                "|",
                BinOp {
                    apply: |a, b| a | b,
                    prio: 0,
                    is_commutative: true
                }
            ),
            Operator::make_unary("!", |a| !a)
        );
        let expr = FlatEx::<_, BitwiseOpsFactory>::from_str("!(a|b)")?;
        let result = expr.eval(&[0, 1])?;
        assert_eq!(result, u32::MAX - 1);
        Ok(())
    }
    assert!(!readme_partial().is_err());
    assert!(!readme().is_err());
    assert!(!readme_int().is_err());
}
#[test]
fn test_variables_curly_space_names() -> ExResult<()>{
    let sut = "{x } + { y }";
    let expr = FlatEx::<f64>::from_str(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[1.0, 1.0])?, 2.0);
    assert_eq!(expr.unparse()?, sut);
    let sut = "2*(4*{ xasd sa } + { y z}^2)";
    let expr = FlatEx::<f64>::from_str(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[2.0, 3.0])?, 34.0);
    assert_eq!(expr.unparse()?, sut);
    Ok(())
}
#[test]
fn test_variables_curly() {
    let sut = "5*{x} +  4*log2(log(1.5+{gamma}))*({x}*-(tan(cos(sin(652.2-{gamma}))))) + 3*{x}";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[1.2, 1.0]).unwrap(), 8.040556934857268);

    let sut = "sin({myvwmlf4i58eo;w/-sin(a)r_25})";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[std::f64::consts::FRAC_PI_2]).unwrap(), 1.0);

    let sut = "((sin({myvar_25})))";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[std::f64::consts::FRAC_PI_2]).unwrap(), 1.0);
}

#[test]
fn test_variables_non_ascii() {
    let sut = "5*ς";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[1.2]).unwrap(), 6.0);

    let sut = "5*{χ} +  4*log2(log(1.5+γ))*({χ}*-(tan(cos(sin(652.2-{γ}))))) + 3*{χ}";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    println!("{}", expr);
    utils::assert_float_eq_f64(expr.eval(&[1.2, 1.0]).unwrap(), 8.040556934857268);

    let sut = "sin({myvwmlf4i😎8eo;w/-sin(a)r_25})";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[std::f64::consts::FRAC_PI_2]).unwrap(), 1.0);

    let sut = "((sin({myvar_25✔})))";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[std::f64::consts::FRAC_PI_2]).unwrap(), 1.0);

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Thumbs {
        val: bool,
    }
    impl BitOr for Thumbs {
        type Output = Self;
        fn bitor(self, rhs: Self) -> Self::Output {
            Self {
                val: self.val || rhs.val,
            }
        }
    }
    impl BitAnd for Thumbs {
        type Output = Self;
        fn bitand(self, rhs: Self) -> Self::Output {
            Self {
                val: self.val && rhs.val,
            }
        }
    }
    impl FromStr for Thumbs {
        type Err = ExError;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            if s == "👍" {
                Ok(Self { val: true })
            } else if s == "👎" {
                Ok(Self { val: false })
            } else {
                Err(Self::Err {
                    msg: format!("cannot parse {} to `Thumbs`", s),
                })
            }
        }
    }
    ops_factory!(
        UnicodeOpsFactory,
        Thumbs,
        Operator::make_bin(
            "ορ",
            BinOp {
                apply: |a, b| a | b,
                prio: 0,
                is_commutative: true,
            }
        ),
        Operator::make_bin(
            "ανδ",
            BinOp {
                apply: |a, b| a & b,
                prio: 0,
                is_commutative: true,
            }
        ),
        Operator::make_constant("γ", Thumbs { val: false })
    );

    let literal_pattern = "^(👍|👎)";

    let sut = "γ ορ 👍ορ👎";
    let expr = FlatEx::<_, UnicodeOpsFactory>::from_pattern(sut, literal_pattern).unwrap();
    assert_eq!(expr.eval(&[]).unwrap(), Thumbs { val: true });

    let sut = "(👍 ανδ👎)ορ 👍";
    let expr = FlatEx::<_, UnicodeOpsFactory>::from_pattern(sut, literal_pattern).unwrap();
    assert_eq!(expr.eval(&[]).unwrap(), Thumbs { val: true });

    let sut = "(👍ανδ 👎)οργαβ23";
    let expr = FlatEx::<_, UnicodeOpsFactory>::from_pattern(sut, literal_pattern).unwrap();
    assert_eq!(
        expr.eval(&[Thumbs { val: true }]).unwrap(),
        Thumbs { val: true }
    );
    assert_eq!(
        expr.eval(&[Thumbs { val: false }]).unwrap(),
        Thumbs { val: false }
    );
}

#[test]
fn test_variables() {
    let sut = "sin  ({x})+(((cos({y})   ^  (sin({z})))*log(cos({y})))*cos({z}))";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    assert_eq!(expr.n_vars(), 3usize);
    let reference =
        |x: f64, y: f64, z: f64| x.sin() + y.cos().powf(z.sin()) * y.cos().ln() * z.cos();

    utils::assert_float_eq_f64(
        expr.eval(&[-0.18961918881278095, -6.383306547710852, 3.1742139703464503])
            .unwrap(),
        reference(-0.18961918881278095, -6.383306547710852, 3.1742139703464503),
    );

    let sut = "sin(sin(x - 1 / sin(y * 5)) + (5.0 - 1/z))";
    let expr = OwnedFlatEx::<f64>::from_str(sut).unwrap();
    let reference =
        |x: f64, y: f64, z: f64| ((x - 1.0 / (y * 5.0).sin()).sin() + (5.0 - 1.0 / z)).sin();
    utils::assert_float_eq_f64(
        expr.eval(&[1.0, 2.0, 4.0]).unwrap(),
        reference(1.0, 2.0, 4.0),
    );

    let sut = "0.02*sin( - (3*(2*(5.0 - 1/z))))";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    let reference = |z: f64| 0.02 * (-(3.0 * (2.0 * (5.0 - 1.0 / z)))).sin();
    utils::assert_float_eq_f64(expr.eval(&[4.0]).unwrap(), reference(4.0));

    let sut = "y + 1 + 0.5 * x";
    let expr = OwnedFlatEx::<f64>::from_str(sut).unwrap();
    assert_eq!(expr.n_vars(), 2usize);
    utils::assert_float_eq_f64(expr.eval(&[3.0, 1.0]).unwrap(), 3.5);

    let sut = " -(-(1+x))";
    let expr = OwnedFlatEx::<f64>::from_str(sut).unwrap();
    assert_eq!(expr.n_vars(), 1usize);
    utils::assert_float_eq_f64(expr.eval(&[1.0]).unwrap(), 2.0);

    let sut = " sin(cos(-3.14159265358979*x))";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[1.0]).unwrap(), -0.841470984807896);

    let sut = "5*sin(x * (4-y^(2-x) * 3 * cos(x-2*(y-1/(y-2*1/cos(sin(x*y))))))*x)";
    let expr = OwnedFlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[1.5, 0.2532]).unwrap(), -3.1164569260604176);

    let sut = "5*x + 4*y + 3*x";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[1.0, 0.0]).unwrap(), 8.0);

    let sut = "5*x + 4*y";
    let expr = OwnedFlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[0.0, 1.0]).unwrap(), 4.0);

    let sut = "5*x + 4*y + x^2";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[2.5, 3.7]).unwrap(), 33.55);
    utils::assert_float_eq_f64(expr.eval(&[12.0, 9.3]).unwrap(), 241.2);

    let sut = "2*(4*x + y^2)";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[2.0, 3.0]).unwrap(), 34.0);

    let sut = "sin(myvar_25)";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[std::f64::consts::FRAC_PI_2]).unwrap(), 1.0);

    let sut = "((sin(myvar_25)))";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[std::f64::consts::FRAC_PI_2]).unwrap(), 1.0);

    let sut = "(0 * myvar_25 + cos(x))";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(
        expr.eval(&[std::f64::consts::FRAC_PI_2, std::f64::consts::PI])
            .unwrap(),
        -1.0,
    );

    let sut = "(-x^2)";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[1.0]).unwrap(), 1.0);

    let sut = "log(x) + 2* (-x^2 + sin(4*y))";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[2.5, 3.7]).unwrap(), 14.992794866624788);

    let sut = "-sqrt(x)/(tanh(5-x)*2) + floor(2.4)* 1/asin(-x^2 + sin(4*sinh(y)))";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(
        expr.eval(&[2.5, 3.7]).unwrap(),
        -(2.5f64.sqrt()) / (2.5f64.tanh() * 2.0)
            + 2.0 / ((3.7f64.sinh() * 4.0).sin() + 2.5 * 2.5).asin(),
    );

    let sut = "asin(sin(x)) + acos(cos(x)) + atan(tan(x))";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[0.5]).unwrap(), 1.5);

    let sut = "sqrt(alpha^ceil(centauri))";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[2.0, 3.1]).unwrap(), 4.0);

    let sut = "trunc(x) + fract(x)";
    let expr = FlatEx::<f64>::from_str(sut).unwrap();
    utils::assert_float_eq_f64(expr.eval(&[23422.52345]).unwrap(), 23422.52345);
}

#[test]
fn test_custom_ops_invert() {
    #[derive(Clone)]
    struct SomeF32Operators;
    impl MakeOperators<f32> for SomeF32Operators {
        fn make<'a>() -> Vec<Operator<'a, f32>> {
            vec![
                Operator::make_unary("invert", |a| 1.0 / a),
                Operator::make_unary("sqrt", |a| a.sqrt()),
            ]
        }
    }
    let expr = OwnedFlatEx::<f32, SomeF32Operators>::from_str("sqrt(invert(a))").unwrap();
    utils::assert_float_eq_f32(expr.eval(&[0.25]).unwrap(), 2.0);
}

#[test]
fn test_custom_ops() {
    #[derive(Clone)]
    struct SomeF32Operators;
    impl MakeOperators<f32> for SomeF32Operators {
        fn make<'a>() -> Vec<Operator<'a, f32>> {
            vec![
                Operator::make_bin(
                    "**",
                    BinOp {
                        apply: |a: f32, b| a.powf(b),
                        prio: 2,
                        is_commutative: false,
                    },
                ),
                Operator::make_bin(
                    "*",
                    BinOp {
                        apply: |a, b| a * b,
                        prio: 1,
                        is_commutative: true,
                    },
                ),
                Operator::make_unary("invert", |a: f32| 1.0 / a),
            ]
        }
    }
    let expr = OwnedFlatEx::<f32, SomeF32Operators>::from_str("2**2*invert(3)").unwrap();
    let val = expr.eval(&[]).unwrap();
    utils::assert_float_eq_f32(val, 4.0 / 3.0);

    #[derive(Clone)]
    struct ExtendedF32Operators;
    impl MakeOperators<f32> for ExtendedF32Operators {
        fn make<'a>() -> Vec<Operator<'a, f32>> {
            let zero_mapper = Operator::make_bin_unary(
                "zer0",
                BinOp {
                    apply: |_: f32, _| 0.0,
                    prio: 2,
                    is_commutative: true,
                },
                |_| 0.0,
            );
            FloatOpsFactory::<f32>::make()
                .iter()
                .cloned()
                .chain(once(zero_mapper))
                .collect::<Vec<_>>()
        }
    }
    let expr = FlatEx::<f32, ExtendedF32Operators>::from_str("2^2*1/(berti) + zer0(4)").unwrap();
    let val = expr.eval(&[4.0]).unwrap();
    utils::assert_float_eq_f32(val, 1.0);
}

#[test]
fn test_partial() {
    fn test(
        var_idx: usize,
        n_vars: usize,
        random_range: Range<f64>,
        flatex: FlatEx<f64>,
        reference: fn(f64) -> f64,
    ) {
        let mut rng = rand::thread_rng();

        assert!(flatex.clone().partial(flatex.n_vars()).is_err());

        // test owned flatex without buffer
        let owned_flatex_wo_buff = OwnedFlatEx::from_flatex(flatex.clone());
        let owned_deri = owned_flatex_wo_buff.partial(var_idx).unwrap();
        for _ in 0..3 {
            let vut = rng.gen_range(random_range.clone());
            let mut vars: SmallVec<[f64; 10]> = smallvec![0.0; n_vars];
            vars[var_idx] = vut;
            println!("value under test {}.", vut);
            utils::assert_float_eq_f64(owned_deri.eval(&vars).unwrap(), reference(vut));
        }

        // test flatex
        let deri = flatex.clone().partial(var_idx).unwrap();
        println!("flatex {}", flatex);
        println!("partial {}", deri);
        for _ in 0..3 {
            let vut = rng.gen_range(random_range.clone());
            let mut vars: SmallVec<[f64; 10]> = smallvec![0.0; n_vars];
            vars[var_idx] = vut;
            println!("value under test {}.", vut);
            utils::assert_float_eq_f64(deri.eval(&vars).unwrap(), reference(vut));
        }

        // test owned flatex with buffer
        let owned_flatex_w_buff = OwnedFlatEx::from_flatex(flatex.clone());
        println!("flatex owned {}", owned_flatex_w_buff);
        let owned_deri = owned_flatex_w_buff.partial(var_idx).unwrap();
        println!("partial owned {}", owned_deri);
        for _ in 0..3 {
            let vut = rng.gen_range(random_range.clone());
            let mut vars: SmallVec<[f64; 10]> = smallvec![0.0; n_vars];
            vars[var_idx] = vut;
            println!("value under test {}.", vut);
            utils::assert_float_eq_f64(owned_deri.eval(&vars).unwrap(), reference(vut));
        }
    }

    let sut = "+x";
    println!("{}", sut);
    let var_idx = 0;
    let n_vars = 1;
    let flatex_1 = FlatEx::<f64>::from_str(sut).unwrap();
    let reference = |_: f64| 1.0;
    test(var_idx, n_vars, -10000.0..10000.0, flatex_1, reference);

    let sut = "++x";
    println!("{}", sut);
    let var_idx = 0;
    let n_vars = 1;
    let flatex_1 = FlatEx::<f64>::from_str(sut).unwrap();
    let reference = |_: f64| 1.0;
    test(var_idx, n_vars, -10000.0..10000.0, flatex_1, reference);

    let sut = "+-+x";
    println!("{}", sut);
    let var_idx = 0;
    let n_vars = 1;
    let flatex_1 = FlatEx::<f64>::from_str(sut).unwrap();
    let reference = |_: f64| -1.0;
    test(var_idx, n_vars, -10000.0..10000.0, flatex_1, reference);

    let sut = "-x";
    println!("{}", sut);
    let var_idx = 0;
    let n_vars = 1;
    let flatex_1 = FlatEx::<f64>::from_str(sut).unwrap();
    let reference = |_: f64| -1.0;
    test(var_idx, n_vars, -10000.0..10000.0, flatex_1, reference);

    let sut = "--x";
    println!("{}", sut);
    let var_idx = 0;
    let n_vars = 1;
    let flatex_1 = FlatEx::<f64>::from_str(sut).unwrap();
    let reference = |_: f64| 1.0;
    test(var_idx, n_vars, -10000.0..10000.0, flatex_1, reference);

    let sut = "sin(sin(x))";
    println!("{}", sut);
    let var_idx = 0;
    let n_vars = 1;
    let flatex_1 = FlatEx::<f64>::from_str(sut).unwrap();
    let reference = |x: f64| x.sin().cos() * x.cos();
    test(var_idx, n_vars, -10000.0..10000.0, flatex_1, reference);

    let sut = "sin(x)-cos(x)+a";
    println!("{}", sut);
    let var_idx = 1;
    let n_vars = 2;
    let flatex_1 = FlatEx::<f64>::from_str(sut).unwrap();
    let reference = |x: f64| x.cos() + x.sin();
    test(
        var_idx,
        n_vars,
        -10000.0..10000.0,
        flatex_1.clone(),
        reference,
    );
    let deri = flatex_1.partial(var_idx).unwrap();
    let reference = |x: f64| -x.sin() + x.cos();
    test(var_idx, n_vars, -10000.0..10000.0, deri.clone(), reference);
    let deri = deri.partial(var_idx).unwrap();
    let reference = |x: f64| -x.cos() - x.sin();
    test(var_idx, n_vars, -10000.0..10000.0, deri.clone(), reference);
    let deri = deri.partial(var_idx).unwrap();
    let reference = |x: f64| x.sin() - x.cos();
    test(var_idx, n_vars, -10000.0..10000.0, deri.clone(), reference);

    let sut = "sin(x)-cos(x)+tan(x)+a";
    println!("{}", sut);
    let var_idx = 1;
    let n_vars = 2;
    let flatex_1 = FlatEx::<f64>::from_str("sin(x)-cos(x)+tan(x)+a").unwrap();
    let reference = |x: f64| x.cos() + x.sin() + 1.0 / (x.cos().powf(2.0));
    test(var_idx, n_vars, -10000.0..10000.0, flatex_1, reference);

    let sut = "log(v)*exp(v)+cos(x)+tan(x)+a";
    println!("{}", sut);
    let var_idx = 1;
    let n_vars = 3;
    let flatex = FlatEx::<f64>::from_str(sut).unwrap();
    let reference = |x: f64| 1.0 / x * x.exp() + x.ln() * x.exp();
    test(var_idx, n_vars, 0.01..100.0, flatex, reference);

    let sut = "a+z+sinh(v)/cosh(v)+b+tanh({v})";
    println!("{}", sut);
    let var_idx = 2;
    let n_vars = 4;
    let flatex = FlatEx::<f64>::from_str(sut).unwrap();
    let reference = |x: f64| {
        (x.cosh() * x.cosh() - x.sinh() * x.sinh()) / x.cosh().powf(2.0)
            + 1.0 / (x.cosh().powf(2.0))
    };
    test(var_idx, n_vars, -100.0..100.0, flatex, reference);

    let sut = "w+z+acos(v)+asin(v)+b+atan({v})";
    println!("{}", sut);
    let var_idx = 1;
    let n_vars = 4;
    let flatex = FlatEx::<f64>::from_str(sut).unwrap();
    let reference = |x: f64| {
        1.0 / (1.0 - x.powf(2.0)).sqrt() - 1.0 / (1.0 - x.powf(2.0)).sqrt()
            + 1.0 / (1.0 + x.powf(2.0))
    };
    test(var_idx, n_vars, -1.0..1.0, flatex, reference);

    let sut = "sqrt(var)*var^1.57";
    println!("{}", sut);
    let var_idx = 0;
    let n_vars = 1;
    let flatex = FlatEx::<f64>::from_str(sut).unwrap();
    let reference = |x: f64| 1.0 / (2.0 * x.sqrt()) * x.powf(1.57) + x.sqrt() * 1.57 * x.powf(0.57);
    test(var_idx, n_vars, 0.0..100.0, flatex, reference);
}

#[test]
fn test_eval() {
    fn test(sut: &str, reference: f64) {
        println!(" === testing {}", sut);
        utils::assert_float_eq_f64(eval_str(sut).unwrap(), reference);
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        utils::assert_float_eq_f64(expr.eval(&[]).unwrap(), reference);
    }

    test("0/0", f64::NAN);
    test("abs(-22/2)", 11.0);
    test("signum(-22/2)", -1.0);
    test("cbrt(8)", 2.0);
    test("2*3^2", 18.0);
    test("cos(PI/2)", 0.0);
    test("cos(π/2)", 0.0);
    test("-3^2", 9.0);
    test("11.3", 11.3);
    test("round(11.3)", 11.0);
    test("+11.3", 11.3);
    test("-11.3", -11.3);
    test("(-11.3)", -11.3);
    test("11.3+0.7", 12.0);
    test("31.3+0.7*2", 32.7);
    test("1.3+0.7*2-1", 1.7);
    test("1.3+0.7*2-1/10", 2.6);
    test("(1.3+0.7)*2-1/10", 3.9);
    test("1.3+(0.7*2)-1/10", 2.6);
    test("1.3+0.7*(2-1)/10", 1.37);
    test("1.3+0.7*(2-1/10)", 2.63);
    test("-1*(1.3+0.7*(2-1/10))", -2.63);
    test("-1*(1.3+(-0.7)*(2-1/10))", 0.03);
    test("-1*((1.3+0.7)*(2-1/10))", -3.8);
    test("sin 3.14159265358979", 0.0);
    test("0-sin(3.14159265358979 / 2)", -1.0);
    test("-sin(π / 2)", -1.0);
    test("3-(-1+sin(PI/2)*2)", 2.0);
    test("3-(-1+sin(cos(-3.14159265358979))*2)", 5.6829419696157935);
    test("-(-1+((-PI)/5)*2)", 2.256637061435916);
    test("((2-4)/5)*2", -0.8);
    test("-(-1+(sin(-PI)/5)*2)", 1.0);
    test("-(-1+sin(cos(-PI)/5)*2)", 1.3973386615901224);
    test("-cos(PI)", 1.0);
    test("1+sin(-cos(-PI))", 1.8414709848078965);
    test("-1+sin(-cos(-PI))", -0.1585290151921035);
    test("-(-1+sin(-cos(-PI)/5)*2)", 0.6026613384098776);
    test("sin(-(2))*2", -1.8185948536513634);
    test("sin(sin(2))*2", 1.5781446871457767);
    test("sin(-(sin(2)))*2", -1.5781446871457767);
    test("-sin(2)*2", -1.8185948536513634);
    test("sin(-sin(2))*2", -1.5781446871457767);
    test("sin(-sin(2)^2)*2", 1.4715655294841483);
    test("sin(-sin(2)*-sin(2))*2", 1.4715655294841483);
    test("--(1)", 1.0);
    test("--1", 1.0);
    test("----1", 1.0);
    test("---1", -1.0);
    test("3-(4-2/3+(1-2*2))", 2.666666666666666);
    test("log(log(2))*tan(2)+exp(1.5)", 5.2825344122094045);
    test("log(log2(2))*tan(2)+exp(1.5)", 4.4816890703380645);
    test("log2(2)", 1.0);
    test("2^log2(2)", 2.0);
    test("2^(cos(0)+2)", 8.0);
    test("2^cos(0)+2", 4.0);
}

#[test]
fn test_error_handling() {
    assert!(eval_str::<f64>("").is_err());
    assert!(eval_str::<f64>("5+5-(").is_err());
    assert!(eval_str::<f64>(")2*(5+5)*3-2)*2").is_err());
    assert!(eval_str::<f64>("2*(5+5))").is_err());
}

#[cfg(feature = "serde")]
#[test]
fn test_serde_public_interface() {
    let s = "{x}^(3.0-{y})";
    let flatex = FlatEx::<f64>::from_str(s).unwrap();
    let serialized = serde_json::to_string(&flatex).unwrap();
    let deserialized = serde_json::from_str::<FlatEx<f64>>(serialized.as_str()).unwrap();
    assert_eq!(s, format!("{}", deserialized));
}
#[test]
fn test_constants() -> ExResult<()> {
    assert_float_eq_f64(eval_str::<f64>("PI")?, std::f64::consts::PI);
    assert_float_eq_f64(eval_str::<f64>("E")?, std::f64::consts::E);
    let expr = parse::<f64>("x / PI * 180")?;
    utils::assert_float_eq_f64(expr.eval(&[std::f64::consts::FRAC_PI_2])?, 90.0);

    let expr = parse::<f32>("E ^ x")?;
    utils::assert_float_eq_f32(expr.eval(&[5.0])?, 1f32.exp().powf(5.0));

    let expr = parse::<f32>("E ^ Erwin");
    assert_eq!(expr?.unparse()?, "E ^ Erwin");
    Ok(())
}

#[test]
fn test_fuzz() {
    assert!(eval_str::<f64>("an").is_err());
    assert!(FlatEx::<f64>::from_str("\n").is_err());
    assert!(FlatEx::<f64>::from_pattern("\n", "\n").is_err());
}
