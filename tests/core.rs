#[cfg(test)]
mod utils;
use exmex::{exerr, DeepEx};
#[cfg(test)]
use exmex::{
    literal_matcher_from_pattern, ops_factory, parse,
    prelude::*,
    Calculate, ExError, ExResult, MatchLiteral, {BinOp, FloatOpsFactory, MakeOperators, Operator},
};
use regex::Regex;
use std::fs::{self, File};
use std::io::{self, BufRead};
use std::iter::repeat;
use utils::assert_float_eq_f64;

use std::fmt::Debug;
#[cfg(test)]
use std::{
    iter::once,
    ops::{BitAnd, BitOr},
    str::FromStr,
};

#[test]
fn test_version() {
    // make sure the version strings in the Cargo.toml and lib.rs coincide
    let file = File::open("src/lib.rs").unwrap();
    let version_line_lib = io::BufReader::new(file)
        .lines()
        .find(|line| line.as_ref().unwrap().contains("html_root_url"))
        .unwrap()
        .unwrap();
    let re_version = Regex::new(r#"[0-9]{1,4}\.[0-9]{1,4}\.[0-9]{1,4}"#).unwrap();
    let match_lib = re_version.find(&version_line_lib).unwrap().as_str();

    let toml_string = fs::read_to_string("Cargo.toml").unwrap();
    let cargo_toml: toml::Value = toml::from_str(&toml_string).unwrap();
    let package = cargo_toml.get("package").unwrap().as_table().unwrap();
    let version = package.get("version").unwrap().as_str().unwrap();
    assert_eq!(match_lib, version);
}

#[test]
fn test_display() -> ExResult<()> {
    let flatex = FlatEx::<f64>::parse("sin(var)/5")?;
    println!("{}", flatex);
    assert_eq!(format!("{}", flatex), "sin(var)/5");
    Ok(())
}

#[test]
fn test_expr() -> ExResult<()> {
    fn test(sut: &str, vars: &[f64], reference: f64) -> ExResult<()> {
        fn inner_test<'a, E: Express<'a, f64>>(
            expr: &E,
            vars: &[f64],
            reference: f64,
            more_vars: bool,
        ) -> ExResult<()> {
            utils::assert_float_eq_f64(expr.eval(vars)?, reference);
            utils::assert_float_eq_f64(expr.eval_relaxed(vars)?, reference);

            let n_vars = vars.len();
            if n_vars > 0 {
                assert!(expr.eval(&vars[0..n_vars - 1]).is_err());
            }
            if more_vars {
                for n_additional_vars in 1..11 {
                    let more_vars = vars
                        .iter()
                        .copied()
                        .chain(
                            repeat(10)
                                .map(|exp| {
                                    (rand::random::<f64>() * n_additional_vars as f64).powi(exp)
                                })
                                .take(n_additional_vars),
                        )
                        .collect::<Vec<_>>();
                    assert!(expr.eval(&more_vars).is_err());
                    utils::assert_float_eq_f64(expr.eval_relaxed(&more_vars)?, reference);
                }
            }
            println!("...ok.");
            Ok(())
        }
        println!("testing {}...", sut);
        let flatex = FlatEx::<f64>::parse(sut)?;
        inner_test(&flatex, vars, reference, true)?;
        inner_test(&flatex.to_deepex()?, vars, reference, false)?;
        let deepex = DeepEx::<f64>::parse(sut)?;
        inner_test(&deepex, vars, reference, false)?;
        for (vidx, vn) in deepex.var_names().iter().enumerate() {
            let sub_num = 3.7;
            fn sub<'a>(v: &str, vn_: &str, x: f64) -> Option<DeepEx<'a, f64>> {
                if v == vn_ {
                    Some(DeepEx::from_num(x))
                } else {
                    None
                }
            }
            let mut sub_closure = |v: &str| sub(v, vn, sub_num);
            let subs = deepex.clone().subs(&mut sub_closure)?;
            let vars_for_subs = vars
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != vidx)
                .map(|(_, x)| *x)
                .collect::<Vec<_>>();
            let mut vars_substituted = vars.iter().map(|x| *x).collect::<Vec<_>>();
            vars_substituted[vidx] = sub_num;
            assert!((subs.eval(&vars_for_subs)? - deepex.eval(&vars_substituted)?).abs() < 1e-12);
        }
        inner_test(&FlatEx::from_deepex(deepex)?, vars, reference, true)?;
        Ok(())
    }
    test("sin(1)", &[], 1.0f64.sin())?;
    test("2*3^2", &[], 2.0 * 3.0f64.powi(2))?;
    test("(3.7)-2.0*1.0/{x}", &[1.5], 3.7 - 2.0 / 1.5)?;
    test("sin(-(sin(2)))*2", &[], (-(2f64.sin())).sin() * 2.0)?;
    test("sin(-(0.7))", &[], (-0.7f64).sin())?;
    test("sin(-0.7)", &[], (-0.7f64).sin())?;
    test("sin(-x)", &[0.7], (-0.7f64).sin())?;
    test("1.3+(-0.7)", &[], 0.6)?;
    test("2-1/2", &[], 2.0 - 1.0 / 2.0)?;
    test("ln(log2(2))*tan(2)+exp(1.5)", &[], 4.4816890703380645)?;
    test("sin(0)", &[], 0f64.sin())?;
    test("1-(1-2)", &[], 2.0)?;
    test("1-(1-x)", &[2.0], 2.0)?;
    test("1*sin(2-0.1) + x", &[1.0], 1.0 + 1.9f64.sin())?;
    test("sin(6)", &[], -0.27941549819892586)?;
    test("sin(x+2)", &[5.0], 0.6569865987187891)?;
    test("sin((x+1))", &[5.0], -0.27941549819892586)?;
    test("sin(y^(x+1))", &[5.0, 2.0], 0.9200260381967907)?;
    test("sin(((a*y^(x+1))))", &[0.5, 5.0, 2.0], 0.5514266812416906)?;
    test(
        "sin(((cos((a*y^(x+1))))))",
        &[0.5, 5.0, 2.0],
        0.7407750251209115,
    )?;
    test("sin(cos(x+1))", &[5.0], 0.819289219220601)?;
    test(
        "5*{χ} +  4*log2(ln(1.5+γ))*({χ}*-(tan(cos(sin(652.2-{γ}))))) + 3*{χ}",
        &[1.2, 1.0],
        8.040556934857268,
    )?;
    test("y-2*1/x", &[1.5, 0.2532], -1.0801333333333334)?;
    test(
        "5*sin(x * (4-y^(2-x) * 3 * cos(x-2*(3.7-1/(y-2*1/cos(sin(x*y))))))*x)",
        &[1.5, 0.2532],
        0.3102594604194633,
    )?;
    test(
        "5*sin(x * (4-y^(2-x) * 3 * cos(x-2*(y-1/(y-2*1/cos(sin(x*y))))))*x)",
        &[1.5, 0.2532],
        -3.1164569260604176,
    )?;
    test("sin(x)+sin(y)+sin(z)", &[1.0, 2.0, 3.0], 1.8918884196934453)?;
    test("x*0.2*5.0/4.0+x*2.0*4.0*1.0*1.0*1.0*1.0*1.0*1.0*1.0+7.0*sin(y)-z/sin(3.0/2.0/(1.0-x*4.0*1.0*1.0*1.0*1.0))",
       &[1.0, 2.0, 3.0], 20.872570916580237)?;
    test("sin(-(1.0))", &[], -0.8414709848078965)?;
    test("x*0.02*(3-(2*y))", &[1.0, 2.0], -0.02)?;
    test("x*((x*1)-0.98)*(0.5*-y)", &[1.0, 2.0], -0.02)?;
    test("x*0.02*sin(3*(2*y))", &[1.0, 2.0], 0.02 * 12.0f64.sin())?;
    test(
        "x*0.02*sin(-(3.0*(2.0*sin(x-1.0/(sin(y*5.0)+(5.0-1.0/z))))))",
        &[1.0, 2.0, 3.0],
        0.01661860154948708,
    )?;

    let n_vars = 65;
    let s = (0..n_vars)
        .map(|i| format!("{{{}}}", i.to_string()))
        .collect::<Vec<_>>()
        .join("+");
    test(s.as_str(), &vec![1.0; n_vars], n_vars as f64)?;
    Ok(())
}

#[test]
fn test_readme() -> ExResult<()> {
    fn readme() -> ExResult<()> {
        let result = exmex::eval_str::<f64>("E^(2*π-τ)")?;
        assert!((result - 1.0).abs() < 1e-12);
        let expr = parse::<f64>("2*x^3-4/y")?;
        let result = expr.eval(&[2.0, 4.0])?;
        assert!((result - 15.0).abs() < 1e-12);
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
        let expr = FlatEx::<_, BitwiseOpsFactory>::parse("!(a|b)")?;
        let result = expr.eval(&[0, 1])?;
        assert_eq!(result, u32::MAX - 1);
        Ok(())
    }
    readme()?;
    readme_int()?;
    Ok(())
}
#[test]
fn test_variables_curly_space_names() -> ExResult<()> {
    let sut = "{x } + { y }";
    let expr = FlatEx::<f32>::parse(sut)?;
    utils::assert_float_eq::<f32>(expr.eval(&[1.0, 1.0])?, 2.0, 1e-6, 0.0, "");
    assert_eq!(expr.unparse(), sut);
    let sut = "2*(4*{ xasd sa } + { y z}^2)";
    let expr = FlatEx::<f32>::parse(sut)?;
    utils::assert_float_eq::<f32>(expr.eval(&[2.0, 3.0])?, 34.0, 1e-6, 0.0, "");
    assert_eq!(expr.unparse(), sut);
    Ok(())
}
#[test]
fn test_variables_curly() -> ExResult<()> {
    let sut = "5*{x} +  4*log2(ln(1.5+{gamma}))*({x}*-(tan(cos(sin(652.2-{gamma}))))) + 3*{x}";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[1.2, 1.0]).unwrap(), 8.040556934857268);

    let sut = "sin({myvwmlf4i58eo;w/-sin(a)r_25})";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[std::f64::consts::FRAC_PI_2]).unwrap(), 1.0);

    let sut = "((sin({myvar_25})))";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[std::f64::consts::FRAC_PI_2]).unwrap(), 1.0);
    Ok(())
}

#[test]
fn test_variables_non_ascii() -> ExResult<()> {
    let sut = "5*ς";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[1.2]).unwrap(), 6.0);

    let sut = "5*{χ} +  4*log2(ln(1.5+γ))*({χ}*-(tan(cos(sin(652.2-{γ}))))) + 3*{χ}";
    let expr = FlatEx::<f64>::parse(sut)?;
    println!("{}", expr);
    utils::assert_float_eq_f64(expr.eval(&[1.2, 1.0]).unwrap(), 8.040556934857268);

    let sut = "sin({myvwmlf4i😎8eo;w/-sin(a)r_25})";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[std::f64::consts::FRAC_PI_2]).unwrap(), 1.0);

    let sut = "((sin({myvar_25✔})))";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[std::f64::consts::FRAC_PI_2]).unwrap(), 1.0);

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
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
                Err(exerr!("cannot parse {} to `Thumbs`", s))
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

    literal_matcher_from_pattern!(ThumbsMatcher, r"^(👍|👎)");

    let sut = "γ ορ 👍ορ👎";
    let expr = FlatEx::<_, UnicodeOpsFactory, ThumbsMatcher>::parse(sut)?;
    assert_eq!(expr.eval(&[]).unwrap(), Thumbs { val: true });

    let sut = "(👍 ανδ👎)ορ 👍";
    let expr = FlatEx::<_, UnicodeOpsFactory, ThumbsMatcher>::parse(sut)?;
    assert_eq!(expr.eval(&[]).unwrap(), Thumbs { val: true });

    let sut = "(👍ανδ 👎)οργαβ23";
    let expr = FlatEx::<_, UnicodeOpsFactory, ThumbsMatcher>::parse(sut)?;
    assert_eq!(expr.eval(&[Thumbs { val: true }])?, Thumbs { val: true });
    assert_eq!(expr.eval(&[Thumbs { val: false }])?, Thumbs { val: false });
    Ok(())
}

#[test]
fn test_variables() -> ExResult<()> {
    let sut = "sin  ({x})+(((cos({y})   ^  (sin({z})))*ln(cos({y})))*cos({z}))";
    let expr = FlatEx::<f64>::parse(sut)?;
    assert_eq!(expr.var_names().len(), 3usize);
    let reference =
        |x: f64, y: f64, z: f64| x.sin() + y.cos().powf(z.sin()) * y.cos().ln() * z.cos();

    utils::assert_float_eq_f64(
        expr.eval(&[-0.18961918881278095, -6.383306547710852, 3.1742139703464503])
            .unwrap(),
        reference(-0.18961918881278095, -6.383306547710852, 3.1742139703464503),
    );

    let sut = "sin(sin(x - 1 / sin(y * 5)) + (5.0 - 1/z))";
    let expr = FlatEx::<f64>::parse(sut)?;
    let reference =
        |x: f64, y: f64, z: f64| ((x - 1.0 / (y * 5.0).sin()).sin() + (5.0 - 1.0 / z)).sin();
    utils::assert_float_eq_f64(
        expr.eval(&[1.0, 2.0, 4.0]).unwrap(),
        reference(1.0, 2.0, 4.0),
    );

    let sut = "0.02*sin( - (3*(2*(5.0 - 1/z))))";
    let expr = FlatEx::<f64>::parse(sut)?;
    let reference = |z: f64| 0.02 * (-(3.0 * (2.0 * (5.0 - 1.0 / z)))).sin();
    utils::assert_float_eq_f64(expr.eval(&[4.0]).unwrap(), reference(4.0));

    let sut = "y + 1 + 0.5 * x";
    let expr = FlatEx::<f64>::parse(sut)?;
    assert_eq!(expr.var_names().len(), 2usize);
    utils::assert_float_eq_f64(expr.eval(&[3.0, 1.0]).unwrap(), 3.5);

    let sut = " -(-(1+x))";
    let expr = FlatEx::<f64>::parse(sut)?;
    assert_eq!(expr.var_names().len(), 1usize);
    utils::assert_float_eq_f64(expr.eval(&[1.0]).unwrap(), 2.0);

    let sut = " sin(cos(-3.14159265358979*x))";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[1.0]).unwrap(), -0.841470984807896);

    let sut = "5*sin(x * (4-y^(2-x) * 3 * cos(x-2*(y-1/(y-2*1/cos(sin(x*y))))))*x)";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[1.5, 0.2532]).unwrap(), -3.1164569260604176);

    let sut = "5*x + 4*y + 3*x";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[1.0, 0.0]).unwrap(), 8.0);

    let sut = "5*x + 4*y";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[0.0, 1.0]).unwrap(), 4.0);

    let sut = "5*x + 4*y + x^2";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[2.5, 3.7]).unwrap(), 33.55);
    utils::assert_float_eq_f64(expr.eval(&[12.0, 9.3]).unwrap(), 241.2);

    let sut = "2*(4*x + y^2)";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[2.0, 3.0]).unwrap(), 34.0);

    let sut = "sin(myvar_25)";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[std::f64::consts::FRAC_PI_2]).unwrap(), 1.0);

    let sut = "((sin(myvar_25)))";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[std::f64::consts::FRAC_PI_2]).unwrap(), 1.0);

    let sut = "(0 * myvar_25 + cos(x))";
    let expr = FlatEx::<f64>::parse(sut)?;
    let expr = expr.to_deepex()?;
    utils::assert_float_eq_f64(
        expr.eval(&[std::f64::consts::FRAC_PI_2, std::f64::consts::PI])
            .unwrap(),
        -1.0,
    );

    let sut = "(-x^2)";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[1.0]).unwrap(), 1.0);

    let sut = "ln(x) + 2* (-x^2 + sin(4*y))";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[2.5, 3.7]).unwrap(), 14.992794866624788);

    let sut = "-sqrt(x)/(tanh(5-x)*2) + floor(2.4)* 1/asin(sin(4*sinh(y)))";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(
        expr.eval(&[2.5, 3.7]).unwrap(),
        -(2.5f64.sqrt()) / (2.5f64.tanh() * 2.0) + 2.0 / ((3.7f64.sinh() * 4.0).sin()).asin(),
    );

    let sut = "asin(sin(x)) + acos(cos(x)) + atan(tan(x))";
    let expr = DeepEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[0.5]).unwrap(), 1.5);

    let sut = "sqrt(alpha^ceil(centauri))";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(expr.eval(&[2.0, 3.1]).unwrap(), 4.0);

    let sut = "trunc(x) + fract(x)";
    let expr = DeepEx::<f64>::parse(sut)?;
    let expr = FlatEx::from_deepex(expr)?;
    utils::assert_float_eq_f64(expr.eval(&[23422.52345]).unwrap(), 23422.52345);
    Ok(())
}

#[test]
fn test_custom_ops_invert() -> ExResult<()> {
    #[derive(Clone, Debug)]
    struct SomeF32Operators;
    impl MakeOperators<f32> for SomeF32Operators {
        fn make<'a>() -> Vec<Operator<'a, f32>> {
            vec![
                Operator::make_unary("invert", |a| 1.0 / a),
                Operator::make_unary("sqrt", |a| a.sqrt()),
            ]
        }
    }
    let expr = FlatEx::<f32, SomeF32Operators>::parse("sqrt(invert(a))")?;
    utils::assert_float_eq::<f32>(expr.eval(&[0.25]).unwrap(), 2.0, 1e-6, 0.0, "");
    Ok(())
}

#[test]
fn test_custom_ops() -> ExResult<()> {
    #[derive(Clone, Debug)]
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
    let expr = FlatEx::<f32, SomeF32Operators>::parse("2**2*invert(3)")?;
    let val = expr.eval(&[])?;
    utils::assert_float_eq::<f32>(val, 4.0 / 3.0, 1e-6, 0.0, "");

    #[derive(Clone, Debug)]
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
    let expr = FlatEx::<f32, ExtendedF32Operators>::parse("2^2*1/(berti) + zer0(4)")?;
    let val = expr.eval(&[4.0])?;
    utils::assert_float_eq::<f32>(val, 1.0, 1e-6, 0.0, "");
    Ok(())
}

#[test]
fn test_eval_str() -> ExResult<()> {
    fn test(sut: &str, reference: f64) -> ExResult<()> {
        println!(" === testing {}", sut);
        utils::assert_float_eq_f64(exmex::eval_str(sut)?, reference);
        let expr = FlatEx::<f64>::parse(sut)?;
        utils::assert_float_eq_f64(expr.eval(&[])?, reference);
        let expr = DeepEx::<f64>::parse(sut)?;
        utils::assert_float_eq_f64(expr.eval(&[])?, reference);
        let expr = FlatEx::from_deepex(expr)?;
        utils::assert_float_eq_f64(expr.eval(&[])?, reference);
        let expr = expr.to_deepex()?;
        utils::assert_float_eq_f64(expr.eval(&[])?, reference);
        Ok(())
    }
    assert!(exmex::eval_str::<f64>("0/0")?.is_nan());
    test("abs(  -22/2)", 11.0)?;
    test("signum(-22/2)", -1.0)?;
    test("cbrt(8)", 2.0)?;
    test("2*3^2", 18.0)?;
    test("PI", std::f64::consts::PI)?;
    test("π", std::f64::consts::PI)?;
    test("TAU", std::f64::consts::TAU)?;
    test("τ", std::f64::consts::TAU)?;
    test("τ", std::f64::consts::PI * 2.0)?;
    test("E", std::f64::consts::E)?;
    test("e", std::f64::consts::E)?;
    test("cos(TAU)", 1.0)?;
    test("sin(τ)", 0.0)?;
    test("cos(PI/2)", 0.0)?;
    test("cos(π/2)", 0.0)?;
    test("-3^2", 9.0)?;
    test("11.3", 11.3)?;
    test("round(11.3)", 11.0)?;
    test("+11.3", 11.3)?;
    test("-11.3", -11.3)?;
    test("(-11.3)", -11.3)?;
    test("11.3+0.7", 12.0)?;
    test("31.3+0.7*2", 32.7)?;
    test("1.3+0.7*2-1", 1.7)?;
    test("1.3+0.7*2-1/10", 2.6)?;
    test("(1.3+0.7)*2-1/10", 3.9)?;
    test("1.3+(0.7*2)-1/10", 2.6)?;
    test("1.3+0.7*(2-1)/10", 1.37)?;
    test("1.3+0.7*(2-1/10)", 2.63)?;
    test("-1*(1.3+0.7*(2-1/10))", -2.63)?;
    test("-1*(1.3+(-0.7)*(2-1/10))", 0.03)?;
    test("-1*((1.3+0.7)*(2-1/10))", -3.8)?;
    test("sin 3.14159265358979", 0.0)?;
    test("0-sin(3.14159265358979 / 2)", -1.0)?;
    test("-sin(π / 2)", -1.0)?;
    test("3-(-1+sin(PI/2)*2)", 2.0)?;
    test("3-(-1+sin(cos(-3.14159265358979))*2)", 5.6829419696157935)?;
    test("-(-1+((-PI)/5)*2)", 2.256637061435916)?;
    test("((2-4)/5)*2", -0.8)?;
    test("-(-1+(sin(-PI)/5)*2)", 1.0)?;
    test("-(-1+sin(cos(-PI)/5)*2)", 1.3973386615901224)?;
    test("-cos(PI)", 1.0)?;
    test("1+sin(-cos(-PI))", 1.8414709848078965)?;
    test("-1+sin(-cos(-PI))", -0.1585290151921035)?;
    test("-(-1+sin(-cos(-PI)/5)*2)", 0.6026613384098776)?;
    test("sin(-(2))*2", -1.8185948536513634)?;
    test("sin(sin(2))*2", 1.5781446871457767)?;
    test("sin(-(sin(2)))*2", -1.5781446871457767)?;
    test("-sin(2)*2", -1.8185948536513634)?;
    test("sin(-sin(2))*2", -1.5781446871457767)?;
    test("sin(-sin(2)^2)*2", 1.4715655294841483)?;
    test("sin(-sin(2)*-sin(2))*2", 1.4715655294841483)?;
    test("--(1)", 1.0)?;
    test("--1", 1.0)?;
    test("----1", 1.0)?;
    test("---1", -1.0)?;
    test("3-(4-2/3+(1-2*2))", 2.666666666666666)?;
    test("ln(ln(2))*tan(2)+exp(1.5)", 5.2825344122094045)?;
    test("log(log(2))*tan(2)+exp(1.5)", 5.2825344122094045)?;
    test("ln(log2(2))*tan(2)+exp(1.5)", 4.4816890703380645)?;
    test("log2(2)", 1.0)?;
    test("2^log2(2)", 2.0)?;
    test("2^(cos(0)+2)", 8.0)?;
    test("2^cos(0)+2", 4.0)?;
    Ok(())
}

#[test]
fn test_error_handling() {
    assert!(exmex::parse::<f64>("z+/Q").is_err());
    assert!(exmex::parse::<f64>("6-^6").is_err());
    assert!(exmex::eval_str::<f64>("").is_err());
    assert!(exmex::eval_str::<f64>("5+5-(").is_err());
    assert!(exmex::eval_str::<f64>(")2*(5+5)*3-2)*2").is_err());
    assert!(exmex::eval_str::<f64>("2*(5+5))").is_err());
}

#[cfg(feature = "serde")]
#[test]
fn test_serde_public_interface() -> ExResult<()> {
    let s = "{x}^(3.0-{y})";
    let flatex = FlatEx::<f64>::parse(s)?;
    let serialized = serde_json::to_string(&flatex).unwrap();
    let deserialized = serde_json::from_str::<FlatEx<f64>>(serialized.as_str()).unwrap();
    assert_eq!(s, format!("{}", deserialized));
    Ok(())
}
#[test]
fn test_constants() -> ExResult<()> {
    utils::assert_float_eq_f64(exmex::eval_str::<f64>("PI")?, std::f64::consts::PI);
    utils::assert_float_eq_f64(exmex::eval_str::<f64>("E")?, std::f64::consts::E);
    let expr = parse::<f64>("x / PI * 180")?;
    utils::assert_float_eq_f64(expr.eval(&[std::f64::consts::FRAC_PI_2])?, 90.0);

    let expr = parse::<f32>("E ^ x")?;
    utils::assert_float_eq::<f32>(expr.eval(&[5.0])?, 1f32.exp().powf(5.0), 1e-6, 0.0, "");

    let expr = parse::<f32>("E ^ Erwin");
    assert_eq!(expr?.unparse(), "E ^ Erwin");
    Ok(())
}

#[test]
fn test_fuzz() {
    assert!(exmex::eval_str::<f64>("an").is_err());
    assert!(FlatEx::<f64>::parse("\n").is_err());
}

#[cfg(feature = "serde")]
#[test]
fn test_to_deepex_non_default() -> ExResult<()> {
    ops_factory!(
        SomeOps,
        i64,
        Operator::make_bin(
            "/",
            BinOp {
                apply: |a, b| a / b,
                prio: 2,
                is_commutative: true
            }
        ),
        Operator::make_bin(
            "*",
            BinOp {
                apply: |a, b| a * b,
                prio: 2,
                is_commutative: true
            }
        ),
        Operator::make_bin(
            "+",
            BinOp {
                apply: |a, b| a + b,
                prio: 1,
                is_commutative: true
            }
        ),
        Operator::make_bin_unary(
            "-",
            BinOp {
                apply: |a, b| a - b,
                prio: 1,
                is_commutative: false
            },
            |a| -a
        ),
        Operator::make_bin(
            "%",
            BinOp {
                apply: |a, b| a % b,
                prio: 2,
                is_commutative: false
            }
        ),
        Operator::make_bin(
            "|",
            BinOp {
                apply: |a, b| a | b,
                prio: 2,
                is_commutative: true
            }
        ),
        Operator::make_bin(
            "&",
            BinOp {
                apply: |a, b| a & b,
                prio: 2,
                is_commutative: true
            }
        )
    );
    let flatex = FlatEx::<i64, SomeOps>::parse("alpha*(-beta-(11-gamma/(omikron*3+zeta)))")?;
    let deepex = flatex.clone().to_deepex()?;
    let input = [7, 5, 4, 3, 2];
    assert_eq!(flatex.eval(&input), deepex.eval(&input));
    let flatex2 = FlatEx::from_deepex(deepex)?;
    assert_eq!(flatex.eval(&input), flatex2.eval(&input));

    let serialized = serde_json::to_string(&flatex).unwrap();
    let deserialized = serde_json::from_str::<FlatEx<i64, SomeOps>>(serialized.as_str()).unwrap();
    let deepex = deserialized.to_deepex()?;
    assert_eq!(flatex.eval(&input), deepex.eval(&input));
    let flatex2 = FlatEx::from_deepex(deepex)?;
    assert_eq!(flatex.eval(&input), flatex2.eval(&input));
    let xpy_f = FlatEx::<i64, SomeOps>::parse("x + y")?;
    let y_f = FlatEx::<i64, SomeOps>::parse("y")?;
    let xpy = xpy_f.to_deepex()?;
    let y = y_f.to_deepex()?;
    let xpy_mod_y = (xpy.clone() % y.clone())?;
    assert_eq!(xpy_mod_y.eval(&[1, 2])?, 1);
    let xpy_or_y = (xpy.clone() | y.clone())?;
    assert_eq!(
        xpy_or_y.eval(&[1, 2])?,
        DeepEx::<i64, SomeOps>::parse("(x+y) | y")?.eval(&[1, 2])?
    );
    let xpy_and_y = (xpy.clone() & y.clone())?;
    assert_eq!(
        xpy_and_y.eval(&[7, 2])?,
        DeepEx::<i64, SomeOps>::parse("(x+y) & y")?.eval(&[7, 2])?
    );
    Ok(())
}

#[test]
fn test_ops() -> ExResult<()> {
    let d1 = DeepEx::<f64>::one();
    let d2 = DeepEx::<f64>::one();
    let sum = (d1 + d2)?;
    utils::assert_float_eq_f64(sum.eval(&[])?, 2.0);
    let mul = (sum.clone() * sum)?;
    utils::assert_float_eq_f64(mul.eval(&[])?, 4.0);
    let div = (mul.clone() / mul)?;
    utils::assert_float_eq_f64(div.eval(&[])?, 1.0);
    let sub = (div.clone() - div)?;
    utils::assert_float_eq_f64(sub.eval(&[])?, 0.0);
    let d = DeepEx::<f64>::parse("x^2")?;
    let d = (d ^ DeepEx::from_num(2.0))?;
    assert_eq!(
        d.eval(&[7.3]),
        DeepEx::<f64>::parse("{x}^4.0")?.eval(&[7.3])
    );
    let d = (-d)?;
    let ref_val = DeepEx::<f64>::parse("-({x}^4.0)")?.eval(&[7.3])?;
    let ref_val_2 = DeepEx::<f64>::parse("-({x}^4.0)")?.eval(&[0.95])?;
    assert_eq!(d.eval(&[7.3])?, ref_val);
    assert!((d.clone() & d.clone()).is_err());
    assert!((d.clone() | d.clone()).is_err());
    assert!((d.clone() % d.clone()).is_err());
    utils::assert_float_eq_f64(d.clone().abs()?.eval(&[7.3])?, ref_val.abs());
    utils::assert_float_eq_f64(d.clone().sin()?.eval(&[7.3])?, ref_val.sin());
    utils::assert_float_eq_f64(d.clone().cos()?.eval(&[7.3])?, ref_val.cos());
    utils::assert_float_eq_f64(d.clone().tan()?.eval(&[7.3])?, ref_val.tan());
    utils::assert_float_eq_f64(d.clone().asin()?.eval(&[0.95])?, ref_val_2.asin());
    utils::assert_float_eq_f64(d.clone().acos()?.eval(&[0.95])?, ref_val_2.acos());
    utils::assert_float_eq_f64(d.clone().atan()?.eval(&[0.95])?, ref_val_2.atan());
    utils::assert_float_eq_f64(d.clone().ceil()?.eval(&[7.3])?, ref_val.ceil());
    utils::assert_float_eq_f64(d.clone().floor()?.eval(&[7.3])?, ref_val.floor());
    utils::assert_float_eq_f64(d.clone().round()?.eval(&[7.3])?, ref_val.round());
    utils::assert_float_eq_f64(d.clone().exp()?.eval(&[7.3])?, ref_val.exp());
    utils::assert_float_eq_f64((-d.clone())?.clone().log()?.eval(&[7.3])?, (-ref_val).ln());
    utils::assert_float_eq_f64(
        (-d.clone())?.clone().log2()?.eval(&[7.3])?,
        (-ref_val).log2(),
    );
    utils::assert_float_eq_f64(
        (-d.clone())?.clone().log10()?.eval(&[7.3])?,
        (-ref_val).log10(),
    );
    utils::assert_float_eq_f64((-d.clone())?.clone().ln()?.eval(&[7.3])?, (-ref_val).ln());
    utils::assert_float_eq_f64(d.clone().signum()?.eval(&[7.3])?, ref_val.signum());
    utils::assert_float_eq_f64(
        (-d.clone())?.clone().sqrt()?.eval(&[7.3])?,
        (-ref_val).sqrt(),
    );
    utils::assert_float_eq_f64(
        (-d.clone())?.clone().cbrt()?.eval(&[7.3])?,
        (-ref_val).cbrt(),
    );
    utils::assert_float_eq_f64(d.clone().trunc()?.eval(&[7.3])?, ref_val.trunc());
    utils::assert_float_eq_f64(d.clone().fract()?.eval(&[7.3])?, ref_val.fract());
    utils::assert_float_eq_f64(DeepEx::<f64>::pi().eval(&[])?, std::f64::consts::PI);
    utils::assert_float_eq_f64(DeepEx::<f64>::e().eval(&[])?, std::f64::consts::E);
    utils::assert_float_eq_f64(DeepEx::<f64>::tau().eval(&[])?, std::f64::consts::TAU);

    Ok(())
}

#[test]
fn test_calculate() -> ExResult<()> {
    let one = DeepEx::<f64>::one();
    let another_one = one.clone();
    let two = one.operate_binary(another_one, "+")?;
    utils::assert_float_eq_f64(two.eval(&[])?, 2.0);
    let expr = DeepEx::<f64>::parse("exp(x)+2*y")?;
    let expr_sub = DeepEx::parse("2*z")?;
    let mut subs = |var: &str| match var {
        "y" => Some(expr_sub.clone()),
        _ => None,
    };
    let subsed = expr.subs(&mut subs)?;
    utils::assert_float_eq_f64(subsed.eval(&[0.0, 2.0])?, 9.0);
    Ok(())
}

#[test]
fn test_var_indices_ordered() {
    fn test<'a>(s: &str, reference: impl Iterator<Item = &'a usize>) {
        let expr = FlatEx::<f64>::parse(s).unwrap();
        let vars_ordered = expr.var_indices_ordered();
        let reference = reference.collect::<Vec<_>>();
        assert_eq!(reference.len(), vars_ordered.len());
        for (a, b) in reference.iter().zip(vars_ordered) {
            assert_eq!(**a, b);
        }
    }
    test("x + x", [0, 0].iter());
    test("x + y + x", [0, 1, 0].iter());
    test("x + y * z", [1, 2, 0].iter());
    test("(x + y) * z", [0, 1, 2].iter());
    test("(x + y) * cos(a - sin(b + z))", [1, 4, 0, 2, 3].iter());
    test("(x + y) * (a - (b * z)) + a", [1, 4, 0, 2, 3, 0].iter());
}
#[test]
fn test_eval_vec_iter() {
    fn test<'a>(s: &str, vars: Vec<f64>, reference: f64) {
        let expr = FlatEx::<f64>::parse(s).unwrap();
        assert!((expr.eval_vec(vars.clone()).unwrap() - reference).abs() < 1e-12);
        assert!((expr.eval_iter(vars.into_iter()).unwrap() - reference).abs() < 1e-12);
    }
    test("x + y * z", vec![1.0, 2.0, 3.0], 7.0);
    test("(x + y) * z", vec![1.0, 2.0, 3.0], 9.0);
    test("x * z * (x + y) - a * a", vec![0.5, 1.0, 2.0, 3.0], 8.75);

    #[derive(Debug, PartialEq, Eq)]
    struct StringContainer {
        data: Vec<String>,
        has_been_cloned: bool,
    }
    impl StringContainer {
        fn new(s: &str) -> Self {
            Self {
                data: vec![s.to_string()],
                has_been_cloned: false,
            }
        }
        fn from_slice(v: &[&str]) -> Self {
            Self {
                data: v.iter().map(|s| s.to_string()).collect(),
                has_been_cloned: false,
            }
        }
    }
    impl Default for StringContainer {
        fn default() -> Self {
            Self::new("default")
        }
    }
    impl Clone for StringContainer {
        fn clone(&self) -> Self {
            Self {
                data: self.data.clone(),
                has_been_cloned: true,
            }
        }
    }
    impl FromStr for StringContainer {
        type Err = ExError;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            Ok(StringContainer::new(s))
        }
    }
    ops_factory!(
        StringOps,
        StringContainer,
        Operator::make_bin(
            "+",
            BinOp {
                apply: |mut s1, mut s2| {
                    s1.data.append(&mut s2.data);
                    s1
                },
                prio: 2,
                is_commutative: false
            }
        ),
        Operator::make_bin(
            "-",
            BinOp {
                apply: |mut s1, mut s2| {
                    s1.data.append(&mut s2.data);
                    s1
                },
                prio: 3,
                is_commutative: false
            }
        ),
        Operator::make_bin(
            "|",
            BinOp {
                apply: |mut s1, mut s2| {
                    s1.data.append(&mut s2.data);
                    s1
                },
                prio: 0,
                is_commutative: false
            }
        ),
        Operator::make_bin(
            "==",
            BinOp {
                apply: |mut s1, mut s2| {
                    s1.data.append(&mut s2.data);
                    s1
                },
                prio: 1,
                is_commutative: false
            }
        )
    );
    let expr = FlatEx::<StringContainer, StringOps>::parse("x+y").unwrap();
    let x = StringContainer::new("x");
    let y = StringContainer::new("y");
    let res = expr.eval_vec(vec![x, y]).unwrap();
    assert_eq!(res, StringContainer::from_slice(&["x", "y"]));
    assert!(!res.has_been_cloned);
    let expr = FlatEx::<StringContainer, StringOps>::parse("x+y+x").unwrap();
    let x = StringContainer::new("x");
    let y = StringContainer::new("y");
    let res = expr.eval_vec(vec![x, y]).unwrap();
    assert!(res.has_been_cloned);
    assert_eq!(res, StringContainer::from_slice(&["x", "y", "x"]).clone());
    let expr = FlatEx::<StringContainer, StringOps>::parse("(x|y-2)-(x|y==2)").unwrap();
    let x = StringContainer::new("alpha");
    let y = StringContainer::new("beta");
    let res = expr.eval_vec(vec![x, y]).unwrap();
    assert_eq!(
        res,
        StringContainer::from_slice(&["alpha", "beta", "2", "alpha", "beta", "2"]).clone()
    );
}
#[test]
fn test_string_ops() {
    literal_matcher_from_pattern!(StringMatcher, r"^[a-zA-z0-9]+");
    ops_factory!(
        StringOpsFactory,
        String,
        Operator::make_bin(
            "+",
            BinOp {
                apply: |mut s1, s2| {
                    s1.push_str(&s2);
                    s1
                },
                prio: 2,
                is_commutative: false
            }
        ),
        Operator::make_constant("-", "MINUS".to_string())
    );
    let expr = FlatEx::<String, StringOpsFactory, StringMatcher>::parse("x+y+{_}+-").unwrap();
    assert_eq!(
        expr.eval(&["abc".to_string()]).unwrap(),
        "xyabcMINUS".to_string()
    );
}

#[test]
fn test_binary_function_style() {
    fn test(s: &str, vars: &[f64], reference: f64) {
        println!("testing {s}");
        fn test_<'a, EX: Express<'a, f64> + Debug>(s: &'a str, vars: &[f64], reference: f64) {
            let expr = EX::parse(s).unwrap();
            assert_float_eq_f64(expr.eval(vars).unwrap(), reference);
        }
        println!("flatex...");
        test_::<FlatEx<f64>>(s, vars, reference);
        println!("deepex...");
        test_::<DeepEx<f64>>(s, vars, reference);
    }
    test(
        "atan2(0.2/y, x)",
        &[1.2, 2.1],
        (0.2 / 2.1_f64).atan2(1.2_f64),
    );
    test("+ (1, -2) / 2", &[], -0.5);
    test("/ 1 2 * 3", &[], 1.5);
    test("atan2(1, 2) * 3", &[], 1.0f64.atan2(2.0) * 3.0);
    test(
        "2 + atan2(1, x / 2) * 3",
        &[1.0],
        2.0 + 1.0f64.atan2(0.5) * 3.0,
    );
    test(
        "sin(atan2(1, x / 2)) * 3",
        &[1.0],
        (1.0f64.atan2(0.5)).sin() * 3.0,
    );
    test(
        "max(sin(atan2(1, x / 2)) * 3, x)",
        &[1.0],
        ((1.0f64.atan2(0.5)).sin() * 3.0).max(1.0),
    );
    assert!(FlatEx::<f64>::parse("atan3(z, y, x").is_err());
}

#[test]
fn test_op_reprs() {
    fn test(s: &str, uo_reference: &[&str], bo_reference: &[&str], ao_reference: &[&str]) {
        println!("testing {s}");
        fn test_<'a, EX: Express<'a, f64> + Debug>(
            s: &'a str,
            uo_reference: &[&str],
            bo_reference: &[&str],
            all_reference: &[&str],
        ) {
            let expr = EX::parse(s).unwrap();
            let uops = expr.unary_reprs().to_vec();
            let bops = expr.binary_reprs().to_vec();
            let aops = expr.operator_reprs().to_vec();
            let mut uo_reference = uo_reference
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            let mut bo_reference = bo_reference
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            let mut ao_reference = all_reference
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            uo_reference.sort();
            bo_reference.sort();
            ao_reference.sort();
            assert_eq!(uops, uo_reference);
            assert_eq!(bops, bo_reference);
            assert_eq!(aops, ao_reference);
        }

        println!("flatex...");
        test_::<FlatEx<f64>>(s, uo_reference, bo_reference, ao_reference);
        println!("deepex...");
        test_::<DeepEx<f64>>(s, uo_reference, bo_reference, ao_reference);
    }
    test("atan2(0.2/y, x)", &[], &["atan2", "/"], &["atan2", "/"]);
    test("-x", &["-"], &[], &["-"]);
    test("sin(-x)", &["-", "sin"], &[], &["-", "sin"]);
    test(
        "sin(tan(cos(x)))",
        &["cos", "sin", "tan"],
        &[],
        &["cos", "sin", "tan"],
    );
    test(
        "sin(1+tan(cos(x)))",
        &["cos", "sin", "tan"],
        &["+"],
        &["+", "cos", "sin", "tan"],
    );
    test(
        "sin(-tan(cos(x)))",
        &["-", "cos", "sin", "tan"],
        &[],
        &["-", "cos", "sin", "tan"],
    );
    test(
        "sin(-tan(y+cos(x-z)))",
        &["-", "cos", "sin", "tan"],
        &["+", "-"],
        &["+", "-", "cos", "sin", "tan"],
    );
    test(
        "sin(-tan(y+sin(cos(x-z))))",
        &["-", "cos", "sin", "tan"],
        &["+", "-"],
        &["+", "-", "cos", "sin", "tan"],
    );
    test(
        "4/3 * a / b * (1.3 / 65.2 / ((18.93+c+d+e) / 111))",
        &[],
        &["*", "+", "/"],
        &["*", "+", "/"],
    );
}
