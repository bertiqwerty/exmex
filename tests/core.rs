#[cfg(test)]
mod utils;
use exmex::{format_exerr, DeepEx};
#[cfg(test)]
use exmex::{
    literal_matcher_from_pattern, ops_factory, parse,
    prelude::*,
    ExError, ExResult, MatchLiteral, {BinOp, FloatOpsFactory, MakeOperators, Operator},
};
use std::iter::repeat;
#[cfg(test)]
use std::{
    iter::once,
    ops::{BitAnd, BitOr},
    str::FromStr,
};

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
        inner_test(&FlatEx::from_deepex(deepex)?, vars, reference, true)?;
        Ok(())
    }
    test("sin(1)", &[], 1.0f64.sin())?;
    test("2*3^2", &[], 2.0 * 3.0f64.powi(2))?;
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
    // test(
    //     "5*sin(x * (4-y^(2-x) * 3 * cos(x-2*(y-1/(y-2*1/cos(sin(x*y))))))*x)",
    //     &[1.5, 0.2532],
    //     -3.1164569260604176,
    // )?;
    // test("sin(x)+sin(y)+sin(z)", &[1.0, 2.0, 3.0], 1.8918884196934453)?;
    // test("x*0.2*5.0/4.0+x*2.0*4.0*1.0*1.0*1.0*1.0*1.0*1.0*1.0+7.0*sin(y)-z/sin(3.0/2.0/(1.0-x*4.0*1.0*1.0*1.0*1.0))",
    // &[1.0, 2.0, 3.0], 20.872570916580237)?;
    // test("sin(-(1.0))", &[], -0.8414709848078965)?;
    // test("x*0.02*(3-(2*y))", &[1.0, 2.0], -0.02)?;
    // test("x*((x*1)-0.98)*(0.5*-y)", &[1.0, 2.0], -0.02)?;
    // test("x*0.02*sin(3*(2*y))", &[1.0, 2.0], 0.02 * 12.0f64.sin())?;
    // test(
    //     "x*0.02*sin(-(3.0*(2.0*sin(x-1.0/(sin(y*5.0)+(5.0-1.0/z))))))",
    //     &[1.0, 2.0, 3.0],
    //     0.01661860154948708,
    // )?;

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
                Err(format_exerr!("cannot parse {} to `Thumbs`", s))
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

    let sut = "-sqrt(x)/(tanh(5-x)*2) + floor(2.4)* 1/asin(-x^2 + sin(4*sinh(y)))";
    let expr = FlatEx::<f64>::parse(sut)?;
    utils::assert_float_eq_f64(
        expr.eval(&[2.5, 3.7]).unwrap(),
        -(2.5f64.sqrt()) / (2.5f64.tanh() * 2.0)
            + 2.0 / ((3.7f64.sinh() * 4.0).sin() + 2.5 * 2.5).asin(),
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
    test("0/0", f64::NAN)?;
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
