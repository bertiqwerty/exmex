use parse::ExParseError;
mod expression;
mod operators;
mod parse;
mod util;
use expression::eval_expr;

pub fn eval_str(text: &str) -> Result<f32, ExParseError> {
    let exp = parse::parse_with_default_ops(text)?;
    let empty_vec: Vec<f32> = vec![];
    Ok(eval_expr(&exp, &empty_vec))
}

#[cfg(test)]
mod tests {

    use std::iter::once;

    use crate::{
        eval_str,
        expression::eval_expr,
        operators::{make_default_operators, BinOp, OperatorPair},
        parse::parse,
        util::tests::assert_float_eq,
    };

    #[test]
    fn test_variables() {
        let operators = make_default_operators::<f32>();

        let to_be_parsed = "5*{x} + 4*{y} + 3*{x}";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq(eval_expr::<f32>(&expr, &[1.0, 0.0]), 8.0);

        let to_be_parsed = "5*{x} + 4*{y}";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq(eval_expr::<f32>(&expr, &[0.0, 1.0]), 4.0);

        let to_be_parsed = "5*{x} + 4*{y} + {x}^2";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq(eval_expr::<f32>(&expr, &[2.5, 3.7]), 33.55);
        assert_float_eq(eval_expr::<f32>(&expr, &[12.0, 9.3]), 241.2);

        let to_be_parsed = "2*(4*{x} + {y}^2)";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq(eval_expr::<f32>(&expr, &[2.0, 3.0]), 34.0);

        let to_be_parsed = "sin({myvar_25})";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq(eval_expr::<f32>(&expr, &[1.5707963267948966]), 1.0);

        let to_be_parsed = "((sin({myvar_25})))";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq(eval_expr::<f32>(&expr, &[1.5707963267948966]), 1.0);

        let to_be_parsed = "(0 * {myvar_25} + cos({X}))";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq(
            eval_expr::<f32>(&expr, &[1.5707963267948966, 3.141592653589793]),
            -1.0,
        );

        let to_be_parsed = "(-{X}^2)";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq(eval_expr::<f32>(&expr, &[1.0]), 1.0);

        let to_be_parsed = "log({x}) + 2* (-{x}^2 + sin(4*{y}))";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq(eval_expr::<f32>(&expr, &[2.5, 3.7]), 14.992794866624788);
    }

    #[test]
    fn test_custom_ops() {
        let custom_ops = vec![
            (
                "**",
                OperatorPair {
                    bin_op: Some(BinOp {
                        op: |a: f32, b| a.powf(b),
                        prio: 2,
                    }),
                    unary_op: None,
                },
            ),
            (
                "*",
                OperatorPair {
                    bin_op: Some(BinOp {
                        op: |a, b| a * b,
                        prio: 1,
                    }),
                    unary_op: None,
                },
            ),
            (
                "invert",
                OperatorPair {
                    bin_op: None,
                    unary_op: Some(|a: f32| 1.0 / a),
                },
            ),
        ];
        let expr = parse::<f32>("2**2*invert(3)", custom_ops).unwrap();
        let val = eval_expr::<f32>(&expr, &vec![]);
        assert_float_eq(val, 4.0 / 3.0);

        let new_op = (
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
            .chain(once(new_op))
            .collect::<Vec<_>>();
        let expr = parse::<f32>("2^2*1/({berti})", extended_operators).unwrap();
        let val = eval_expr::<f32>(&expr, &[4.0]);
        assert_float_eq(val, 1.0);
    
    }

    #[test]
    fn test_eval() {
        assert_float_eq(eval_str(&"2*3^2").unwrap(), 18.0);
        assert_float_eq(eval_str(&"-3^2").unwrap(), 9.0);
        assert_float_eq(eval_str(&"11.3").unwrap(), 11.3);
        assert_float_eq(eval_str(&"+11.3").unwrap(), 11.3);
        assert_float_eq(eval_str(&"-11.3").unwrap(), -11.3);
        assert_float_eq(eval_str(&"(-11.3)").unwrap(), -11.3);
        assert_float_eq(eval_str(&"11.3+0.7").unwrap(), 12.0);
        assert_float_eq(eval_str(&"31.3+0.7*2").unwrap(), 32.7);
        assert_float_eq(eval_str(&"1.3+0.7*2-1").unwrap(), 1.7);
        assert_float_eq(eval_str(&"1.3+0.7*2-1/10").unwrap(), 2.6);
        assert_float_eq(eval_str(&"(1.3+0.7)*2-1/10").unwrap(), 3.9);
        assert_float_eq(eval_str(&"1.3+(0.7*2)-1/10").unwrap(), 2.6);
        assert_float_eq(eval_str(&"1.3+0.7*(2-1)/10").unwrap(), 1.37);
        assert_float_eq(eval_str(&"1.3+0.7*(2-1/10)").unwrap(), 2.63);
        assert_float_eq(eval_str(&"-1*(1.3+0.7*(2-1/10))").unwrap(), -2.63);
        assert_float_eq(eval_str(&"-1*(1.3+(-0.7)*(2-1/10))").unwrap(), 0.03);
        assert_float_eq(eval_str(&"-1*((1.3+0.7)*(2-1/10))").unwrap(), -3.8);
        assert_float_eq(eval_str(&"sin(3.14159265358979)").unwrap(), 0.0);
        assert_float_eq(eval_str(&"0-sin(3.14159265358979 / 2)").unwrap(), -1.0);
        assert_float_eq(eval_str(&"-sin(3.14159265358979 / 2)").unwrap(), -1.0);
        assert_float_eq(eval_str(&"3-(-1+sin(1.5707963267948966)*2)").unwrap(), 2.0);
        assert_float_eq(
            eval_str(&"3-(-1+sin(cos(-3.14159265358979))*2)").unwrap(),
            5.6829419696157935,
        );
        assert_float_eq(
            eval_str(&"-(-1+((-3.14159265358979)/5)*2)").unwrap(),
            2.256637061435916,
        );
        assert_float_eq(
            eval_str(&"-(-1+(sin(-3.14159265358979)/5)*2)").unwrap(),
            1.0,
        );
        assert_float_eq(
            eval_str(&"-(-1+sin(cos(-3.14159265358979)/5)*2)").unwrap(),
            1.3973386615901224,
        );
        assert_float_eq(eval_str(&"-cos(3.14159265358979)").unwrap(), 1.0);
        assert_float_eq(
            eval_str(&"1+sin(-cos(-3.14159265358979))").unwrap(),
            1.8414709848078965,
        );
        assert_float_eq(
            eval_str(&"-1+sin(-cos(-3.14159265358979))").unwrap(),
            -0.1585290151921035,
        );
        assert_float_eq(
            eval_str(&"-(-1+sin(-cos(-3.14159265358979)/5)*2)").unwrap(),
            0.6026613384098776,
        );
        assert_float_eq(eval_str(&"sin(-(2))*2").unwrap(), -1.8185948536513634);
        assert_float_eq(eval_str(&"sin(sin(2))*2").unwrap(), 1.5781446871457767);
        assert_float_eq(eval_str(&"sin(-(sin(2)))*2").unwrap(), -1.5781446871457767);
        assert_float_eq(eval_str(&"-sin(2)*2").unwrap(), -1.8185948536513634);
        assert_float_eq(eval_str(&"sin(-sin(2))*2").unwrap(), -1.5781446871457767);
        assert_float_eq(eval_str(&"sin(-sin(2)^2)*2").unwrap(), 1.4715655294841483);
        assert_float_eq(
            eval_str(&"sin(-sin(2)*-sin(2))*2").unwrap(),
            1.4715655294841483,
        );
        assert_float_eq(eval_str(&"--(1)").unwrap(), 1.0);
        assert_float_eq(eval_str(&"--1").unwrap(), 1.0);
        assert_float_eq(eval_str(&"----1").unwrap(), 1.0);
        assert_float_eq(eval_str(&"---1").unwrap(), -1.0);
        assert_float_eq(eval_str(&"3-(4-2/3+(1-2*2))").unwrap(), 2.666666666666666);
        assert_float_eq(
            eval_str(&"log(log(2))*tan(2)+exp(1.5)").unwrap(),
            5.2825344122094045,
        );
        assert_float_eq(
            eval_str(&"log(log2(2))*tan(2)+exp(1.5)").unwrap(),
            4.4816890703380645,
        );
        assert_float_eq(eval_str(&"log2(2)").unwrap(), 1.0);
    }

    #[test]
    fn test_error_handling() {
        assert!(eval_str(&"").is_err());
        assert!(eval_str(&"5+5-(").is_err());
        assert!(eval_str(&")2*(5+5)*3-2)*2").is_err());
        assert!(eval_str(&"2*(5+5))").is_err());
    }
}
