use parse::EvilParseError;
mod parse;
mod expression;
mod util;
use expression::eval_expr;


pub fn eval_str(text: &str) -> Result<f32, EvilParseError> {
    let exp = parse::parse(text)?;
    Ok(eval_expr(&exp))
}

#[cfg(test)]
mod tests {

    use crate::{eval_str};

    pub fn assert_float_eq(f1: f32, f2: f32) {
        if (f1 - f2).abs() >= 1e-5 {
            panic!("Floats not almost equal.\nf1: {}\nf2: {}\n", f1, f2);
        }
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
        assert_float_eq(eval_str(&"3-(-1+sin(cos(-3.14159265358979))*2)").unwrap(), 5.6829419696157935);
        assert_float_eq(eval_str(&"-(-1+((-3.14159265358979)/5)*2)").unwrap(), 2.256637061435916);
        assert_float_eq(eval_str(&"-(-1+(sin(-3.14159265358979)/5)*2)").unwrap(), 1.0);
        assert_float_eq(eval_str(&"-(-1+sin(cos(-3.14159265358979)/5)*2)").unwrap(), 1.3973386615901224);
        assert_float_eq(eval_str(&"-cos(3.14159265358979)").unwrap(), 1.0);
        assert_float_eq(eval_str(&"1+sin(-cos(-3.14159265358979))").unwrap(), 1.8414709848078965);
        assert_float_eq(eval_str(&"-1+sin(-cos(-3.14159265358979))").unwrap(), -0.1585290151921035);
        assert_float_eq(eval_str(&"-(-1+sin(-cos(-3.14159265358979)/5)*2)").unwrap(), 0.6026613384098776);
        assert_float_eq(eval_str(&"sin(-(2))*2").unwrap(), -1.8185948536513634);
        assert_float_eq(eval_str(&"sin(sin(2))*2").unwrap(), 1.5781446871457767);
        assert_float_eq(eval_str(&"sin(-(sin(2)))*2").unwrap(), -1.5781446871457767);
        assert_float_eq(eval_str(&"-sin(2)*2").unwrap(), -1.8185948536513634);
        assert_float_eq(eval_str(&"sin(-sin(2))*2").unwrap(), -1.5781446871457767);
        assert_float_eq(eval_str(&"sin(-sin(2)^2)*2").unwrap(), 1.4715655294841483);
        assert_float_eq(eval_str(&"sin(-sin(2)*-sin(2))*2").unwrap(), 1.4715655294841483);
        assert_float_eq(eval_str(&"--(1)").unwrap(), 1.0);
        assert_float_eq(eval_str(&"--1").unwrap(), 1.0);
        assert_float_eq(eval_str(&"----1").unwrap(), 1.0);
        assert_float_eq(eval_str(&"---1").unwrap(), -1.0);
        assert_float_eq(eval_str(&"3-(4-2/3+(1-2*2))").unwrap(), 2.666666666666666);
        assert_float_eq(eval_str(&"log(log(2))*tan(2)+exp(1.5)").unwrap(), 5.2825344122094045);
    }

    #[test]
    fn test_error_handling() {
        assert!(eval_str(&"").is_err());
        assert!(eval_str(&"5+5-(").is_err());
        assert!(eval_str(&")2*(5+5)*3-2)*2").is_err());
        assert!(eval_str(&"2*(5+5))").is_err());
    }
}
