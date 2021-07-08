use crate::expression::{Expression, Node};
use crate::operators::{make_default_operators, BinOp, Operator};
use crate::util::apply_unary_ops;
use itertools::Itertools;
use num::Float;
use regex::{Regex, RegexSet};
use std::error::Error;
use std::fmt;
use std::iter::once;
use std::str::FromStr;

/// This will be thrown at you if the parsing went wrong. Ok, obviously it is not an 
/// exception, so thrown needs to be understood figuratively.
#[derive(Debug)]
pub struct ExParseError {
    pub msg: String,
}
impl fmt::Display for ExParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
impl Error for ExParseError {}

#[derive(Debug, PartialEq, Eq)]
enum Paren {
    Open,
    Close,
}

#[derive(Debug, PartialEq, Eq)]
enum ParsedToken<'a, T: Float + FromStr> {
    Num(T),
    Paren(Paren),
    Op(Operator<'a, T>),
    Var(String),
}

/// Parses tokens of a text with regexes and returns them as a vector
///
/// # Arguments
///
/// * `text` - text to be parsed
/// * `ops_in` - vector of operator-pairs
///
fn apply_regexes<'a, T: Float + FromStr>(text: &str, ops_in: Vec<Operator<'a, T>>) -> Vec<ParsedToken<'a, T>>
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let regex_escapes_ops = r"\|?^*+.";

    // We sort operators inverse alphabetically such that log2 has higher priority than log (wlog :D).
    let mut ops_tmp = ops_in;
    ops_tmp.sort_by(|o1, o2| o2.repr.partial_cmp(o1.repr).unwrap());
    let ops = ops_tmp; // from now on const

    let pattern_ops = ops
        .iter()
        .map(|op| {
            let mut s_tmp = op.repr.to_string();
            for c in regex_escapes_ops.chars() {
                s_tmp = s_tmp.replace(c, format!("\\{}", c).as_str());
            }
            s_tmp
        })
        .collect::<Vec<_>>()
        .join("|");
    let pattern_nums = r"\.?[0-9]+(\.[0-9]+)?";
    let pattern_var = r"\{[a-zA-Z_]+[a-zA-Z_0-9]*\}";
    let pattern_parens = r"\(|\)";
    let patterns = [
        pattern_var,
        pattern_ops.as_str(),
        pattern_nums,
        pattern_parens,
    ];
    let pattern_any = patterns.join("|");
    let any = Regex::new(pattern_any.as_str()).unwrap();

    let which_one = RegexSet::new(&patterns).unwrap();

    any.captures_iter(text)
        .map(|c| c[0].to_string())
        .map(|elt_string| {
            let elt_str = elt_string.as_str();
            let matches = which_one.matches(elt_str);
            if matches.matched(0) {
                ParsedToken::<T>::Var(elt_str[1..elt_str.len() - 1].to_string())
            } else if matches.matched(1) {
                let wrapped_op = ops.iter().find(|op| op.repr == elt_str);
                ParsedToken::<T>::Op(match wrapped_op {
                    Some(op) => *op,
                    None => {
                        panic!("Could not find operator {}.", elt_str);
                    }
                })
            } else if matches.matched(2) {
                ParsedToken::<T>::Num(elt_str.parse::<T>().unwrap())
            } else if matches.matched(3) {
                let c = elt_str.chars().next().unwrap();
                ParsedToken::<T>::Paren(if c == '(' {
                    Paren::Open
                } else if c == ')' {
                    Paren::Close
                } else {
                    panic!("Paren {} is neither ( nor ). Check the paren-regex.", c);
                })
            } else {
                panic!("Internal regex mismatch!");
            }
        })
        .collect()
}

/// Returns an expression that is created recursively and can be evaluated
///
/// # Arguments
///
/// * `parsed_tokens` - parsed tokens created with [`apply_regexes`]
/// * `parsed_vars` - elements of `parsed_tokens` that are variables
/// * `unary_ops` - unary operators of the expression to be build
///
fn make_expression<T>(
    parsed_tokens: &[ParsedToken<T>],
    parsed_vars: &[String],
    unary_ops: Vec<fn(T) -> T>,
) -> Result<(Expression<T>, usize), ExParseError>
where
    T: Float + FromStr + std::fmt::Debug,
{
    fn unpack_binop<S>(bo: Option<BinOp<S>>) -> Result<BinOp<S>, ExParseError>
    where
        S: Float + FromStr + std::fmt::Debug,
    {
        match bo {
            Some(bo) => Ok(bo),
            None => Err(ExParseError {
                msg: "Expected binary operator but there was None.".to_string(),
            }),
        }
    }

    let find_var_index = |name: &str| {
        let idx = parsed_vars
            .iter()
            .enumerate()
            .find(|(_, n)| n.as_str() == name);
        match idx {
            Some((i, _)) => i,
            None => {
                panic!("I don't know variable {}", name)
            }
        }
    };
    // this closure handles the case that a token is a unary operator and accesses the
    // variable 'tokens' from the outer scope
    let process_unary = |i: usize, uo| {
        // gather subsequent unary operators from the beginning
        let uops = once(uo)
            .chain(
                (i + 1..parsed_tokens.len())
                    .map(|j| match parsed_tokens[j] {
                        ParsedToken::Op(op) => op.unary_op,
                        _ => None,
                    })
                    .take_while(|uo_| uo_.is_some())
                    .flatten(),
            )
            .collect::<Vec<_>>();
        let n_uops = uops.len();

        match &parsed_tokens[i + n_uops] {
            ParsedToken::Paren(p) => match p {
                Paren::Close => Err(ExParseError {
                    msg: "I do not understand a closing parenthesis after an operator.".to_string(),
                }),
                Paren::Open => {
                    let (expr, i_forward) =
                        make_expression::<T>(&parsed_tokens[i + n_uops + 1..], &parsed_vars, uops)?;
                    Ok((Node::Expr(expr), i_forward + n_uops + 1))
                }
            },
            ParsedToken::Var(name) => {
                let expr = Expression {
                    nodes: vec![Node::Var(find_var_index(&name))],
                    bin_ops: vec![],
                    unary_ops: uops,
                };
                Ok((Node::Expr(expr), n_uops + 1))
            }
            ParsedToken::Num(n) => Ok((Node::Num(apply_unary_ops(&uops, *n)), n_uops + 1)),
            ParsedToken::Op(_) => Err(ExParseError {
                msg: "A unary operator cannot be followed by a binary operator.".to_string(),
            }),
        }
    };

    let mut result = Expression::<T> {
        bin_ops: Vec::<BinOp<T>>::new(),
        nodes: Vec::<Node<T>>::new(),
        unary_ops,
    };

    // The main loop checks one token after the next whereby sub-expressions are
    // handled recursively. Thereby, the token-position-index idx_tkn is increased
    // according to the length of the sub-expression.
    let mut idx_tkn: usize = 0;
    while idx_tkn < parsed_tokens.len() {
        match &parsed_tokens[idx_tkn] {
            ParsedToken::Op(b) => match b.unary_op {
                None => {
                    result.bin_ops.push(unpack_binop(b.bin_op)?);
                    idx_tkn += 1;
                }
                Some(uo) => {
                    // might the operator be unary?
                    if idx_tkn == 0 {
                        // if the first element is an operator it must be unary
                        let (node, idx_forward) = process_unary(idx_tkn, uo)?;
                        result.nodes.push(node);
                        idx_tkn += idx_forward;
                    } else {
                        // decide type of operator based on predecessor
                        match &parsed_tokens[idx_tkn - 1] {
                            ParsedToken::Num(_) | ParsedToken::Var(_) => {
                                // number or variable as predecessor means binary operator
                                result.bin_ops.push(unpack_binop(b.bin_op)?);
                                idx_tkn += 1;
                            }
                            ParsedToken::Paren(p) => match p {
                                Paren::Open => {
                                    let msg = "Opening parenthesis next to operator must not occur here."
                                        .to_string();
                                    return Err(ExParseError { msg: msg });
                                }
                                Paren::Close => {
                                    result.bin_ops.push(unpack_binop(b.bin_op)?);
                                    idx_tkn += 1;
                                }
                            },
                            ParsedToken::Op(_) => {
                                let (node, idx_forward) = process_unary(idx_tkn, uo)?;
                                result.nodes.push(node);
                                idx_tkn += idx_forward;
                            }
                        }
                    }
                }
            },
            ParsedToken::Num(n) => {
                result.nodes.push(Node::Num(*n));
                idx_tkn += 1;
            }
            ParsedToken::Var(name) => {
                result.nodes.push(Node::Var(find_var_index(&name)));
                idx_tkn += 1;
            }
            ParsedToken::Paren(p) => match p {
                Paren::Open => {
                    idx_tkn += 1;
                    let (expr, i_forward) =
                        make_expression::<T>(&parsed_tokens[idx_tkn..], &parsed_vars, vec![])?;
                    result.nodes.push(Node::Expr(expr));
                    idx_tkn += i_forward;
                }
                Paren::Close => {
                    idx_tkn += 1;
                    break;
                }
            },
        }
    }
    Ok((result, idx_tkn))
}

/// Tries to give useful error messages for invalid constellations of the parsed tokens
///
/// # Arguments
///
/// * `parsed_tokens` - parsed tokens
///
fn check_preconditions<T>(parsed_tokens: &[ParsedToken<T>]) -> Result<u8, ExParseError>
where
    T: Float + FromStr + std::fmt::Debug,
{
    if parsed_tokens.len() == 0 {
        return Err(ExParseError {
            msg: "Cannot parse empty string.".to_string(),
        });
    };
    let num_pred_succ = |idx: usize, forbidden: Paren| match &parsed_tokens[idx] {
        ParsedToken::Num(_) => Err(ExParseError {
            msg: "A number/variable cannot be next to a number/variable.".to_string(),
        }),
        ParsedToken::Paren(p) => {
            if p == &forbidden {
                Err(ExParseError {
                    msg: "Wlog, a number/variable cannot be on the right of a closing parenthesis."
                        .to_string(),
                })
            } else {
                Ok(0)
            }
        }
        _ => Ok(0),
    };
    let binop_pred_succ = |idx: usize| match parsed_tokens[idx] {
        ParsedToken::Op(op) => {
            if op.unary_op == None {
                Err(ExParseError {
                    msg: "A binary operator cannot be next to a binary operator.".to_string(),
                })
            } else {
                Ok(0)
            }
        }
        _ => Ok(0),
    };
    let paren_pred_succ = |idx: usize, forbidden: Paren| match &parsed_tokens[idx] {
        ParsedToken::Paren(p) => {
            if p == &forbidden {
                Err(ExParseError {
                    msg: "Wlog an opening paren cannot be next to a closing paren.".to_string(),
                })
            } else {
                Ok(0)
            }
        }
        _ => Ok(0),
    };
    let mut open_paren_cnt = 0i8;
    parsed_tokens
        .iter()
        .enumerate()
        .map(|(i, expr_elt)| -> Result<usize, ExParseError> {
            match expr_elt {
                ParsedToken::Num(_) | ParsedToken::Var(_) => {
                    if i < parsed_tokens.len() - 1 {
                        num_pred_succ(i + 1, Paren::Open)?;
                    }
                    if i > 0 {
                        num_pred_succ(i - 1, Paren::Close)?;
                    }
                    Ok(0)
                }
                ParsedToken::Paren(p) => {
                    if i < parsed_tokens.len() - 1 {
                        match p {
                            Paren::Open => paren_pred_succ(i + 1, Paren::Close)?,
                            Paren::Close => paren_pred_succ(i + 1, Paren::Open)?,
                        };
                    }
                    open_paren_cnt += match p {
                        Paren::Close => -1,
                        Paren::Open => 1,
                    };
                    if open_paren_cnt < 0 {
                        return Err(ExParseError {
                            msg: format!("To many closing parentheses until position {}.", i)
                                .to_string(),
                        });
                    }
                    Ok(0)
                }
                ParsedToken::Op(_) => {
                    if i < parsed_tokens.len() - 1 {
                        binop_pred_succ(i + 1)?;
                        Ok(0)
                    } else {
                        Err(ExParseError {
                            msg: "The last element cannot be an operator.".to_string(),
                        })
                    }
                }
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if open_paren_cnt != 0 {
        Err(ExParseError {
            msg: "Parentheses mismatch.".to_string(),
        })
    } else {
        Ok(0)
    }
}

/// Parses a string and a vector of operators into an expression that can be evaluated
pub fn parse<'a, T>(text: &str, ops: Vec<Operator<'a, T>>) -> Result<Expression<T>, ExParseError>
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
    T: Float + FromStr + std::fmt::Debug,
{
    let parsed_tokens = apply_regexes::<T>(text, ops);
    let parsed_vars = parsed_tokens
        .iter()
        .filter_map(|pt| match pt {
            ParsedToken::Var(name) => Some(name.clone()),
            _ => None,
        })
        .unique()
        .collect::<Vec<_>>();
    check_preconditions(&parsed_tokens[..])?;
    let (expr, _) = make_expression(&parsed_tokens[0..], &parsed_vars, vec![])?;
    Ok(expr)
}

/// Parses a string into an expression that can be evaluated using default operators
pub fn parse_with_default_ops<T>(text: &str) -> Result<Expression<T>, ExParseError>
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
    T: Float + FromStr + std::fmt::Debug,
{
    let ops = make_default_operators::<T>();
    Ok(parse(&text, ops)?)
}

#[cfg(test)]
mod tests {
    use crate::parse::{apply_regexes, check_preconditions, make_default_operators};

    #[test]
    fn test_preconditions() {
        fn test(text: &str, msg_part: &str) {
            let ops = make_default_operators::<f32>();
            let elts = apply_regexes::<f32>(text, ops);
            let err = check_preconditions(&elts[..]);
            match err {
                Ok(_) => assert!(false),
                Err(e) => {
                    println!("{}", e.msg);
                    assert!(e.msg.contains(msg_part));
                }
            }
        }

        test("", "empty string.");
        test("++", "The last element cannot be an operator.");
        test(
            "{12} (",
            "Wlog, a number/variable cannot be on the right of a closing paren",
        );
        test("++)", "closing parentheses until");
        test(")12-(1+1) / (", "closing parentheses until position");
        test("12-()+(", "Wlog an opening paren");
        test("12-() ())", "Wlog an opening paren");
        test("12-(3-4)*2+ (1/2))", "closing parentheses until");
        test("12-(3-4)*2+ ((1/2)", "Parentheses mismatch.");
    }
}
