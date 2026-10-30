//! Stateful tensor calculator with a small, non-executing expression grammar.
//! Shape requirements are checked by the tensor/matrix APIs before assignment.
use crate::{Matrix, Tensor, Vector};
use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, BufRead, Write};

/// An expression syntax, name, shape, or numerical error. Failed assignments do
/// not replace the existing variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplError(
    /// Human-readable diagnostic produced by parsing or library validation.
    pub String,
);
impl fmt::Display for ReplError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ReplError {}
type Result<T> = std::result::Result<T, ReplError>;
fn error(e: impl fmt::Display) -> ReplError {
    ReplError(e.to_string())
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Number(f64),
    Name(String),
    Mark(char),
}
fn tokenize(line: &str) -> Result<Vec<Token>> {
    let bytes = line.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if b.is_ascii_digit() || b == b'.' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'.' {
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                i += 1;
                if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                    i += 1;
                }
                let exponent = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if exponent == i {
                    return Err(error("missing exponent digits"));
                }
            }
            let n = line[start..i].parse::<f64>().map_err(error)?;
            if !n.is_finite() {
                return Err(error("number must be finite"));
            }
            out.push(Token::Number(n));
        } else if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            i += 1;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.')
            {
                i += 1;
            }
            out.push(Token::Name(line[start..i].to_owned()));
        } else if b"[]()+-*/,=".contains(&b) {
            out.push(Token::Mark(b as char));
            i += 1;
        } else {
            return Err(error(format!("unexpected character at byte {i}")));
        }
    }
    if out.len() > 4096 {
        return Err(error("input exceeds 4096 tokens"));
    }
    Ok(out)
}
#[derive(Clone, Debug)]
enum Expr {
    Number(f64),
    Name(String),
    Array(Vec<Expr>),
    Call(String, Vec<Expr>),
    Neg(Box<Expr>),
    Binary(char, Box<Expr>, Box<Expr>),
}
struct Parser {
    tokens: Vec<Token>,
    at: usize,
    depth: usize,
}
impl Parser {
    fn mark(&mut self, c: char) -> bool {
        if self.tokens.get(self.at) == Some(&Token::Mark(c)) {
            self.at += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, c: char) -> Result<()> {
        if self.mark(c) {
            Ok(())
        } else {
            Err(error(format!("expected '{c}'")))
        }
    }
    fn expression(&mut self) -> Result<Expr> {
        let mut value = self.term()?;
        loop {
            let op = if self.mark('+') {
                Some('+')
            } else if self.mark('-') {
                Some('-')
            } else {
                None
            };
            if let Some(op) = op {
                value = Expr::Binary(op, Box::new(value), Box::new(self.term()?));
            } else {
                break;
            }
        }
        Ok(value)
    }
    fn term(&mut self) -> Result<Expr> {
        let mut value = self.atom()?;
        loop {
            let op = if self.mark('*') {
                Some('*')
            } else if self.mark('/') {
                Some('/')
            } else {
                None
            };
            if let Some(op) = op {
                value = Expr::Binary(op, Box::new(value), Box::new(self.atom()?));
            } else {
                break;
            }
        }
        Ok(value)
    }
    fn atom(&mut self) -> Result<Expr> {
        self.depth += 1;
        if self.depth > 128 {
            return Err(error("expression nesting exceeds 128"));
        }
        let result = self.atom_inner();
        self.depth -= 1;
        result
    }
    fn atom_inner(&mut self) -> Result<Expr> {
        if self.mark('-') {
            return Ok(Expr::Neg(Box::new(self.atom()?)));
        }
        if self.mark('+') {
            return self.atom();
        }
        if self.mark('(') {
            let value = self.expression()?;
            self.expect(')')?;
            return Ok(value);
        }
        if self.mark('[') {
            let mut values = Vec::new();
            if !self.mark(']') {
                loop {
                    values.push(self.expression()?);
                    if self.mark(']') {
                        break;
                    }
                    self.expect(',')?;
                }
            }
            return Ok(Expr::Array(values));
        }
        match self.tokens.get(self.at).cloned() {
            Some(Token::Number(n)) => {
                self.at += 1;
                Ok(Expr::Number(n))
            }
            Some(Token::Name(name)) => {
                self.at += 1;
                if !self.mark('(') {
                    return Ok(Expr::Name(name));
                }
                let mut args = Vec::new();
                if !self.mark(')') {
                    loop {
                        args.push(self.expression()?);
                        if self.mark(')') {
                            break;
                        }
                        self.expect(',')?;
                    }
                }
                Ok(Expr::Call(name, args))
            }
            _ => Err(error("expected number, variable, array, or function call")),
        }
    }
}
fn scalar(n: f64) -> Tensor<f64> {
    Tensor::new(vec![n], vec![]).expect("scalar shape")
}
fn integer(t: &Tensor<f64>) -> Result<usize> {
    if t.rank() != 0 {
        return Err(error("expected a scalar integer"));
    }
    let n = t.to_vec()[0];
    if !n.is_finite() || n < 0. || n.fract() != 0. || n >= usize::MAX as f64 {
        return Err(error("expected a nonnegative representable integer"));
    }
    Ok(n as usize)
}
fn dimensions(t: &Tensor<f64>) -> Result<Vec<usize>> {
    if t.rank() != 1 {
        return Err(error("shape/axes must be a vector of integers"));
    }
    t.to_vec()
        .into_iter()
        .map(|n| integer(&scalar(n)))
        .collect()
}
fn count(shape: &[usize]) -> Result<usize> {
    if shape.iter().any(|&n| n > 10_000_000) {
        return Err(error("interactive dimension limit is 10000000"));
    }
    shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d).ok_or_else(|| error("shape overflow"))
    })
}
fn output_guard(a: &Tensor<f64>, b: &Tensor<f64>, matmul: bool) -> Result<()> {
    let shape = if matmul && a.rank() == 2 && b.rank() == 2 {
        vec![a.shape()[0], b.shape()[1]]
    } else {
        let rank = a.rank().max(b.rank());
        let mut shape = vec![1; rank];
        for (i, dim) in shape.iter_mut().enumerate() {
            let ad = if i + a.rank() >= rank {
                a.shape()[i + a.rank() - rank]
            } else {
                1
            };
            let bd = if i + b.rank() >= rank {
                b.shape()[i + b.rank() - rank]
            } else {
                1
            };
            *dim = if ad == 1 { bd } else { ad };
            if ad != bd && ad != 1 && bd != 1 {
                return Err(error("incompatible broadcasting shapes"));
            }
        }
        shape
    };
    if count(&shape)? > 10_000_000 {
        return Err(error("interactive allocation limit is 10000000 elements"));
    }
    Ok(())
}
fn arity(args: &[Tensor<f64>], n: usize) -> Result<()> {
    if args.len() == n {
        Ok(())
    } else {
        Err(error(format!("expected {n} arguments, got {}", args.len())))
    }
}

/// Persistent numeric variables and expression evaluation. Runtime values are
/// float64 tensors: scalar rank 0, vector rank 1, matrix rank 2, or n-D.
#[derive(Default)]
pub struct Session {
    variables: BTreeMap<String, Tensor<f64>>,
}
impl Session {
    /// Create an empty session. Shape requirements: none.
    pub fn new() -> Self {
        Self::default()
    }
    /// Inspect a stored tensor without copying. Shape requirements: none.
    pub fn get(&self, name: &str) -> Option<&Tensor<f64>> {
        self.variables.get(name)
    }
    /// Evaluate one expression/assignment or `help`, `vars`, `clear`. Returns
    /// formatted output, or None for blank input. Array siblings must have equal
    /// shapes. `*` is rank-2 matmul and otherwise broadcasting multiplication.
    /// Shape requirements: rectangular array siblings and operands compatible with each called operation.
    pub fn eval(&mut self, line: &str) -> Result<Option<String>> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(None);
        }
        if line.len() > 65536 {
            return Err(error("input exceeds 65536 bytes"));
        }
        match line {
            "help" => return Ok(Some(Self::help().to_owned())),
            "vars" => {
                return Ok(Some(
                    self.variables
                        .iter()
                        .map(|(n, v)| format!("{n}: {:?}", v.shape()))
                        .collect::<Vec<_>>()
                        .join("\n"),
                ));
            }
            "clear" => {
                self.variables.clear();
                return Ok(Some("variables cleared".to_owned()));
            }
            _ => {}
        }
        let tokens = tokenize(line)?;
        let mut parser = Parser {
            tokens,
            at: 0,
            depth: 0,
        };
        let target = if let (Some(Token::Name(name)), Some(Token::Mark('='))) =
            (parser.tokens.first(), parser.tokens.get(1))
        {
            if name.contains('.') {
                return Err(error("assignment requires a plain variable name"));
            }
            let name = name.clone();
            parser.at = 2;
            Some(name)
        } else {
            None
        };
        let expr = parser.expression()?;
        if parser.at != parser.tokens.len() {
            return Err(error("unexpected trailing input"));
        }
        let value = self.evaluate(&expr)?;
        let text = value.to_string();
        if let Some(name) = target {
            self.variables.insert(name.clone(), value);
            Ok(Some(format!("{name} = {text}")))
        } else {
            Ok(Some(text))
        }
    }
    /// Grammar and available functions. Shape requirements: documented in text.
    pub fn help() -> &'static str {
        "Tensor calculator: name = expression; + - * /, parentheses, nested arrays.\n* multiplies rank-2 matrices; otherwise it broadcasts elementwise. Use hadamard(A,B) for explicit elementwise products.\nFunctions: matrix(array), vector(array), tensor(data,shape), random(rows,cols[,seed]), random(shape[,seed]), zeros(shape), ones(shape), eye(n), shape(A), transpose(A[,axes]), reshape(A,shape), slice(A,axis,start,stop[,step]), dot(a,b), cross(a,b), norm1(a), norm2(a), sum(A).\nMatrix.random(3,3) aliases random(3,3). vars, clear, help, quit/exit."
    }
    fn evaluate(&self, expr: &Expr) -> Result<Tensor<f64>> {
        self.evaluate_at(expr, 0)
    }
    fn evaluate_at(&self, expr: &Expr, depth: usize) -> Result<Tensor<f64>> {
        if depth > 256 {
            return Err(error("evaluation nesting exceeds 256"));
        }
        match expr {
            Expr::Number(n) => Ok(scalar(*n)),
            Expr::Name(n) => self
                .variables
                .get(n)
                .cloned()
                .ok_or_else(|| error(format!("unknown variable '{n}'"))),
            Expr::Neg(x) => Ok(self.evaluate_at(x, depth + 1)?.scalar_mul(-1.)),
            Expr::Array(items) => {
                if items.is_empty() {
                    return Tensor::new(vec![], vec![0]).map_err(error);
                }
                let values = items
                    .iter()
                    .map(|x| self.evaluate_at(x, depth + 1))
                    .collect::<Result<Vec<_>>>()?;
                let child = values[0].shape().to_vec();
                if values.iter().any(|v| v.shape() != child) {
                    return Err(error("ragged array: sibling shapes differ"));
                }
                let mut shape = vec![values.len()];
                shape.extend(child);
                if count(&shape)? > 10_000_000 {
                    return Err(error("interactive allocation limit is 10000000 elements"));
                }
                Tensor::new(values.into_iter().flat_map(|v| v.to_vec()).collect(), shape)
                    .map_err(error)
            }
            Expr::Binary(op, a, b) => {
                let a = self.evaluate_at(a, depth + 1)?;
                let b = self.evaluate_at(b, depth + 1)?;
                output_guard(&a, &b, *op == '*')?;
                match op {
                    '+' => a.try_add(&b).map_err(error),
                    '-' => a.try_sub(&b).map_err(error),
                    '*' if a.rank() == 2 && b.rank() == 2 => Ok(Tensor::from(
                        a.to_matrix()
                            .map_err(error)?
                            .try_matmul(&b.to_matrix().map_err(error)?)
                            .map_err(error)?,
                    )),
                    '*' => a.try_hadamard(&b).map_err(error),
                    '/' => a.try_element_div(&b).map_err(error),
                    _ => unreachable!(),
                }
            }
            Expr::Call(name, args) => {
                let args = args
                    .iter()
                    .map(|a| self.evaluate_at(a, depth + 1))
                    .collect::<Result<Vec<_>>>()?;
                self.call(name, &args)
            }
        }
    }
    fn call(&self, name: &str, args: &[Tensor<f64>]) -> Result<Tensor<f64>> {
        let function = name.rsplit('.').next().unwrap_or(name).to_ascii_lowercase();
        match function.as_str() {
            "matrix" | "vector" => {
                arity(args, 1)?;
                let rank = if function == "matrix" { 2 } else { 1 };
                if args[0].rank() != rank {
                    return Err(error(format!("{function} requires rank {rank}")));
                }
                Ok(args[0].clone())
            }
            "tensor" => {
                if args.len() == 1 {
                    return Ok(args[0].clone());
                }
                arity(args, 2)?;
                let shape = dimensions(&args[1])?;
                if count(&shape)? > 10_000_000 {
                    return Err(error("interactive allocation limit exceeded"));
                }
                Tensor::new(args[0].to_vec(), shape).map_err(error)
            }
            "random" | "zeros" | "ones" => {
                if args.is_empty() {
                    return Err(error("constructor requires dimensions"));
                }
                let (shape, seed) = if args[0].rank() == 1 {
                    if args.len() > 2 {
                        return Err(error("expected shape and optional seed"));
                    }
                    (
                        dimensions(&args[0])?,
                        if args.len() == 2 {
                            integer(&args[1])? as u64
                        } else {
                            1
                        },
                    )
                } else {
                    if !(2..=3).contains(&args.len()) {
                        return Err(error("expected rows, cols, and optional seed"));
                    }
                    (
                        vec![integer(&args[0])?, integer(&args[1])?],
                        if args.len() == 3 {
                            integer(&args[2])? as u64
                        } else {
                            1
                        },
                    )
                };
                let n = count(&shape)?;
                if n > 10_000_000 {
                    return Err(error("interactive allocation limit is 10000000 elements"));
                }
                if function == "random" {
                    Tensor::<f64>::random(shape, seed).map_err(error)
                } else {
                    Tensor::new(vec![if function == "ones" { 1. } else { 0. }; n], shape)
                        .map_err(error)
                }
            }
            "eye" | "identity" => {
                arity(args, 1)?;
                let n = integer(&args[0])?;
                if count(&[n, n])? > 10_000_000 {
                    return Err(error("interactive allocation limit exceeded"));
                }
                Ok(Tensor::from(Matrix::<f64>::try_identity(n).map_err(error)?))
            }
            "shape" => {
                arity(args, 1)?;
                Tensor::new(
                    args[0].shape().iter().map(|&n| n as f64).collect(),
                    vec![args[0].rank()],
                )
                .map_err(error)
            }
            "transpose" => {
                if !(1..=2).contains(&args.len()) {
                    return Err(error("transpose requires tensor and optional axes"));
                }
                let axes = if args.len() == 2 {
                    dimensions(&args[1])?
                } else {
                    (0..args[0].rank()).rev().collect()
                };
                args[0].transpose(&axes).map_err(error)
            }
            "reshape" => {
                arity(args, 2)?;
                let shape = dimensions(&args[1])?;
                if count(&shape)? > 10_000_000 {
                    return Err(error("interactive allocation limit exceeded"));
                }
                args[0].make_contiguous().reshape(shape).map_err(error)
            }
            "slice" => {
                if !(4..=5).contains(&args.len()) {
                    return Err(error("slice(A,axis,start,stop[,step])"));
                }
                args[0]
                    .slice(
                        integer(&args[1])?,
                        integer(&args[2])?,
                        integer(&args[3])?,
                        if args.len() == 5 {
                            integer(&args[4])?
                        } else {
                            1
                        },
                    )
                    .map_err(error)
            }
            "hadamard" => {
                arity(args, 2)?;
                output_guard(&args[0], &args[1], false)?;
                args[0].try_hadamard(&args[1]).map_err(error)
            }
            "dot" | "cross" => {
                arity(args, 2)?;
                if args[0].rank() != 1 || args[1].rank() != 1 || args[0].shape() != args[1].shape()
                {
                    return Err(error("dot/cross require equally sized rank-1 vectors"));
                }
                let a = Vector::new(args[0].to_vec());
                let b = Vector::new(args[1].to_vec());
                if function == "dot" {
                    Ok(scalar(a.try_dot(&b).map_err(error)?))
                } else {
                    Ok(a.cross(&b).map_err(error)?.to_tensor())
                }
            }
            "norm1" | "norm2" => {
                arity(args, 1)?;
                if args[0].rank() != 1 {
                    return Err(error("vector norm requires rank 1"));
                }
                let v = Vector::new(args[0].to_vec());
                Ok(scalar(if function == "norm1" {
                    v.norm_l1()
                } else {
                    v.norm_l2()
                }))
            }
            "sum" => {
                arity(args, 1)?;
                let m = Matrix::new(args[0].to_vec(), 1, args[0].len());
                Ok(scalar(m.sum_compensated()))
            }
            _ => Err(error(format!("unknown function '{name}'"))),
        }
    }
}
/// Run a line-oriented session. Shape requirements follow `Session::eval`.
/// Errors are printed and the next line is still read; EOF and quit terminate.
/// Shape requirements: expressions follow Session::eval; the input/output streams have no dimensions.
pub fn run<R: BufRead, W: Write>(mut input: R, mut output: W, prompt: bool) -> io::Result<()> {
    let mut session = Session::new();
    if prompt {
        writeln!(output, "Tensor calculator. Type help or quit.")?;
    }
    loop {
        if prompt {
            write!(output, "> ")?;
            output.flush()?;
        }
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            break;
        }
        if matches!(line.trim(), "quit" | "exit") {
            break;
        }
        match session.eval(&line) {
            Ok(Some(value)) => writeln!(output, "{value}")?,
            Ok(None) => {}
            Err(e) => writeln!(output, "error: {e}")?,
        }
    }
    Ok(())
}
