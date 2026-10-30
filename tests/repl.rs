use matrix_calc::repl::{Session, run};
#[test]
fn persistent_variables_arrays_and_operations() {
    let mut s = Session::new();
    s.eval("A = [[1,2],[3,4]]").unwrap();
    s.eval("B = A * transpose(A)").unwrap();
    assert_eq!(s.get("B").unwrap().to_vec(), vec![5., 11., 11., 25.]);
    s.eval("C = A + [10,20]").unwrap();
    assert_eq!(s.get("C").unwrap().to_vec(), vec![11., 22., 13., 24.]);
    s.eval("r = Matrix.random(3,3,7)").unwrap();
    assert_eq!(s.get("r").unwrap().shape(), &[3, 3]);
    s.eval("v = cross([1,0,0],[0,1,0])").unwrap();
    assert_eq!(s.get("v").unwrap().to_vec(), vec![0., 0., 1.]);
    s.eval("n = norm2([3,4])").unwrap();
    assert_eq!(s.get("n").unwrap().to_vec(), vec![5.]);
    s.eval("x = reshape(tensor([1,2,3,4],[2,2]),[4])").unwrap();
    assert_eq!(s.get("x").unwrap().shape(), &[4]);
    s.eval("z = slice(A,1,0,2,2)").unwrap();
    assert_eq!(s.get("z").unwrap().to_vec(), vec![1., 3.]);
}
#[test]
fn errors_do_not_destroy_state_and_scripts_recover() {
    let mut s = Session::new();
    s.eval("x = 2 + 3 * 4").unwrap();
    assert_eq!(s.get("x").unwrap().to_vec(), vec![14.]);
    for line in [
        "x = [[1],[2,3]]",
        "x = missing",
        "x = ones([-1])",
        "x = 1e999",
        "x = reshape([1,2],[3])",
        "x = slice([1,2],0,0,2,0)",
        "x = 1 / 0",
        "x = (1+2",
        "x = [1] trailing",
        "x = 1; exit()",
    ] {
        assert!(s.eval(line).is_err(), "{line}");
        assert_eq!(s.get("x").unwrap().to_vec(), vec![14.]);
    }
    let mut out = Vec::new();
    run(
        &b"A = [1,2]\nbroken(\ndot(A,A)\nquit\nnever\n"[..],
        &mut out,
        false,
    )
    .unwrap();
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("error:"));
    assert!(out.contains('5'));
    assert!(!out.contains("never"));
    assert!(s.eval("vars").unwrap().unwrap().contains("x"));
    s.eval("clear").unwrap();
    assert!(s.get("x").is_none());
}
#[test]
fn resource_and_nesting_guards_are_recoverable() {
    let mut s = Session::new();
    for line in [
        vec!["1"; 300].join("+"),
        format!("{}1{}", "(".repeat(130), ")".repeat(130)),
        "zeros([4000,1]) + zeros([1,4000])".to_owned(),
        "hadamard(zeros([4000,1]),zeros([1,4000]))".to_owned(),
        "zeros([0,10000001])".to_owned(),
    ] {
        assert!(s.eval(&line).is_err());
    }
    s.eval("ok = 7").unwrap();
    assert_eq!(s.get("ok").unwrap().to_vec(), vec![7.]);
}
#[test]
fn nested_array_cannot_bypass_output_limit() {
    let mut session = Session::new();
    session.eval("a = zeros([100000])").unwrap();
    let input = format!("result = [{}]", vec!["a"; 101].join(","));
    assert!(
        session
            .eval(&input)
            .unwrap_err()
            .to_string()
            .contains("allocation limit")
    );
    assert!(session.get("result").is_none());
    assert!(session.eval("tensor([], [0,10000001])").is_err());
}
