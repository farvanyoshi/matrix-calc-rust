use matrix_calc::{Matrix, Tensor, Vector};
#[test]
fn direct_vector_arithmetic_and_conversion() {
    let a = Vector::from([1, 2, 3]);
    let b = Vector::from([4, 5, 6]);
    assert_eq!(a.try_dot(&b).unwrap(), 32);
    assert_eq!(a.dot(&b), 32);
    assert_eq!(a.cross(&b).unwrap(), Vector::from([-3, 6, -3]));
    assert_eq!(&a + &b, Vector::from([5, 7, 9]));
    assert_eq!(&b - &a, Vector::from([3, 3, 3]));
    assert_eq!(a.scalar_mul(3), Vector::from([3, 6, 9]));
    assert_eq!(-a.clone(), Vector::from([-1, -2, -3]));
    assert!(a.try_dot(&Vector::from([1])).is_err());
    assert!(Vector::from([1, 2]).cross(&Vector::from([3, 4])).is_err());
    assert_eq!(a.to_row_matrix(), Matrix::from([[1, 2, 3]]));
    assert_eq!(a.to_col_matrix(), Matrix::from([[1], [2], [3]]));
    assert_eq!(Vector::try_from(a.to_row_matrix()).unwrap(), a);
    assert!(Vector::try_from(Matrix::<i32>::zeros(2, 2)).is_err());
    let t: Tensor<_> = a.clone().into();
    assert_eq!(t.shape(), &[3]);
    assert_eq!(t.to_vector().unwrap(), a);
    assert_eq!(Vector::<i32>::new(vec![]).dot(&Vector::new(vec![])), 0);
    assert_eq!(
        Vector::try_from(Matrix::<i32>::zeros(0, 1)).unwrap().len(),
        0
    );
}
#[test]
fn vector_norms_are_stable_and_specialized() {
    let a = Vector::from([3f64, -4.]);
    assert_eq!(a.norm_l1(), 7.);
    assert!((a.norm_l2() - 5.).abs() < 1e-14);
    assert_eq!(a.norm_inf(), 4.);
    assert!((a.normalized().unwrap().norm_l2() - 1.).abs() < 1e-15);
    let large = Vector::from([1e300f64, 1e300]);
    assert!((large.norm_l2() / 1e300 - 2f64.sqrt()).abs() < 1e-14);
    assert!((large.normalized().unwrap().norm_l2() - 1.).abs() < 1e-14);
    assert!(Vector::from([f64::NAN]).normalized().is_err());
    assert!(Vector::from([0f64, 0.]).normalized().is_err());
    assert!(Vector::from([f64::NAN]).norm_l1().is_nan());
    assert!(Vector::from([f64::NAN]).norm_inf().is_nan());
    let empty = Vector::<f32>::new(vec![]);
    assert_eq!(empty.norm_l1(), 0.);
    assert_eq!(empty.norm_l2(), 0.);
}
#[test]
fn vectors_support_nonnumeric_and_mutation() {
    let mut v = Vector::from([String::from("hello"), String::from("world")]);
    assert_eq!(v.map(String::len), Vector::from([5, 5]));
    v[0] = String::from("new");
    v.as_mut_slice()[1].push('!');
    assert_eq!(v.get(0).unwrap(), "new");
    assert!(v.get(2).is_none());
    assert_eq!(v.into_col_matrix().shape(), (2, 1));
}
