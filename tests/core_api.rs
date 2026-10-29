use matrix_calc::{Matrix, MatrixError};
#[test]
fn storage_mapping_and_construction() {
    let m = Matrix::from([[String::from("a"), String::from("b")]]);
    let mut calls = 0;
    let lengths = m.map(|s| {
        calls += 1;
        s.len()
    });
    assert_eq!(calls, 2);
    assert_eq!(lengths.into_vec(), vec![1, 1]);
    let error = m.try_map(|_| Err::<(), _>("stop"));
    assert_eq!(error.unwrap_err(), "stop");
    assert!(Matrix::<i32>::try_from_rows(vec![vec![1], vec![]]).is_err());
    assert_eq!(
        Matrix::<i32>::try_from_rows(vec![]).unwrap().shape(),
        (0, 0)
    );
    assert!(Matrix::<i32>::try_new(std::iter::repeat(0), 1, 1).is_err());
    assert!(Matrix::<u8>::try_filled(usize::MAX, 1, 0).is_err());
    let mut m = Matrix::from_fn(2, 3, |r, c| r * 3 + c);
    for r in m.iter_rows_mut() {
        r.reverse();
    }
    assert_eq!(m.data(), &[2, 1, 0, 5, 4, 3]);
    *m.iter_col_mut(1).next_back().unwrap() = 9;
    assert_eq!(m.row(1), Some(&[5, 9, 3][..]));
    let mut z = Matrix::<u8>::zeros(3, 0);
    assert_eq!(z.iter_rows_mut().rev().count(), 3);
    assert!(matches!(m.try_get(9, 0), Err(MatrixError::IdxErr(_))));
    assert!(Matrix::from([[300i32]]).try_cast::<u8>().is_err());
    assert!(Matrix::from([[1.5f64]]).try_cast_exact::<i32>().is_err());
    assert_eq!(
        Matrix::from([[1.5f64]]).try_cast::<i32>().unwrap(),
        Matrix::from([[1]])
    );
}
#[test]
fn operators_and_assignment() {
    let a = Matrix::from([[1, 2], [3, 4]]);
    assert_eq!(a.clone() + &a, a.scalar_mul(2));
    assert_eq!(&a + a.clone(), a.scalar_mul(2));
    assert_eq!(-a.clone(), a.negate());
    assert_eq!(a.clone() * Matrix::<i32>::identity(2), a);
    let mut b = a.clone();
    b += &a;
    b -= &a;
    b *= 3;
    b /= 3;
    assert_eq!(b, a);
    assert_eq!(a.scalar_add(1).data(), &[2, 3, 4, 5]);
    assert_eq!(a.scalar_sub(1).data(), &[0, 1, 2, 3]);
    assert!(Matrix::<f64>::zeros(0, 0).try_scalar_div(-0.0).is_err());
}
