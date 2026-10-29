use matrix_calc::Matrix;
#[test]
fn shape_views_and_mutation() {
    let a = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    assert_eq!(a.transpose(), Matrix::from([[1, 4], [2, 5], [3, 6]]));
    assert_eq!(a.clone().into_transpose().into_transpose(), a);
    let s = Matrix::from([[String::from("a"), String::from("b")]]).into_transpose();
    assert_eq!(s.shape(), (2, 1));
    assert_eq!(
        a.slice(0..2, 1..3).unwrap().transpose().to_owned(),
        Matrix::from([[2, 5], [3, 6]])
    );
    let v = a.view().transpose();
    assert_eq!(v.try_matmul(&a).unwrap(), a.transpose().matmul(&a));
    let mut b = a.clone();
    {
        let mut v = b.slice_mut(0..2, 1..3).unwrap().transpose();
        v[(1, 0)] = 9;
        v.reborrow().slice(0..1, 0..2).unwrap().fill(7);
    }
    assert_eq!(b, Matrix::from([[1, 7, 9], [4, 7, 6]]));
    assert!(a.slice(2..1, 0..0).is_err());
    let mut z = Matrix::<i32>::zeros(2, 0);
    assert_eq!(z.view().transpose().shape(), (0, 2));
    assert_eq!(z.slice_mut(2..2, 0..0).unwrap().shape(), (0, 0));
    let mut r = a.clone();
    r.try_reshape(3, 2).unwrap();
    assert_eq!(r.data(), a.data());
    assert!(r.try_reshape(1, 2).is_err());
    assert_eq!(r.shape(), (3, 2));
}
#[test]
fn blocks_vectors_reductions_and_powers() {
    let mut a = Matrix::from([[1, 2], [3, 4]]);
    a.swap_cols(0, 1).unwrap();
    a.swap_rows(0, 1).unwrap();
    assert_eq!(a, Matrix::from([[4, 3], [2, 1]]));
    a.set_row(1, &[8, 9]).unwrap();
    a.set_col(0, &[6, 7]).unwrap();
    assert_eq!(a, Matrix::from([[6, 3], [7, 9]]));
    a.insert_block(1, 1, &Matrix::from([[10]])).unwrap();
    assert_eq!(a[(1, 1)], 10);
    let before = a.clone();
    assert!(a.insert_block(2, 2, &Matrix::from([[1]])).is_err());
    assert_eq!(a, before);
    assert_eq!(a.hcat(&a).unwrap().shape(), (2, 4));
    assert_eq!(a.vcat(&a).unwrap().shape(), (4, 2));
    assert_eq!(a.diagonal(), vec![6, 10]);
    assert_eq!(a.trace().unwrap(), 16);
    let a = Matrix::from([[1, 2], [3, 4]]);
    assert_eq!(a.sum(), 10);
    assert_eq!(a.product(), 24);
    assert_eq!(a.row_sums(), Matrix::from([[3], [7]]));
    assert_eq!(a.col_products(), Matrix::from([[3, 8]]));
    assert_eq!(a.argmax().unwrap(), (1, 1));
    assert_eq!(a.min().unwrap(), 1);
    assert_eq!(a.pow(0).unwrap(), Matrix::identity(2));
    assert_eq!(a.pow(3).unwrap(), a.matmul(&a).matmul(&a));
    let x = Matrix::from([[1, 2, 3]]);
    let y = Matrix::from([[4], [5], [6]]);
    assert_eq!(x.dot(&y).unwrap(), 32);
    assert_eq!(
        x.outer(&y).unwrap(),
        Matrix::from([[4, 5, 6], [8, 10, 12], [12, 15, 18]])
    );
    let z = Matrix::<i32>::zeros(2, 0);
    assert_eq!(z.row_sums(), Matrix::zeros(2, 1));
    assert_eq!(z.row_products(), Matrix::ones(2, 1));
    assert!(z.min().is_err());
}
