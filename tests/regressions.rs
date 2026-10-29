use matrix_calc::{Matrix, MatrixError};
use matrix_calc::error::{ShapeError, NumericError};
#[test]
fn coordinates_and_empty_shapes() {
    let mut m = Matrix::new([1,2,3,4],2,2);
    assert!(!m.is_empty());
    for (r,c) in [(0,2),(2,0),(usize::MAX,1),(1,usize::MAX)] {
        assert!(m.get(r,c).is_none()); assert!(m.get_mut(r,c).is_none());
    }
    assert_eq!(m.data(), &[1,2,3,4]);
    for (r,c) in [(0,0),(2,0),(0,3)] {
        let z=Matrix::<i32>::zeros(r,c); assert!(z.is_empty());
        assert_eq!(z.iter_rows().count(),r);
    }
    assert_eq!(Matrix::<i32>::zeros(2,0).matmul(&Matrix::<i32>::zeros(0,3)), Matrix::zeros(2,3));
}
#[test]
fn invalid_shapes_and_division() {
    let m=Matrix::new([1i32,2,3,4],2,2);
    for n in [Matrix::<i32>::ones(1,4),Matrix::ones(2,3),Matrix::ones(1,2)] {
        assert!(matches!(m.try_add(&n),Err(MatrixError::ShapeErr(ShapeError::Dimension{..}))));
        assert!(m.try_sub(&n).is_err()); assert!(m.try_hadamard(&n).is_err());
        assert!(m.try_el_div(&n).is_err());
    }
    assert!(matches!(m.try_scalar_div(0),Err(MatrixError::NumErr(NumericError::DivideByZero))));
    assert!(Matrix::<i32>::zeros(0,0).try_scalar_div(0).is_err());
    assert!(matches!(Matrix::<i32>::try_new([],usize::MAX,2),Err(MatrixError::ShapeErr(ShapeError::Overflow{..}))));
}
