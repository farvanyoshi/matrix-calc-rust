#![cfg(feature = "serde")]
use matrix_calc::Matrix;
#[test]
fn validated_round_trips() {
    for a in [
        Matrix::from([[1, 2], [3, 4]]),
        Matrix::zeros(2, 0),
        Matrix::zeros(0, 3),
    ] {
        let text = serde_json::to_string(&a).unwrap();
        let restored: Matrix<i32> = serde_json::from_str(&text).unwrap();
        assert_eq!(a, restored);
    }
    for text in [
        r#"{"rows":2,"cols":2,"data":[1]}"#,
        r#"{"rows":18446744073709551615,"cols":2,"data":[]}"#,
        r#"{"rows":0,"cols":0,"data":[],"typo":1}"#,
    ] {
        assert!(serde_json::from_str::<Matrix<i32>>(text).is_err());
    }
}
