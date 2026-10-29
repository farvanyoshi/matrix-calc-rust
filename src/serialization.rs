//! The serde representation is `{ rows, cols, data }`, with row-major data.
//! Deserialization always validates shape and overflow before returning a matrix.
use crate::Matrix;
impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for Matrix<T> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Storage<T> {
            rows: usize,
            cols: usize,
            data: Vec<T>,
        }
        let raw = <Storage<T> as serde::Deserialize>::deserialize(deserializer)?;
        Matrix::try_from_vec(raw.data, raw.rows, raw.cols).map_err(serde::de::Error::custom)
    }
}
