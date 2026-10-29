# Migrating from 0.1 to 0.2

- `try_map` now maps callbacks returning `Result<U,E>` and preserves their error
  type. Use `map` for ordinary callbacks, or `try_map(|x| Ok::<_, E>(...))`.
- `map`, `try_map`, and zip mapping accept `FnMut`.
- `element_div` / `try_element_div` are canonical names; `el_div` / `try_el_div`
  remain compatibility aliases.
- `try_scalar_mul` remains a compatibility wrapper. `scalar_mul` is ordinary
  scalar multiplication; `checked_scalar_mul` checks primitive integer overflow.
- Constructors retain argument order `(input, rows, cols)` and special-matrix
  order `(rows, cols)`. `try_from_vec` takes ownership without copying.
- All elementwise operations require identical dimensions. Unequal shapes no
  longer truncate or accept equal-length storage with different dimensions.
- Out-of-range columns are rejected rather than aliasing the next row.
- `is_empty` returns true for zero element count, irrespective of logical shape.
- Row iteration has an opaque iterator type supporting exact length and reverse
  iteration, including zero-width rows. Code naming `slice::Chunks` must adapt.
- Arithmetic permits cloned elements where Copy was previously required.
  Borrowed and owned operands and unary negation are supported.
- Both scalar and elementwise division reject zero, including floating signed
  zero and an empty matrix divided by zero. Ordinary integer overflow semantics
  remain ordinary; use checked operations for recoverable overflow errors.
- `ShapeError` and `NumericError` have additional variants. Update exhaustive
  matches. Error messages avoid arithmetic that could itself overflow.
- The placeholder greeting binary is removed; this release is a library.
- Optional serde deserialization validates private-field invariants.
