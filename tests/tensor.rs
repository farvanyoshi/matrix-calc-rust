use matrix_calc::{Matrix, Tensor, TensorError, Vector};
#[test]
fn rank_zero_empty_shapes_and_validation() {
    let scalar = Tensor::scalar(7);
    assert_eq!(scalar.rank(), 0);
    assert_eq!(scalar.shape(), &[]);
    assert_eq!(scalar.strides(), &[]);
    assert_eq!(scalar.get(&[]), Some(&7));
    assert!(scalar.get(&[0]).is_none());
    assert!(Tensor::<i32>::new(vec![], vec![]).is_err());
    assert!(Tensor::new(vec![1], vec![2]).is_err());
    assert!(matches!(
        Tensor::<u8>::new(vec![], vec![usize::MAX, 2]),
        Err(TensorError::Overflow)
    ));
    let empty = Tensor::<u8>::new(vec![], vec![0, usize::MAX, 2]).unwrap();
    assert_eq!(empty.len(), 0);
    assert!(empty.is_contiguous());
    assert_eq!(empty.iter().count(), 0);
    assert_eq!(empty.strides(), &[0, 0, 0]);
    assert!(empty.get(&[0, 0, 0]).is_none());
    assert!(empty.as_slice().unwrap().is_empty());
    let empty = Tensor::<i32>::new(vec![], vec![2, 0, 3]).unwrap();
    assert_eq!(empty.reshape(vec![0]).unwrap().shape(), &[0]);
    assert!(scalar.transpose(&[0]).is_err());
    assert!(scalar.slice(0, 0, 0, 1).is_err());
}
#[test]
fn strides_views_reshape_and_copy_on_write() {
    let a = Tensor::new((0..24).collect(), vec![2, 3, 4]).unwrap();
    assert_eq!(a.strides(), &[12, 4, 1]);
    assert_eq!(a.get(&[1, 2, 3]), Some(&23));
    let t = a.transpose(&[2, 0, 1]).unwrap();
    assert_eq!(t.shape(), &[4, 2, 3]);
    assert_eq!(t.strides(), &[1, 12, 4]);
    assert!(!t.is_contiguous());
    for i in 0..4 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(t.get(&[i, j, k]), a.get(&[j, k, i]));
            }
        }
    }
    assert!(t.reshape(vec![24]).is_err());
    let c = t.make_contiguous();
    assert!(c.is_contiguous());
    assert_eq!(c.to_vec(), t.to_vec());
    assert_eq!(c.reshape(vec![24]).unwrap().to_vec(), t.to_vec());
    let sliced = a.slice(2, 1, 4, 2).unwrap();
    assert_eq!(sliced.shape(), &[2, 3, 2]);
    assert_eq!(sliced.get(&[1, 2, 1]), Some(&23));
    assert_eq!(
        sliced.to_vec(),
        vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    );
    let mut detached = sliced.clone();
    detached.set(&[0, 0, 0], 99).unwrap();
    assert_eq!(detached[[0, 0, 0]], 99);
    assert_eq!(sliced[[0, 0, 0]], 1);
    assert_eq!(a[[0, 0, 1]], 1);
    assert!(detached.set(&[2, 0, 0], 1).is_err());
    assert!(a.transpose(&[0, 0, 1]).is_err());
    assert!(a.slice(1, 0, 4, 1).is_err());
    assert!(a.slice(1, 0, 3, 0).is_err());
    assert!(a.slice(1, 2, 1, 1).is_err());
    let offset = a.slice(0, 1, 2, 1).unwrap();
    assert!(offset.is_contiguous());
    assert_eq!(
        offset.reshape(vec![12]).unwrap().as_slice().unwrap(),
        &(12..24).collect::<Vec<_>>()
    );
    let empty = a.slice(0, 1, 1, 1).unwrap();
    assert_eq!(empty.shape(), &[0, 3, 4]);
    assert_eq!(empty.iter().count(), 0);
}
#[test]
fn broadcasting_aligns_axes_and_respects_zero_dimensions() {
    let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    let b = Tensor::new(vec![10, 20, 30], vec![3]).unwrap();
    assert_eq!(
        a.try_add(&b).unwrap().to_vec(),
        vec![11, 22, 33, 14, 25, 36]
    );
    assert_eq!(
        a.try_hadamard(&Tensor::new(vec![2, 3], vec![2, 1]).unwrap())
            .unwrap()
            .to_vec(),
        vec![2, 4, 6, 12, 15, 18]
    );
    assert_eq!(
        a.try_sub(&Tensor::scalar(1)).unwrap().to_vec(),
        vec![0, 1, 2, 3, 4, 5]
    );
    let x = Tensor::new(vec![1, 2, 3], vec![1, 3, 1]).unwrap();
    let y = Tensor::new(vec![10, 20], vec![1, 1, 2]).unwrap();
    let z = x.try_add(&y).unwrap();
    assert_eq!(z.shape(), &[1, 3, 2]);
    assert_eq!(z.to_vec(), vec![11, 21, 12, 22, 13, 23]);
    assert!(
        a.try_add(&Tensor::new(vec![1, 2], vec![2]).unwrap())
            .is_err()
    );
    let empty = Tensor::<i32>::new(vec![], vec![0, 3]).unwrap();
    let result = empty.try_add(&b).unwrap();
    assert_eq!(result.shape(), &[0, 3]);
    assert!(result.is_empty());
    assert_eq!(
        Tensor::<i32>::broadcast_shape(&[0, 3], &[1, 3]).unwrap(),
        vec![0, 3]
    );
    assert!(Tensor::<i32>::broadcast_shape(&[0, 3], &[2, 3]).is_err());
    assert!(a.try_element_div(&Tensor::scalar(0)).is_err());
    assert!(a.scalar_div(0).is_err());
    assert_eq!(a.scalar_div(1).unwrap(), a);
    assert_eq!((&a + &b).to_vec(), vec![11, 22, 33, 14, 25, 36]);
    assert_eq!((&a * &Tensor::scalar(2)).to_vec(), vec![2, 4, 6, 8, 10, 12]);
}
#[test]
fn conversion_and_pretty_print_do_not_flatten_away_shape() {
    let m = Matrix::from([[1, 2], [3, 4]]);
    let t = m.to_tensor();
    assert_eq!(t.to_matrix().unwrap(), m);
    assert_eq!(Tensor::from(m.clone()).shape(), &[2, 2]);
    assert_eq!(
        t.transpose(&[1, 0]).unwrap().to_matrix().unwrap(),
        m.transpose()
    );
    assert!(t.to_vector().is_err());
    assert!(Tensor::scalar(1).to_matrix().is_err());
    let v = Vector::from([1, 2]);
    assert_eq!(v.to_tensor().to_vector().unwrap(), v);
    let pretty = format!("{t}");
    assert!(pretty.contains("[2, 2]"));
    assert!(pretty.contains("[[1, 2], [3, 4]]"));
    assert_eq!(format!("{}", Tensor::scalar(2)), "Tensor[] 2");
    let t = Tensor::new(vec![String::from("a"), String::from("bc")], vec![2]).unwrap();
    assert_eq!(t.map(String::len).to_vec(), vec![1, 2]);
}
#[test]
fn seeded_random_is_reproducible() {
    let a = Tensor::random(vec![2, 3, 4], 42).unwrap();
    assert_eq!(a, Tensor::random(vec![2, 3, 4], 42).unwrap());
    assert_ne!(a, Tensor::random(vec![2, 3, 4], 43).unwrap());
    assert!(a.iter().all(|x| *x >= 0. && *x < 1.));
    assert_eq!(Tensor::random(vec![], 42).unwrap().len(), 1);
    assert_eq!(Tensor::random(vec![0, 2], 42).unwrap().len(), 0);
    assert!(matches!(
        Tensor::random(vec![usize::MAX], 42),
        Err(TensorError::Allocation)
    ));
}
#[test]
fn generated_broadcast_and_noncontiguous_indices_match_independent_loops() {
    for m in 0..5 {
        for n in 0..5 {
            for p in 1..4 {
                let a =
                    Tensor::new((0..m * n * p).map(|x| x as i32).collect(), vec![m, n, p]).unwrap();
                let b = Tensor::new((0..p).map(|x| x as i32 + 1).collect(), vec![p]).unwrap();
                let out = a.try_add(&b).unwrap();
                for i in 0..m {
                    for j in 0..n {
                        for k in 0..p {
                            assert_eq!(out[[i, j, k]], ((i * n + j) * p + k) as i32 + k as i32 + 1);
                        }
                    }
                }
                let t = a.transpose(&[2, 0, 1]).unwrap();
                assert_eq!(t.transpose(&[1, 2, 0]).unwrap(), a);
                let broadcast = t.try_add(&Tensor::scalar(9)).unwrap();
                for k in 0..p {
                    for i in 0..m {
                        for j in 0..n {
                            assert_eq!(broadcast[[k, i, j]], a[[i, j, k]] + 9);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn display_summarizes_huge_empty_or_deep_shapes_without_recursive_overflow() {
    let empty = Tensor::<i32>::new(vec![], vec![usize::MAX, 0]).unwrap();
    assert!(format!("{empty}").ends_with(" []"));
    let deep = Tensor::new(vec![7], vec![1; 1000]).unwrap();
    assert!(format!("{deep}").ends_with(" [7]"));
    let large = Tensor::new((0..1000).collect(), vec![10, 10, 10]).unwrap();
    assert!(format!("{large}").contains('…'));
}
