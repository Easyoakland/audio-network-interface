use reed_solomon_erasure::galois_8::ReedSolomon;
use reed_solomon_erasure::shards;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // or use the following for Galois 2^16 backend
    // use reed_solomon_erasure::galois_16::ReedSolomon;

    let r = ReedSolomon::new(3, 2).unwrap(); // 3 data shards, 2 parity shards

    let mut master_copy = shards!(
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [0, 0, 0, 0], // last 2 rows are parity shards
        [0, 0, 0, 0]
    );

    // Construct the parity shards
    r.encode(&mut master_copy).unwrap();

    println!("{:?}", &master_copy);

    // Make a copy and transform it into option shards arrangement
    // for feeding into reconstruct_shards
    let mut shards: Vec<_> = master_copy.iter().cloned().map(Some).collect();

    shards[0] = Some(vec![0, 1, 2, 4]);
    println!("{:?}", &shards);
    dbg!(r
        .verify(
            &(shards
                .iter()
                .cloned()
                .map(Option::unwrap)
                .collect::<Vec<_>>()),
        )
        .unwrap());

    // We can remove up to 2 shards, which may be data or parity shards
    shards[0] = None;
    shards[4] = None;

    // Try to reconstruct missing shards
    r.reconstruct(&mut shards).unwrap();

    println!("{:?}", &shards);

    // Convert back to normal shard arrangement
    let result: Vec<_> = shards.into_iter().flatten().collect();

    assert!(r.verify(&result).unwrap());
    assert_eq!(master_copy, result);

    Ok(())
}
