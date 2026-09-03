use p_gpt_program::processor::shared::seeds_for;
use solana_pubkey::Pubkey;

#[test]
fn shard_seeds_match() {
    let program = Pubkey::new_from_array(p_gpt_interface::PROGRAM_ID);
    for k in 0..p_gpt_interface::SHARD_COUNT {
        let expected = Pubkey::find_program_address(&[b"shard", &[k as u8]], &program).0;
        let mut buf: [&'static [u8]; 2] = [&[], &[]];
        let seeds = seeds_for(6 + k, &mut buf).unwrap();
        println!("k={k} seeds={:?}", seeds.iter().map(|s| s.to_vec()).collect::<Vec<_>>());
        let got = pinocchio::Address::find_program_address(seeds, &p_gpt_program::ID).0;
        assert_eq!(got.as_array(), &expected.to_bytes(), "shard {k}");
    }
}
