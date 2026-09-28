//! Authoritative CPython vectors for the standalone `py_random` module.

use kaggriculture_engine::py_random::{PyRandom, RandomError};

struct Vector {
    seed: &'static str,
    state_words: &'static [u32],
    random_bits: &'static [u64],
    bits: &'static [&'static str],
    ranges: &'static [u64],
    choices: &'static [usize],
    shuffle1: &'static [usize],
    shuffle2: &'static [usize],
}

// Generated directly with this repository's `.venv/bin/python`: for each decimal
// seed, construct a fresh `random.Random(int(seed))` per operation; serialize float
// results with `struct.unpack(">Q", struct.pack(">d", value))`, arbitrary bit results
// with `hex()`, and the remaining integer/list results directly.
// Python: 3.11.15 (main, Mar  3 2026, 15:47:15) [Clang 21.1.4 ]
// Resolved interpreter SHA-256: eecb5f80bb58cf0411c91d17437d923fc95de6e9fe0d070e86f9a4280cc72bae
// (`_random` is built into that interpreter, rather than a separately hashable module.)
// Lib/random.py SHA-256: b931a2d452db89ab9e1f79d59b66ce194dbd6748049cfe44699726da52b9c956
const BIT_WIDTHS: &[usize] = &[
    0, 1, 2, 5, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 257,
];
const RANGE_BOUNDS: &[u64] = &[
    1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 255, 256, 257, 65535, 65536, 65537, 1000000000, 4294967296,
    4294967297,
];
const CHOICE_LENGTHS: &[usize] = &[1, 2, 3, 5, 8, 16, 17, 31, 32, 33];
const ROLLOVER_INDICES: &[usize] = &[0, 1, 622, 623, 624, 625, 1246, 1247, 1248, 1249, 9999];
const ROLLOVER_VECTORS: &[(&str, &[u32])] = &[
    (
        "0",
        &[
            3626764237, 1654615998, 1691392275, 2390040247, 2229104038, 1244770883, 2232670494,
            577331751, 2465233080, 2246525520, 3292398474,
        ],
    ),
    (
        "1",
        &[
            577090037, 2444712010, 1405491179, 802355090, 1360367077, 3404757168, 3227136675,
            1721233950, 1032912167, 497515921, 586364410,
        ],
    ),
    (
        "-1",
        &[
            577090037, 2444712010, 1405491179, 802355090, 1360367077, 3404757168, 3227136675,
            1721233950, 1032912167, 497515921, 586364410,
        ],
    ),
    (
        "-9223372036854775808",
        &[
            2377109768, 1691753470, 1281902390, 3239629188, 346131666, 1859507527, 3995760549,
            3126662580, 3566289627, 4274427590, 2798221440,
        ],
    ),
    (
        "18446744073709551617",
        &[
            437050517, 3681013637, 441468705, 476122439, 2658536893, 1419036053, 3381334523,
            42808965, 2505169973, 1726303071, 4161686695,
        ],
    ),
    (
        "-1606938044258990275541962092341162602522202993782792835313721",
        &[
            632829219, 3476216445, 3361582370, 342211395, 1594797085, 3811294247, 755428290,
            1099820769, 1268351431, 710329047, 32975495,
        ],
    ),
    (
        "6864797660130609714981900799081393217269435300143305409394463459185543183397656052122559640661454554977296311391480858037121987999716643812574028291115057151",
        &[
            2072548960, 36195863, 4016140458, 542502175, 182176742, 820514609, 2625485163,
            3073082194, 1022877396, 1735002255, 4148466512,
        ],
    ),
];
const TEN_THOUSAND_WORD_FNV1A: &[(&str, u64)] = &[
    ("0", 0xaf74_b5c0_c79f_9e21),
    ("1", 0xec4c_b8d9_5469_9654),
    ("-1", 0xec4c_b8d9_5469_9654),
    ("-9223372036854775808", 0x51a7_98ea_b8af_f04e),
    ("18446744073709551617", 0xe931_8070_e637_4042),
    (
        "-1606938044258990275541962092341162602522202993782792835313721",
        0x4a53_29f5_2634_4d2e,
    ),
    (
        "6864797660130609714981900799081393217269435300143305409394463459185543183397656052122559640661454554977296311391480858037121987999716643812574028291115057151",
        0xc361_3906_3760_c51b,
    ),
];
const VECTORS: &[Vector] = &[
    Vector {
        seed: "0",
        state_words: &[
            3626764237_u32,
            1654615998_u32,
            3255389356_u32,
            3823568514_u32,
            1806341205_u32,
            173879092_u32,
            1112038970_u32,
            4146640122_u32,
        ],
        random_bits: &[
            4605781095417019838_u64,
            4605002265878567962_u64,
            4601247963976755576_u64,
            4598335849438463398_u64,
            4602780372834555655_u64,
            4600966264840031036_u64,
        ],
        bits: &[
            "0x0",
            "0x1",
            "0x1",
            "0x18",
            "0x71",
            "0x6b",
            "0x52e",
            "0x4248",
            "0x7b945a7d",
            "0x82e2e662",
            "0x7c65c1e5",
            "0x6453831ceb1167b3",
            "0x4da5e709d4713d60",
            "0x7a024204f7c1bd87",
            "0xe87a1613e443df789558867f",
            "0x2413433923a7711a8133287637ebdcd9",
            "0x9e4d6e3c1846d424c17c627923c6612f",
            "0x1e8e5216afcbd04c340212ef7cca5a5a1",
            "0x119488dec4f65d4d9259f4329e6f4590b9a164106cf6a659eb4862b21fb97d435",
        ],
        ranges: &[
            0_u64,
            1_u64,
            0_u64,
            2_u64,
            4_u64,
            3_u64,
            6_u64,
            4_u64,
            7_u64,
            11_u64,
            6_u64,
            129_u64,
            71_u64,
            144_u64,
            9158_u64,
            12429_u64,
            32834_u64,
            976832602_u64,
            631194409_u64,
            1418186270_u64,
        ],
        choices: &[0, 1, 0, 2, 7, 12, 9, 30, 30, 22],
        shuffle1: &[10, 14, 5, 1, 9, 2, 3, 11, 13, 7, 8, 4, 0, 6, 15, 12],
        shuffle2: &[1, 11, 6, 4, 3, 2, 10, 8, 15, 13, 9, 12, 14, 0, 5, 7],
    },
    Vector {
        seed: "1",
        state_words: &[
            577090037_u32,
            2444712010_u32,
            3639700191_u32,
            3445702192_u32,
            3280387012_u32,
            271041745_u32,
            1095513148_u32,
            506456969_u32,
        ],
        random_bits: &[
            4594009002368267652_u64,
            4605808224069059832_u64,
            4605054689724112659_u64,
            4598266534995001180_u64,
            4602596585012360058_u64,
            4601768931085461634_u64,
        ],
        bits: &[
            "0x0",
            "0x0",
            "0x2",
            "0x1b",
            "0x66",
            "0xc3",
            "0x813",
            "0x414c",
            "0xf17f5c4",
            "0x7ed4d57b",
            "0xc2ce6f44",
            "0x5367660d78e51061",
            "0xc9e9c616612e7696",
            "0x18072e8c35bf992d",
            "0xd5f4b3b2e4b06ce60741c7a8",
            "0x619264c29b810e766ec9d28663ca828d",
            "0x7204e52db2221a58008a05a6c4647159",
            "0x13a902931cd447e35b8b6d8fe442e3d43",
            "0x1a648a7dd06839eb905b6e6e307d4bedc51431193e6c3f3391a2b8f1ff1fd42a2",
        ],
        ranges: &[
            0_u64,
            0_u64,
            1_u64,
            0_u64,
            3_u64,
            6_u64,
            7_u64,
            7_u64,
            10_u64,
            12_u64,
            6_u64,
            24_u64,
            249_u64,
            14_u64,
            58544_u64,
            51093_u64,
            56723_u64,
            652231581_u64,
            1912923437_u64,
            4059906722_u64,
        ],
        choices: &[0, 0, 1, 0, 7, 14, 15, 20, 24, 13],
        shuffle1: &[2, 10, 0, 14, 6, 5, 3, 8, 7, 11, 15, 1, 12, 13, 9, 4],
        shuffle2: &[0, 14, 1, 15, 7, 10, 5, 6, 8, 4, 11, 12, 3, 13, 9, 2],
    },
    Vector {
        seed: "-1",
        state_words: &[
            577090037_u32,
            2444712010_u32,
            3639700191_u32,
            3445702192_u32,
            3280387012_u32,
            271041745_u32,
            1095513148_u32,
            506456969_u32,
        ],
        random_bits: &[
            4594009002368267652_u64,
            4605808224069059832_u64,
            4605054689724112659_u64,
            4598266534995001180_u64,
            4602596585012360058_u64,
            4601768931085461634_u64,
        ],
        bits: &[
            "0x0",
            "0x0",
            "0x2",
            "0x1b",
            "0x66",
            "0xc3",
            "0x813",
            "0x414c",
            "0xf17f5c4",
            "0x7ed4d57b",
            "0xc2ce6f44",
            "0x5367660d78e51061",
            "0xc9e9c616612e7696",
            "0x18072e8c35bf992d",
            "0xd5f4b3b2e4b06ce60741c7a8",
            "0x619264c29b810e766ec9d28663ca828d",
            "0x7204e52db2221a58008a05a6c4647159",
            "0x13a902931cd447e35b8b6d8fe442e3d43",
            "0x1a648a7dd06839eb905b6e6e307d4bedc51431193e6c3f3391a2b8f1ff1fd42a2",
        ],
        ranges: &[
            0_u64,
            0_u64,
            1_u64,
            0_u64,
            3_u64,
            6_u64,
            7_u64,
            7_u64,
            10_u64,
            12_u64,
            6_u64,
            24_u64,
            249_u64,
            14_u64,
            58544_u64,
            51093_u64,
            56723_u64,
            652231581_u64,
            1912923437_u64,
            4059906722_u64,
        ],
        choices: &[0, 0, 1, 0, 7, 14, 15, 20, 24, 13],
        shuffle1: &[2, 10, 0, 14, 6, 5, 3, 8, 7, 11, 15, 1, 12, 13, 9, 4],
        shuffle2: &[0, 14, 1, 15, 7, 10, 5, 6, 8, 4, 11, 12, 3, 13, 9, 2],
    },
    Vector {
        seed: "-9223372036854775808",
        state_words: &[
            2377109768_u32,
            1691753470_u32,
            4039908392_u32,
            899083848_u32,
            1039357674_u32,
            4162741883_u32,
            3656249883_u32,
            2156693347_u32,
        ],
        random_bits: &[
            4603160380059113583_u64,
            4606647521506646969_u64,
            4597886784565798500_u64,
            4605842931276984861_u64,
            4590771714562803792_u64,
            4604966893295839929_u64,
        ],
        bits: &[
            "0x0",
            "0x1",
            "0x1",
            "0x1e",
            "0x1a",
            "0x3d",
            "0x7c0f",
            "0xd9ed",
            "0x404643b1",
            "0x15b28d35",
            "0x15b6a42be",
            "0x4066fcb8554aae48",
            "0x7559f9364e23b0e5",
            "0x1ab5f1254a6b4ea86",
            "0x6bec7753e34c7e730b42e7d",
            "0x2ddeb2b1d08810b017f58e26ab8859be",
            "0xc70d7ba3cf88823c8c1fc336f80f615b",
            "0x1404b97592f8744e6bb61f89d29d68682",
            "0xb13a80ede3ab0f866c193c8555e5f82e5d6f1df7670d8346a9e58962fdccdf92",
        ],
        ranges: &[
            0_u64,
            0_u64,
            0_u64,
            0_u64,
            2_u64,
            6_u64,
            5_u64,
            8_u64,
            4_u64,
            14_u64,
            6_u64,
            62_u64,
            13_u64,
            47_u64,
            53384_u64,
            46970_u64,
            21421_u64,
            785940007_u64,
            797394150_u64,
            2850392418_u64,
        ],
        choices: &[0, 0, 0, 4, 1, 11, 10, 16, 19, 29],
        shuffle1: &[6, 2, 0, 7, 11, 9, 13, 4, 15, 10, 5, 1, 8, 14, 3, 12],
        shuffle2: &[4, 10, 2, 3, 6, 7, 5, 9, 13, 11, 12, 0, 14, 8, 15, 1],
    },
    Vector {
        seed: "18446744073709551617",
        state_words: &[
            437050517_u32,
            3681013637_u32,
            3113036029_u32,
            1867347033_u32,
            1076864185_u32,
            3866848357_u32,
            2653509278_u32,
            2884483247_u32,
        ],
        random_bits: &[
            4591996911697590768_u64,
            4604703729247925713_u64,
            4598188315692489570_u64,
            4603740031816808162_u64,
            4583417296367973504_u64,
            4600011554035523364_u64,
        ],
        bits: &[
            "0x0",
            "0x0",
            "0x3",
            "0x17",
            "0x37",
            "0x40",
            "0x733d",
            "0x9e29",
            "0x55f6dc57",
            "0x6e46fd6",
            "0x7ec9e937",
            "0x44f2e9141f5be4b9",
            "0x9bb9a6e25ef5043b",
            "0x188d2f3ae2dd8510c",
            "0x801cf887d783a1d86a50c005",
            "0x3a66baf4ed3ec405ff91bac6dee2b1b8",
            "0x6bf9763b2cbafb1ac504dff1238cfd76",
            "0x179579682ac9af658728f71131968e4f6",
            "0xa36bfe74fc71f889c7e7ca4efc79555890129cdadd5b56c4fd4056a8657b9047",
        ],
        ranges: &[
            0_u64,
            1_u64,
            1_u64,
            0_u64,
            3_u64,
            2_u64,
            1_u64,
            8_u64,
            5_u64,
            5_u64,
            13_u64,
            215_u64,
            233_u64,
            71_u64,
            50436_u64,
            22901_u64,
            55282_u64,
            106576189_u64,
            529969575_u64,
            2914704993_u64,
        ],
        choices: &[0, 1, 1, 4, 0, 15, 11, 3, 23, 11],
        shuffle1: &[12, 15, 8, 1, 2, 14, 10, 5, 7, 0, 9, 4, 6, 11, 13, 3],
        shuffle2: &[14, 9, 1, 11, 12, 2, 7, 4, 0, 5, 15, 10, 3, 6, 8, 13],
    },
    Vector {
        seed: "-1606938044258990275541962092341162602522202993782792835313721",
        state_words: &[
            632829219_u32,
            3476216445_u32,
            2945453961_u32,
            1585401214_u32,
            3981780140_u32,
            3936536862_u32,
            2327137416_u32,
            2437764533_u32,
        ],
        random_bits: &[
            4594476576731770276_u64,
            4604352284216393013_u64,
            4606525617765780260_u64,
            4603055580452828502_u64,
            4597737616388352660_u64,
            4607015098842848292_u64,
        ],
        bits: &[
            "0x0",
            "0x0",
            "0x3",
            "0x15",
            "0x2f",
            "0xed",
            "0x7551",
            "0x8ab5",
            "0x48a6aada",
            "0x3ce40124",
            "0x13476a966",
            "0x4ec9bebf58e70936",
            "0xb782899990082e91",
            "0x10eac7d8e4497371a",
            "0x2dbbde45685274779a532baf",
            "0x38c6375816a572e3b39082f39c04f1d",
            "0x19d03f0508e835cc64b1691ebb125c93",
            "0xaafe616d99d98ab9dcfd5829bf8e2720",
            "0x4e7c3930dbce038b96df85c3dcc02bf9cb11934e2dd199cc0bcd27854bc16532",
        ],
        ranges: &[
            0_u64,
            1_u64,
            2_u64,
            1_u64,
            1_u64,
            2_u64,
            4_u64,
            0_u64,
            8_u64,
            13_u64,
            5_u64,
            57_u64,
            118_u64,
            14_u64,
            47890_u64,
            51554_u64,
            4560_u64,
            108269505_u64,
            941767782_u64,
            197994373_u64,
        ],
        choices: &[0, 1, 2, 4, 3, 6, 11, 19, 17, 3],
        shuffle1: &[11, 1, 7, 0, 2, 6, 15, 14, 13, 3, 9, 8, 5, 10, 12, 4],
        shuffle2: &[14, 2, 4, 9, 5, 3, 7, 10, 1, 8, 15, 11, 13, 12, 0, 6],
    },
    Vector {
        seed: "6864797660130609714981900799081393217269435300143305409394463459185543183397656052122559640661454554977296311391480858037121987999716643812574028291115057151",
        state_words: &[
            2072548960_u32,
            36195863_u32,
            3086341171_u32,
            4265590159_u32,
            275675485_u32,
            3589822997_u32,
            2504225322_u32,
            3368063154_u32,
        ],
        random_bits: &[
            4602364520312160880_u64,
            4604647746131525366_u64,
            4589289487783103296_u64,
            4603426960719413826_u64,
            4599614988071165176_u64,
            4588961746785067376_u64,
        ],
        bits: &[
            "0x0",
            "0x0",
            "0x0",
            "0x16",
            "0x7f",
            "0x10",
            "0x6afc",
            "0x9543",
            "0x64604859",
            "0x5475d8e6",
            "0x716b9f2f",
            "0xdc5bf45d3eb85c5",
            "0x9a7ef5a7c7725dd1",
            "0x87656ee959ac789b",
            "0xc015ef2a9a878f20c86bd023",
            "0x40b89c9fb40f32be539f7ea3ecf42644",
            "0x5030478a35a29a20fe136c499e53b892",
            "0x19608e1bbed8e08933b4d53fa77768efc",
            "0x1ec053171d718723fc0b1facb5525afaae6cb22654a0f1e437fcaec482cd16b0f",
        ],
        ranges: &[
            0_u64,
            0_u64,
            2_u64,
            0_u64,
            4_u64,
            6_u64,
            5_u64,
            7_u64,
            0_u64,
            3_u64,
            11_u64,
            135_u64,
            92_u64,
            167_u64,
            46095_u64,
            27461_u64,
            41056_u64,
            501064639_u64,
            751921935_u64,
            1894177445_u64,
        ],
        choices: &[0, 0, 2, 0, 5, 14, 1, 26, 6, 22],
        shuffle1: &[8, 3, 10, 13, 2, 4, 6, 12, 14, 7, 5, 9, 1, 11, 0, 15],
        shuffle2: &[14, 8, 3, 1, 5, 9, 15, 0, 2, 11, 10, 7, 13, 12, 4, 6],
    },
];

#[test]
fn integer_seed_and_mt19937_state_match_cpython() {
    for vector in VECTORS {
        let mut random = PyRandom::from_seed_decimal(vector.seed).unwrap();
        let actual: Vec<u32> = (0..vector.state_words.len())
            .map(|_| random.getrandbits_u64(32).unwrap() as u32)
            .collect();
        assert_eq!(actual, vector.state_words, "seed {}", vector.seed);
    }
}

#[test]
fn mt19937_twists_match_cpython_across_multiple_state_rollovers() {
    for (seed, expected) in ROLLOVER_VECTORS {
        let mut random = PyRandom::from_seed_decimal(seed).unwrap();
        let mut actual = Vec::with_capacity(expected.len());
        let last = *ROLLOVER_INDICES.last().unwrap();
        for index in 0..=last {
            let word = random.getrandbits_u64(32).unwrap() as u32;
            if ROLLOVER_INDICES.binary_search(&index).is_ok() {
                actual.push(word);
            }
        }
        assert_eq!(actual, *expected, "seed {seed}");
    }
}

#[test]
fn first_ten_thousand_tempered_words_match_cpython_stream_hashes() {
    for (seed, expected) in TEN_THOUSAND_WORD_FNV1A {
        let mut random = PyRandom::from_seed_decimal(seed).unwrap();
        let mut hash = 14_695_981_039_346_656_037_u64;
        for _ in 0..10_000 {
            let word = random.getrandbits_u64(32).unwrap() as u32;
            for byte in word.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(1_099_511_628_211);
            }
        }
        assert_eq!(hash, *expected, "seed {seed}");
    }
}

#[test]
fn random_53_bit_fractions_match_cpython_bit_for_bit() {
    for vector in VECTORS {
        let mut random = PyRandom::from_seed_decimal(vector.seed).unwrap();
        let actual: Vec<u64> = (0..vector.random_bits.len())
            .map(|_| random.random().to_bits())
            .collect();
        assert_eq!(actual, vector.random_bits, "seed {}", vector.seed);
    }
}

#[test]
fn getrandbits_matches_cpython_across_word_boundaries() {
    for vector in VECTORS {
        let mut random = PyRandom::from_seed_decimal(vector.seed).unwrap();
        let actual: Vec<String> = BIT_WIDTHS
            .iter()
            .map(|width| random.getrandbits(*width).to_hex_string())
            .collect();
        assert_eq!(actual, vector.bits, "seed {}", vector.seed);
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn randrange_rejection_sampling_matches_cpython() {
    for vector in VECTORS {
        let mut random = PyRandom::from_seed_decimal(vector.seed).unwrap();
        let actual: Vec<u64> = RANGE_BOUNDS
            .iter()
            .map(|bound| random.randrange(*bound as usize).unwrap() as u64)
            .collect();
        assert_eq!(actual, vector.ranges, "seed {}", vector.seed);
    }
}

#[test]
fn choice_and_two_successive_shuffles_match_cpython() {
    for vector in VECTORS {
        let mut random = PyRandom::from_seed_decimal(vector.seed).unwrap();
        let actual: Vec<usize> = CHOICE_LENGTHS
            .iter()
            .map(|length| {
                let values: Vec<usize> = (0..*length).collect();
                *random.choice(&values).unwrap()
            })
            .collect();
        assert_eq!(actual, vector.choices, "choice seed {}", vector.seed);

        let mut random = PyRandom::from_seed_decimal(vector.seed).unwrap();
        let mut shuffled: Vec<usize> = (0..16).collect();
        random.shuffle(&mut shuffled);
        assert_eq!(
            shuffled, vector.shuffle1,
            "first shuffle seed {}",
            vector.seed
        );
        random.shuffle(&mut shuffled);
        assert_eq!(
            shuffled, vector.shuffle2,
            "second shuffle seed {}",
            vector.seed
        );
    }
}

#[test]
fn primitive_and_word_seed_adapters_use_the_same_python_integer_path() {
    let cases = [
        (0_i64, VECTORS[0].state_words),
        (1_i64, VECTORS[1].state_words),
        (-1_i64, VECTORS[2].state_words),
        (i64::MIN, VECTORS[3].state_words),
    ];
    for (seed, expected) in cases {
        let mut random = PyRandom::from_seed_i64(seed);
        let actual: Vec<u32> = (0..expected.len())
            .map(|_| random.getrandbits_u64(32).unwrap() as u32)
            .collect();
        assert_eq!(actual, expected, "i64 seed {seed}");
    }

    let mut from_i128 = PyRandom::from_seed_i128((1_i128 << 64) + 1);
    let mut from_words = PyRandom::from_seed_words_le(&[1, 0, 1]);
    let i128_words: Vec<u32> = (0..8)
        .map(|_| from_i128.getrandbits_u64(32).unwrap() as u32)
        .collect();
    let explicit_words: Vec<u32> = (0..8)
        .map(|_| from_words.getrandbits_u64(32).unwrap() as u32)
        .collect();
    assert_eq!(i128_words, VECTORS[4].state_words);
    assert_eq!(explicit_words, VECTORS[4].state_words);

    let mut positive = PyRandom::from_seed_decimal("  +1 ").unwrap();
    let mut negative = PyRandom::from_seed_decimal(" -1 ").unwrap();
    assert_eq!(positive.getrandbits(257), negative.getrandbits(257));
}

#[test]
fn python_domain_edges_are_explicit_and_zero_bit_draws_consume_nothing() {
    assert!(PyRandom::from_seed_decimal("").is_err());
    assert!(PyRandom::from_seed_decimal("-").is_err());
    assert!(PyRandom::from_seed_decimal("12x").is_err());

    let mut random = PyRandom::from_seed_i64(7);
    assert!(random.getrandbits(0).words_le().is_empty());
    let after_zero = random.getrandbits_u64(32).unwrap();
    let direct = PyRandom::from_seed_i64(7).getrandbits_u64(32).unwrap();
    assert_eq!(after_zero, direct);

    assert_eq!(random.getrandbits_u64(65), Err(RandomError::BitsTooWide));
    assert_eq!(random.randrange(0), Err(RandomError::EmptyRange));
    assert_eq!(random.choice_index(0), Err(RandomError::EmptySequence));
    assert_eq!(random.choice::<u8>(&[]), Err(RandomError::EmptySequence));

    let mut empty: Vec<u8> = vec![];
    let mut singleton = vec![9_u8];
    random.shuffle(&mut empty);
    random.shuffle(&mut singleton);
    assert!(empty.is_empty());
    assert_eq!(singleton, [9]);
}
