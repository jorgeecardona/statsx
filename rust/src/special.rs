//! Special functions: inverse normal CDF (D-6), log-gamma and digamma (D-10).
// The coefficients keep every digit printed in AS 241, so they can be checked against the paper.
#![allow(clippy::excessive_precision)]

/// Inverse of the standard normal CDF, Wichura (1988), "Algorithm AS 241: The percentage points
/// of the normal distribution", Applied Statistics 37(3):477-484, DOI 10.2307/2347330 (PPND16,
/// about 1e-16 relative accuracy). Decision D-6. Returns -inf at 0, +inf at 1, NaN outside.
pub fn norm_ppf(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q * poly(&A, r) / poly(&B, r);
    }
    let r = (-(p.min(1.0 - p)).ln()).sqrt();
    let x = if r <= 5.0 {
        let r = r - 1.6;
        poly(&C, r) / poly(&D, r)
    } else {
        let r = r - 5.0;
        poly(&E, r) / poly(&F, r)
    };
    if q < 0.0 {
        -x
    } else {
        x
    }
}

/// Horner evaluation, coefficients from the constant term up.
fn poly(c: &[f64; 8], x: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &k| acc * x + k)
}

const A: [f64; 8] = [
    3.387_132_872_796_366_608,
    1.331_416_678_917_843_774_5e2,
    1.971_590_950_306_551_442_7e3,
    1.373_169_376_550_946_112_5e4,
    4.592_195_393_154_987_145_7e4,
    6.726_577_092_700_870_085_3e4,
    3.343_057_558_358_812_810_5e4,
    2.509_080_928_730_122_672_7e3,
];
const B: [f64; 8] = [
    1.0,
    4.231_333_070_160_091_125_2e1,
    6.871_870_074_920_579_083e2,
    5.394_196_021_424_751_107_7e3,
    2.121_379_430_158_659_586_7e4,
    3.930_789_580_009_271_061e4,
    2.872_908_573_572_194_267_4e4,
    5.226_495_278_852_854_561e3,
];
const C: [f64; 8] = [
    1.423_437_110_749_683_577_34,
    4.630_337_846_156_545_295_9,
    5.769_497_221_460_691_405_5,
    3.647_848_324_763_204_605_04,
    1.270_458_252_452_368_382_58,
    2.417_807_251_774_506_117_7e-1,
    2.272_384_498_926_918_458_33e-2,
    7.745_450_142_783_414_076_4e-4,
];
const D: [f64; 8] = [
    1.0,
    2.053_191_626_637_758_821_87,
    1.676_384_830_183_803_849_4,
    6.897_673_349_851_000_045_5e-1,
    1.481_039_764_274_800_745_9e-1,
    1.519_866_656_361_645_719_66e-2,
    5.475_938_084_995_344_946e-4,
    1.050_750_071_644_416_843_24e-9,
];
const E: [f64; 8] = [
    6.657_904_643_501_103_777_2,
    5.463_784_911_164_114_369_9,
    1.784_826_539_917_291_335_8,
    2.965_605_718_285_048_912_3e-1,
    2.653_218_952_657_612_309_3e-2,
    1.242_660_947_388_078_438_6e-3,
    2.711_555_568_743_487_578_15e-5,
    2.010_334_399_292_288_132_65e-7,
];
const F: [f64; 8] = [
    1.0,
    5.998_322_065_558_879_376_9e-1,
    1.369_298_809_227_358_053_1e-1,
    1.487_536_129_085_061_485_25e-2,
    7.868_691_311_456_132_591e-4,
    1.846_318_317_510_054_681_8e-5,
    1.421_511_758_316_445_888_7e-7,
    2.044_263_103_389_939_785_64e-15,
];

/// Lanczos coefficients for g = 7, n = 9 (Godfrey's set), Lanczos (1964), "A precision
/// approximation of the gamma function", SIAM J. Numer. Anal. Ser. B 1:86-96,
/// DOI 10.1137/0701008. Decision D-10.
#[allow(dead_code)]
const LANCZOS_G: f64 = 7.0;
#[allow(dead_code)]
const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_571_6e-6,
    1.505_632_735_149_311_6e-7,
];

/// ln |Gamma(x)| for x > 0 (Lanczos, D-10). NaN for x <= 0.
#[allow(dead_code)]
pub fn lgamma(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    if x < 0.5 {
        // Reflection: Gamma(x) Gamma(1 - x) = pi / sin(pi x).
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - lgamma(1.0 - x);
    }
    let z = x - 1.0;
    let mut sum = LANCZOS[0];
    for (k, c) in LANCZOS.iter().enumerate().skip(1) {
        sum += c / (z + k as f64);
    }
    let t = z + LANCZOS_G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + sum.ln()
}

/// Digamma psi(x) for x > 0: the recurrence psi(x) = psi(x + 1) - 1/x up to x >= 12, then the
/// asymptotic series, Bernardo (1976), "Algorithm AS 103: Psi (digamma) function", Applied
/// Statistics 25(3):315-317, DOI 10.2307/2347257, with more series terms. Decision D-10.
#[allow(dead_code)]
pub fn digamma(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let mut x = x;
    let mut shift = 0.0;
    while x < 12.0 {
        shift -= 1.0 / x;
        x += 1.0;
    }
    let r = 1.0 / (x * x);
    // Bernoulli terms B_2k / (2k x^2k), k = 1..6.
    let series = r
        * (1.0 / 12.0
            - r * (1.0 / 120.0
                - r * (1.0 / 252.0 - r * (1.0 / 240.0 - r * (1.0 / 132.0 - r * 691.0 / 32_760.0)))));
    shift + x.ln() - 0.5 / x - series
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_quantiles() {
        // Reference values from mpmath at 50 digits (root of the normal CDF).
        let cases = [
            (0.5, 0.0),
            (0.975, 1.959_963_984_540_054),
            (0.025, -1.959_963_984_540_054),
            (0.841_344_746_068_542_9, 1.0),
            (1e-10, -6.361_340_902_404_056),
            (1e-300, -37.047_096_299_361_2),
        ];
        for (p, x) in cases {
            let got = norm_ppf(p);
            assert!((got - x).abs() <= 1e-12 * x.abs().max(1.0), "p={p}: {got} vs {x}");
        }
    }

    #[test]
    fn lgamma_and_digamma_match_mpmath() {
        // (x, lgamma, digamma), mpmath at 40 digits.
        let cases = [
            (0.1, 2.252_712_651_734_206, -10.423_754_940_411_077),
            (0.5, 0.572_364_942_924_700_1, -1.963_510_026_021_423_5),
            (1.0, 0.0, -0.577_215_664_901_532_9),
            (1.5, -0.120_782_237_635_245_22, 0.036_489_973_978_576_52),
            (3.7, 1.428_072_326_665_388, 1.167_153_539_361_511_4),
            (10.0, 12.801_827_480_081_47, 2.251_752_589_066_721),
            (123.456, 469.605_547_129_929_47, 4.811_829_323_828_985),
            (1e5, 1_051_287.708_973_656_9, 11.512_920_464_961_895),
            (1e-8, 18.420_680_738_180_21, -100_000_000.577_215_65),
        ];
        for (x, lg, dg) in cases {
            let tol = |v: f64| 1e-13 * v.abs().max(1.0);
            assert!((lgamma(x) - lg).abs() <= tol(lg), "lgamma({x}) = {} vs {lg}", lgamma(x));
            assert!((digamma(x) - dg).abs() <= tol(dg), "digamma({x}) = {} vs {dg}", digamma(x));
        }
    }
}
