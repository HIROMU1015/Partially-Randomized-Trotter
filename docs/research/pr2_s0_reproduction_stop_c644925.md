# PR-2 S0 input-reproduction stop report

- date: 2026-09-28
- source commit: `c644925b50587072784846df09bf02c39e8453e1`
- authorization commit: `e9bffb85f9ed57712bb83150172a6a4662cecf7f`
- S0 status: `STOP_INPUT_REPRODUCTION_MISMATCH`
- S1 executed: no
- S2/S3 authorized: no

## 結論

結果前amendment v3に従ってS0を実行したが、再生成したH4 1.00 Å、STO-3G、DF rank 12
development Hamiltonianのcanonical byte-level hashがpilot hashと一致しなかった。

- expected pilot hash:
  `d8b4aaf21afcc3935d5b5aa4d0805b358c5ec670d8104d25807c7cd0620a3dc3`
- observed S0 hash:
  `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`

事前登録はarray bytesを含む完全一致を要求し、近似一致を同一snapshotとして扱うことを禁止している。
従ってS0はterminal STOPとし、prefix identity、candidate signal、compile、trajectory sampling、S1 correctnessを
実行しなかった。

## 実行範囲

S0で許可された入力freezeだけを実行した。

| item | count / status |
|---|---:|
| molecular builds / ground-state solves | 2 |
| development input | H4 1.00 Å、STO-3G、8 qubits、4-electron singlet、DF rank 12 |
| held-out input | H4 1.30 Å、同basis/sector/rank。snapshotだけをfreeze |
| candidate signal evaluations | 0 |
| circuit compilations | 0 |
| trajectory samples | 0 |
| quantum shots | 0 |
| held-out signal/cost/ranking | unopened |

held-out snapshotのHamiltonian hashは
`a70d9619e794a0238aae57096a33759bf72d7b294a251b235508cd2ccc6b16c0`である。このhashは入力freezeの
provenanceであり、held-out transfer結果ではない。

## 限定診断

新しい分子buildを追加せず、保存済みdevelopment snapshotとpilot artifactの既存recordだけを比較した。

- Python、NumPy、SciPy、Qiskit、OpenFermion、OpenFermion-PySCF、PySCFは事前固定版と完全一致した。
- pilot source 3ファイルのSHA-256はpilot provenance記録と一致した。
- Hamiltonian builderとhash実装はpilot commit `71169d8`からsource差分がない。
- pilot ground energy `-2.1663874486347736` Haに対し、保存snapshot/stateのenergyは
  `-2.16638744863476` Haで、絶対差は約`1.38e-14` Haだった。
- rank 3/6/9 residualのcomponent数はpilotと同じ324/216/108だった。
- residual $\lambda_R$ のpilot/new差はそれぞれ約`1.2e-15`、`9.2e-16`、`4.6e-17`だった。
- ただし、Hamiltonian hashとrank 3/6/9 tail hashはいずれも不一致だった。

以上は微小な数値表現差またはfactor gauge差と整合的だが、原因を確定したものではない。物理的不変量の
近さを理由にbyte-level gateを事後緩和しない。旧pilot入力と新snapshotを混ぜず、再試行やhash規約変更には
新しい結果前amendmentと外部レビューを必要とする。

## Evidence packet

| path | SHA-256 / fingerprint |
|---|---|
| `artifacts/pr2_s0_s1_validation/2026-09-28/pr2_s0_validation_v1.json` | file `cf082a81ed70dcee774906ff2391683811cbdf544c1217127e100a977a11fbd7`; result `6d44888a1b806bc3b6418b49a18fdbcee09b621dd345838dc0d426abb1005182` |
| `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz` | `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a` |
| `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p30_rank12_held_out_v1.npz` | `ad7e3e7165c55dbaa395eef7a1dd74db89e1f7ab29a69ac64333f4aebf8b3e37` |
| `artifacts/pr2_s0_s1_validation/2026-09-28/pr2_s0_s1_tests_c644925.xml` | `fa2e1ca2ea2c12a3700cbe4c47437f4016bc5412d490892c3b5fb86a16acdd7b`; 123 passed |

このpacketはlocal dirty-worktree validationであり、immutable CIまたは外部再現ではない。

## Mandatory stop

`S1_authorized=false`である。S1 summaryは存在しない。S2/S3、32/128 expected-cost Monte Carlo、resource
winner、materiality判定、state-preparation break-even、別rank/geometry/precision、H12、長RPE、final total
costへ進まない。
