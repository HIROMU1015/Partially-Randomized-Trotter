# 次の限定科学実行計画：Controlled IID直接Hoeffding

2026-11-05 JST。**準備済み・実H4未実行。公開後の別途明示実行許可が必要。**
今回の技術準備や過去stageの実行許可は、この実行を許可しない。

## 固定SOURCE・入力

SOURCEは本ディレクトリとbyte-identical旧helper3個を含む、今回公開する準備commitそのもの。
正確なSHAは最終chatとremote branch
`codex/rpe-controlled-iid-success-energy-preparation-20261105`で提示する。
将来の実行担当はそのSHAをcheckoutし、SOURCEを変更せず実行する。
`frozen_source`照合に加え、RESULTに実際のHEADとruntimeを記録する。
準備commitを変更する必要がある場合、変更と科学条件への影響を先に示す。

規範的な入力識別子は[FIXED_BINDINGS.json](FIXED_BINDINGS.json)のcommit/path/blob/SHA256/bytes。
旧D/A/Cのmanifest中の依存SOURCEと入力もbyte照合する。

| 主入力 | commit | SHA256（原ファイル） |
|---|---|---|
| mean RESULT | 1789a8ca731a1de631e80a34d2b8c90657d24458 | df9b0082b6b6f51ff01058639f92e0d873d5b1f28c51cc2cc666078b3746f84e |
| 固定18round入力 | f4905fff592351d33ba07b927892fb9532c99711 | 72644832c1c8a159809efc30ff2909aaa576e49d69197d7ee007bf8a19d57673 |
| 元36軸表 | d2e7a7db683e116243eae061c0e4c87865054e9e | 0346da604f125e41c5afd073a91e8eb46afe4c2e1c950937d5de9eb33e588e0d |
| D RESULT | 89cb94fe4db1d39d574877c3bc2db79a6d7a5f9f | a80ce0376ff2e29241911fee901f327a49bfb7a33495d340caf7236370712b68 |
| A/C RESULT | 61f569c32a94c2b698df10a62a51e515b12160ed | 707c22df15d690348bc081d9249a5598deeec1353447a4e5c85a12a9be5434f3 |

A_RESULT、C_RESULT、MEAN_AXES、旧alpha/common/D/Ball、旧証明等のhashはmanifestを参照。
全36軸はm=0..17×{cosine,sine}の重複なしの一意なjoinとし、
各m/q/r/K、N/ε/α/μ₀、元U_cat、Δ_meanが原証拠と一致することを実行直前に確認する。
defaultでこの確認のみ実行できる。defaultはU_ctrl/t/確率/総和を計算しない。

科学条件・保証対象・信号符号・G・alias prior・energy対象は[CONNECTION_PROOF.md](CONNECTION_PROOF.md)通り。
εとαは旧包含のexact decimal、μ₀/E_ref/δ/energy目標は元binary64のexact値。
平均上端は原exact rationalのまま使用し、表示約値やFloat64へ縮めない。
旧m17や旧Aの確率値を新controlled確率として流用しない。

## 許可を求める最小の数値評価

Arb **160 bit、1 pass**。Python3.11、NumPy1.26.4、Qiskit1.3.0、
python-flint0.9.0／FLINT3.6.0、binary64と最近接丸めの既存環境を維持する。
NumPy/Qiskitは既存照合の依存だけで、新しい配列・回路は構築しない。

| 新規科学call | 最大範囲 |
|---|---|
| U_ctrl=U_cat+Δ_mean | 36 exact rational加算。平均差を一度だけ計上 |
| t=ε−U_ctrlとstrict正値 | 36セル、旧alpha_adapterのArb経路 |
| 2exp(−Nt²/2)の上端 | 36セル、各1回、dyadic上端を保存 |
| exact rational総和・成功下界・≤1/20判定 | 全36上端で1集約、旧aggregateを無変更で使用 |
| D/C接続 | 保存判定・上端の転記のみ。新規幾何/branch/energy/reference評価0 |

原理上の全体計算量はO(36)。旧Aの同じscalar経路を再利用し、q個のroot列挙、matrix、
trajectory、二項格子を避ける。36個のscalar Arbと小さい有理数和にworker pool/GPUは不要。
これはCPU/RAM/worker上限ではない。affinityとMemAvailableを読み、共有環境へ変更を加えない。
wall-time、CPU、RAMの人工上限は設けず、科学call数と1 passを保全する。

新しいHamiltonian/DF/state生成・decode、categorical law/mean enclosure、support差、
固有値reference、Dの幾何/branch/alias/energy、Float64 decoder、CDF/PMF、
RNG/sampling/shot、回路build/transpile、noise/cost/state-preparation評価は全て**0**。

## 実行入口・原結果

別途許可後だけ、指定SOURCEの専用実行worktreeで次の形を使う。
`PY`は上記固定環境のinterpreter。以下は**未実行の将来用command**。

```bash
"$PY" docs/research/rpe_controlled_iid_success_energy_preparation_20261105/run_controlled_success.py \
  --run --output artifacts/rpe_controlled_iid_success_energy/<execution-date>/run01
```

出力先は上記namespaceの新規ディレクトリだけ。既存ディレクトリは拒否し、再開・上書きしない。
RESULT.jsonにはSOURCE/inputs/precision、全36軸の保存literal、U_ctrl、t包含、tail上端、
call数、exact総和、成功下界、既存energy上端と出典、formal statusを保存する。
各セル後に部分結果を保存し、異常時はUNKNOWN_MANDATORY_STOPにする。
stdout/stderrと`time -v`等の資源記録は実行担当が原byteで別保存する。
結果を再計算して整形せず、表示表は原endpointから転記する。

## 判定とmandatory STOP

全36セルのt>0と有効上端がそろい、exact総和≤1/20の場合のみ、
`CONDITIONAL_IDEAL_CONTROLLED_IID_MATHEMATICAL_ENERGY_95_SUFFICIENT`。
これは指定IID・ideal logical測定・数学的decoder・外部alias条件の下で、
同じ保存DF sector ground energy目標に対する95%以上の条件付き十分保証を意味する。
旧energy上端は変えず、Δ_meanをenergyへ再加算しない。

以下はmandatory STOP：

- SOURCE/input/hash/schema/sign/policy/36軸identity/runtimeが不一致、欠測、重複。
- tのstrict正値が不明、非有限、exp/endpoint不明、例外・中断・部分結果。
- 全36上端がそろわない、またはexact総和>1/20。
- 新しい科学callや予算/状態/法則/精度の変更が必要になる。
- 正常完了・結果公開とremote SHA/成果物照合の完了。

非成立はUNKNOWN/STOPであり、実際の成功率が95%未満という証明ではない。
自動retry/resume、精度追加、shift調整、N/α/ε再設計、別candidate/namespaceでの追加実行はしない。
旧STOPを解除しない。詳細科学レビューも開始しない。

この計画が許可されてもformal全体認定は自動変更しない。
`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
`U_virtual_IID_upper=null`、`U_exec_upper=null`を維持する。
