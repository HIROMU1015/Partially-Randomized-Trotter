# SOURCE再利用と検証の範囲

## 固定参照の役割

| 固定commit | 役割 |
|---|---|
| `d829ead865e7ff5defe38ba9d97a5662c75441aa` | 第一計画・p8原因調査。指定5資料を確認 |
| `0c49a88134066177816feca2741f36f6b88d23ba` | H8 loader、保存restore、DBA binding等の参照carrier |
| `745a674546f2cb6258526fbfde1d9a00601689f8` | H8独立確認の実行SOURCE。再起動しない |
| `1549b6f891a15ff48379996c34b344118ba2606c` | 元dataset/seed SOURCE。再実行しない |
| `155529e20a8fac39e00786c40b57830b71203c4f` | H46成果物・コードのcarrier。誤記短縮SHAは不使用 |
| `c31ef424175c02379eafebe8fc6c7548c1d9aabe` | 保存H8 configのprovider SOURCE |
| `37694741ee356fd09b5dfcb56130c55ab381044c` | G04 metadata補完の固定証拠 |
| `862d732fba0a5f353c522a9219dec78ae2f21a4b` | H8 rare-RTE正式終了、既存STOP維持 |

H46 carrierのprovenanceが記録する実際の科学SOURCEはsignal
`e417ffc9f52b13c18f5493193ce510d858664259`、native
`27ebabdd2943a4de787ec8b976c38e06fda5a497`、extension signal
`2069360497a7849b63cb102e02ca1ca412462f52`、extension native
`f1e95576a8a924face4213b86172455f4f993d7c`、continuation
`ff323f1b09f96c0ce01955728832ff6b0f4c230f`。
carrierを全旧科学callのSOURCEと混同しない。
新adapterは公開した新commitをSOURCEとする別scopeである。

## 関数の接続

| 再利用元 | 新adapterでの使い方・変更 |
|---|---|
| h8_ab_native.load/prefix_preparation | H8 snapshot、state、保存Full restore、prefix7/8/15/16。Fullはrestoreされたblock viewを追加 |
| h8_b0_anchor_adapter.restore / h8_strengthened_export.validate_export | 既存記述からgateを復元。入力再生成や新Gaussian synthesisなし |
| h8_sparse_reference_v1.OccupationCSR/df_operator/SpinPrimitive/reference_pair | H8 sector4900のsparse full/tail、70×70 spin primitive、独立CSR gate。dense H46 contextは不使用 |
| h8_dba_direct.scalar_for、DBA saved_binding | policyを静的確認し、p16既存rational scalar receiptを新実行時の照合先にする。旧evaluate/permitは起動しない |
| h6_native_b0i_bridge_v1.exact_identity | 新scopeでp15/16のexact stored-binary64 scalar。旧T=.8 overlay関数は不使用 |
| h46_m5_plan.weights/schedule、h46_new2_core.allocation | 固定binary64 signed PF、R_outer整数配分をそのまま再利用 |
| h46_new2_core.polynomial、h8_sparse_reference_v1.recurrence_paths、h6_dba_signal.forward_finite | 既存K+1次数・raw/corrected規約を参照。新T/signed piece用のmatrix-free forward/Horner wrapperへ最小拡張 |
| h46_pr_pilot_repair.canonical_tau | symbolic lambdaとtime/stepsの演算順をsamplerに合わせる |
| run_pf_common_run.qualify_record | allowance、axis headroom、analytic shots、Decimal照合。旧Stage/grant launcherは不使用 |
| h46_long_time_native.compose/sample_requests/compile_cached/rarity | H8 preparationを渡すgeneric部分だけ。SOURCE/input/cohortで新seedを作るtask-local関数へ接続。cacheは新stage内に限定 |
| ax2b_native_df_v5 / df_rte_qiskit | native directional loweringとbasisのactual gate inverseを維持。単に係数符号を反転してadjointの代わりにしない |
| ax2b_molecular_ports_v3.explicit_event_action / make_native_block_action | 新Tのalgebraic vector oracleへ接続。旧h6_dba_science.event_oracleの`.8`固定scalarは不使用 |
| h246_saved_cost_evaluation.paired | 既存paired会計の設計を参照。新関数は固定N付きG sampleのexact mean/varianceを直接計算 |

新コードは既存scriptに実装された研究別処理を呼ぶ薄いdriverで、共通ライブラリの既存API・原SOURCEは変更しない。
関係するlocal/frozen SOURCEのhashと入力receiptは[VALIDATION.json](VALIDATION.json)。
private configは絶対path、除外seed実値、専用run rootを含むため公開対象外。
ハッシュは科学SOURCE/inputの再現性確認に使い、利用者承認の証明には使わない。

## p8 flagの扱い

保存exportに`faithful:p8`がないため、旧`saved_exact_export_match=false`は直接比較未実施を表す。
`EXPLAINED_NO_SCIENTIFIC_IMPACT`はこのflagの原因と影響についての分類。
p8の独立export比較完了や全科学入力の一致証明ではない。
新adapterも旧falseや旧exportを書き換えず、p8準備hashの固定と新scopeの作用/reference gateを別に確認する。
新不一致は影響scopeをHOLDする。

## 人工検証

新しい専用testだけをunittestで指定して実施。既存の科学setupを起動するtest全体は実行しない。
21件PASS：passive import、候補domain/id、fragment順序、S2/m5 signed schedule、R_outer/q、入力hash不一致、旧reference拒否、SOURCE/cohort、exclusive保存、K2/K4の最高K+1次、2×2 finite/exact-tail作用、raw/corrected closure、scalar一回、mock native順序/control、保存basisのactual adjoint、paired Gの共分散、cohort/条件付き/重複混合拒否、未測定価格の拒否、mock price jobの両軸measurement/compile会計、確認精度未達時の無条件top-up拒否、同q診断の適格S2限定。

```bash
"$H8_PYTHON" -B -m unittest discover -s tests -p test_h8_long_time_adapter.py -v
"$H8_PYTHON" -B scripts/h8_long_time_adapter.py --config "$H8_PRIVATE/config.json"
```

H8_PYTHONは確認済み専用venv、H8_PRIVATEはこの専用cloneのprivate設定領域。
二つ目はmetadata-only inspectである。実行承認後の科学入口は同scriptの`--stage A|B|C --execute --scope ... --output ...`。
新コードのimport自体はnumpy/scipy/qiskit/provider/H8 loaderを必要としないことをimport blockerで確認した。
小行列testが使うNumPyは人工fixtureだけ。mock composeは実Qiskit回路を作らない。

## 実H8では未検証

H8 loaderから新wrapperへのrestore実行、p15/16 exact scalar再照合、新signed-timeのprimitive、T1.6 full-H reference、finite meanの精度、eligibility/N、sample法則、native/control/compiler/cacheの実H8一致、価格分布、weighted-G精度、性能/RSS/GPUは未実行。
Stage実装は準備済みだが、実科学pathのend-to-end PASSは主張しない。
人工・mock PASSは技術的証拠で、科学性能・数値ランキング・population保証の証拠ではない。
