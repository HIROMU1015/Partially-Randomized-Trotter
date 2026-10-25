# C1：保存H4 XDF prefix1〜9・DF予算評価の技術準備

2026-10-26 JST。起点 `8c5b7e903b27464e8253f23db7c476201be4c176`。
Branch：`codex/research-c-c1-h4-prefix-preparation-20261026`。

**案Aはユーザーが次の検証対象として採択した。sector36の限定entry・人工検証・passive preflight・356-call未承認契約を準備した。実H4の数値decode／新norm／新rank判定は0。** 新人工29件＋関連する旧Stage1の16件＋旧Stage3の12件がPASS。旧Stage1のfull256回帰1件には環境上の未解決異常があり、全回帰PASSとは表示しない。

[SOURCE](../../src/trottertracks/c1_h4_xdf_prefix_budget.py)／[CLI](../../scripts/run_research_c_c1_h4_xdf_prefix_budget.py)／[人工tests](../../tests/management/test_research_c_c1_h4_xdf_prefix_budget.py)／[未承認契約](research_c_c1_h4_xdf_prefix_budget_contract_unapproved_20261026.json)／[技術証拠・SOURCE manifest](../../artifacts/research_c_c1_h4_xdf_prefix_budget/2026-10-26/technical/)。前回の[監査](research_c_c1_h2_h4_h6_comparison_audit_20261026.md)と[案A設計](research_c_c1_minimal_experiment_design_20261026.md)を引き継ぎ、旧SOURCE・契約・結果・STOPは変更していない。

## 固定条件と保存入力

直鎖H4・隣接距離1.00 Å・STO-3G、Nα=Nβ=2／sector36。OpenFermion1.6.1 eigh由来の保存λ/Gの**元順prefix1..9**だけを評価する。通常number表現、元g補正κ、核scalar、frame／spin／bit順、保存hermitizationを維持する。CA=.00159 Ha、DF予算CA/8=.00019875 Ha、既存`binary64_operation_scale_guard8_v1`。

| 役割 | 固定identity |
|---|---|
| 元P-H4 | commit `86b8b002704aced8ca239414080f53793de81658`、`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz` |
| 元input SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 元入力生成SOURCE | `0c329b30a81accb35c212be8defa66fee0dd0f12`のh4_molecular_inputs.py／df_hamiltonian.py。df_rank=None／df_tol1e-8→rank10の来歴 |
| 保存physical sector | `artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/run01/operators/physical_sector.npz`、matrix complex128・36×36 |
| Stage1評価SOURCE／scalar | `bec41ac7dc7e034bd664817acfdcfed858ddadd0`、同runのresult.json／provenance.json／evaluation_SOURCE.json／STOP.json |
| Stage3予算値 | `artifacts/research_c_c1_stage3_feasibility_science/2026-10-25/run01/prepared.json`。精度pilotの公開execution_summaryもbyte固定 |

公開Git objectから元packetの同一bytesをtask-local `.external/P-H4.npz`へ取得した。旧pathを移動せず、CLIの`--input`は固定hashと同じbytesを受け付ける。今回確認したのはZIP／NPY headerとhashだけ。h4×4、g4⁴、κ8×8、λ10、G10×8×8、nuclear scalar、保存physicalのFortran-order metadataを確認し、数値payloadはdecodeしていない。

保存anchorは次の既報scalarを再利用する。今回の新規測定ではない。

| anchor | norm Ha | allowance Ha | 保存Stage3 norm＋allowance Ha | 保存判定 |
|---|---:|---:|---:|---|
| XDF10 | 3.3751558354542566e-15 | 1.4812598092146554e-9 | 1.4812631843704909e-9 | PASS |
| CDF8 | 7.666027761392501e-5 | 1.881549094133745e-9 | 7.666215916301914e-5 | PASS |

## 再利用と最小差分

`load_input(evaluation=True)`のH4 whitelist・shape・dtype・finite、`legacy_df_module`／PhysicalSector／df_linear_operator、bit_reverse_indices、sector_indices、operation_lengths／allowanceを無変更で使用する。spectral評価も旧Stage1と同じSciPy `svd(...,compute_uv=False,lapack_driver='gesdd')[0]`への薄い呼出しで、独自SVDやCARは実装していない。

旧`saved_xdf_sector`は36 matvecを実処理前に一括計上する。今回はactionごとの完了・FAILを記録するため、そのhelperを変更せず、同じDFHamiltonian／Python backend／_linear_operator_to_denseへcounted LinearOperatorを挟んだ。人工rank1／3／9で旧helperとの行列完全一致・scale一致・allowance一致・36実actionを確認した。新しいsector simulatorやDF分解ではない。

κ確認は実行時のidentity gateで一度だけ行い、全prefixは保存one-bodyとscalarをそのまま使う。再補正・再ordering・再hermitizationはない。input／physicalはread-onlyとしてworkerへ共有する。journal、STOP event、call上限、resource_planは既存`integrated.Calls`等を再利用する。

## correctness・判定・STOP

順序はbyte preflight→fresh output→保存load→sector context→κ／scalar／basis gate→保存physical attach→全9prefix→選定報告。basisは旧Stage1と同じascending Qiskit little-endian interleaved sector_indices(4,2,2)。OF sectorから既存bit反転で対応を作り、indices／permutationを照合する。保存physical自体に独立のbasis listはないため、過去SOURCE・ファイルhashの来歴も確認する。元physicalを新規生成して検算しない。

κ・physical／one-body／G／prefixのHermiticityはmax-entry1e-10／rtol0、finite必須。physical scaleは元h/gの`|c|+Σ|spin_one|+Σ|spin_two/2|`、XDF scaleは`|c|+Σ|saved_one|+Σ_prefix |λ|(Σ|G|)^2`。rank rごとのoperation_lengths(4,36,r)と差行列を既存allowanceへ渡す。保存rank10 guardを低prefixへ流用しない。

判定は`norm_r+allowance_r<=.00019875 Ha`のPASS／NOT_MET。最初のPASS後も全9件を評価し、単調性は仮定しない。全NOT_METなら保存rank10がこの保存順prefix1..10中で最初の適格rankとなる。費用最適rankや他順序／他分解を含む最小rankとは呼ばない。

最初のcorrectness FAILで新規依存callを止め、開始済みcallを回収する。rank別完了JSON、partial_prefixes、例外詳細、SOURCE、契約、journal、資源、STOPを保存する。retry／resume0、条件変更・fallback・別候補追加は0。fresh pathは`artifacts/research_c_c1_h4_xdf_prefix_budget/<execution-date>/run01`。存在済みrun、run02、family外pathは拒否する。

## 最終356-call定義

| logical unit | 最大 |
|---|---:|
| saved bundle load | 1 |
| sector context | 1 |
| input κ／scalar／basis identity | 1 |
| 保存physical attachment／formation scale | 1 |
| prefix model | 9 |
| 実matvec（9×36） | 324 |
| sector error norm | 9 |
| 数値allowance／Hermiticity／budget | 9 |
| 全prefix選定報告 | 1 |
| **合計** | **356** |

正常人工E2Eで356 attempt／356 COMPLETE、324実matvecとprefix9件を確認した。basis／permutation処理、SVD内部演算、guard内Frobeniusは各複合callに含む。新fit／分解・元physical生成・rank10 norm・PF・native／compile・合成誤差・shots／費用はすべて0。失敗時は到達したcallだけを保存する。

## 人工・回帰検証と残る環境制約

| 検証 | 結果 |
|---|---:|
| 新prefix人工unit／E2E | 29 PASS |
| 旧Stage1関連回帰（full256以外） | 16 PASS、1件deselected |
| 旧Stage3 deterministic accuracy | 12 PASS |
| **関係する完了試験** | **57 PASS、最終FAIL0** |
| 実固定bytesのpassive CLI | PASS、science_calls0、NumPyもimportしない |

λ/G pair順、非零gのκ確認一度・scalar一度、sector36・36基底、独立occupation reference、旧helper直接比較、rank別guard、非単調列・全NOT_MET、SOURCE／input／環境／basis／permutation／finite／Hermiticity／matvec不一致、並列FAIL回収、356上限、fresh output、rank10等の拒否、passive importを検証した。人工PASSを科学H4の適格性と扱わない。

初回12 FAILはGit mockがMatplotlibの外部コマンドにも作用したためで、Gitだけに限定して修正した。拡張κ人工試験の1 FAILは独立な和の丸め差に対する期待値2e-16が狭すぎたため、人工検算を2e-15とした。実行用1e-10、DF予算、既存allowanceは変更していない。失敗log／XMLもlossless gzipで保全する。

旧Stage1全件のcold実行とbackend先読込み実行は、既存`test_artificial_eight_mode_full256_and_sector36`の256×256 SVDでexit139になった。tracebackは旧Stage1の32-thread SciPy gesddで、今回のprefix moduleへ入る前の試験である。**原因確定・修正・当該1件のPASSは未完了**。旧SOURCE／環境を変更せず、full256経路を新sector36 runnerへ持ち込まない。これを全回帰PASSとは表示しない。新経路は独立sector36 E2E・旧helper比較を完了した。旧full256経路を再利用する場合は環境問題の技術対応が残る。

## 環境・固定SOURCE

既存eval venvをread-onlyで再利用：Python3.11.1、NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1、openfermionpyscf0.5、PySCF2.7.0、Qiskit1.3.0、Nature0.7.2、threadpoolctl3.6.0。stage2依存lockを参照し、新fit環境は不要。共有venv・package・cache・他ユーザーjobは変更していない。

affinity32 CPUを確認。将来は9独立prefixと実際のCPU／memory／quotaから既存resource_planでworkerを選び、複数worker時はlocal BLAS1でoversubscriptionを避ける。36次元・Python matvecでJIT不要、immutable packet／physical／contextを共有する。CPU／RAM／worker／wallの人工上限はない。大規模full行列やGPUは不要で、speedup倍率は未測定。

- 契約SHA256：`56ad290738e1faec290bd66e896fc7da8370e4f31760ddd3429d5fea1c6787d0`。
- SOURCE manifest SHA256：`b63ef45279a27c09fcb0626628b025e4891c6ea58f602ffddce9489cf022a516`。
- SOURCE15ファイル・保存入力7ファイル・元packet・過去SOURCE3点のbytes／環境をpassive確認。実装は公開commitに固定し、将来のrunにも実commitを保存する。承認証明・原文保存・新grantは追加しない。

以下は**passive確認だけ**を行う。

```sh
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python \
  scripts/run_research_c_c1_h4_xdf_prefix_budget.py \
  --contract docs/research/research_c_c1_h4_xdf_prefix_budget_contract_unapproved_20261026.json \
  --input .external/P-H4.npz
```

## 引渡し

限定runnerの技術準備は完了した。許可後に全9prefixのnorm／guard／合計／予算判定、保存rank10との関係、この保存順のfirst-passingを取得できる。低rank PASSでもPF／native／合成／CA／shots／費用は未資格である。

**ユーザー／研究C**が公開SOURCE・契約・入力identityと旧full256回帰制約を確認し、最大356-callの実H4評価を別途明示許可する。**Codex**はその許可後に同じSOURCE・入力でfresh run01を一度実行する。旧full256環境問題を今後再利用する際の技術対応は、このsector36科学契約と分ける。

今回はcommit／push・remote SHA／SOURCE／input／成果物照合後に**mandatory STOP**。実H4 prefix、H6 rank13、PF／native／shots／費用、詳細科学レビューへ自動進行しない。
