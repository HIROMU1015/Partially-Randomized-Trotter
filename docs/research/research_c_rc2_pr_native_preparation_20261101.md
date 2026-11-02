# 研究C：保存RC2／PR-tailのnative価格・数値同値性検証準備

2026-11-01 JST。**技術準備完了、実H4科学call 0、mandatory STOP。**

## 1. 範囲と固定資料

起点1542bf0bece5a10f20a3c5f28101df170515ec94の[条件付きshots／反復重み結果](research_c_rc2_pr_conditional_shots_execution_20261101.md)から、同じ保存RC2／PR-tail終点をnative校正へ接続した。今回は人工・mock検証とbyte/header/metadata確認だけを実施した。共通運用指示2026-10-25版§8/10とAGENTS.mdに従い、承認証明・承認原文保存・新grantを作成せず、重要GPT科学レビューも行っていない。

| 固定対象 | Identity |
| --- | --- |
| 実装commit | 3bb00c1caddc30f21a292927b2d2effbacbf2164 |
| 元2-fit SOURCE | e2f7dfd44357510d54ab9c6c9c5032e0072ab376 |
| 保存Stage B SOURCE | d6b17b79461a39bd8ba2e6b7389a1f4fad4d74e6 |
| 保存shots SOURCE | e88fa599af7724f34e11a0f2eab959f92135cf13 |
| 新SOURCE manifest SHA256 | b17d0236f2ed03ca57ad0b84d314344ee1ed5e0a93aa6b987f84717a076a45b1 |
| 未承認契約SHA256 | 43d41fece2fa958627f908eaec352ee0c78dde8c6846d5a07f91e773cd53d9b9 |

新経路は[module](../../src/trottertracks/c_rc2_pr_native.py)、[CLI](../../scripts/run_research_c_rc2_pr_native.py)、[人工テスト](../../tests/management/test_research_c_rc2_pr_native.py)。[未承認655-call契約](research_c_rc2_pr_native_contract_unapproved_20261101.json)、[SOURCE130](../../artifacts/research_c_rc2_pr_native/2026-11-01/technical/SOURCE.json)、[依存pins](../../artifacts/research_c_rc2_pr_native/2026-11-01/technical/evaluation-pins.txt)を固定した。

保存H4＝直鎖1.00 Å／STO-3G、CDF rank8、Nα=Nβ=2、sector36／full256／controlled512。元frame順・L_D=2、partial ordered symmetric S2、δ=h=0.1、s=1／r=128／K=0。13 round m=0..12、共通CA=0.00159 Ha／DF reserve=CA/8、own-DF ground-state oracle／状態準備費用除外を保持する。今回の新契約で実測する単位は**1 h-step kernel**で、13 roundの再評価は含まない。

RC2 exportはd6cb5d56dfc6a83090a6ad1bc1207fdaee62a14cf120542fcb5a7d116644a798、PR exportは3fdb0cb77afc770294cbe7aebf47d8a4529e3567ef9e886895495f4490b7ee21。元input snapshotは1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db。26入力を契約へhash固定した。両終点はDF予算の保存判定PASSだが、optimizerは両方UNQUALIFIEDのまま。最適RC-DF／最適PR-tailの比較へ昇格させない。

## 2. 再利用と最小差分

| 既存資産 | 再利用方法・適合性 |
| --- | --- |
| `c_rc2_pr_stage_b.prepare_candidate` | 新RC2／PRのJ/U・元g補正κ・scalar・元frame順・source identityを保持。旧baseline/proposedは内部API適合だけに使い、結果identityは新armへ戻す |
| `c_pr_aware_df_stage_b` | 既存signed finite-RTE sampler／event順／identity phase／controlled kernel。新しいfitterやsamplerは実装しない |
| `c_pr_aware_df_stage_b2` | full CAR/event reference、raw/P0/all-entry gate、actual adjoint／return、X/Y、shell、6費用指標・記述的校正 |
| `c_pr_aware_df_m2.sector_match` | full CARのbase＋8 frameを保存sector36の各generatorと対応させる。保存NPZにindex列がないため既存PhysicalSector＋bit-reversalから同じ順を作る |
| StrongBuilder／precision Gaussian／P0 wrapper | 決定論head内の既存相対Gaussian融合、tailの厳密basis_key共有・角度結合。追加最適化は導入しない |
| `SharedQiskitCalls` | kernel build／compile／actual-adjoint構築のprocess-local排他を再利用。独立referenceと数値gateは並列 |
| 既存Calls／atomic JSON export | call上限・途中STOP・partial evidence・exclusive出力。新しい実行管理／承認基盤は作らない |

旧one-fit／M4／B2 runnerは再起動しない。旧M4の総資源接続やbaseline/proposed専用比較は旧候補の価格・目標に依存するため再利用せず、新armのdispatch／保存law schema変換／native-only比較だけを追加した。既存libraryと旧SOURCEの編集は0。

SOURCE130は保存shots、B2互換修正、M4-nativeの依存集合と新3ファイルのunion。B2歴史版と起点に既に存在した4差分を`reuse_sources.historical_to_base`へ記録した。新SOURCEは現在の起点bytesを固定し、歴史版と同一とは扱わない。

## 3. 次回限定実行の推奨案

**各arm32、計64の非paired・非adaptive calibration trajectory**を新しいcohort／SOURCE digest／arm／export hash／sample indexで固定する。各trajectoryは128個の独立microstepを含む。旧cohortはpoolしない。これはclassical校正であり、quantum shotsは0。将来のcoherent各occurrenceにもfresh IID trajectoryが必要で、校正1件を2^m回繰り返す費用モデルにはしない。

P0はQiskit1.3.0、opt1、seed17、basis rz/sx/x/cx、approximation_degree1、qubits_initially_zero=Falseを既存profile通り保持する。両armで同じ合法的head融合・tail共有を適用し、head/tail境界融合や新規相対Gaussian最適化は行わない。

1. 保存SOURCE／入力／旧Stage B・shots資格を照合する。
2. head/X/Y外部shell3件を新しくbuild／P0 compileし、同一register・ancilla-only・位相を検証する。旧shell価格は転用しない。
3. armごとにbase＋8 frameの独立full256 CARを保存sector36 generatorsへ照合し、保存groundを同じindex順でembedする。再eigensolve／物理DF誤差再計算はしない。
4. 新armのcomponent ID・符号・順・probability、λ、identityと保存tail/law記録を照合する。
5. 64 taskでsample→native build→独立reference→raw gate→P0→compiled gate→actual adjoint／return→X/Y→metricsを実行する。
6. arm別平均／分散／SE／観測min/maxとPR-minus-RC2の記述的差を保存しSTOPする。

| Logical unit群 | 最大数 |
| --- | ---: |
| 保存evidence／identity／input／CAR／両armのmodel・law・sector・ground・18 factor照合 | 38 |
| Gaussian basis要求18＋head相対Gaussian cache miss最大8 | 26 |
| shell build／compile／assessment／metrics各3 | 12 |
| 64 trajectory × 上記9段階 | 576 |
| arm校正2＋native比較1 | 3 |
| **合計** | **655** |

Compileは64 kernel＋3 shell＝67件。QPYは64×(raw／compiled／actual raw adjoint／actual compiled adjoint)＋3×(raw／compiled)＝262件。cache hit時に架空callを足さず、relative Gaussianが8未満なら総数も655未満となる。固定32件を結果に応じて増減しない。

## 4. 数値資格と限界

raw／compiledは各trajectoryのfull512全成分を比較し、control=0 identity、control=1 CAR/event evolution、相対phase、unitarity、actual adjoint、X/Y複素signalを確認する。個別Gaussianのsector射影やglobal phase fitで代用しない。

既存B2 allowanceをそのまま使う。operator／unitarity component abs=1e-10、complex signal=2e-10、normalization=1e-12、mass rel=1e-12／abs=1e-14、rtol=0。frame full/sector照合も既存1e-10。実行後の閾値変更・fallbackは禁止。

これはbinary64の**有限cohort・全成分数値pilot**で、validated enclosureでも全law保証でもない。未知のwhole-law synthesis／implemented-law距離／cost support／ideal-law期待費用／actual CAはnullを維持する。SEは有限bit IID校正の記述量で、同時保証区間ではない。観測最大値をsupport上界へ昇格しない。

主指標はsample mean RZ count、副指標はRZ depth／CX count／CX depth／total depth／circuit size。両候補内のkernel価格差を測定できるが、保証付き総資源優位とは扱わない。

保存shots／Wの3 laneはidentityと資格だけを読み、式・N・Wを再計算しない。将来の条件付き接続は`C_metric = W × μ_metric + S_metric`で、μは完全h-step kernel価格、Sはhead/readoutのshots加重費用。r=128をWへ重ねない。ただし**初回655-callにはSの加重集計・interrogation総費用算定を含めない**。evidence laneの未知shotsや誤差を0で補わない。合成誤差資格が有限cohortに留まる場合、総資源への接続も明示仮定付きに留まる。

## 5. 人工・回帰検証とpassive確認

[pytest結果](../../artifacts/research_c_rc2_pr_native/2026-11-01/technical/pytest.txt)：**105 PASS、88.69秒**。新17件と既存Stage B／conditional-shots／B2／mutex／M4-nativeの適合テストを実行した。

- 発明した両armのr128 full-controlled reference／P0／adjoint／X/Y、符号・相対phase・compile不一致。既存M4の人工full512試験もPASS。
- 新cohort128 seedの独立性・再現性、sample index／r不一致の拒否。
- 保存Stage B形式のsector mappingとlaw catalog adapter、PF payloadを使わない読込。
- 64-task mock end-to-endで655 call／compile67／QPY262、未知誤差と総費用null、FAIL時の後続停止・partial STOP・fresh/no-resume、原本保全。
- passive preflightでarray load／sample／build／reference／校正／accuracy entryを呼出し禁止にし、header／hash／metadataだけで成功。

[実保存入力のpassive確認](../../artifacts/research_c_rc2_pr_native/2026-11-01/technical/passive_preflight.json)でSOURCE130／入力26／依存pinsを照合した。numpy／scipy／qiskit／ffsim／jaxのimport自体0。実H4 J/U decode／sampling／回路build-compile／保存値の新規数値再解析／科学callはすべて0。fresh run01は未作成。

## 6. 資源・STOP・次の判断

[read-only環境記録](../../artifacts/research_c_rc2_pr_native/2026-11-01/technical/environment.json)ではaffinity32 CPU、利用可能RAM約56 GB。既存resource_plan(64,256)の作業量見積り256 MiB/taskから、その時点の空き資源・cgroup quotaに応じてworkersを選ぶ。並列時BLAS1は過剰多重化を防ぐprocess-local調整で、CPU上限の追加ではない。実行直前に再確認する。共有venv／cache／global設定は変更しない。

既存M4-native32件の保存runtimeは919.83秒、B2互換64件は613.55秒だったが、rや候補が異なるため今回の実行時間・速度向上の保証には使わない。Qiskitの必要な排他を維持し、CAR/reference・独立taskを並列化する。任意のCPU/RAM/worker/wall上限は設けない。

新契約のcorrectness FAILではqueued taskをcancelし、開始済み処理を回収してpartial evidenceとfailed STOPを保存する。retry/resume=0、初期値fallback／追加fit／候補拡張／別namespaceでのやり直しはしない。optimizer UNQUALIFIEDと全law未資格は記録するが、捏造PASSにもcorrectness FAILにも置き換えない。

[保全postflight](../../artifacts/research_c_rc2_pr_native/2026-11-01/technical/postflight.json)で旧SOURCE・raw・STOP、新入力と新SOURCEを照合し、commit/push後にremote SHA/treeと主要本文を確認する。

技術blockerは現時点でなし。科学的には両fit未収束、全law合成誤差・実装law距離・費用support・actual CAが未資格で残る。

**次の担当：ユーザー／研究Cが固定655-call案の科学実行を別途明示許可するか判断する。Codexは許可後にfresh `artifacts/research_c_rc2_pr_native/2026-11-01/run01`を一度だけ実行する。今回の公開照合後はmandatory STOPし、新規実H4評価・総資源算定・重要レビューへ自動進行しない。**
