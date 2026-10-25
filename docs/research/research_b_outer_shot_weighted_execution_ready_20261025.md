# 研究B：Shot加重資源費用の実装・科学実行準備

2026-10-25 JST。**IMPLEMENTED／MOCK_VALIDATED／SCIENCE_NOT_EXECUTED**。作業起点は `488d6ab544295905dd99fa2c6357202d9b60ea31`。[固定設計](research_b_outer_shot_weighted_resource_design_20261025.md)と今回の指示に基づき、保存値adapter・入口・人工テストを実装した。実候補のnormalization enclosure、shot数、G、rare確率追加4件、費用比較・逆転境界は未評価。旧科学SOURCE・原本・STOPと科学的分類を変更していない。

## 実装と再利用

- [`shot_weighted.py`](../../src/trottertracks/b2d_evaluator/shot_weighted.py)：固定保存値の取得・結合、scalar interval・shot・paired資源費用・未知rare費用感度と資格のadapter。
- [`run_b2d_outer_shot_weighted.py`](../../scripts/run_b2d_outer_shot_weighted.py)：`--inspect`、`--preflight`、`--execute`の排他的な薄い入口。
- [`test_b2d_shot_weighted.py`](../../tests/management/test_b2d_shot_weighted.py)：toy保存schema／scalar値／fake256 paired観測だけの人工検証。

新規production変更は上記adapterと入口だけ。既存の`rpe_resource_accounting._hoeffding_shots`の数学式、`pr_dictionary_validation.intervals`の`scalar_context/rational_interval/endpoints`、`gates.shot_count_from_enclosure`を再利用する。`f2_statistics`のscalar計算方針と旧PRのpaired会計を確認したが、RPE位相・半径・round、F2の4arm、他研究のcohort固定・grant条件は持ち込まない。既存module・guardは変更していない。

`rare_sensitivity.dec/pack/require`、B0 compiler profile、`followup_io.registered_scope/FollowupLedger`、`preflight._check_source`、`launcher.Evidence`の出力・部分結果・失敗・STOPも再利用した。新しい認証・承認証明・grant機構はない。ユーザーのチャット上の許可後にCodexが通常の`--execute`を起動する運用であり、実行プログラムが許可を証明する構造ではない。

## 固定条件と入力

Synthetic 2-mode one-body＋3 DF因子、N=1／mode0 occupation、L_D=2 weight-ranked prefix、n_D=1、T=±0.18。分子geometry／AO basis／分子DF rank探索は非該当。単一physical complex coherent signalの直接絶対誤差を対象とし、operator誤差・raw誤差・RPE位相へ置換しない。

| ID | Outer | D内部 | q/f | r/K |
|---|---|---|---|---|
| B3 | Yoshida4 | A2 | q2 | 2/2 |
| D1 | Yoshida4 | A3 | q2 | 2/2 |
| R2 | Yoshida4 | A3 | q2 | 4/2 |
| R4 | Yoshida4 | A3 | q2 | 2/4 |
| B4 | SABA2 | A2 | q2 | 2/2 |
| F1 | Refined-S2 | A2 | q2,f8 | 2/2 |

ε=1e-8／1e-9、X/Y。Full prepared wrapper単価はB1のQiskit1.3.0、rz/sx/x/cx、opt1、seed17、approximation1、`qubits_initially_zero=False`、topologyなし、`official_definitions_angle_preserving_native_v5`を維持する。回路は再生成・再compileしない。

Preflightは次の13入力fileを各固定結果commitのgit blobと照合する。全依存のhashやbitwise環境一致ではなく、再利用する科学原本の同一性確認である。

| 入力 | 固定結果commit | file／用途 |
|---|---|---|
| 22行 | `33715848bfb83d49ec4e54d6eef481012a7441e4` | `analytic-once/{result,inputs}.json`：全6候補共通bias・normalization・全R stage |
| A0 | `31662b49157b15f2d1c03eef1c2faa07e2f905e7` | `independent-once/{result,inputs}.json`：B3/F1/R4・両Tの補助記録。主biasへ差替えない |
| B0 | `0db9b3ed189eb6cd9d66e070993dbd0e72d29104` | `corrected-2/result.json`：native profile・preparation／tail provenance |
| B1 | `bef63edbd8cc564444b44e4e8cf23baa90f05fe5` | `iid256-once/{result,inputs,graph,binding}.json`、`SAMPLES.csv`、`STATISTICS.csv`、`iid256-once/trajectories.jsonl`：7file |
| Rare | `cae3cd784dbdfcf2fe96902345379aca4b9606c7` | `saved-once/result.json`：既存8確率・CONDITIONAL_SENSITIVITY_ONLY |

実行時は全12条件の結合を数値会計より先に終える。科学SOURCE、Hamiltonian／sector／state／DF split／array／fixture identity、T・outer・内部法・q/f・n_D・r/K、全signed occurrence／seed／law／normalization、X/Y sample ID・physical_key・n256、cost metricとprofileを照合する。A0と22行の観測bias予算flagが異なる場合は`INCONCLUSIVE_INPUT_NUMERICS`を残し、有利なbiasへ差し替えない。

## 数学式と数値表現

各occurrenceのsigned binary64 τとKに対し、既存paired有限Taylor scalar定義からB_jをoutward評価し、`𝒩_model=∏_j B_j^r_j`を囲む。負τはeventの符号に残し、normalizationのweightはabs(τ)を使う。Cacheは同じabs(τ),KのB_jだけを共有し、繰返しoccurrenceの因子を省かない。実planの88 occurrence・200 short-step因子を全て反映する。

22行の保存全体normalizationを`N_used`とし、`[N_model_low,N_model_high]`と区別する。`U=max(N_model_high,abs(N_used))`をshot rangeの安全側上界に使う。演算は既存mpmath1.3.0の90dps outward interval、端点はexact Fraction。保存binary64値を正確にliftし、入力モデルを高精度物理計算へ置換しない。

`s=2ε/5`、`alpha_X=alpha_Y=1/40`、`eta_a=s/sqrt(2)`。各quantum shotでfresh IIDの全RTE trajectoryを抽出するbounded-outcome ±1モデルの下で、

`m_a=ceil(4*U²/s²*ln(2/alpha_a))`

を保守的な十分shot数として評価する。sqrtを消した同値式で、RPE半径・phase仮定は使わない。Logをoutward評価した式の下端／上端のceilが一致した場合だけPython整数を返す。非有限・ceil straddleは拒否し、`SHOT_COUNT_UNRESOLVED`として失敗・部分結果を保存してSTOPする。実際のquantum shotは実行しない。

N_usedのscale差は`rescaling_bias_upper=sqrt(2)_upper*max(|N_used-N_model_low|,|N_used-N_model_high|)`へ別記し、shot上界を広げただけで平均biasが消えたとしない。これはsignal全体のu_numやsampling bit-law／compiled meanのγ証明ではない。

Bias＋u_num予算はε/2、γ予算はε/10、統計は2ε/5。各(candidate,T,ε)のX/Y union boundで95%とし、候補・T・精度間familywise配分はnullのまま。量子測定αと費用MCのSEを区別する。

Shot数はdecimal文字列、費用整数積・mean・variance・covarianceはPython整数／Fractionで保持し、巨大整数をfloatに変換しない。標本SEのsqrtはinterval端点として保存する。保存B1のbinary64 mean／不偏variance／SEも原値のまま併記する。同じ整数原本からexact mean／joint momentsを求めるため末尾丸めが異なり得るが、新しい標本や別cohortへ置換したものではない。

## Paired費用と未知rare項

各caseの256 paired IDを保ち、各metricのjoint momentをε間で共有する。`W_i=m_X*C_iX+m_Y*C_iY`の分散と同値な

`Var_sample(W)=m_X²*s_X²+m_Y²*s_Y²+2*m_X*m_Y*s_XY`

を整数raw momentsからexact評価する。`SE_S0=sqrt(Var_sample(W)/256)`。同じtrajectoryから得たX/Yを512独立標本とは扱わない。

`G_S0=m_X*mu0_X+m_Y*mu0_Y`はS0条件付き参考値。未知rare平均を`u_X,u_Y>=0`に残し、

`G_hat(u)=(1-p_rare)*G_S0+p_rare*m_X*u_X+p_rare*m_Y*u_Y`

を出力する。真の無条件母平均にはS0経験平均と真のS0母平均の差も残る。未知平均・未知variance・無条件G／SE／winnerはnull。R2／F1の両Tだけ、保存p0から追加4件のscalar p_rare評価を将来実行する。B3/B4/D1/R4の既存8確率は読取り利用し、旧感度解析自体は再実行しない。

比較A=P:B3/Q:B4、B=P:R4/Q:D1、C=P:R2/Q:D1とし、`A=(1-p_P)*G_S0,P-(1-p_Q)*G_S0,Q`、`M_i=m_iX+m_iY`から

`u_Q^m*=A/(p_Q*M_Q)+(p_P*M_P)/(p_Q*M_Q)*u_P^m`

をshot加重後に新たに計算する。`u_i^m`はshot加重rare混合平均の定義で、両軸同費用仮定ではない。未知X/Y別係数も保持し、旧単価の境界を無条件転記しない。S0の±1／2SE感度は、候補間の共分散や独立性を仮定せず矩形端点で表示する。Confidence intervalやrare tail保証とは呼ばない。

## 固定scopeと出力・STOP

| 登録境界 | 件数 |
|---|---:|
| `shot.bind`：保存binding／case取得 | 12 |
| `shot.normalization`：range enclosure | 12 |
| `shot.rare`：R2/F1追加確率 | 4 |
| `shot.joint`：paired moments | 72 |
| `shot.bias`：観測bias予算比較 | 24 |
| `shot.count`：X/Y shot要求 | 48 |
| `shot.resource`：資源行 | 144 |
| `shot.compare`：3組×2T×2ε×RZ/CX | 24 |
| `shot.qualification`：資格組立て | 24 |

高位境界の内部scalar演算・cacheは別の科学callへ二重計上しない。`--inspect`で全identityを数値importなしに列挙できる。上記以外の候補・精度・時間・metric・call追加、retry／resumeは拒否する。

Fresh namespaceは `artifacts/research_b_outer_shot_weighted/20261025/saved-once/`。今回未作成。許可後の実行は排他的mkdirと新規ファイル作成だけを用い、既存結果を上書きしない。

- `provenance.json`：SOURCE、13固定入力blob、scope、依存版・Python・platform・資源・policy。
- `result.json`：12 normalization／rare、72 joint moments、48 axis shot、144資源、24比較・資格。
- `SHOTS.csv`48行、`RESOURCES.csv`144行、`COMPARISONS.csv`24行。分数は文字列、巨大mはdecimal文字列、未知値は`null`。比較とG_S0の±1/2SE値は各行内に5端点を保持し、追加仮想費用gridを作らない。
- `events.jsonl`、`partial_results.json`、`status.json`、`STOP.json`、失敗時のみ`failure.json`。各CSV行はflush/fsyncし、完了したnormalization・joint・bias・shot・資源・比較・資格を部分結果へ保全する。出力上限は総32MiB、JSON単体8MiB、STOP reserve256KiB。失敗保存も人工確認した。

予定された`u_num=null/gamma=null`・rare未取得は、完了状態`COMPLETED_DESCRIPTIVE_MANDATORY_STOP`、科学状態`INCONCLUSIVE_QUALIFICATIONS_AND_RARE_COSTS`として分ける。観測bias予算外でも参考shot・費用は保存し、同精度適格候補へ認定しない。全24条件のdeterministic／implementation／end-to-end資格は未取得ならnullを維持する。

入力・correctness不一致、数値失敗、ceil未確定、scope不一致、技術停止は予定INCONCLUSIVEと別のfailure categoryで保存する。証拠保存失敗は`EVIDENCE_SAVE_FAILED_MANDATORY_STOP`。失敗後に別outputで自動attemptを開始せずSTOPする。

## 人工検証と性能方針

Focused人工45件PASS：新adapter10、既存設計5、既存interval5、FollowupLedger10、既存launcher15。新adapter最終変更後の10件もPASSした。

確認内容はtoy signed/K4 scalar normalization、interval／巨大整数ceil、straddle・非有限拒否、SOURCE・H／sector／state／DF split・candidate結合不一致、paired ID、covarianceを含むexact W分散、unknown rare係数／shot加重境界、null資格と予算外参考表示、全固定scopeのmock出力48/144/24行、scope超過／未完了、保存失敗、fresh output・partial・STOP、inspect import barrier／明示実行指定。

全条件mockではnormalizationとshot評価をopaque stubへ置換し、実候補の数値を一切読んでいない。個別scalarテストはtoy値だけを用いた。NumPy／SciPy／Qiskit／PySCF／trotterlib packageはimportしていない。Mpmathはtoy interval試験内だけで使用する。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B tests/management/test_b2d_shot_weighted.py -q
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B tests/management/test_b2d_shot_design.py -q
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B tests/management/test_pr_dictionary_f2_scalar_intervals.py -q
/usr/bin/python3.11 -I -B -S tests/management/test_b2d_followup.py -q
/usr/bin/python3.11 -I -B -S tests/management/test_b2d_stage1_launcher.py -q
```

保存データを一回readし、B_jとjoint momentをcache・ε間共有する。dominant workは既存JSONL/CSV走査と小さなscalar処理であり、process poolやIPCを追加する合理性はないため直列を選んだ。CPU／RAM／worker／wall timeに人工capは設けず、preflightでaffinity・空きRAM・loadを記録する。共有設定・他jobへ干渉しない。実候補の実測性能は未取得で、speedupは認定しない。

## SOURCE固定・read-only確認・許可後command

科学実行SOURCE：`9bebdc46ebd60039f315da9b6c6e57bd9c004b85`。公開branchは `codex/research-b-outer-shot-ready-20261025`。このSOURCEの実装を変更せず実行するclean detached checkoutを `/tmp/prt-research-b-outer-shot-science-20261025` に準備した。報告書追補commitは科学SOURCEと区別する。科学実行は未実施。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-research-b-outer-shot-science-20261025/scripts/run_b2d_outer_shot_weighted.py \
  --execute --source-commit 9bebdc46ebd60039f315da9b6c6e57bd9c004b85 \
  --output /tmp/prt-research-b-outer-shot-science-20261025/artifacts/research_b_outer_shot_weighted/20261025/saved-once
```

同じcommandの`--execute`を`--preflight`へ変更すると、SOURCE・13入力blob・scope・mpmath version・環境・fresh outputだけをread-only確認する。`--inspect`はSOURCE引数なしでmetadataだけを出力する。どちらも`read_payload`、実normalization、shot／G／比較・追加確率を呼ばずoutputも作らない。

固定SOURCEのclean detached checkoutでCLI `--preflight` が **READY_READ_ONLY_NO_SCIENCE**。13入力blobが各指定commitと一致し、上表の全scope、mpmath metadata version1.3.0、fresh output未使用を確認した。確認時は利用可能CPU32、`MemAvailable: 52712728 kB`。環境全体のbitwise一致は要求していない。

加えてread_payload／normalization_enclosure／scalar_shots／evaluateを呼べば例外にするstubに置換したread-only確認もPASS。NumPy／SciPy／Qiskit／PySCF／mpmathのimportがなく、出力namespaceも未作成、science checkoutはcleanのままである。これは保存値の数値結合・enclosure・shot／Gを実行したという意味ではない。上の `--execute` commandは今回起動していない。

## 未解決事項とmandatory STOP

人工PASSは実候補のnormalization enclosure・ceil確定やscientific適格性のPASSではない。実候補のscalar数値とjoint統計・比較はユーザーの別途明示的許可後に初めて評価する。u_num／γ、finite-bit sampling・実装meanの保証、未観測rare平均・全support上界、候補間familywise confidenceは未取得のまま。

SOURCE・人工テスト・技術報告の公開とremote内容照合後は**mandatory STOP**。次の担当は研究Bのユーザー：固定24条件・48 shot要求・144資源行・24比較の一回の科学実行を許可するか判断する。Codexはその指示後にだけ固定SOURCEの通常commandを起動する。今回、shot数・G・追加MC／A1／γ評価や次stageへの自動移行は行わない。
