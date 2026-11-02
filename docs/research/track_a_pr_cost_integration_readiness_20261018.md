# Track A Codex 1：S2 PR A_COST・既存B/D会計の接続準備

2026-10-18。**実装・保存証拠のmetadata照合・人工検証まで。実科学実行は未認可、未seal。**
この文書は実測結果・新しいPR winner・科学的採否の報告ではない。

資料は [準備bundle](../../artifacts/resource_applicability/track_a_pr_cost_integration/2026-10-18/)、
候補・元資料のbyte来歴は [機械可読索引](../../references/pr_cost_integration_evidence_v1.json)。
SOURCE、検査結果、保全確認はbundleの `source_closure.json`、`artificial_tests.json`、
`preservation_audit.json`、公開状態は `publication_status.json` を参照する。

## 完了した技術作業

- S2 PR専用のA_SIGNAL→A_COST adapter、元RTEの非条件付き抽選、semantic seed登録、paired full wrapper、
  元scalar＋tail identityの一回適用、branch/X/Yと独立event oracle、固定compilerへの接続。
- 追加Dの実測優先・固定モデルの区別、保存shotに基づくpaired RZ総費用・SE、旧集合と強化集合の別比較。
- Bの条件付き標本・元law質量・通常標本を分離。完全な層分割証拠と元質量が不足すれば無条件平均はnull。
- H4/H6/H8×A_COST/D_ACCOUNTING/B_ANALYSISの9契約と**無効grant案**。
  SOURCE/input/runtime/compilerの不一致、欠測、cohort混合、重複実行を拒否する。
- 指定7commitをGitHubのbranch SHAと照合。29固定fileをblob/bytes/SHAで索引化し、
  Aの完了receipt・B/Dのnative終了と最終gateを照合した。既存資料そのものは再commitしていない。

実装は `src/trotterlib/pr_cost_integration.py`、`scripts/pr_cost_{contract,native,saved}.py`、
`scripts/run_pr_cost_integration.py`。metadata専用は `scripts/prepare_pr_cost_integration.py`。
旧runnerやモデルを並行して再実装せず、元law・native lowering・fingerprint cache・owned pool・
保存shot acceptanceを利用する。再利用APIと固定blobの対応は `source_reuse.json` に記録する。

## A_SIGNALからA_COSTへの接続

| 系列 | A_SIGNAL固定結果 | 保存候補 | 接続状態 |
|---|---|---|---|
| H4 legacy DF12 | [2588bb1f / A結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2588bb1ff8a03120050fdaa67b9cd30739131bec/artifacts/resource_applicability/track_a_h4_native_validation/2026-10-18/A_result.json.gz) | p2/4×q1/4/8×r4×K2/4の12、p3/q1/r2/K2、p3/q4/r8/K4、p3/q8/r4/K2/4の4 | 保存archive優先。なければ旧固定生成器で3 prefixを各一度だけ復元する案をA_COSTへ明記 |
| H6 tol-only DF19 | [648184eb / A結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/648184eb2a6f1c2406df17c1796eb5cc195908a8/artifacts/resource_applicability/track_a_h6_dba/2026-10-18/A_result.json.gz) | p3/4×q2/4/8×r4×K2/4の12、p5/q4・8/r8/K4、p5/q16/r2/K2、p4/q8/r8/K4 | 保存full archiveからp3/4 view、p5は元のfaithful:p5を保持する実装 |
| H8 tol-only DF28 | [839c4030 / A結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/839c403004cf9343730c43adb346024aea41999b/artifacts/resource_applicability/track_a_h8_ab/2026-10-18/A_result.json.gz) | p6/8×q2/4/8×r4×K2/4の12、p7/q4・8/r8/K4、p7/q16/r4/K4、p6/q8/r8/K4 | 元のexport/restoreとprefix_preparationを再利用する実装 |

すべてlinear 1Å/STO-3G/T=.8/P=0。H4はlegacy DF12の保存状態。
H4新tol-only DF10/新状態は別系列であり、このA_COSTやlegacy比較に混入しない。
H4のR追加12候補やT/FP panelも今回のA16へ追加しない。

保存済み通常/経験的非相殺×5精度×倍率1のどれかが適格な候補だけを費用取得対象にする。
適格flagの投影はmetadata照合であり、新しいbias・shot・normalizationの計算ではない。
新標本は各系の**探索n8**として提案する。既存確認n32へのpoolingや「独立確認済み」への昇格は禁止。
将来のwinner確認は別のSOURCE/標本/実行承認へ分離する。

### 個別に導出したA_COSTの提案call数

H4/H6のn8は、それぞれの既存探索cohortとの対応を根拠とした独立の提案。
H8契約の数値をそのまま他系の認可上限に移していない。以下は保存flagで対象となる16候補と
各候補のq/rを静的に数えた値で、**実行許可・実績ではない**。

| 系列 | trajectory / evolution | wrapper compile | occurrence | RTE event | control probe / 独立event oracle |
|---|---:|---:|---:|---:|---:|
| H4 | 128 / 128 | 256 | 584 | 2,448 | 64 / 16 |
| H6 | 128 / 128 | 256 | 736 | 3,328 | 64 / 16 |
| H8 | 128 / 128 | 256 | 736 | 3,584 | 64 / 16 |

signal/SCF/DF/state/reference生成・fit・量子measurement shotsはすべて0。
H4だけは保存archiveがなければnative preparationの復元を最大3件提案する。
旧H4 provider217c971dとc31ef424の `pr2_matched_accuracy_m1_execution.py` は同じSHA
`7528ebae754e14ee623d28212a668e4edc237575775a78d0d2bf4d34dc53c293`。
この固定関数と同じsnapshotからp2/3/4を各一度構築し、hash・scalar・tail identityを保存し、
元A_SIGNALのnormalizationとcontrol/oracleを照合する。新SCF・DF分解・state生成ではない。
この復元も将来のA_COST個別承認に含める案であり、今回は実H4入力で実行していない。
各候補の先頭trajectoryで4論理probeとevent oracleを通し、同じstageの先頭群がPASSしてから残りを並列取得する。
before-call予約を残し、失敗・未完了分は返却しない。独立した他stageまで一括停止しない。

## D：実測単価を優先するbinding

| 系列 | 新しい保存実測 | 既存比較へ加える範囲 |
|---|---|---|
| H4 legacy DF12 | [2588bb1f / D完了receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2588bb1ff8a03120050fdaa67b9cd30739131bec/artifacts/resource_applicability/track_a_h4_native_validation/2026-10-18/D_completion_receipt.json) の8構成16軸 | 旧固定16＋実測追加部分集合。未測定88 Dを含む全96最良は認定しない |
| H6 DF19 | [648184eb / D保存結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/648184eb2a6f1c2406df17c1796eb5cc195908a8/artifacts/resource_applicability/track_a_h6_dba/2026-10-18/completion/D_terminal_result.json)、B0I p11/q4 S4、両軸258,653 RZ | V1の保存signal/shot・固定modelとjoinし、この構成の単価だけ実測優先 |
| H8 DF28 | [bc9cceac / 追加D](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc9cceac6088d798dc0c6dca92490066258860e5/docs/research/track_a_h8_dba_20261018.md)、B0 p15/q8 S2=446,288、B0I p16/q4 S4=713,432 RZ/軸 | 6軸反映済みの通常/非相殺保存会計を起点に、追加2構成の単価を置換 |

上の数値は既存報告の再掲。今回新しい差・費用・順位・削減率は計算していない。
compiler hashは共通でも、environment profile・DF入力・状態・control・preparation来歴の一致を別々に要求する。
同一単価に見える異系列は混ぜない。実測同士が衝突すればSTOPし、モデルへ黙って戻さない。

scalar会計runnerは5精度、通常/経験的非相殺、倍率1/10、cosine/sine別保存shot、
shot重み付きfull-wrapper RZを扱う。PRのSEは軌道ごとの `N_cos*C_cos + N_sin*C_sin` から計算する。
軸のSEを独立として足さない。モデル条件付きの点比較と、すべて直接実測した比較を混同しない。
保存shot/mode/factorが欠ける場合は `ACCOUNTING_MISSING` とし、別modeから複写したり0補完したりしない。
今回のA_SIGNAL結果は倍率1の登録なので、倍率10の欠けた保存会計をこの準備段階で生成していない。

## B：条件付き費用と元lawの分離

| 系列 | 取得済み条件付き群 | 元証拠と不足 |
|---|---|---|
| H4 legacy DF12 | p3/r4/K4、q4/8×s2_multi/s4、各n8 | [97d25cc4 / B](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/97d25cc431516d720ce91c7cb4f148e061d8ea34/artifacts/resource_applicability/track_a_h4_followup/2026-10-18/B_completion_receipt.json)。元distribution/log_stratum_massのprivate receiptと通常標本の層分類を未結合 |
| H6 DF19 | p5/q8/r2、K2 s2_multi、K4 s4、各n8 | [648184eb / B](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/648184eb2a6f1c2406df17c1796eb5cc195908a8/artifacts/resource_applicability/track_a_h6_dba/2026-10-18/completion/B_terminal_result.json)。K4 s2_multiは未実測。K2群とK4群は別lawなので互いを補完しない |
| H8 DF28 | p7/r4/K4、q4/8×s2_multi/s4、各n8 | [839c4030 / B](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/839c403004cf9343730c43adb346024aea41999b/artifacts/resource_applicability/track_a_h8_ab/2026-10-18/completion/B_terminal_result.json)。旧s2 pilot、通常n32、元distributionと層分類を別receiptで結合する必要 |

`s2_multi` はN2>=2,N4=0、`s4` はN4>=1,N2任意。
conditional meanをそのまま通常n32平均へ代入しない。通常の有限標本平均・SEは独立に保持する。
将来のlaw-weighted診断は、zero/single/multi/s4等の重複しない全support分割、元lawの質量、
各層の費用がすべて揃う場合だけ行う。浮動小数の曖昧な質量を事後的に再正規化しない。
未観測層・未測定費用はnull。層間共分散/独立性の証拠がなければ無条件SEもnull。
この補助点推定を母平均保証・厳密bias保証と呼ばない。

## 実行前に残るtechnical bindingと認可

1. **A_COST各系**：privateなSOURCE/input/runtime/元preparation identity/旧master・派生seed除外台帳を結合する。
   H4の公開入力receiptにarchive roleはなく、private原本の有無は未確定。
   既存archiveか、planに登録した旧固定生成器による最大3 prefix復元かを新configへ明示する。
   暗黙のfallbackは行わず、個別に承認されるA_COST planへモードとsourceを結合する。
2. **D_ACCOUNTING各系**：旧signal/保存shot/固定modelとDのinput・scalar・preparation来歴を結合した
   scalar payloadをsealする。投影adapterは実装済みだが、実データへの適用・新費用再会計は未実施。
3. **B_ANALYSIS各系**：元確率receipt・既存通常標本の層分類・完全性を結合する。不足なら診断は未確定として終了できる。
   新しいsamplingやcompileで不足を補う承認は含めない。
4. **ユーザー**：行う単位について新SOURCE、個別call予算、input/profile、fresh資源とSTOP条件を明示承認する。
   この準備依頼を有効grantへ転用しない。9草案はapproved=false/runtime_authorization=false/allowed_cpus=[]。
5. **Codex**：承認された単位だけfresh gateとexclusive output/one-shot lockを通し、一回実行してSTOPする。
   新規wrapper compileと、保存値の総費用再会計は別認可。B/D/Aの完了から他stageへ自動進行しない。

新しい環境のinstall/upgradeやvenv生成は不要。人工検証は既存private venvを変更せず実施した。
本番compiler採用や旧compilerとの完全なoutput同一性を今回の人工検査から認定しない。

## 並列化・資源・STOP

A_COSTはsystem別実行領域を使う。preparation・DF blocks・basis metadataはprefixごとに一度復元し、
owned fork workerへCOWで共有する。cosine/sine間ではevent列・evolutionを共有し、fingerprint cacheを利用する。
master/派生seedはsystem/input/T/candidate/phase/replica/SOURCEから固定し、worker番号や完了順は使わない。
既存n8/n32/希少層のmaster・派生seedと衝突すればSTOPする。

CPU/RAM/worker/wallに過去の恒常上限を移送しない。起動前の利用可能資源と並行jobを確認し、
使用中CPUを除いてworker数を選ぶ。worker内部thread1で入れ子並列を避ける。
B/Dの小さなscalar集計は単一processを基本とし、同じ集計や入力読込を繰り返さない。
PSI/無関係worktree更新だけで正常workerを止めない。科学的不一致、実OOM、own worker異常、
実際に使うSOURCE/input/runtime破損は当該stageをfail closedし、後続を起動しない。
old STOP・grant・ledger・markerは保持する。共有OS/他ユーザー/他jobを変更しない。

## 人工検証・保全

会計/入力・SOURCE改竄拒否/9scope/無効grantの実CLI/one-shot/first STOP/条件付き混合防止と、
2モード合成DFの元RTE→scalar/control/event oracle→paired native compileを検査した。
正確な件数・log hash・実行環境はbundleの人工検査receiptへ固定する。
人工fixtureだけのwrapper compileは2、実H4/H6/H8の新しい科学callは0。
これは作者自身の技術検査であり、別担当の独立レビューとは呼ばない。AI subagentは0。

専用独立cloneで作業し、既存rootの958 staged削除を含む状態を保全する。
高次PF/new_2研究・理論資源比較・他branch/worktreeのSOURCE・成果・grant・ledger・lockには触れない。
大容量raw、NPZ、QPY、venv、有効grant、credentialは公開対象にしない。

次の担当：ユーザーがA_COSTと保存B/D会計のどの単位を承認するか判断する。
Codexはその指示まで科学実行を行わず、準備成果公開後にmandatory STOPする。
