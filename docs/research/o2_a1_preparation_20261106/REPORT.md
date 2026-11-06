# A1 Control Variates：最小実装・実行前準備

2026-11-06 JST。主エージェントのみ、独立AIレビューなし。
branch：`codex/o2-a1-preparation-20261106`。基点：`1bcf70ba69f91c62763a8739ec620b040b097ed9`。

**保存canonical PRから一roundの候補を特定し、古典CV後処理・同targetのraw/層別接口を実装した。人工テスト15件PASS。科学実行は未開始で、実行条件はまだ閉じていない。**
推奨候補はlinear H4/1.00 Å/STO-3G/legacy DF12/L_D3、保存最終opt2のm0/q1/δ.02/r1/K2、X/Y両軸。
新しいscienceの採択や改善認定ではない。物理Hへの結合と半径、実CDF/finite-target bias、full価格・confidence・有限測定scopeの確定が先に必要。

## 1. 成果物

| 成果物 | 内容 |
|---|---|
| [A1_PREFLIGHT.md](A1_PREFLIGHT.md) | 入力原本・state/model限定、保存round/予算、実law、phase/回路policy、価格、centeringと障害 |
| [A1_EXECUTION_PLAN.md](A1_EXECUTION_PLAN.md) | 独立train/cal/prod、強いbaseline、bounded confidence/費用、未固定条件、成功/不改善/STOP |
| [a1_control_variates.py](../../../src/trottertracks/algorithm_codesign/a1_control_variates.py) | science入口を持たない最小古典adapter |
| [人工test](../../../tests/tracks/algorithm_codesign/test_a1_control_variates.py) | invented metadata・手書きbit・小さい古典列挙だけを使用 |

このdirectoryの3文書と上記2ファイルだけを追加する。新しい科学結果・正式採択・科学validation経路を作っていないため、旧概要/manifest/科学runnerの索引は変更しない。

## 2. 実施した技術作業と入力可用性

指定の固定調査4文書、適用AGENTS、PROJECT_MAP、研究概要全文、関連validation status/manifest、共通指示2026-10-25と個別opt2契約を確認した。
CV4Quantum v1の実測CV/factorization/fitとMaurer–Pontil v1 Theorem4/10の適用条件を一次SOURCEで確認した。網羅的新規性調査・重要科学レビューはしていない。

保存V1/V2/opt2/信号gridの入力metadataとround欄を静的に照合した。既存判定を引用し、新しい統計・数値再解析はしない。
保存NPZはGit blobのbyte hashだけを確認し、記録SHA256に一致した。H/state/operator生成、NPZ decode、保存meanからvarianceや費用を計算する処理は0件。
元DF modelのphysical sector groundとPF control stateは区別する。未近似の物理Hへの結合、厳密単位半径は未保証である。

標準short wrapperとfull価格providerのq受理は1,2,4。最終opt2 m0は一つのoccurrenceで最小再利用に適する。
ただし元opt2はsupport_run_le_1と同じcompilerのcomplete wrapper価格を使うため、標準full-basis providerの価格をそのまま代用しない。
T/native価格は未取得。元の主指標RZと混同せず、未知価格を0にしない。

## 3. 再利用と新規実装の理由

| 既存SOURCE/関数・test | 実際の扱い |
|---|---|
| rte.py finite_rte_distribution/_make_event/iter_sample_rte_events | 静読しorder→labels、selected順、phase、finite lawを接続。samplerは呼ばず変更なし |
| Hadamard wrapper、df_rpe_hadamard_compiled_cost、partial_s2/repeated | staticに軸/符号/full scopeと版を確認。carrier4591b8dと基点の6 coreファイルは同じ。build/compile呼出しなし |
| full_scope/sequence-policy、opt2契約・保存結果 | 既存selected回路と価格profileを確認。量子機構の再実装なし |
| parent_allocation.py rational/integer | **直接importして再利用**。浮動小数点の無言変換を避ける入出力検査 |
| return_aggregation.py Interval | **直接再利用**したrange表現。dyadic_indexは人工testで再利用。新proposal生成はしない |
| Parent event_statistics、design.ledger、S2 evaluate_law | M2/range/bias/full費分離を静読。未知CV covarianceを扱わない旧ledgerを無変更でCVへ流用しない。科学/S2 runnerは起動なし |
| NumPy1.26.4 lstsq | pilotのintercept付きOLSに**既存実装を利用**。今回は人工bitのみ。独自solver/regularizer探索を作らない |
| 既存test_rpe_hadamard_interrogation.py、test_rte.py、sampling_law_freedom、S2/Parent test | 適合性を静読。既存wrapperのbit mapping/Sdg→HをASTで確認する人工検査を追加。量子build/生成や科学scopeを含む旧test一式は実行しない |
| V1/V2 decoder | 符号・branchの静的対応を再利用。旧binary tail関数はCVへ不適合のため呼び出さない |

既存コードには「実bitとlocal controlを結び、凍結係数・独立3標本区分・補正rangeを持つ」部品がなかったため、新しい小moduleを作った。
量子回路・sampler・汎用pipeline・承認機構は追加しない。

実装した接口：

- law_from_choice_cdf：実binary64 CDFから53-bit格子の質量をexact rationalで取得する接口。実CDF未入力。ideal pやParent dyadic proposalを実qと偽装しない。
- EventMetadata/bind_wrapper_measurement：既存products→rotation順、path/wrapper、axis、bit、weightを保持。phase/signを実測値へ二度掛けない。
- ControlBank：order条件付きlocal/pair indicator、factorized既知平均・joint moment、小さいfeature stratumの質量。全path表の生成なし。
- fit_and_freeze/FrozenCV：trainだけで係数を学習・凍結。λ=0でrawへ戻る。cal/prodへの補正、range enclosureとcentering bias入力。
- Batch/evaluation_inputs：scope一致とstream/trial再利用を検査。stream名の差だけで科学的独立性を証明したとはしない。
- raw_mean/stratified_mean：同じtargetでrawと全正層を扱う。層内取得/配分は外部の事前固定処理で、無補正のIID poststratificationはしない。
- TrialCost/account_costs：train/cal/prodのfull no-prep価格、T/CX/RZ/1Q/depth、古典秒・memoryを分ける。未知価格はNone、同方式の重複trialは拒否。

Fractionによる後処理は正しいcentering/rangeのための最小参照実装で、速度やmemoryの科学評価は未実施。
finite集合は実際のoutcome前に選ぶ必要があるが、今回のmock bankを科学bankとして採択していない。

## 4. 人工テスト

環境：Python3.12.3、NumPy1.26.4（既存環境を変更せず使用）。実行コマンド：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover \
  -s tests/tracks/algorithm_codesign -p test_a1_control_variates.py -v
```

最終結果：**15 tests / OK**。主エージェントによる自己確認で、独立科学レビュー・CIではない。

| 確認 | 範囲 |
|---|---|
| centering | invented lawでknown mean/joint momentを小さい古典和と照合。重複position/矛盾条件も確認 |
| finite law | artificial CDFの境界、1/10と格子質量の差、既存dyadic_indexのbit mapping。RNGなし |
| phase/axis/weight | 手書きmetadata/bit、mock wrapper、既存readoutのAST。X/Y、signed weight、相対phaseの二重補正なし |
| raw/fit/freeze | λ=0、手書きlinear応答の学習係数、immutable係数、train限定 |
| independence | stage/scope不一致、同stream再利用、duplicate trial拒否。数学的IIDや未実行campaignの証明ではない |
| range/error | mock全indicator組合せ・両端をenclosureが包含。centering誤差の増幅を検査 |
| support/invalid | 不正CDF/law/label/order、非有限値、過大weight、欠けたpath |
| 強い対照/費用 | 全正strata、欠層/mismatched targetの拒否。全3段費用、重複・非full価格拒否、未知Tの保持、rawへの不要train非強制 |

人工PASSはPRでの有用なcovariance、分散削減、shot節約、総費用改善の実証ではない。
実Hamiltonian、科学保存結果、RNG、Hadamard simulation、量子回路、native synthesis、科学runnerをtestへ混ぜていない。
confidence公式・N・科学variance/covariance・科学総費用は今回数値計算していない。

## 5. 科学実行の準備状態と次の担当

| 問い | 今回の回答 |
|---|---|
| 入力/roundを特定できたか | 保存cohortとして可能。推奨m0を提示。物理Hを保証した新実行契約ではない |
| νは取得できるか | 実CDFと既存IIDモデルを固定できれば有限格子lawから取得する接口あり。実科学CDF/momentは未取得、actual-q target biasも残る |
| raw/層別/CV比較は成立するか | 同targetの古典接口と公平な計画は成立。実際の条件付き層取得、価格、confidence/failure配分が未接続 |
| 今すぐ科学実行できるか | **できない**。入力保証範囲・実law/centering・残余幅・control bank・有限train/cal/prod/層配分・価格scopeが未固定で、明示的科学実行許可もまだない |

次の**ユーザー／GPT**は、モデル限定診断の採否、m0、具体的bank/比較、精度・失敗配分・測定数と価格取得の範囲を判断する。
次の**Codex**は、その固定scopeを明示的に許可された後だけ、不足する小adapterと科学測定・統計/費用評価を進める。
重要科学レビューの許可は科学実行許可と独立。今回の実装許可・commit/push許可を科学実行許可に読み替えない。

## 6. 環境保全・公開・mandatory STOP

開始時のmainは `0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`、多数のstaged deletionがあった。既存index/filesにcheckout/reset/clean/stageを行わず、指定固定commitから専用worktreeを新設した。
作業領域は `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/o2-a1-preparation-20261106`、originは `https://github.com/HIROMU1015/Partially-Randomized-Trotter`。
物理SERVER1/2はIDEのTSV名やdirectory名から推定していない。

main・旧branch・他sessionのdirty/untracked・科学SOURCE/input/raw/grant/ledger/marker/STOPを変更しない。C1凍結、S2/R1負結果、旧one-shot/retry制限を維持する。
共有venv/package/cache、scheduler/quota/OS、他ユーザーのprocess/filesを変更しない。科学workerを起動せず、他job停止・再起動もしていない。
恣意的なCPU/RAM/worker上限は新設せず、sub-agentも使っていない。

この5ファイルだけをcommit/pushし、公開後にremote SHAと5本文を再取得・照合する。確定SHA・照合実績は最終チャットに報告し、自己SHAを埋める追加commitは不要とする。
通常pushの認証設定を変更せず、必要なら許可されたGitHub接続を利用する。

**preflight・最小adapter・人工テスト・実行計画・公開照合でmandatory STOP。新しい科学測定、variance/N/費用評価、build/compile、重要科学レビュー、C1再開へ自動進行しない。**
