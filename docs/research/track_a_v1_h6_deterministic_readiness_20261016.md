# Track A V1：H6決定論的打切り対照の強化・実行準備

2026-10-16 JST。**実装・人工検証・実行案を準備した。新H6 signal、full-wrapper compile、grant発行・消費は0。**
起点はmetadata監査 `46c06e5653152eb4ce786d69d927ecbf12f9a2de`。
専用branchは `track-a-v1-h6-deterministic-readiness-20261016`。
[証跡・計画入口](../../artifacts/resource_applicability/track_a_v1_h6_readiness/2026-10-16/README.md)。

## 比較契約

linear H6/1.00 Å/STO-3G、保存DF rank19、12 modes/sector400、保存Hamiltonian/state、signed generation order、T=.8を維持する。
主費用はP=0の `Σ_axis N_axis × compiled measured-full-wrapper RZ_axis`。
symmetric-directional control、ancilla12、cosine/sine、rz/sx/x/cx・opt1・seed17を保持する。
入力は既存NPZ 150,052 bytes/SHA `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`、
archiveは既存5,670,237 bytes/SHA `ad52f9f6a3150981d7be00191d8b3a9f9af992c37c9102cba6d94fcffc8e3010`。
今回NPZはbyte/hash確認だけで配列を開いていない。

既存104候補（B0/B0I各12、full S2/S4計8、PR72）を保持する。
S2 p6〜14×q1/2/4/8×B0/B0Iのうちp10の8候補は既存再利用。
新規はS2 64候補、S4 p11〜15の40候補、計104候補。
PR確認union8/n32を主比較とし、探索72/n8は保存して別scopeとする。
PR p3/p4×4qはexact-tailを入れた外側S2 biasの8診断で、正式候補・normalization・shot・費用比較には追加しない。

分母は各精度・allowance factor・bias規則で適格なB0/B0I/full S2/full S4の最小。
旧格子S2、細格子S2、細格子S2＋打切りS4の3集合を別に再選択し、格子とS4の効果を分離する。
ε=.05/.01/.005/.001/.0005、factor1/10。
費用同値はbinary64保存値をFractionで扱った正確な点費用→candidate IDの辞書順とする。許容差tieは作らない。
prefixに対するbias適格性の単調性は仮定しない。候補欠測を0費用にしない。

通常は軸bias `|z_candidate-z_reference|_axis`。
既存のsigned分解 `discard + PF`、又はPRの `outer_PF + finite_RTE` から、軸ごとの絶対値和を補助診断として計算できるようにした。
これは相殺を選ぶ効果を調べる**経験的な非相殺envelope**であり、厳密なbias上界のcertificateではない。
既存SOURCE・契約から全候補に認証済みのbias上界は得られない。
**厳密上界による適格性は未解決の科学的項目**としてnull/UNRESOLVEDを保持する。
分解未保存の候補を落として「全候補に対する保守的最良」を表示しない。厳密保証を新設するには別の科学的判断が必要。

shotは既存 `d0_frozen_shot_rule_v1.py` / `d0_saved_value_reanalysis_v1.accounting` を再利用する。
各軸α=.025、`h=ε/√2-|bias|-factor*u`、`log_B_upper=log_B+factor*u_log`、
`N=ceil(2 B_upper² / h² * log(80))`。経験的u、整数算術順序、u/h≤.01、log-domain-only時の未解決扱いを維持する。
no-cancellation診断でもuを別に加える。u・局所数値一致を数学的な誤差証明へ変更しない。

## 実装とS2/S4の正しさ

[契約・候補](../../src/trotterlib/track_a_v1_contract.py)、[runner](../../scripts/track_a_v1_runner.py)、
[signal adapter](../../scripts/track_a_v1_signal.py)、[compile adapter](../../scripts/track_a_v1_compile.py)、
[費用モデル・比較会計](../../scripts/track_a_v1_costs.py)、[private設定materializer](../../scripts/prepare_track_a_v1.py)。
数値部分はpermit確認後に限り固定provider `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7` をimportする。
既存full preparationのblockを切り出すprefix viewを使い、元preparationのhashで新prefixを偽装しない。
新SCF/DF/state・Gaussian再生成・samplingは不要。新prefix viewは新規科学signalの入力であり、実行は今回行わない。

S2は正順half→逆順half、S4はglobal Yoshidaのw/(1−2w)/w、負時間pieceを含む元scheduleを使う。
元signalのordinary scheduleとnative directional実装は別表現なので、将来の両軸control照合を必須とする。
人工X+Zの2×2行列でS2の二次、S4の四次収束を確認した。scalarは全Tに1回だけのrelative phase。
B0Iはfull-Fock tail identityの解析回転でB0から得る。full-H referenceは回転しない。
既存rotated-allowance規則とexact stored-coefficient算術を再利用し、新保証を付けない。

## 線形費用モデルと4軸上限

S2は `a+b*q+d*p*q` を保存実測anchorからFraction最小二乗で凍結する。
method/formula/axis/control/compiler/profile/phase規約をstratumとし、3個以上の非共線anchorを要求する。
anchor残差も各軸2%未満が必要。B0/B0Iを自動的に同費用にせず、SOURCE/旧profile/現profileを混ぜない。
S4は別のschedule-composition仮説 `a+3*b*q+3*d*p*q` とし、S2測定済みとは表示しない。
この仮説を使う前に元nativeの未merge schedule、scalar、phase、compilerとprofile対応のreceiptが必要。

signalのfactor1で1精度以上適格なS4候補から、B0は(p,q)辞書順最小、B0Iは最大を選び、各両軸を検証する。
全domain適格ならB0 p11/q1、B0I p15/q8。該当familyに適格候補がなければ、その検証・費用推定は不要。
**合計最大4軸**。新compile費用を見て検証候補を選び直さない。
各軸 `abs(C_model-C_measured)/C_measured >= .02` で直ちにSTOPする（ちょうど2%も停止）。
4軸平均・総費用で誤差を相殺しない。model SHA変更・identity不一致・欠測・非正値もSTOP。
検証後refit、自動追加compile、別compiler/control/phaseへの転用はしない。
4軸の一致は全prefixでの厳密誤差保証ではなく、宣言した補間政策の限定検査。
全候補へ同政策を適用できない場合は費用未解決を残す。

実測はMEASURED_COMPLETE、検証済み推定はVALIDATED_MODEL_ESTIMATE、未取得はMISSING/UNRESOLVED。
表示には費用の来歴を残し、eligibleな費用欠測があれば全domain最小の報告を停止する。
将来計算できる量は5精度の決定論最良、PR比、旧/細格子差、S4追加差、緩/中間/厳精度での優劣。
今回は新しい最良値・費用比・候補再順位を取得していない。

## 予算・grant案・gate

将来signal stageの新規call：B0 signal52、B0I解析overlay52、prefix別exact-truncated reference pair9、full reference pair1、
reference action400、tail scalar9、outer S2診断8、snapshot load1、archive decode1。
primitive validation20,000上限、各cell deterministic action100,000上限。
compile stageは新native evolution最大2、measured wrapper最大4、control probe最大8、compile最大4。
stageごとに停止し、signal完了からcompileへ自動移行しない。

未承認STOP ceilingは各stage process1/thread1/CPU1枠、AS32GiB/RSS9GiB、CPU8時間/wall12時間、
output512MiB、stage保存枠2GiB。これは完走予測・実測速度ではない。旧H6 familyのRSS約6GiBを参考に保守的に設定した。
CPU/RAMのfresh admission、外部RSS/PSI監視、profile34000余fileの最終hash、source/input不変、owned PID/reap・既存job非干渉を実行前に再確認する。
今回のhost snapshotには高いloadがあり、将来CPUを事前予約しない。

private configは未認可で具体化済み。公開[入力binding](../../artifacts/resource_applicability/track_a_v1_h6_readiness/2026-10-16/input_binding_v1.json)にbytes/SHAを記録。
新grantは作らず、`approved=false`、`runtime_authorization=false`、CPU[]、command=null。
旧grant/ledger/消費markerは変更しない。新scopeのone-shot lockはoutput外、retry/resume禁止、失敗分をrefundしない。
科学gateは厳密bias保証の未解決明示、model profile移行の証拠、S4仮説の凍結。運用gateは新明示承認・最終SOURCE/config固定・fresh資源・監視である。

## 検証・保全・停止

[人工テスト](../../tests/test_track_a_v1_readiness.py)はstdlibのみ、実H6入力を使わない。件数・wall/RSSは証跡JSONを正本とする。
新signal/compile/native full-wrapper・trajectory・grant消費0、H5作業への操作0、サブエージェント0。
著者である主担当の技術検査であり、別担当の独立レビューとは呼ばない。
Claude資料は今回の利用者指示の範囲で採用し、H5停止やV3/V4実行を自動採択しない。
旧 `check_validation_manifest.py` は起点46c06e56のmanifest自体にある拡張キーも拒否する。
この既存不整合は今回の変更前後で再現した。今回のentry・旧entry不変・リンク・bytesの限定検査は別に行い、全体CI成功とは表示しない。
commit・pushとremote bytes/SHA確認の後、**MANDATORY STOP**。実計算は範囲と予算への別途明示承認後に限る。
