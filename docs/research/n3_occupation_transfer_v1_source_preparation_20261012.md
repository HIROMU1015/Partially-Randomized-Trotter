# N3 occupation-aware v1：実装・実行準備

2026-10-12 JST。完了済み設計科学レビューのGOを受領し、限定追補を実装した。
状態：`N3_OCC_SOURCE_READY_AWAITING_EXPLICIT_BATCH_AUTHORIZATION`。
新8入力の科学batchは0、Mandatory STOP、中心テーマnull。N1追加探索停止・N2保留を維持。

## 1. 固定資料と実装

固定設計c2955ba0と入力SHA
`c8df923eebfd999bf389ece2eb1728e7897c90e0950b51137785c708404489f4`（48,201 bytes）を保持。
元v1・T3・旧12入力の結果を変更しない。
[追補v2](n3_occupation_transfer_amendment_v2_20261012.md)が実装済みの比較・counter・費用契約。

- [generator](../../src/trottertracks/representation_exploration/n3_occupation_transfer_v1.py)：係数trial、三下界、四方法、継承/fallback、cost DAG、条件付きseparator。
- [独立checker](../../src/trottertracks/representation_exploration/n3_occupation_transfer_check_v1.py)：stdlibのみ、trialと下界を再計算、元行列のLDL/全要素再構成、binding/継承/gain/費用counterを検査。
- [runner](../../scripts/run_n3_occupation_transfer_v1.py)：read-only preflightと別認可付きone-shotを分離。
- [保存検査入口](../../scripts/verify_n3_occupation_transfer_v1.py)：既存ledgerの検査だけ。
- [synthetic packet作成](../../scripts/prepare_n3_occupation_transfer_v1_source.py)：固定3fixtureのみ、transfer packを読まない。
- [24 synthetic tests](../../tests/test_n3_occupation_transfer_v1.py)。

## 2. レビュー修正への対応

raw96recordの独自生成を保持し、既存全三行のPSD-valid候補をc_starで再検証する。
弱い下界でaffine/boxだけが不成立だった候補も含める。
source/input/U/δとκ/μ/τ/Zを固定し、新数値proposalを継承へ混ぜない。
gainはraw_OCC_at_c_star AND NOT transported_BASE_OR。
raw、same-method inheritance、他法fallback、operational unionを別fieldで保存する。
fallbackにはNORMのκ0/τ0/Zvvと旧証拠のτμ/Z0という実際のOCC証拠を付ける。

テスト仕様はκ=0でNORMと一致、ν=1で旧LMI族と一致へ修正。
ν=1でNORMが落ちてもOCC/旧LMIが通るcontrolと、μ0による元行列の一致を検査する。
finite生成器同士の発見集合が必ず等しいとは仮定しない。

source-bound planはsource commit後に作成する
[execution plan](n3_occupation_transfer_v1_execution_plan_20261012.json)。
planは準備状態を記録するだけで実行を認可しない。
source/input/planへbindingした別authorizationを受領した場合だけrun1を作る。

## 3. 検証済みと未評価の区分

24 test methodsのfixture scopeで、改変τ/Z/U/c・source binding・PSD flags・継承改変の拒否、
zero pivot/row、400-bit tiny negativeのzero tolerance、rational cap、stage予約、
m=9 synthetic PSD/eig kernel、cache無効対cached一致、二worker spawn、authorization拒否を検査。
一次ledger後のseparatorについて両端点・nonzero/PSD・NORM不成立を確認する。
G=0、G=1、G≥2を区別し、finite gainとfamily separationを分ける。

[現在sourceのsynthetic保存検査](../../artifacts/n3_occupation_transfer_v1/2026-10-12/preparation/synthetic_saved_audit.json)
は87,765 checks PASS。固定fixtureはsynthetic_normal、synthetic_nu1、synthetic_gainの3個。
これは移送成績、新しい科学batch、新8入力の成功例ではない。
初期の未commit synthetic probeもdevelopment_before_*へ保持し、現在sourceの検証と区別する。

固定8入力はbyte/hash/schemaを確認するだけで、U/c/κ/保証predicate/受理は未評価。
旧12入力・T3の再探索、many-body truth、ground/gap、quantum compile/samplingは0。
数学的soundnessの証拠と、有限generatorの完全性・取得費用の一般的有用性は別である。

## 4. 性能準備と実測範囲

CPU32/affinity32、利用可能memory約51.8 GiBを確認。GPU driver接続不能、GPU0。
Python3.12.3、NumPy1.26.4、binary64 proposal/exact rational acceptance。
旧の小行列/係数処理を参考に新versionへ実装し、元sourceは変更しない。
case内の数値matrix・PSD・旧λ/LDL・trial diagonalをcache、独立caseを最大2workersで処理。
BLAS/OMP/MKL/Numba/Rayon各1thread、spawnをsyntheticで確認。

現在sourceの固定3fixture packet観測：wall約0.254秒、parent CPU約0.223秒、children CPU約0.240秒。
計測fixtureは実one-body＋非負density、外部1mode、active2/3、ν1/2、full3/4、scale1、δ=.004。
geometry・化学basis・DF rank・L_D・time-evolution/PF窓は適用なし。係数はsynthetic_packに全保存。
parent max RSS 50,659,328 bytes、最大child max RSS 46,661,632 bytes。
parent/helperを含むsampled aggregate peak 142,606,336 bytes。
目標poll50ms、観測最大gap約95.0ms。瞬間peakやhard OS aggregate上限の実測とは呼ばない。
CPU/RSSはこのsource-preparation processの観測で、単独科学算法のcold測定ではない。
速度倍率・大系scaling・新8入力のruntimeをこの小fixtureから推定しない。

段階別予約の最大eig817/PSD448はcase上限850/512内。
OCC spectral pair180と外部pair12は別counter。checker費用もproducer予算と別。
4096-bit guardはexplicit rational operands/resultsに適用、Fraction内部一時整数は未計測。
worker address-space4 GiB、sampled aggregate budget8 GiB、JSON32 MiB、output64 MiB、log8 MiB。
wall/CPU-time capはなし。raw/factor wallと経路再構成費用を分け、後段cacheを無料にしない。

## 5. 停止位置と次の手順

source・契約・検証資料を固定し、source-bound plan、hash/schema preflight、公開・再取得を確認する。
source commit `83f51cd8a4848572c66a5535c67d0d0a5f3868cd`、plan SHA
`a825ece44d9bf8266a9c83d390572139d3799b748419b644515d79230e5c0d0f`。
preflight1692 schema checks PASS、new8のpredicate/eigensolver/ground callsはいずれも0。
本taskの終端は実行認可待ち。詳細科学レビューの同内容再実施は要求しない。
次の判断は準備が揃ったかの確認と、一回の固定8入力batchへの明示認可。
sourceに従った科学batchは未実行。run1 directoryは作成しない。
新結果取得後に研究上の意味をGPTへ返す。

既存global manifestのrelevant_commits[7].description欠落FAILを保持する。
local synthetic/source checksをglobal PASS、immutable CI、外部科学再現へ読み替えない。
