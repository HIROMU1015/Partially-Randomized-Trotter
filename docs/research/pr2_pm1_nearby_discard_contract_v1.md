# Track A PM-1：近接discard反証の結果前契約・実装

2026-10-04。**契約・source実装のみ。本実行は未認可、未実施。**
利用者が承認した「1. 契約固定、2. 実装・synthetic tests」だけを行う。
別source commitの固定後にも、別execution authorizationと実行前review・利用者指示が必要である。

## 目的と比較集合

[PM-0](pr2_post_m2_evidence_attribution.md)はM1のB0 rank4/5が未登録であることを確認した。
rank3不適格・rank6適格だけから、rank4/5のaccuracyとcompiled費用は推測できない。
この穴を最小限で反証する。新しいselector、sampling、synthesis policyを開発しない。
strong deterministic method一般への優位性や全prefix最適性を、この試験で証明するわけではない。

将来の対象は**H4 linear 1.00 Å、STO-3G、DF rank12、8 system qubits、同じsnapshot/state、T=0.8**。
固定fragment順、one-body correction、scalar/identity処理をM1から維持する。
B0 discard rank4/5 × q=1/2/4/8、r=K=0の8構成のみ。
deltaはT/q=0.8/0.4/0.2/0.1。追加・除外・置換・再探索を禁止する。
H4 1.30 Åの使用済みM2 snapshot、別geometry/分子/basis/PF、Track Bに触れない。

比較相手は保存済みM1-A/M1-B1の次の5構成だけから固定する。

- B2 rank3/q1/r4/K2、B2 rank3/q1/r8/K2。
- B0 rank6/q1/r0/K0、B1 rank12/q1/r0/K0、B3 rank0/q8/r32/K4。

保存JSONのSHAとevidence commit `b6e65c6123475add5e620ec1064f361378bead95`のblob一致を要求する。
signal/cost candidate identityとsignal-record fingerprintを照合する。既存B2/B3を再sample/compileしない。
新8構成＋旧5構成のpoint比較であり、held-out再最適化、全finite gridや一般methodのwinnerとは呼ばない。
元M1/M2 artifact、source、authorization、正式statusを変更しない。

## 数値・費用の契約

full-H targetは固定M1-A JSONの保存値を使い、新しいfull-H ground stateを求めない。
discard-prefix二次PF state actionを各構成一度だけ計算する。
準備ではprefix0..4と同じone-body/scalarだけを構築し、最大6 component eigensystemsを共有する。
exact truncated-H signalは追加計算しない。
保存するbiasは**full-Hに対するdiscard＋PF総bias**。pure discardとpure PFはnullであり、混同しない。
state-action normは1から1e-10以内、signal radiusは1+1e-10以下、nonfiniteはgate failure。

epsilon_complex=0.05、epsilon_axis=0.05/sqrt(2)、alpha_axis=0.025、normalization=1。
各軸でa=epsilon_axis-|mean_axis-target_axis|、a<=0ならaccuracy不適格、shots=null。
a>0ならN_axis=ceil(2/a^2*log(2/0.025))。M1のcorrected Hoeffding式と同じ。
ineligibleは科学結果であり実装failureではない。8構成とも記録する。

全8件のsignal ledgerを確定後、各cosine/sine full measured Hadamard wrapperをexact compileする。
不適格構成もbaseline completeness用にcompileするが、matched-accuracy workには入れない。
controlled evolutionへ再controlしない。状態準備を除き、ancilla readoutとmeasurementを含める。
既存boundary_optimized builder、identity phase、固定二次DF-PFを使い、compiler policyを変えない。

compilerはQiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routingなし。
Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1、openfermionpyscf0.5、PySCF2.7.0を維持する。
CPU単一process、BLAS各1、PYTHONNOUSERSITE=1、PYTHONDONTWRITEBYTECODE=1、PYTHONPATH=src。
環境install/変更、GPU query/useはしない。

primaryはN_real*C_cosine,RZ+N_imag*C_sine,RZ。6指標のmatched-workを保存する。
新B0／各固定referenceのprimary ratioはpoint comparisonとして保存する。
既存random costは32 trajectoryの点推定であり、point ratioから厳密winnerやformal CIを主張しない。
10%を新しい自動GO基準として導入しない。研究主張の維持・縮小は結果後の外部reviewで判断する。
epsilon/P sweep、追加trajectory、energy/RPE接続をPM-1へ混ぜない。

## 固定上限と停止

- candidate signal最大8、full wrapper build/transpile最大16、development hash/load各一回。
- random trajectory/sample、held-out access、量子shot、GPU操作、cache reuseは0。
- CPU process最大1。各cellでexact trajectory1、wrapper2、最大untranspiled size1,000,000。
- 各cellのplanned instruction applications最大2,000,000、build/transpile request各2。
- 一回限り。resume/retry、別output/root/serverでの同authorization再利用、cache移送は禁止。

wrapper keyはplan fingerprint、actual source commit/全source SHA、candidate fingerprint、axis、
trajectory index0/seed=null、compiler、wrapper semanticsを含む。16 unique keysを事前固定する。
cache/checkpointを再利用しないため、旧M1/M2 runtimeにはアクセスしない。
sourceとsnapshotのidentityが違えば、代替データや再構築で救済しない。

成功は `PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW`、失敗は `IMPLEMENTATION_GATE_FAILED` のみ。
どちらもmandatory STOP、next_stage_authorized=false、research_decision=null。
科学runnerはCONTINUE/NARROW/STOP等の研究判断を出さない。
failureではpartial/null ledgerとattempt/reservation/completedを分離し、未解決予約のactual件数はnull。
不完全結果を0計算や科学的NOT_SUPPORTEDに読み替えない。成功markerは成功だけで作る。
exclusive registryは固定root内の一回制限であり、全checkout横断global lockではない。

## 実装・検査・次のbarrier

- [contract module](../../src/trottertracks/resource_applicability/pm1_discard_contract.py)：stdlib-only。
- [future science module](../../src/trottertracks/resource_applicability/pm1_discard_execution.py)：科学importはprivate boundary後。
- [zero-science planner](../../scripts/resource_applicability/run_pr2_pm1_discard_contract.py)：stdoutだけ、NPZ pathはliteral。
- [future runner](../../scripts/resource_applicability/run_pr2_pm1_discard.py)：別committed authorizationなしでは実行不可。
- [synthetic/mock tests](../../tests/tracks/resource_applicability/test_pm1_discard.py)。
- [preparation artifacts](../../artifacts/resource_applicability/pr2_pm1_discard_preparation/2026-10-04/)。

前の準備attemptでは確認scriptの範囲ミスにより、追跡NPZ4件をstat/hashしてしまった。
load、signal、compileは0。この違反は利用者へ報告し、停止後に文書-only条件で再開指示を受けた。
今回のauditでは前attemptの4件と再開後の0件を別記録し、過去もaccess0だったとは主張しない。
使用済みM2 geometryは今後もfresh held-outにはならない。

今回のテストは合成/mockと保存JSON回帰のみ。tiny2-qubit DFのsynthetic wrappersをH4科学compileと区別する。
全repository tests、既存science runner、元runtime validatorは実行しない。
準備の限定7-file suiteは201 passed、fail/skip 0。NPZ/NPY/pickleおよびruntimeのopen/stat/lstatを
pytest import前に拒否する診断guardで、禁止pathへのアクセス試行0を確認した。
これはlocal testであり、OS sandbox、immutable CI、科学的PM-1結果ではない。
正確なcommand・UTC時刻・stdoutと初期実装testの修正履歴はpreparation auditへ保存する。
source commit→そのblobに結合したsealed plan→別authorization→final review→明示launchの順序を守る。
今回はauthorizationを作らず、PM-1科学結果も研究判断も作らない。
