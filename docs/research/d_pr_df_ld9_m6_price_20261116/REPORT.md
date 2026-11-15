# PR-DF LD9対LD7：m6の直接compiled単価（2026-11-16 JST）

**m6の指定有限cohortでは、LD9の1 shot平均はX/Y各492,573 RZで、LD7の393,837.25 RZより98,735.75 RZ高かった。比LD9/LD7は1.25070190795817307783862。** 同じT=6.4・q=64・R=1・予定64 eventでも、full wrapperのcompiled価格は同じではなかった。

新規はLD9 normal **4 path・X/Y計8 compile**、全8 receipt有効。LD7は保存4 path・8軸receiptを再利用し、再compile0。新規信号・Q3・fit・別価格点・科学retryは0。**LD9の13 round価格表・総費用はnullのまま**で、この一点からモデルを作っていない。公開・remote一致確認後mandatory STOP。

## 実測値と比較規則

| 軸 | LD7の4 path平均 (RZ/shot) | LD9の4 path平均 | LD9−LD7 | LD9/LD7 |
|---|---:|---:|---:|---:|
| X | 393,837.25 | 492,573 | +98,735.75 | 1.25070190795817307783862 |
| Y | 393,837.25 | 492,573 | +98,735.75 | 1.25070190795817307783862 |

exact値はLD7=`1575349/4`、LD9=`492573`、差=`394943/4`、比=`1970292/1575349`。軸別に有効な4 pathのcompiled RZを算術平均し、X/Yはpoolしていない。Sは元pipelineが同一eventから保存する診断量であり、Cに再加算しない。

| 分割 | cohort | path index | X compiled RZ | Y compiled RZ |
|---|---|---:|---:|---:|
| LD7（保存） | D_PR_DF_LD7_PRICE_CAL_20261116_V1 | 0 | 392,817 | 392,817 |
| LD7（保存） | 同上 | 1 | 394,877 | 394,877 |
| LD7（保存） | 同上 | 2 | 394,134 | 394,134 |
| LD7（保存） | 同上 | 3 | 393,521 | 393,521 |
| LD9（新規） | D_PR_DF_LD9_M6_DIRECT_PRICE_20261116_V1 | 0 | 493,211 | 493,211 |
| LD9（新規） | 同上 | 1 | 492,415 | 492,415 |
| LD9（新規） | 同上 | 2 | 492,159 | 492,159 |
| LD9（新規） | 同上 | 3 | 492,507 | 492,507 |

**二つの別cohortによる記述的比較**であり、同じindexのLD7/LD9をpaired標本にはしていない。各分割内では同一pathのevent列をX/Yで共有する。旧backendの「paired path price」は同一pathのS/C保存を指し、分割間のpairingではない。各path・軸のseed、event digest、S/C、fingerprint、元receiptと実行SOURCEは[COMPARISON.json](COMPARISON.json)と[PATH_PRICES.csv](PATH_PRICES.csv)で追跡できる。

## 同一条件と分割固有条件

保存H4 linear1.00Å/STO-3G/tol-only DF10、元H/state/snapshot、元fragment/basis順、S2・Δ=.1・K2・共通κを維持。m6はT=6.4・q=64・R=1。price quantityはP0の**測定付きfull controlled Hadamard wrapper全体のcompiled RZ/shot**。symmetric-directional control、CONTROLLED_ZZ_CX_CRZ_CX_V1、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17を元LD7経路のまま使用した。保存LD7と新LD9のH/state/snapshotとprice_profileの一致を元requestから照合している。

| 条件 | LD7 | LD9 |
|---|---|---|
| configuration | ce9c7f08234089f0b50fd786c20403fe02c5de9073e89d6d35a12ee9d966c5e1 | ea087a7e3b311be7efae27558347e9eb471f616b8b55f59cbef8e506dda7a833 |
| physical ID | e7194af048772d7160c6 | 09fa97187d6eee510d0c |
| one-body | 別の決定論head、L_D本数外 | 同じ定義 |
| DF head / tail | 0..6 / 7..9 | 0..8 / 9 |
| λ_R | 0.0003129539143720037 | 1.5605698403540042e-07 |
| tail identity s_R | 4.5302064712666724e-05 | 5.193351122457893e-08 |
| 総scalar c+s_R | 2.293146549384713 | 2.2931012992535114 |

c=`2.29310124732` と各s_Rは別SOURCE。各分割の総scalarを総Tに一回だけ適用する。LD9は固定configuration460d77d8fのm6からgeneric `from_b` requestを作り、実NativeRuntimeのhead0..8/tail9・lambda・scalarをbuild前に照合した。[INPUTS.json](INPUTS.json)、[acquisition/PREPARATION.json](acquisition/PREPARATION.json)。LD7のrequestやtailをLD9として読み替えていない。

価格を生成するのは各分割自身の元normal sampler Q。configuration上の理想finite-RTE law、実装samplerの保存法則・丸め・seedを元のまま保持し、LD9のtail component/準備/時刻/K/control/lawとrequestを対応させた。各pathは固定trial_seed(own physical, cohort, index, outer, 0)の64 seed、事前固定4 path合計256 seed。新cohortはLD7と異なり、seed重複もない。特定event/orderを得る引き直し・追加pathは0。

## SOURCE固定・重複照合・取得とgate

実行・比較SOURCEは **8f0a82842481270e0581658b5294ecd59e79104f**。元LD7 helper8d4503e00のrequest/restore、ea9e6609 path worker、e8273676のprefix-capable NativeRuntime/Adapterと対応fastを再利用。新しい科学library・sampler・loweringは作っていない。SOURCEを先にcommitし、[INPUTS.json](INPUTS.json)を **e733f6beaa6471c63062ef2379f3f56cb8f2499a** で固定・pushしてから実取得した。

§10.5はf89・460dの旧候補/receipt調査を再利用し、以後の公開成果・同候補/cohort/job・SOURCE・出力先を差分確認。受入可能なLD9 m6 receiptや進行中jobは確認範囲で未同定、未解決0。PR-Pauli LD5と決定論Yoshida4の作業は別方式として除外し、他worktree/jobは変更していない。[DELTA_CHECK.json](DELTA_CHECK.json)。

固定入力を親で一回restore/prepareし、独立4 pathをfork並列、X/Yは各worker内で個別compileした。専用の空cacheを使い、new compile8・cache hit0・無断retry0。計256 eventは実観測order0のみ（すべてZZ rotation）、order2/productは未観測。保存event列は原rawに保持し、未観測を補うsamplingはしていない。

全4 path/8軸は元P2・LD7と同範囲のinput/preparation/request/lowering/compiler/fingerprint/metrics gateを通過。実full-wrapper statevectorとnumerical action gateは元P2と同じく未実施で、新しい作用保証を付けない。人工prefix9試験は実取得leafとwrapper/receipt経路をfake compilerで通した技術検査であり、H4での性能証明ではない。旧保存H4 P0/operator数値再検査suiteは実行0。[TESTS.txt](TESTS.txt)、[CHECKS.json](CHECKS.json)。照合は主担当自身で、独立レビューではない。

取得wall248.674s、worker CPU合計968.346s、単一worker最大RSS1,082,800 KiB。外側entry wall250.190s、全worker/entry exit0・signalなし。協調stage logにsample/structural/compose/X,Y wrapper/fingerprint/compile/保存検査の開始終了と所要時間を保存。過去のLD7実行と負荷条件を揃えた速度benchmarkではない。共有設定変更や周期faulthandlerは0。

## 固定成果と終了範囲

- LD9原raw公開：**6392aefdad0438e619cba0c0e9e55d170eb99a49**、当directoryの `acquisition/RESULT.json`、`acquisition/m6/p0..3/PATH_RECEIPT.json`、軸別 `cosine.json` / `sine.json`。
- LD7参照：**5c5d87ab32d03bb107a9a6bfed3cba253aef2aa0**、`docs/research/d_pr_df_ld7_price_results_20261116/calibration/acquisition/m6/p0..3/` の元receipt。
- 比較・C入力固定commit：**d85150cd1b51f87da4e767b32cf52f532765d358**、[COMPARISON.json](COMPARISON.json) / [C_HANDOFF.json](C_HANDOFF.json)。入力・実行codeと結果公開commitは[SOURCE_LOCATORS.json](SOURCE_LOCATORS.json)で区別する。

この有限cohortのm6ではLD9のcompiled平均が高かった。head/tailの変更と異なるevent内容・compiler相殺を含んだfull-wrapper単価の比較であり、差を純粋なhead増分として分離同定していない。4 path平均を全lawのE[C]や全supportの上下界と認定せず、CI/価格モデルを作らない。LD9の13 round総費用・方式winnerは未判定、nullのまま。

SOURCE・原結果・旧STOP、LD7較正/holdout/凍結価格、LD9の保存信号/Q3を保全した。m6以外の追加取得、fit、全26価格、shot配分/総費用、新信号/Q3には進まない。[HANDOFF.md](HANDOFF.md)にロードマップ差分を記載。commit/push・remote照合後mandatory STOP。
