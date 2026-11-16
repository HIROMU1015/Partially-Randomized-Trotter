# P0 no-drop builder：固定科学範囲

利用者は診断・最小修正案 `bf7b4dc17455f20eec3d646e7e461fed51237917` を採択し、
新しい固定SOURCEによるH4 LD7・8・9のP0再評価と、合格分割の基準P1 configuration生成を承認した。
承認の証明書やgrantを新規発行しない。

- 通常入口 `scripts/prepare_pauli_candidate.py --prepare-missing --p0-builder no-drop` を各LD一回使用する。
- 各分割自身の保存DF descriptor/factorsを再利用。H4 linear1.00Å/STO-3G、保存tol-only DF10、
  weighted-Hermitian completion、full256/sector(2,2)36、同H/state/snapshot/constantを維持する。
- 元P0診断の共通・state・operator・identity/scalar・sector・Pauli確率検査を全て実行する。
  単一分割では共通51＋分割15＝66検査。別分割の実計算をcontrol名目で追加しない。
- 変更はone-body/Fermion部分和/JW部分和の局所 `+=` のみ。元factor順・積・weight・scalar・
  global EQ_TOLERANCE=1e-8、絶対Frobenius gate1e-9、既存imaginary toleranceを維持する。
  最終の非zero微小項にも新cutoffを適用しない。
- 旧P0/旧辞書/旧STOPは上書きせず、新builder IDと固定SOURCEをpayload/hash/cacheに保持する。
- gate失敗・入力不一致・非有限では部分結果を保存してSTOP。retry・閾値変更・候補変更なし。
- P0合格後は同じ通常入口から元基準S2/ΔPF=.1/K2/h=.1/M12/exactκ/P=0のcanonical configurationと
  13round scheduleのみ生成する。同要求の再入力は保存辞書/configurationの読取りのみ。
- signal/normalizer/reference/N/RMSE/sampling/compile/価格/fit/Q3/Gは全て0。

## §10.5照合

Aの旧P0 `9fa9525b3`、原因追跡 `655e74212` と結果 `bf7b4dc17`、
P0固定 `fd111f989`、P1・B2 mapper `98b6f8e1a`、通常入口公開 `6e4b6bd6a` を確認した。
D2 `194e00a8a` の候補完成入口は旧失敗LD7–9を自動retryしないが、Aから既成configurationが返れば後段へ進む。
D2最新 `69ef01cdc` までの変更はDF価格/receipt接続等であり、旧LD7–9 STOPとpauli_workspaceの入口は維持されている。本P0 builder変更と重複しない。Bのdescriptor/intake準備も重複しない。
既存診断はno-drop shadowの演算子一致であり、通常入口の新辞書生成・全P0合格の代用にはしない。
LD7/8/9の新P0生成各1件のみを不足分として実施する。他worktree/run/STOPは変更しない。

SOURCE固定後、先行分割の結果commitだけでHEADが進んだ場合は、科学SOURCEが祖先であり、
使用する数値builder・依存関数・入口のbytesが固定SOURCEと同じである条件で継続する。
新SOURCEでの係数変更や科学retryを認めるものではない。
